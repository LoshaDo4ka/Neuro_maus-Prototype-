"""
NeuroMouse — мульти-сессионный декодер (Stage III-B): обучение на пуле сессий.

Зачем это нужно
--------------
Декодер Этапа III и раздел E обучались на ОДНОЙ сессии (~200 испытаний при
778 нейронах), то есть в режиме p ≫ n. Единственный по-настоящему большой
резерв качества — больше данных. В DANDI 000017 (Steinmetz et al., 2019)
39 сессий от 10 мышей, но просто «сложить испытания в кучу» нельзя: признак
«нейрон × бины» привязан к конкретному набору нейронов сессии, а он у каждой
сессии свой.

Как это обходится
-----------------
Переходим к признакам **«структура мозга × бины»**: для каждой структуры
(из `electrodes.location`) считаем частоту спайков на один нейрон структуры
в каждом бине окна

    r_{s,b}(t) = -------------------------------------------------
                 n_units(s) · ширина_бина

Такой признак определён одинаково во всех сессиях, поэтому испытания
объединяются в один пул (тысячи испытаний вместо ~200), а вместе с ним
появляется честная проверка обобщения на НОВУЮ сессию и НОВУЮ мышь.

Мелкие структуры (74 акронима CCF) при сборке пула укрупняются в системы
(REGION_GROUPS: кора, гиппокамп, таламус, средний мозг, стриатум, лимбическая
система), метки `root`/`unknown` отбрасываются. Иначе признак «структура ×
бины» привязан к зоне имплантации конкретной мыши и между животными не
переносится.

Что считается
-------------
  remote      — извлекает по одной сессии на мышь (10 сессий, ~0.7 ГБ): по каждой
                сессии читает из DANDI только нужные датасеты (спайки,
                электроды, испытания) прямо по HTTP и кладёт признаки в кэш
                (npz). Полная выкачка NWB не нужна: ~3/4 файла — это LFP,
                который нам не требуется;
  remote-all  — то же, но для всех 39 сессий (~3.5 ГБ спайков);
  train       — собирает пул и считает три схемы оценки для каждой цели:
                • pooled        — StratifiedKFold на объединённых испытаниях;
                • cross-session — LeaveOneGroupOut по сессиям (обобщение на сессию);
                • cross-subject — LeaveOneGroupOut по мышам (обобщение на мышь).

Метрики: accuracy, balanced_accuracy, f1 (среднее ± std по фолдам). Дополнительно
печатается уровень случая (доля мажоритарного класса).

Артефакты (results/mouse_stage/multisession/)
---------------------------------------------
  cache/<сессия>_w<окно>_b<бины>.npz — кэш признаков одной сессии
  ms_dataset.json                    — состав пула (сессии, мыши, испытания)
  ms_performance.csv / .png          — точность по схемам оценки и моделям
  ms_progress.json                   — сколько сессий уже извлечено

Запуск из корня проекта (окружение neuromouse):
    python -m src.mouse_multisession_decoder remote     # по одной сессии на мышь (HTTP)
    python -m src.mouse_multisession_decoder remote-all # все 39 сессий (HTTP)
    python -m src.mouse_multisession_decoder train      # обучение на пуле сессий
    python -m src.mouse_multisession_decoder plot       # график точности
    python -m src.mouse_multisession_decoder all        # remote + train + plot
    python -m src.mouse_multisession_decoder download   # полная выкачка NWB (~15 ГБ, резерв)
    python -m src.mouse_multisession_decoder extract    # извлечение из локальных NWB
"""

import json
import os
import tempfile
import threading
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_TMP = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_TMP / "nm_mpl"))
# Маленькие данные + Windows: порождение joblib-процессов дороже самого обучения.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

import aiohttp
import fsspec
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, MOUSE_STAGE_DIR, RANDOM_STATE

# ----------------------------------------------------------------------
# Параметры
# ----------------------------------------------------------------------
STEINMETZ_DIR = DATA_DIR / "steinmetz"          # сюда качается DANDI 000017
OUTPUT_DIR = MOUSE_STAGE_DIR / "multisession"
CACHE_DIR = OUTPUT_DIR / "cache"

# Загрузка датасета: CLI `dandi download` в песочнице виснет без диагностики, а
# соединение к S3 нестабильно (всплеск, затем простой). Поэтому качаем напрямую
# через REST API: несколько соединений, мелкие блоки и докачка с места обрыва —
# таймаут на блоке стоит не потерянных данных, а всего одного переподключения.
DANDISET_ID = "000017"
DANDI_VERSION = "draft"
DANDI_API = "https://api.dandiarchive.org/api"
DOWNLOAD_WORKERS = 12                            # замер: 6 потоков ≈ 2.8 МБ/с, 12 ≈ 3.2 МБ/с
CHUNK_BYTES = 1 << 16                            # 64 КБ: крупный read «съедает» всплеск
SOCKET_TIMEOUT_S = 20
MAX_ATTEMPTS_PER_FILE = 150

# Удалённое извлечение: вместо полной выкачки NWB (15 ГБ, из которых 3/4 — LFP,
# который нам не нужен) читаем по HTTP только нужные датасеты HDF5. Объём
# передачи падает ~в 4 раза, а на диске почти ничего не остаётся.
REMOTE_BLOCK_BYTES = 1 << 20                     # блок кэша fsspec (1 МБ)
# Замер в песочнице: 1 соединение в простое держит ~6.7 МБ/с, а 6 потоков
# «съедают» себя — срабатывает троттлинг и суммарная скорость падает до 0.5 МБ/с.
REMOTE_WORKERS = 2
REMOTE_RETRIES = 3
TRIAL_COLUMNS = [
    "visual_stimulus_time", "go_cue", "feedback_time",
    "visual_stimulus_left_contrast", "visual_stimulus_right_contrast",
    "response_choice", "feedback_type",
]

WINDOW_S = 1.0                                  # окно после события, с
N_BINS = 10                                     # бинов в окне (по 100 мс)
MIN_TRIALS_VALID = 30                           # сессию с меньшим числом испытаний пропускаем
POOLED_SPLITS = 5

# Выравнивание к событию испытания (как в mouse_ai_experiments).
ALIGNMENT_COLUMNS = {
    "stimulus": "visual_stimulus_time",
    "choice": "go_cue",
    "feedback": "feedback_time",
}
TASK_TITLES = {
    "stimulus": "сторона стимула",
    "choice": "моторный выбор",
    "feedback": "исход испытания",
}

# Укрупнение структур: 74 мелких акронима CCF -> крупные системы.
# Разные мыши имплантированы в разные зоны, поэтому «структура × бины» плохо
# переносится между ними. Грубое деление даёт признак, который означает одно и
# то же у всех животных. Частота системы — средневзвешенное по числу нейронов.
REGION_GROUPS = {
    "cortex": ["MOs", "MOp", "SSp", "SSs", "VISp", "VISa", "VISam", "VISpm", "VISl",
               "VISrl", "ACA", "PL", "ILA", "ORB", "ORBm", "RSP", "AUD", "TT", "DP"],
    "hippocampus": ["CA1", "CA2", "CA3", "CA", "DG", "SUB", "POST"],
    "thalamus": ["TH", "LP", "MD", "LD", "PO", "POL", "VPM", "VPL", "VAL", "RT", "PT",
                 "MG", "LGd", "SPF", "CL", "LH"],
    "midbrain": ["MRN", "MB", "SCig", "SCsg", "SCm", "SCs", "PAG", "SNr", "ZI", "RN",
                 "IC", "APN", "NB"],
    "striatum": ["CP", "ACB", "LS", "LSc", "LSr", "GPe", "SI", "OT", "MS"],
    "limbic": ["BLA", "BMA", "MEA", "COA", "OLF", "PIR", "EP", "EPd"],
}
# Метки, которые структурами мозга не являются: корешок пробы и неопознанные.
DROPPED_REGIONS = {"root", "unknown"}

_REGION_TO_GROUP = {r: g for g, rs in REGION_GROUPS.items() for r in rs}


def coarse_region(name):
    """Акроним CCF -> крупная система; None, если метку надо выбросить."""
    if name in DROPPED_REGIONS:
        return None
    return _REGION_TO_GROUP.get(name, "other")


# ======================================================================
# Загрузка датасета DANDI 000017
# ======================================================================
def list_assets():
    """Список ассетов датасета (путь, размер, asset_id) через REST API."""
    url = (f"{DANDI_API}/dandisets/{DANDISET_ID}/versions/{DANDI_VERSION}"
           f"/assets/?page_size=100")
    out = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        out += [{"path": a["path"], "size": a["size"], "asset_id": a["asset_id"]}
                for a in page["results"]]
        url = page.get("next")
    return out


def download_asset(asset, root):
    """Скачать один .nwb, восстанавливаясь после обрывов. Возвращает (статус, байт)."""
    dest = root / asset["path"]
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == asset["size"]:
        return "skip", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (f"{DANDI_API}/dandisets/{DANDISET_ID}/versions/{DANDI_VERSION}"
           f"/assets/{asset['asset_id']}/download/")
    got = 0
    for _ in range(MAX_ATTEMPTS_PER_FILE):
        have = part.stat().st_size if part.exists() else 0
        if have >= asset["size"]:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=SOCKET_TIMEOUT_S) as r:
                # Если сервер проигнорировал Range (200 вместо 206) — пишем с нуля.
                mode = "ab" if (have and getattr(r, "status", 200) == 206) else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
        except Exception:                                # noqa: BLE001 — обрыв: продолжаем с места
            time.sleep(0.5)
    if part.exists() and part.stat().st_size == asset["size"]:
        part.replace(dest)
        return "ok", got
    return "fail", got


def _disk_total(root):
    """Сколько байт уже лежит на диске (готовые файлы + недокачанные .part)."""
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def download_all():
    """Скачать все сессии датасета (параллельно), с журналом прогресса."""
    root = STEINMETZ_DIR / DANDISET_ID
    root.mkdir(parents=True, exist_ok=True)
    assets = list_assets()
    total = sum(a["size"] for a in assets)
    print(f"Датасет {DANDISET_ID}: {len(assets)} файлов, {total / 1e9:.1f} ГБ -> {root}")

    lock = threading.Lock()
    state = {"done": 0, "start": time.time(), "stop": False}

    def monitor():                                       # прогресс раз в 30 с по байтам на диске
        while not state["stop"]:
            time.sleep(30)
            have = _disk_total(root)
            el = time.time() - state["start"]
            rate = have / max(el, 1e-9)
            print(f"    ... {have / 1e9:.2f}/{total / 1e9:.2f} ГБ, "
                  f"{rate / 1e6:5.2f} МБ/с, ETA {(total - have) / max(rate, 1e-9) / 60:5.0f} мин",
                  flush=True)

    def worker(a):
        status, _ = download_asset(a, root)
        with lock:
            state["done"] += 1
            print(f"  [{state['done']:>2}/{len(assets)}] {a['path'].split('/')[-1]:<45} {status}",
                  flush=True)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        list(ex.map(worker, assets))
    state["stop"] = True
    have = _disk_total(root)
    print(f"Готово: {state['done']}/{len(assets)} файлов, {have / 1e9:.2f} ГБ "
          f"за {(time.time() - state['start']) / 60:.1f} мин")


# ======================================================================
# Чтение сессий и построение признаков
# ======================================================================
def session_paths():
    """Все .nwb скачанных сессий DANDI 000017 (отсортированы по имени)."""
    if not STEINMETZ_DIR.exists():
        return []
    return sorted(STEINMETZ_DIR.rglob("*.nwb"))


def _unit_regions(units_df, electrodes):
    """Структура мозга каждого юнита: пиковый канал -> electrodes.location."""
    loc = electrodes["location"]
    regions = []
    for v in units_df["peak_channel"]:
        arr = np.asarray(v).ravel()
        j = int(arr[0]) if len(arr) else -1
        regions.append(str(loc.loc[j]) if j in loc.index else "unknown")
    return np.asarray(regions)


def build_targets(trials):
    """Маска и метки для трёх целей (как в одиночно-сессионном модуле)."""
    cl = trials["visual_stimulus_left_contrast"].to_numpy(dtype=float)
    cr = trials["visual_stimulus_right_contrast"].to_numpy(dtype=float)
    choice = trials["response_choice"].to_numpy(dtype=float)
    feedback = trials["feedback_type"].to_numpy(dtype=float)

    return {
        "stimulus": {
            "mask": (~np.isnan(cl)) & (~np.isnan(cr)) & (cl != cr),
            "y": (cr > cl).astype(int),
        },
        "choice": {
            "mask": np.isin(choice, [1.0, -1.0]),
            "y": (choice > 0).astype(int),
        },
        "feedback": {
            "mask": np.isin(feedback, [1.0, -1.0]),
            "y": (feedback > 0).astype(int),
        },
    }


def _unit_bin_counts(spk, onsets, window_s, n_bins):
    """[испытания × бины] — число спайков одного нейрона в каждом бине окна.

    Кумулятивные счётчики на границах бинов считаются бинарным поиском
    (searchsorted), а счётчик в бине — их разность: O(испытаний·log(спайков)).
    """
    if spk.size == 0:
        return np.zeros((len(onsets), n_bins), dtype=np.float32)
    edges = np.linspace(0.0, window_s, n_bins + 1)
    lo = np.searchsorted(spk, onsets, side="left")
    cum = np.stack([np.searchsorted(spk, onsets + e, side="left") - lo for e in edges], axis=1)
    return np.diff(cum, axis=1).astype(np.float32)


def region_bin_rates(spike_times, unit_regions, onsets, window_s, n_bins):
    """Признаки «структура × бины»: частота спайков на нейрон структуры (спайков/с).

    Возвращает (rates, unit_counts): rates — {структура: [испытания × бины]},
    unit_counts — {структура: число юнитов} (нужно для нормировки на нейрон).
    """
    sums, counts = {}, {}
    for spk, reg in zip(spike_times, unit_regions):
        c = _unit_bin_counts(np.asarray(spk, dtype=float), onsets, window_s, n_bins)
        sums[reg] = c if reg not in sums else sums[reg] + c
        counts[reg] = counts.get(reg, 0) + 1
    bin_w = window_s / n_bins
    rates = {r: (sums[r] / (counts[r] * bin_w)).astype(np.float32) for r in sums}
    return rates, counts


def extract_session(path):
    """Извлечь признаки и метки одной сессии; результат кэшируется в npz.

    Кэш зависит от (окно, число бинов) — при смене параметров пересчитывается.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{path.stem}_w{WINDOW_S:g}_b{N_BINS}.npz"
    if cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}

    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units
        spike_times = [np.asarray(units["spike_times"][i], dtype=float) for i in range(len(units))]
        units_df = units.to_dataframe()
        trials = nwb.trials.to_dataframe()
        electrodes = nwb.electrodes.to_dataframe()
        subject = str(nwb.subject.subject_id) if nwb.subject else "?"

    regions = _unit_regions(units_df, electrodes)
    targets = build_targets(trials)

    payload = {
        "subject": np.array(subject),
        "session": np.array(path.stem),
        "n_units": np.array(len(spike_times)),
    }
    for tname, align_col in ALIGNMENT_COLUMNS.items():
        mask = targets[tname]["mask"].copy()
        onsets = trials[align_col].to_numpy(dtype=float)
        mask &= ~np.isnan(onsets)                     # нет момента события — испытание не годится
        if int(mask.sum()) < MIN_TRIALS_VALID:
            continue
        rates, counts = region_bin_rates(spike_times, regions, onsets[mask], WINDOW_S, N_BINS)
        regs = sorted(rates)
        payload[f"{tname}__regions"] = np.array(regs)
        payload[f"{tname}__counts"] = np.array([counts[r] for r in regs], dtype=int)
        payload[f"{tname}__X"] = np.stack([rates[r] for r in regs], axis=1)   # [испытания × структуры × бины]
        payload[f"{tname}__y"] = targets[tname]["y"][mask].astype(int)

    np.savez_compressed(cache_file, **payload)
    return payload


def extract_all(force=False):
    """Обойти все скачанные сессии и заполнить кэш признаков."""
    paths = session_paths()
    if not paths:
        raise SystemExit(f"Нет .nwb в {STEINMETZ_DIR}. Сначала: dandi download DANDI:000017")
    print(f"Сессий найдено: {len(paths)}")
    done = []
    for i, p in enumerate(paths, 1):
        cache_file = CACHE_DIR / f"{p.stem}_w{WINDOW_S:g}_b{N_BINS}.npz"
        if cache_file.exists() and not force:
            print(f"  [{i:>2}/{len(paths)}] {p.stem:<40} из кэша")
            done.append(p.stem)
            continue
        try:
            d = extract_session(p)
            nt = {t: int(len(d[f"{t}__y"])) for t in ALIGNMENT_COLUMNS if f"{t}__y" in d}
            print(f"  [{i:>2}/{len(paths)}] {p.stem:<40} юнитов={int(d['n_units']):>4}  испытаний={nt}")
            done.append(p.stem)
        except Exception as exc:                       # noqa: BLE001 — сессию лучше пропустить, чем упасть
            print(f"  [{i:>2}/{len(paths)}] {p.stem:<40} ОШИБКА: {exc}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "ms_progress.json").write_text(
            json.dumps({"extracted": done, "total": len(paths)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return done


# ======================================================================
# Удалённое извлечение: читаем из NWB только нужные датасеты (без выкачки файла)
# ======================================================================
def _download_url(asset_id):
    return (f"{DANDI_API}/dandisets/{DANDISET_ID}/versions/{DANDI_VERSION}"
            f"/assets/{asset_id}/download/")


def _decode(arr):
    """vlen/bytes-строки HDF5 -> numpy-массив Python-строк."""
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(arr).ravel()])
    return np.asarray(arr)


def _remote_fs():
    """HTTP-файловая система для чтения HDF5: только нужные блоки, без полной выкачки."""
    return fsspec.filesystem("http", client_kwargs={
        "timeout": aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=180)})


def _read_remote(path_url):
    """Открыть удалённый NWB и вытащить только нужные датасеты (спайки, электроды, триалы)."""
    fs = _remote_fs()
    with fs.open(path_url, "rb", block_size=REMOTE_BLOCK_BYTES) as f:
        with h5py.File(f, "r") as h5:
            spikes = h5["units/spike_times"][:].astype(np.float64)
            index = h5["units/spike_times_index"][:]
            peak = np.asarray(h5["units/peak_channel"]).ravel()
            location = _decode(h5["general/extracellular_ephys/electrodes/location"][:])
            subject = _decode(np.asarray([h5["general/subject/subject_id"][()]]))[0]
            tkeys = set(h5["intervals/trials"].keys())
            trials = {k: np.asarray(h5[f"intervals/trials/{k}"][:]) for k in TRIAL_COLUMNS if k in tkeys}
    return spikes, index, peak, location, subject, trials


def extract_session_remote(asset, force=False):
    """Извлечь признаки одной сессии прямо из DANDI (без скачивания NWB). Кэшируется в npz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = Path(asset["path"]).stem
    cache_file = CACHE_DIR / f"{session}_w{WINDOW_S:g}_b{N_BINS}.npz"
    if cache_file.exists() and not force:
        d = np.load(cache_file, allow_pickle=True)
        return {k: d[k] for k in d.files}

    spikes, index, peak, location, subject, trials = _read_remote(_download_url(asset["asset_id"]))

    # Ragged-массив спайков: index — концы отрезков по юнитам.
    bounds = np.concatenate([[0], index])
    n_units = len(index)
    spike_times = [spikes[bounds[i]:bounds[i + 1]] for i in range(n_units)]

    regions = _unit_regions(pd.DataFrame({"peak_channel": peak}),
                            pd.DataFrame({"location": location}))
    targets = build_targets(pd.DataFrame(trials))

    payload = {
        "subject": np.array(subject),
        "session": np.array(session),
        "n_units": np.array(n_units),
    }
    for tname, align_col in ALIGNMENT_COLUMNS.items():
        if align_col not in trials:
            continue
        mask = targets[tname]["mask"].copy()
        onsets = np.asarray(trials[align_col], dtype=float)
        mask &= ~np.isnan(onsets)
        if int(mask.sum()) < MIN_TRIALS_VALID:
            continue
        rates, counts = region_bin_rates(spike_times, regions, onsets[mask], WINDOW_S, N_BINS)
        regs = sorted(rates)
        payload[f"{tname}__regions"] = np.array(regs)
        payload[f"{tname}__counts"] = np.array([counts[r] for r in regs], dtype=int)
        payload[f"{tname}__X"] = np.stack([rates[r] for r in regs], axis=1)
        payload[f"{tname}__y"] = targets[tname]["y"][mask].astype(int)

    np.savez_compressed(cache_file, **payload)
    return payload


def _summarize(session, payload):
    nt = {t: int(len(payload[f"{t}__y"])) for t in ALIGNMENT_COLUMNS if f"{t}__y" in payload}
    return f"{session:<42} юнитов={int(payload['n_units']):>4}  испытаний={nt}"


def _pick_one_per_subject(assets):
    """По одной (самой лёгкой) сессии на субъекта — база для честной кросс-мышиной оценки.

    Каждая сессия — отдельная мышь со своим набором нейронов, поэтому для
    проверки обобщения «на новую мышь» нужно, чтобы каждая мышь была
    представлена, но не обязательно всеми своими сессиями.
    """
    best = {}
    for a in assets:
        subj = Path(a["path"]).parts[0]                 # 'sub-Richards/...' -> 'sub-Richards'
        if subj not in best or a["size"] < best[subj]["size"]:
            best[subj] = a
    return [best[s] for s in sorted(best)]


def extract_all_remote(force=False, limit=None, one_per_subject=False):
    """Обойти сессии датасета и заполнить кэш признаков через удалённое чтение.

    one_per_subject=True берёт по одной сессии на мышь (10 сессий, ~0.7 ГБ
    вместо ~3.5 ГБ спайков на все 39): покрывает всех субъектов и остаётся
    посильным для троттлящегося канала песочницы.
    """
    assets = [a for a in list_assets() if a["path"].endswith(".nwb")]
    if one_per_subject:
        assets = _pick_one_per_subject(assets)
    if limit:
        assets = assets[:limit]
    print(f"Сессий к извлечению: {len(assets)}; параллельно {REMOTE_WORKERS}")
    lock = threading.Lock()
    state = {"done": 0, "cached": 0, "failed": 0}

    def job(asset):
        session = Path(asset["path"]).stem
        err = None
        for attempt in range(1, REMOTE_RETRIES + 1):
            try:
                payload = extract_session_remote(asset, force=force)
                return session, payload, None
            except Exception as exc:                   # noqa: BLE001 — сессию лучше пропустить
                err = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * attempt)
        return session, None, err

    done = []
    with ThreadPoolExecutor(max_workers=REMOTE_WORKERS) as ex:
        futures = [ex.submit(job, a) for a in assets]
        for fut in as_completed(futures):
            session, payload, err = fut.result()
            with lock:
                state["done"] += 1
                if err:
                    state["failed"] += 1
                    print(f"  [{state['done']:>2}/{len(assets)}] {session:<42} ОШИБКА: {err}", flush=True)
                else:
                    done.append(session)
                    print(f"  [{state['done']:>2}/{len(assets)}] {_summarize(session, payload)}", flush=True)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                (OUTPUT_DIR / "ms_progress.json").write_text(
                    json.dumps({"extracted": sorted(done), "failed": state["failed"],
                                "total": len(assets)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    print(f"Готово: {len(done)} сессий в кэше, ошибок {state['failed']}")
    return done


# ======================================================================
# Сборка пула
# ======================================================================
def build_dataset(target):
    """Собрать пул испытаний по всем кэшированным сессиям для одной цели.

    Мелкие акронимы CCF сначала укрупняются в системы (REGION_GROUPS), чтобы
    признак «система × бины» означал одно и то же у разных мышей. Частота
    системы — средневзвешенное по числу нейронов входящих в неё структур:

        r_{g,b} = Σ_{s∈g} n_s · r_{s,b} / Σ_{s∈g} n_s

    Затем сессии приводятся к общему списку систем (объединение), а система,
    отсутствующая в сессии, даёт нулевые столбцы.
    """
    per_session = []
    for cf in sorted(CACHE_DIR.glob(f"*_w{WINDOW_S:g}_b{N_BINS}.npz")):
        d = np.load(cf, allow_pickle=True)
        if f"{target}__X" not in d.files:
            continue
        regs = [str(r) for r in d[f"{target}__regions"]]
        counts = np.asarray(d[f"{target}__counts"], dtype=np.float64)   # нейронов на структуру
        Xb = np.asarray(d[f"{target}__X"], dtype=np.float64)            # [испытания × структуры × бины]

        acc, wsum = {}, {}
        for j, r in enumerate(regs):
            g = coarse_region(r)
            if g is None:                              # root / unknown — не структуры
                continue
            w = counts[j]
            acc[g] = acc[g] + Xb[:, j, :] * w if g in acc else Xb[:, j, :] * w
            wsum[g] = wsum.get(g, 0.0) + w
        if not acc:
            continue
        groups = sorted(acc)
        per_session.append({
            "session": str(d["session"]),
            "subject": str(d["subject"]),
            "regions": groups,
            "X": np.stack([acc[g] / wsum[g] for g in groups], axis=1).astype(np.float32),
            "y": np.asarray(d[f"{target}__y"], dtype=int),
        })
    if not per_session:
        raise SystemExit("Кэш пуст. Сначала: python -m src.mouse_multisession_decoder remote")

    region_order = sorted({r for s in per_session for r in s["regions"]})
    ridx = {r: i for i, r in enumerate(region_order)}

    n_trials = sum(len(s["y"]) for s in per_session)
    n_reg = len(region_order)
    X = np.zeros((n_trials, n_reg, N_BINS), dtype=np.float32)
    y = np.zeros(n_trials, dtype=int)
    ses, sub = [], []
    pos = 0
    for s in per_session:
        cols = [ridx[r] for r in s["regions"]]
        n = len(s["y"])
        X[pos:pos + n, cols, :] = s["X"]
        y[pos:pos + n] = s["y"]
        ses += [s["session"]] * n
        sub += [s["subject"]] * n
        pos += n
    return {
        "X": X.reshape(n_trials, n_reg * N_BINS),
        "y": y,
        "session": np.asarray(ses),
        "subject": np.asarray(sub),
        "regions": region_order,
    }


# ======================================================================
# Оценка
# ======================================================================
def make_models():
    """Декодеры для пула: линейные (со shrinkage) и бустинг."""
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=0.1, random_state=RANDOM_STATE)),
        ]),
        "HistGBM": Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def _score(model, X, y, cv, groups=None):
    res = cross_validate(
        model, X, y, cv=cv, groups=groups,
        scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1"},
    )
    out = {}
    for m in ("accuracy", "balanced_accuracy", "f1"):
        v = res[f"test_{m}"]
        out[f"{m}_mean"] = float(np.mean(v))
        out[f"{m}_std"] = float(np.std(v))
    out["n_folds"] = int(len(res["test_accuracy"]))
    return out


def evaluate_target(target, data, schemes=("pooled", "cross-session", "cross-subject")):
    """Три схемы оценки: пул, новая сессия, новая мышь."""
    X, y = data["X"], data["y"]
    ses, sub = data["session"], data["subject"]
    chance = float(max(np.bincount(y)) / len(y))

    cv_map = {}
    if "pooled" in schemes:
        cv_map["pooled"] = (StratifiedKFold(POOLED_SPLITS, shuffle=True,
                                            random_state=RANDOM_STATE), None)
    if "cross-session" in schemes and len(set(ses)) > 1:
        cv_map["cross-session"] = (LeaveOneGroupOut(), ses)
    if "cross-subject" in schemes and len(set(sub)) > 1:
        cv_map["cross-subject"] = (LeaveOneGroupOut(), sub)

    rows = []
    for sname, (cv, groups) in cv_map.items():
        for mname, model in make_models().items():
            m = _score(model, X, y, cv, groups)
            rows.append({"target": target, "scheme": sname, "model": mname,
                         "n_samples": int(len(y)), "n_features": int(X.shape[1]),
                         "n_groups": int(len(set(groups))) if groups is not None else int(POOLED_SPLITS),
                         "chance_level": chance, **m})
            print(f"  [{target:<8}|{sname:<13}] {mname:<8} bacc={m['balanced_accuracy_mean']*100:5.2f}% "
                  f"(±{m['balanced_accuracy_std']*100:.2f}) acc={m['accuracy_mean']*100:5.2f}%")
    return rows


def plot_performance(df):
    """Столбики balanced accuracy по (цель × схема × модель)."""
    fig, axes = plt.subplots(1, len(df["target"].unique()), figsize=(5 * df["target"].nunique(), 4.2),
                             squeeze=False)
    for ax, (tname, g) in zip(axes[0], df.groupby("target")):
        schemes = list(dict.fromkeys(g["scheme"]))
        models = list(dict.fromkeys(g["model"]))
        width = 0.8 / len(models)
        for k, m in enumerate(models):
            sub = g[g["model"] == m].set_index("scheme").reindex(schemes)
            xs = np.arange(len(schemes)) + k * width
            ax.bar(xs, sub["balanced_accuracy_mean"] * 100, width,
                   yerr=sub["balanced_accuracy_std"] * 100, capsize=3, label=m)
        ax.axhline(g["chance_level"].iloc[0] * 100, color="grey", ls="--", lw=1, label="уровень случая")
        ax.set_xticks(np.arange(len(schemes)) + 0.4 - width / 2)
        ax.set_xticklabels(schemes, fontsize=8)
        ax.set_title(f"{tname} — {TASK_TITLES[tname]}")
        ax.set_ylabel("balanced accuracy, %")
        ax.set_ylim(0, 100)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Мульти-сессионный декодер: пул / новая сессия / новая мышь")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ms_performance.png", dpi=130)
    plt.close(fig)


def main(sections=None):
    sections = sections or ["all"]
    if "all" in sections:
        sections = ["remote", "train", "plot"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "download" in sections:
        print("=" * 72)
        print(f"Мульти-сессионный декодер: загрузка DANDI {DANDISET_ID}")
        print("=" * 72)
        download_all()

    if "extract" in sections:
        print("=" * 72)
        print("Мульти-сессионный декодер: извлечение признаков (структура × бины)")
        print("=" * 72)
        print(f"Окно {WINDOW_S:g} с, бинов {N_BINS} (по {WINDOW_S / N_BINS * 1000:.0f} мс)")
        extract_all()

    if "remote" in sections:
        print("=" * 72)
        print(f"Мульти-сессионный декодер: извлечение по HTTP из DANDI {DANDISET_ID}")
        print("=" * 72)
        print(f"Окно {WINDOW_S:g} с, бинов {N_BINS} (по {WINDOW_S / N_BINS * 1000:.0f} мс)")
        print("Режим: по одной сессии на мышь (для кросс-субъектной оценки)")
        extract_all_remote(one_per_subject=True)

    if "remote-all" in sections:
        print("=" * 72)
        print(f"Мульти-сессионный декодер: извлечение ВСЕХ сессий DANDI {DANDISET_ID}")
        print("=" * 72)
        print(f"Окно {WINDOW_S:g} с, бинов {N_BINS} (по {WINDOW_S / N_BINS * 1000:.0f} мс)")
        extract_all_remote(one_per_subject=False)

    if "train" in sections:
        print("\n" + "=" * 72)
        print("Мульти-сессионный декодер: обучение на пуле сессий")
        print("=" * 72)
        rows, dataset_info = [], {}
        for target in ALIGNMENT_COLUMNS:
            try:
                data = build_dataset(target)
            except SystemExit as exc:
                print(f"  [{target}] пропуск: {exc}")
                continue
            dataset_info[target] = {
                "n_sessions": int(len(set(data["session"]))),
                "n_subjects": int(len(set(data["subject"]))),
                "n_trials": int(len(data["y"])),
                "n_features": int(data["X"].shape[1]),
                "n_regions": int(len(data["regions"])),
                "class_balance": [int(v) for v in np.bincount(data["y"])],
            }
            print(f"\n[{target}] сессий={dataset_info[target]['n_sessions']}, "
                  f"мышей={dataset_info[target]['n_subjects']}, "
                  f"испытаний={dataset_info[target]['n_trials']}, "
                  f"признаков={dataset_info[target]['n_features']} "
                  f"({dataset_info[target]['n_regions']} структур × {N_BINS} бинов)")
            rows += evaluate_target(target, data)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_DIR / "ms_performance.csv", index=False)
            (OUTPUT_DIR / "ms_dataset.json").write_text(
                json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nСохранено: {OUTPUT_DIR / 'ms_performance.csv'}")

    if "plot" in sections:
        csv = OUTPUT_DIR / "ms_performance.csv"
        if csv.exists():
            plot_performance(pd.read_csv(csv))
            print(f"Сохранено: {OUTPUT_DIR / 'ms_performance.png'}")


if __name__ == "__main__":
    main(sections=[a.lower() for a in os.sys.argv[1:]] or None)
