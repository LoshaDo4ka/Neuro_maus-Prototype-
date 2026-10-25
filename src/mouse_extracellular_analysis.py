"""
NeuroMouse — Этап IV: внеклеточная электрофизиология и spike sorting.

Что делает модуль
-----------------
Часть A (задачи 4.2–4.5, 4.11): спайк-сортировка на синтетической записи
Neuropixels-подобного масштаба с известным ground truth.
  4.2 — визуализация сырых трейсов: спайк / шум / артефакт.
  4.3 — используется SpikeInterface.
  4.4 — запуск сортировщика на записи.
  4.5 — метрики качества (SNR, ISI-нарушения, пропуски/ложные срабатывания)
        и сверка с ground truth.

Часть B (задачи 4.6–4.10): анализ уже отсортированных юнитов реальной сессии
мыши (Steinmetz 2019, 778 юнитов).
  4.6 — firing rate каждой единицы.
  4.7 — ISI-гистограммы.
  4.8 — PSTH относительно предъявления зрительного стимула.
  4.9 — временные кросс-корреляции между нейронами.
  4.10 — сравнение активности популяции при разных условиях стимула.

Почему синтетика для сортировки: локальные NWB (Allen, Steinmetz) не содержат
сырого AP-напряжения, поэтому реальную сортировку запустить не на чем.
Синтетическая запись даёт эталон (ground truth) для честной оценки качества.

Запуск из корня проекта:
    python -m src.mouse_extracellular_analysis
"""

import json
import os
import tempfile
from pathlib import Path

# Кэш numba и matplotlib — в записываемую временную папку (иначе среды-песочницы
# блокируют запись в каталог пакета). Должно быть задано до импорта numba/SI.
_TMP = Path(tempfile.gettempdir())
os.environ.setdefault("NUMBA_CACHE_DIR", str(_TMP / "nm_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP / "nm_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import spikeinterface.full as si
from scipy import stats

from .config import (
    MOUSE_STAGE_DIR,
    MOUSE_STEINMETZ_NWB,
    SORTER_NAME,
    STAGE4_DIR,
    SYNTHETIC_DURATION_S,
    SYNTHETIC_FIRING_RATE_HZ,
    SYNTHETIC_N_CHANNELS,
    SYNTHETIC_N_UNITS,
    SYNTHETIC_NOISE_LEVEL,
    SYNTHETIC_SEED,
    SYNTHETIC_SFREQ_HZ,
)

OUTPUT_DIR = STAGE4_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Порог рефрактерности: ISI короче него для одного нейрона физиологически
# невозможен и считается нарушением (типично 1–2 мс).
REFRACTORY_MS = 2.0
QUALITY_METRICS = [
    "num_spikes",
    "firing_rate",
    "snr",
    "isi_violation",
    "amplitude_cutoff",
    "presence_ratio",
]
PERFORMANCE_COLUMNS = [
    "accuracy",
    "recall",
    "precision",
    "miss_rate",
    "false_discovery_rate",
]
# Метрика isi_violation в отчёте раскрывается в две колонки; берём долю нарушений.
QUALITY_METRIC_COLUMNS = {
    "num_spikes": "num_spikes",
    "firing_rate": "firing_rate",
    "snr": "snr",
    "isi_violation": "isi_violations_ratio",
    "amplitude_cutoff": "amplitude_cutoff",
    "presence_ratio": "presence_ratio",
}


# ======================================================================
# Часть A — spike sorting на синтетике с ground truth
# ======================================================================
def generate_synthetic_recording():
    """Синтетическая внеклеточная запись с известными юнитами (4.1/4.4)."""
    return si.generate_ground_truth_recording(
        durations=[SYNTHETIC_DURATION_S],
        sampling_frequency=SYNTHETIC_SFREQ_HZ,
        num_channels=SYNTHETIC_N_CHANNELS,
        num_units=SYNTHETIC_N_UNITS,
        generate_sorting_kwargs={
            "firing_rates": SYNTHETIC_FIRING_RATE_HZ,
            "refractory_period_ms": 4.0,
        },
        noise_kwargs={"noise_levels": SYNTHETIC_NOISE_LEVEL, "strategy": "on_the_fly"},
        seed=SYNTHETIC_SEED,
    )


def preprocess(recording):
    """Полосовая фильтрация AP-полосы + общая референция (общий шум)."""
    rec = si.bandpass_filter(recording, freq_min=300.0, freq_max=6000.0)
    rec = si.common_reference(rec, operator="median", reference="global")
    return rec


def plot_raw_traces(recording, ground_truth_sorting, path):
    """4.2 — сырые трейсы: шум, спайк и артефакт."""
    fs = recording.get_sampling_frequency()
    n_show = min(4, recording.get_num_channels())
    traces = recording.get_traces(start_frame=0, end_frame=int(3.0 * fs), channel_ids=recording.channel_ids[:n_show])
    t = np.arange(traces.shape[0]) / fs

    # Берём первый настоящий спайк из ground truth и окно вокруг него.
    unit0 = ground_truth_sorting.unit_ids[0]
    spikes = ground_truth_sorting.get_unit_spike_train(unit0)
    spike_t = float(spikes[0]) if len(spikes) else 0.1
    w0, w1 = spike_t - 0.02, spike_t + 0.02
    m = (t >= w0) & (t <= w1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    for i in range(n_show):
        axes[0].plot(t, traces[:, i] * 1e6 + i * 60, lw=0.6, color="#1e293b")
    axes[0].set_title("Сырые трейсы (шум + спайки), 3 с")
    axes[0].set_xlabel("Время, с")
    axes[0].set_ylabel("Канал (смещено), мкВ")
    axes[0].grid(alpha=0.3)

    for i in range(n_show):
        axes[1].plot(t[m], traces[m, i] * 1e6 + i * 60, lw=1.0, color="#b91c1c")
    axes[1].set_title(f"Спайк крупным планом (окно 40 мс вокруг {spike_t:.3f} с)")
    axes[1].set_xlabel("Время, с")
    axes[1].grid(alpha=0.3)

    # Артефакт — искусственно добавленный большой медленный сдвиг (иллюстрация).
    art = traces.copy()
    a0, a1 = int(1.5 * fs), int(1.55 * fs)
    win = np.hanning(a1 - a0) * 250.0  # мкВ
    for i in range(n_show):
        art[a0:a1, i] += win
    for i in range(n_show):
        axes[2].plot(t, art[:, i] * 1e6 + i * 60, lw=0.6, color="#7c3aed")
    axes[2].set_title("Артефакт vs спайк: медленный большой сдвиг (добавлен искусственно)")
    axes[2].set_xlabel("Время, с")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


def run_sorting(recording, folder):
    """4.4 — сохранить запись в binary и запустить сортировщик (с кэшем)."""
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    rec_saved = recording.save(folder=folder.parent / "preprocessed", format="binary", overwrite=True)
    if folder.exists() and (folder / "spikeinterface_log.json").exists():
        try:
            return rec_saved, si.read_sorter_folder(folder)
        except Exception:
            pass
    sorting = si.run_sorter(SORTER_NAME, rec_saved, folder=folder, verbose=False, remove_existing_folder=True)
    return rec_saved, sorting


def evaluate_sorting(ground_truth_sorting, sorting, recording):
    """4.5 — метрики качества и сверка с ground truth."""
    analyzer = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=True)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes"])
    si.compute_quality_metrics(analyzer, metric_names=QUALITY_METRICS)
    si.compute_isi_histograms(analyzer, window_ms=50.0, bin_ms=1.0)

    qm = analyzer.get_extension("quality_metrics").get_data()

    comparison = si.compare_sorter_to_ground_truth(
        ground_truth_sorting, sorting, exhaustive_gt=True)
    perf = comparison.get_performance()

    mean_isi = analyzer.get_extension("isi_histograms").get_data()
    return analyzer, qm, perf, mean_isi


def plot_sorting_qc(analyzer, path):
    """Волновые формы отсортированных юнитов + ISI-гистограммы."""
    unit_ids = analyzer.unit_ids
    n = min(6, len(unit_ids))
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 6))

    waveforms = analyzer.get_extension("waveforms").get_data()
    for ax, uid in zip(axes[0], unit_ids[:n]):
        wf = waveforms[uid]  # (n_samples, n_channels)
        ax.plot(np.arange(wf.shape[0]) / analyzer.sampling_frequency * 1000.0,
                wf[:, 0] * 1e6, lw=0.7, color="#0f766e", alpha=0.6)
        ax.set_title(f"Юнит {uid}", fontsize=9)
        ax.set_xlabel("мс", fontsize=8)
    axes[0][0].set_ylabel("мкВ", fontsize=9)

    isi, _ = analyzer.get_extension("isi_histograms").get_data()
    bins = np.arange(isi.shape[1]) * 1.0
    unit_ids_list = list(analyzer.unit_ids)
    for ax, uid in zip(axes[1], unit_ids[:n]):
        counts = isi[unit_ids_list.index(uid)]
        ax.bar(bins, counts, width=1.0, color="#1d4ed8")
        ax.set_xlim(0, 30)
        ax.set_title(f"ISI {uid}", fontsize=9)
        ax.set_xlabel("мс", fontsize=8)
    axes[1][0].set_ylabel("число пар", fontsize=9)

    fig.suptitle("Волновые формы и ISI отсортированных юнитов (синтетика)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


# ======================================================================
# Часть B — одиночные юниты реальной сессии Steinmetz
# ======================================================================
def load_steinmetz():
    """Спайковые времена 778 юнитов и таблица испытаний."""
    with pynwb.NWBHDF5IO(str(MOUSE_STEINMETZ_NWB), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units
        spike_times = [np.asarray(units["spike_times"][i], dtype=float) for i in range(len(units))]
        trials = nwb.trials.to_dataframe()
        subject = nwb.subject.subject_id if nwb.subject else "?"
        return spike_times, trials, subject


def firing_rate_table(spike_times):
    """4.6 — firing rate каждой единицы (Гц)."""
    rows = []
    for i, spk in enumerate(spike_times):
        if len(spk) < 2:
            rows.append({"unit_id": i, "n_spikes": len(spk), "duration_s": np.nan, "firing_rate_hz": np.nan})
            continue
        dur = float(spk[-1] - spk[0])
        rows.append({
            "unit_id": i,
            "n_spikes": int(len(spk)),
            "duration_s": dur,
            "firing_rate_hz": float(len(spk) / dur) if dur > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def isi_summary_table(spike_times):
    """4.7 — сводка по межспайковым интервалам и нарушениям рефрактерности."""
    rows = []
    for i, spk in enumerate(spike_times):
        if len(spk) < 3:
            rows.append({"unit_id": i, "n_isi": 0, "median_isi_ms": np.nan,
                         "isi_violation_ratio": np.nan, "refractory_ms": REFRACTORY_MS})
            continue
        isi_ms = np.diff(np.sort(spk)) * 1000.0
        viol = float(np.mean(isi_ms < REFRACTORY_MS))
        rows.append({
            "unit_id": i,
            "n_isi": int(len(isi_ms)),
            "median_isi_ms": float(np.median(isi_ms)),
            "isi_violation_ratio": viol,
            "refractory_ms": REFRACTORY_MS,
        })
    return pd.DataFrame(rows)


def plot_firing_rate_distribution(fr_df, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = fr_df["firing_rate_hz"].dropna()
    ax.hist(vals, bins=40, color="#0ea5e9", edgecolor="white")
    ax.axvline(vals.median(), color="#b91c1c", ls="--", label=f"медиана {vals.median():.2f} Гц")
    ax.set_xlabel("Firing rate, Гц")
    ax.set_ylabel("Число юнитов")
    ax.set_title(f"Распределение firing rate ({len(vals)} юнитов, Steinmetz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


def plot_isi_examples(spike_times, isi_df, path, n_units=6):
    # Берём юниты с наименьшей медианой ISI (самые «занятые») для наглядности.
    ids = list(isi_df.dropna(subset=["median_isi_ms"]).sort_values("median_isi_ms")["unit_id"].iloc[:n_units])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, uid in zip(axes.ravel(), ids):
        isi_ms = np.diff(np.sort(spike_times[uid])) * 1000.0
        ax.hist(isi_ms[isi_ms < 100], bins=np.arange(0, 100, 2), color="#7c3aed")
        ax.axvline(REFRACTORY_MS, color="#b91c1c", ls="--", lw=1)
        ax.set_title(f"Юнит {uid}: медиана ISI {np.median(isi_ms):.1f} мс", fontsize=10)
        ax.set_xlabel("ISI, мс")
    axes[0][0].set_ylabel("Пар спайков")
    fig.suptitle("4.7 ISI-гистограммы: рефрактерный провал слева от красной линии", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


def psth(spike_times, trials, path, pre_s=0.1, post_s=0.5, bin_ms=10.0, n_units=6):
    """4.8 — PSTH относительно visual_stimulus_time, усреднённо по испытаниям."""
    onsets = trials["visual_stimulus_time"].dropna().to_numpy()
    edges = np.arange(-pre_s, post_s + 1e-9, bin_ms / 1000.0)
    centers = (edges[:-1] + edges[1:]) / 2 * 1000.0

    # Берём активные юниты с наибольшим числом спайков для наглядности.
    order = np.argsort([-len(s) for s in spike_times])[:n_units]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    psth_rows = []
    for ax, uid in zip(axes.ravel(), order):
        spk = np.sort(spike_times[uid])
        counts = np.zeros(len(centers))
        for t0 in onsets:
            sel = spk[(spk >= t0 - pre_s) & (spk < t0 + post_s)] - t0
            h, _ = np.histogram(sel * 1000.0, bins=(edges * 1000.0))
            counts += h
        rate = counts / len(onsets) / (bin_ms / 1000.0)
        ax.bar(centers, rate, width=bin_ms, color="#0f766e")
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_title(f"Юнит {uid}", fontsize=10)
        psth_rows.append({"unit_id": int(uid), "mean_rate_pre_hz": float(rate[centers < 0].mean()),
                          "mean_rate_post_hz": float(rate[(centers >= 0) & (centers <= 200)].mean())})
    for ax in axes[1]:
        ax.set_xlabel("Время от стимула, мс")
    for ax in axes[:, 0]:
        ax.set_ylabel("Частота, Гц")
    fig.suptitle("4.8 PSTH относительно предъявления зрительного стимула", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return pd.DataFrame(psth_rows)


def cross_correlation(spike_times, path, pairs=6, window_ms=50.0, bin_ms=1.0, max_dur_s=300.0):
    """4.9 — кросс-коррелограммы для выбранных пар нейронов."""
    active = sorted(range(len(spike_times)), key=lambda i: -len(spike_times[i]))
    sel = active[: max(2, pairs)]
    edges = np.arange(-window_ms, window_ms + 1e-9, bin_ms)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    rows = []
    used = 0
    for k in range(len(sel) - 1):
        if used >= 6:
            break
        a, b = sel[k], sel[k + 1]
        sa = np.sort(spike_times[a][spike_times[a] < max_dur_s])
        sb = np.sort(spike_times[b][spike_times[b] < max_dur_s])
        # простая гистограмма разностей времён
        diffs = []
        for t in sb:
            d = sa - t
            d = d[(d > -window_ms / 1000.0) & (d < window_ms / 1000.0)] * 1000.0
            if len(d):
                diffs.append(d)
        diffs = np.concatenate(diffs) if diffs else np.array([])
        ax = axes.ravel()[used]
        if len(diffs):
            ax.bar(centers, np.histogram(diffs, bins=edges)[0], width=bin_ms, color="#1d4ed8")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{a} → {b}", fontsize=10)
        ax.set_xlabel("Задержка, мс")
        rows.append({"unit_a": int(a), "unit_b": int(b), "n_pairs": int(len(diffs))})
        used += 1
    for ax in axes.ravel()[used:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Число спайков")
    fig.suptitle("4.9 Временные кросс-корреляции между нейронами", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return pd.DataFrame(rows)


def population_condition_comparison(spike_times, trials, path, window_s=0.4):
    """4.10 — активность популяции при левом vs правом стимуле."""
    c_left = trials["visual_stimulus_left_contrast"].to_numpy()
    c_right = trials["visual_stimulus_right_contrast"].to_numpy()
    onsets = trials["visual_stimulus_time"].to_numpy()

    valid = (c_left != c_right) & ~np.isnan(c_left) & ~np.isnan(c_right)
    left_trials = onsets[valid & (c_left > c_right)]
    right_trials = onsets[valid & (c_right > c_left)]

    def trial_rates(t0s):
        X = np.zeros((len(t0s), len(spike_times)))
        for j, spk in enumerate(spike_times):
            spk = np.sort(spk)
            for i, t0 in enumerate(t0s):
                lo = np.searchsorted(spk, t0)
                hi = np.searchsorted(spk, t0 + window_s)
                X[i, j] = (hi - lo) / window_s
        return X

    Xl = trial_rates(left_trials)
    Xr = trial_rates(right_trials)
    mean_l = Xl.mean(axis=0)
    mean_r = Xr.mean(axis=0)
    tstat, pval = stats.ttest_ind(Xl, Xr, axis=0, equal_var=False)

    table = pd.DataFrame({
        "unit_id": np.arange(len(spike_times)),
        "mean_rate_left_hz": mean_l,
        "mean_rate_right_hz": mean_r,
        "diff_hz": mean_r - mean_l,
        "p_value": pval,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(mean_l, bins=40, alpha=0.6, label="Стимул слева", color="#2563eb")
    axes[0].hist(mean_r, bins=40, alpha=0.6, label="Стимул справа", color="#dc2626")
    axes[0].set_xlabel("Средняя частота в окне 0–400 мс, Гц")
    axes[0].set_ylabel("Число юнитов")
    axes[0].set_title("Популяция: распределение частот по условиям")
    axes[0].legend()

    sig = table["p_value"] < 0.05
    axes[1].scatter(mean_l[~sig], mean_r[~sig], s=10, color="#94a3b8", label="p ≥ 0.05")
    axes[1].scatter(mean_l[sig], mean_r[sig], s=14, color="#b91c1c", label="p < 0.05")
    lim = max(mean_l.max(), mean_r.max()) * 1.05 if len(mean_l) else 1
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[1].set_xlabel("Частота при стимуле слева, Гц")
    axes[1].set_ylabel("Частота при стимуле справа, Гц")
    axes[1].set_title(f"Юниты, различающие условие: {int(sig.sum())} из {len(table)}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return table


# ======================================================================
# Оркестрация
# ======================================================================
def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def main():
    si.set_global_job_kwargs(n_jobs=1, chunk_duration="1s", progress_bar=False)
    summary = {"part_a": {}, "part_b": {}}

    # ---------- Часть A ----------
    print("ЭТАП IV, часть A: синтетическая запись + spike sorting")
    rec, gt = generate_synthetic_recording()
    print(f"  запись: {rec.get_num_channels()} каналов, {SYNTHETIC_SFREQ_HZ:.0f} Гц, "
          f"{SYNTHETIC_DURATION_S:.0f} с; ground truth: {gt.get_num_units()} юнитов")
    summary["part_a"]["n_channels"] = int(rec.get_num_channels())
    summary["part_a"]["n_ground_truth_units"] = int(gt.get_num_units())

    plot_raw_traces(rec, gt, OUTPUT_DIR / "stage4_raw_traces.png")
    print("  4.2 сохранено: stage4_raw_traces.png")

    rec_pre = preprocess(rec)
    rec_saved, sorting = run_sorting(rec_pre, OUTPUT_DIR / "sorter_output")
    print(f"  4.4 сортировщик {SORTER_NAME}: {sorting.get_num_units()} юнитов")
    summary["part_a"]["sorter"] = SORTER_NAME
    summary["part_a"]["n_sorted_units"] = int(sorting.get_num_units())

    analyzer, qm, perf, _ = evaluate_sorting(gt, sorting, rec_saved)
    qm.to_csv(OUTPUT_DIR / "sorting_quality_metrics.csv")
    perf.to_csv(OUTPUT_DIR / "sorting_ground_truth_performance.csv")
    plot_sorting_qc(analyzer, OUTPUT_DIR / "stage4_sorted_units_qc.png")
    print("  4.5 метрики качества и сверка с ground truth сохранены")

    summary["part_a"]["quality_metrics_median"] = {
        name: float(np.nanmedian(qm[col])) for name, col in QUALITY_METRIC_COLUMNS.items() if col in qm}
    summary["part_a"]["ground_truth_performance_median"] = {
        c: float(np.nanmedian(perf[c])) for c in PERFORMANCE_COLUMNS if c in perf}

    # ---------- Часть B ----------
    print("\nЭТАП IV, часть B: юниты реальной сессии (Steinmetz)")
    spike_times, trials, subject = load_steinmetz()
    print(f"  субъект {subject}: {len(spike_times)} юнитов, {len(trials)} испытаний")

    fr = firing_rate_table(spike_times)
    fr.to_csv(OUTPUT_DIR / "single_unit_firing_rates.csv", index=False)
    plot_firing_rate_distribution(fr, OUTPUT_DIR / "stage4_firing_rate_distribution.png")
    print(f"  4.6 медиана firing rate: {np.nanmedian(fr['firing_rate_hz']):.2f} Гц")

    isi = isi_summary_table(spike_times)
    isi.to_csv(OUTPUT_DIR / "isi_summary.csv", index=False)
    plot_isi_examples(spike_times, isi, OUTPUT_DIR / "stage4_isi_histograms.png")
    print(f"  4.7 медиана ISI: {np.nanmedian(isi['median_isi_ms']):.1f} мс; "
          f"доля нарушений рефрактерности: {np.nanmedian(isi['isi_violation_ratio']):.4f}")

    psth_df = psth(spike_times, trials, OUTPUT_DIR / "stage4_psth.png")
    psth_df.to_csv(OUTPUT_DIR / "psth_summary.csv", index=False)
    print("  4.8 PSTH сохранён")

    xc = cross_correlation(spike_times, OUTPUT_DIR / "stage4_cross_correlation.png")
    xc.to_csv(OUTPUT_DIR / "cross_correlation_pairs.csv", index=False)
    print("  4.9 кросс-корреляции сохранены")

    cond = population_condition_comparison(spike_times, trials, OUTPUT_DIR / "stage4_population_conditions.png")
    cond.to_csv(OUTPUT_DIR / "population_condition_comparison.csv", index=False)
    n_sig = int((cond["p_value"] < 0.05).sum())
    print(f"  4.10 условия различимы у {n_sig} из {len(cond)} юнитов (p < 0.05)")

    summary["part_b"] = {
        "subject": subject,
        "n_units": len(spike_times),
        "n_trials": int(len(trials)),
        "median_firing_rate_hz": float(np.nanmedian(fr["firing_rate_hz"])),
        "median_isi_ms": float(np.nanmedian(isi["median_isi_ms"])),
        "median_isi_violation_ratio": float(np.nanmedian(isi["isi_violation_ratio"])),
        "n_units_condition_sensitive_p05": n_sig,
    }

    with open(OUTPUT_DIR / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_jsonable)

    print("\nЭтап IV завершён. Результаты в:", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
