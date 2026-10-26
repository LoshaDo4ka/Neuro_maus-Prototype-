"""
NeuroMouse — «Эксперименты и ИИ»: расширенный анализ декодирования и юнитов.

Назначение
----------
Развитие ИИ-части проекта поверх базового декодера Этапа III. Базовый декодер
(`src/mouse_spikes_decoder.py`) обучал LogReg и LDA на 778 признаках (спайк-каунты
на нейрон) и достигал 65.31% (LDA) — но без кросс-валидации, без подбора
гиперпараметров, без отбора признаков и с выравниванием спайков по началу
испытания (а не по моменту стимула). Этот модуль проводит строгий эксперимент.

Данные
------
Steinmetz et al. (2019), сессия Richards_2017-10-31 (DANDI 000017):
  778 отсортированных юнитов, 2 Neuropixels-зонда, 260 поведенческих испытаний.

Что проверяем (и почему)
------------------------
A. Сравнение семейств моделей (e2). 5-кратная стратифицированная кросс-валидация
   честно сравнивает LDA, LogReg, SVM-RBF, RandomForest, GradientBoosting, MLP.
   Кросс-валидация даёт несмещённую оценку обобщения — в отличие от одного
   train/test-сплита из Этапа III.
B. Гиперпараметры и отбор признаков (e3). GridSearchCV подбирает C/gamma для SVM,
   C для LogReg; анализ ANOVA F-value и mutual information показывает, сколькими
   нейронами достаточно для максимальной точности (борьба с переобучением p≫n).
C. Свип окна и целевой переменной (e4). Как меняется точность при окне
   интегрирования спайков 0.05–1.5 с, и что декодируется лучше — сторона
   зрительного стимула (сенсорная задача), моторный выбор мыши (решение) или
   исход испытания (правильно/ошибка, выравнивание к моменту обратной связи).
D. Углублённый анализ юнитов (e5). Попарные Welch t-тесты «стимул слева/справа»
   для каждого нейрона с FDR-поправкой Бенджамини–Хохберга, разбивка значимых
   юнитов по структурам мозга (из electrodes.location) и PCA популяционного отклика.

Метрики
-------
  accuracy          = (TP+TN)/(TP+TN+FP+FN);
  balanced_accuracy = (TPR + TNR)/2  (устойчива к дисбалансу классов);
  f1                = 2·precision·recall/(precision+recall).
Итог по кросс-валидации: среднее ± стандартное отклонение по фолдам.

Артефакты (results/mouse_stage/ai_experiments/)
-----------------------------------------------
  ai_model_comparison.csv/.png     — сравнение моделей (e2)
  ai_hyperparam_search.csv         — GridSearchCV (e3)
  ai_feature_selection.csv/.png    — кривая точности от числа признаков (e3)
  ai_feature_scores.csv            — ANOVA F / mutual information по юнитам (e3)
  ai_window_sweep.csv/.png         — свип окна и целевой переменной (e4)
  ai_unit_stats.csv                — t-тест по каждому юниту + FDR (e5)
  ai_region_stats.csv              — агрегация по структурам мозга (e5)
  ai_unit_fdr.png                  — вулкан-график отклика юнитов (e5)
  ai_pca_population.png            — PCA популяционного отклика (e5)
  ai_experiments_summary.json      — сводка параметров и ключевых результатов

Запуск из корня проекта (окружение neuromouse):
    python -m src.mouse_ai_experiments
"""

import json
import os
import tempfile
from pathlib import Path

# Кэш matplotlib — в записываемую временную папку песочницы.
_TMP = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_TMP / "nm_mpl"))

# Ограничиваем число потоков BLAS/OpenMP: данные маленькие (≈200×778), а
# многопоточность и порождаемые joblib-процессы в песочнице дают огромный
# оверхед (запуск процессов на Windows дороже самого обучения).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import MOUSE_STAGE_DIR, MOUSE_STEINMETZ_NWB, RANDOM_STATE

# ----------------------------------------------------------------------
# Параметры эксперимента
# ----------------------------------------------------------------------
OUTPUT_DIR = MOUSE_STAGE_DIR / "ai_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5                       # фолды кросс-валидации
MAIN_WINDOW_S = 0.4                # основное окно интегрирования спайков
# Сетка окна расширена до 1.5 с: в базовой версии (≤0.8 с) точность ещё росла
# на границе сетки, оптимум не был локализован.
WINDOW_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]   # свип окна (e4)
TOP_K_GRID = [5, 10, 25, 50, 100, 200, 400, 778]     # свип числа признаков (e3)
ALPHA = 0.05                       # уровень значимости (FDR)

# Выравнивание спайков: к какому событию испытания привязано окно.
# Сенсорное решение (сторона стимула) — к моменту предъявления стимула;
# моторное решение (выбор) — к go-cue (сигнал к движению);
# исход испытания (правильно/ошибка) — к моменту обратной связи.
ALIGNMENT_COLUMNS = {
    "stimulus": "visual_stimulus_time",
    "choice": "go_cue",
    "feedback": "feedback_time",
}
TARGET_LABELS = {
    "stimulus": ["Стимул слева", "Стимул справа"],
    "choice": ["Выбор влево", "Выбор вправо"],
    "feedback": ["Ошибка", "Правильно"],
}
TASK_TITLES = {
    "stimulus": "сторона стимула",
    "choice": "моторный выбор",
    "feedback": "исход испытания",
}


# ======================================================================
# Загрузка данных
# ======================================================================
def load_session():
    """Спайковые времена, таблицы испытаний/электродов и карта структур мозга."""
    with pynwb.NWBHDF5IO(str(MOUSE_STEINMETZ_NWB), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units
        spike_times = [np.asarray(units["spike_times"][i], dtype=float) for i in range(len(units))]
        units_df = units.to_dataframe()
        trials = nwb.trials.to_dataframe()
        electrodes = nwb.electrodes.to_dataframe()
        subject = nwb.subject.subject_id if nwb.subject else "?"
        identifier = nwb.identifier

    # Структура мозга юнита: пиковый канал -> строка electrodes -> колонка location.
    loc = electrodes["location"]
    regions = []
    for v in units_df["peak_channel"]:
        arr = np.asarray(v).ravel()
        j = int(arr[0]) if len(arr) else -1
        regions.append(str(loc.loc[j]) if j in loc.index else "unknown")

    return {
        "identifier": str(identifier),
        "subject": str(subject),
        "spike_times": spike_times,
        "trials": trials,
        "regions": np.asarray(regions),
        "n_units": len(spike_times),
    }


def build_targets(trials):
    """Целевые переменные: сторона стимула (A), моторный выбор мыши (B), исход (C)."""
    cl = trials["visual_stimulus_left_contrast"].to_numpy(dtype=float)
    cr = trials["visual_stimulus_right_contrast"].to_numpy(dtype=float)
    choice = trials["response_choice"].to_numpy(dtype=float)
    feedback = trials["feedback_type"].to_numpy(dtype=float)

    stim_mask = (~np.isnan(cl)) & (~np.isnan(cr)) & (cl != cr)
    y_stim = (cr > cl).astype(int)                    # 1 = справа сильнее

    choice_mask = np.isin(choice, [1.0, -1.0])        # 1 = влево, -1 = вправо
    y_choice = (choice > 0).astype(int)               # 1 = выбрал влево

    feedback_mask = np.isin(feedback, [1.0, -1.0])    # 1 = правильно, -1 = ошибка
    y_feedback = (feedback > 0).astype(int)           # 1 = правильно

    return {
        "stimulus": {"mask": stim_mask, "y": y_stim},
        "choice": {"mask": choice_mask, "y": y_choice},
        "feedback": {"mask": feedback_mask, "y": y_feedback},
    }


def spike_count_matrix(spike_times, onsets, window_s):
    """Матрица [испытания × нейроны] — число спайков в окне [onset, onset+window).

    Для каждого нейрона используем бинарный поиск (searchsorted) вместо цикла по
    испытаниям: O(испытаний·нейронов) → O(нейронов·log(спайков)).
    """
    onsets = np.asarray(onsets, dtype=float)
    X = np.zeros((len(onsets), len(spike_times)), dtype=np.float32)
    for i, spk in enumerate(spike_times):
        if spk.size == 0:
            continue
        lo = np.searchsorted(spk, onsets, side="left")
        hi = np.searchsorted(spk, onsets + window_s, side="left")
        X[:, i] = hi - lo
    return X


# ======================================================================
# Модели и метрики
# ======================================================================
def make_models():
    """Набор декодеров: линейные, ядровые, ансамблевые, нейросетевой."""
    scaler = StandardScaler
    return {
        "LDA": Pipeline([
            ("scaler", scaler()),
            # shrinkage='auto' (лемма Ледойта–Вольфа) критично при p ≫ n.
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "LogReg": Pipeline([
            ("scaler", scaler()),
            ("clf", LogisticRegression(max_iter=5000, C=0.1, random_state=RANDOM_STATE)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", scaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE)),
        ]),
        "GradientBoosting": Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "MLP": Pipeline([
            ("scaler", scaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500, random_state=RANDOM_STATE)),
        ]),
    }


def cv_scores(model, X, y):
    """Кросс-валидация: accuracy/balanced_accuracy/f1 (среднее и std по фолдам)."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(
        model, X, y, cv=cv,
        scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1": "f1"},
    )
    out = {}
    for metric in ("accuracy", "balanced_accuracy", "f1"):
        v = res[f"test_{metric}"]
        out[f"{metric}_mean"] = float(np.mean(v))
        out[f"{metric}_std"] = float(np.std(v))
    return out


# ======================================================================
# e2 — сравнение моделей
# ======================================================================
def experiment_model_comparison(targets, spike_times, window_s=MAIN_WINDOW_S):
    """Сравнить 6 моделей на обеих целевых переменных через 5-кратную CV."""
    models = make_models()
    rows = []
    for tname, tgt in targets.items():
        mask = tgt["mask"]
        X = spike_count_matrix(spike_times, tgt["onsets"][mask], window_s)
        y = tgt["y"][mask]
        chance = float(max(np.bincount(y)) / len(y))
        for mname, model in models.items():
            m = cv_scores(model, X, y)
            rows.append({
                "target": tname, "model": mname,
                "n_samples": int(len(y)), "n_features": int(X.shape[1]),
                "chance_level": chance, **m,
            })
            print(f"  [{tname}] {mname:<18} acc={m['accuracy_mean']*100:5.2f}% "
                  f"(±{m['accuracy_std']*100:.2f}) bacc={m['balanced_accuracy_mean']*100:5.2f}%")
    return pd.DataFrame(rows)


def plot_model_comparison(df):
    """Столбчатая диаграмма точности моделей (с планками std) для двух задач."""
    targets = sorted(df["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), squeeze=False)
    for ax, tname in zip(axes[0], targets):
        sub = df[df["target"] == tname].sort_values("accuracy_mean")
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sub)))
        ax.barh(sub["model"], sub["accuracy_mean"] * 100,
                xerr=sub["accuracy_std"] * 100, color=colors, capsize=4)
        chance = sub["chance_level"].iloc[0] * 100
        ax.axvline(chance, color="crimson", ls="--", lw=1.2, label=f"уровень случая {chance:.1f}%")
        ax.set_xlabel("Точность (accuracy), %")
        ax.set_title(f"Задача: {TASK_TITLES.get(tname, tname)}\n"
                     f"5-кратная кросс-валидация, окно {MAIN_WINDOW_S*1000:.0f} мс")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "ai_model_comparison.png"
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path


# ======================================================================
# e3 — гиперпараметры и отбор признаков
# ======================================================================
def experiment_hyperparameters(X, y):
    """GridSearchCV (balanced_accuracy) по C/gamma для SVM и C для LogReg/LDA."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grids = {
        "SVM_RBF": (
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.001, 0.01]},
        ),
        "LogReg": (
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "LDA": (
            Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr"))]),
            {"clf__shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]},
        ),
    }
    rows = []
    for name, (pipe, grid) in grids.items():
        gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=cv, n_jobs=1)
        gs.fit(X, y)
        n_combo = int(np.prod([len(v) for v in grid.values()]))
        rows.append({
            "model": name,
            "best_params": json.dumps(gs.best_params_),
            "best_balanced_accuracy": float(gs.best_score_),
            "n_combinations": n_combo,
        })
        print(f"  {name:<10} best bacc={gs.best_score_*100:5.2f}%  {gs.best_params_}")
    return pd.DataFrame(rows)


def experiment_feature_selection(X, y):
    """Кривая точности LDA/SVM от числа отобранных нейронов (ANOVA F и MI)."""
    methods = {"ANOVA_F": f_classif, "MutualInfo": mutual_info_classif}
    models = {"LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
              "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)}
    rows = []
    for mname, score_func in methods.items():
        for model_name, clf in models.items():
            for k in TOP_K_GRID:
                k_actual = min(k, X.shape[1])
                pipe = Pipeline([
                    ("select", SelectKBest(score_func=score_func, k=k_actual)),
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ])
                m = cv_scores(pipe, X, y)
                rows.append({"feature_method": mname, "model": model_name, "k_features": k_actual,
                             **{keys: m[keys] for keys in m}})
                print(f"  {mname:<11} {model_name:<8} k={k_actual:<4} acc={m['accuracy_mean']*100:5.2f}%")

    # Оценки значимости каждого признака (для карты нейронов).
    f_val, p_val = f_classif(X, y)
    mi_val = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    scores_df = pd.DataFrame({
        "unit_id": np.arange(X.shape[1]),
        "anova_f": f_val,
        "anova_p": p_val,
        "mutual_info": mi_val,
    }).fillna(0.0).sort_values("anova_f", ascending=False).reset_index(drop=True)
    scores_df["anova_rank"] = np.arange(1, len(scores_df) + 1)
    return pd.DataFrame(rows), scores_df


def plot_feature_selection(df):
    """Точность против числа признаков для двух методов отбора и двух моделей."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (method, model), sub in df.groupby(["feature_method", "model"]):
        sub = sub.sort_values("k_features")
        ax.errorbar(sub["k_features"], sub["accuracy_mean"] * 100, yerr=sub["accuracy_std"] * 100,
                    marker="o", capsize=3, label=f"{method} + {model}")
    ax.set_xscale("log")
    ax.set_xlabel("Число используемых нейронов (признаков), log-шкала")
    ax.set_ylabel("Точность (accuracy), %")
    ax.set_title("Отбор признаков: сколько нейронов нужно для декодирования?\n"
                 "5-кратная кросс-валидация")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "ai_feature_selection.png"
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path


# ======================================================================
# e4 — свип окна и целевой переменной
# ======================================================================
def experiment_window_sweep(targets, spike_times):
    """Точность LDA/LogReg в зависимости от длины окна и целевой переменной."""
    models = {
        "LDA": Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "LogReg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=5000, C=0.1, random_state=RANDOM_STATE))]),
    }
    rows = []
    for tname, tgt in targets.items():
        mask = tgt["mask"]
        y = tgt["y"][mask]
        o = tgt["onsets"][mask]
        for w in WINDOW_GRID:
            X = spike_count_matrix(spike_times, o, w)
            for mname, model in models.items():
                m = cv_scores(model, X, y)
                rows.append({"target": tname, "model": mname, "window_ms": int(w * 1000),
                             "n_samples": int(len(y)), **m})
        print(f"  [{tname}] окно {WINDOW_GRID} — готово")
    return pd.DataFrame(rows)


def plot_window_sweep(df):
    """Точность против длины окна для трёх целевых переменных."""
    linestyles = {"stimulus": "-", "choice": "--", "feedback": ":"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (tname, model), sub in df.groupby(["target", "model"]):
        sub = sub.sort_values("window_ms")
        ax.errorbar(sub["window_ms"], sub["accuracy_mean"] * 100, yerr=sub["accuracy_std"] * 100,
                    marker="o", ls=linestyles.get(tname, "-"), capsize=3, label=f"{tname} / {model}")
    ax.axhline(50, color="crimson", ls=":", lw=1, label="уровень случая 50%")
    ax.set_xlabel("Длина окна интегрирования спайков, мс")
    ax.set_ylabel("Точность (accuracy), %")
    ax.set_title("Свип окна декодирования и целевой переменной\n5-кратная кросс-валидация")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "ai_window_sweep.png"
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path


# ======================================================================
# e5 — углублённый анализ юнитов
# ======================================================================
def benjamini_hochberg(pvals):
    """FDR-поправка Бенджамини–Хохберга: q-значения из набора p-значений."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]     # монотонность q-значений
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def experiment_unit_analysis(spike_times, regions, targets, window_s=MAIN_WINDOW_S):
    """t-тест «стимул слева vs справа» по каждому юниту + FDR + структуры мозга."""
    mask = targets["stimulus"]["mask"]
    X = spike_count_matrix(spike_times, targets["stimulus"]["onsets"][mask], window_s)  # [trials × units]
    y = targets["stimulus"]["y"][mask]

    rate_l = X[y == 0].mean(axis=0)      # средний спайк-каунт при стимуле слева
    rate_r = X[y == 1].mean(axis=0)

    t_stat = np.zeros(X.shape[1])
    p_val = np.ones(X.shape[1])
    for i in range(X.shape[1]):
        a, b = X[y == 0, i], X[y == 1, i]
        if a.std() == 0 and b.std() == 0:
            t_stat[i], p_val[i] = 0.0, 1.0
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        t_stat[i] = 0.0 if np.isnan(t) else t
        p_val[i] = 1.0 if np.isnan(p) else p

    q_val = benjamini_hochberg(p_val)
    unit_df = pd.DataFrame({
        "unit_id": np.arange(X.shape[1]),
        "region": regions,
        "mean_count_left": rate_l,
        "mean_count_right": rate_r,
        "mean_rate_hz_left": rate_l / window_s,
        "mean_rate_hz_right": rate_r / window_s,
        "delta_mean": rate_r - rate_l,
        "t_stat": t_stat,
        "p_value": p_val,
        "q_value_fdr": q_val,
        "significant_fdr": q_val < ALPHA,
    })

    reg_rows = []
    for reg, sub in unit_df.groupby("region"):
        reg_rows.append({
            "region": reg,
            "n_units": int(len(sub)),
            "n_significant": int(sub["significant_fdr"].sum()),
            "fraction_significant": float(sub["significant_fdr"].mean()),
            "mean_abs_delta_rate_hz": float((sub["delta_mean"].abs() / window_s).mean()),
        })
    region_df = pd.DataFrame(reg_rows).sort_values("fraction_significant", ascending=False)

    # PCA популяционного отклика: испытания в пространстве 778 нейронов.
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    from sklearn.decomposition import PCA
    pcs = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xz)
    return unit_df, region_df, pcs, y


def plot_unit_fdr(unit_df):
    """Вулкан-график: значимость отклика каждого юнита."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sig = unit_df["significant_fdr"]
    ax.scatter(unit_df.loc[~sig, "delta_mean"], -np.log10(unit_df.loc[~sig, "q_value_fdr"].clip(lower=1e-12)),
               s=14, c="#94a3b8", label="не значим (FDR q≥0.05)")
    ax.scatter(unit_df.loc[sig, "delta_mean"], -np.log10(unit_df.loc[sig, "q_value_fdr"].clip(lower=1e-12)),
               s=18, c="#dc2626", label="значим (FDR q<0.05)")
    ax.axhline(-np.log10(ALPHA), color="k", ls="--", lw=1)
    ax.set_xlabel("Δ среднего числа спайков (справа − слева) за окно")
    ax.set_ylabel("−log10(q-значение, FDR)")
    ax.set_title(f"Селективность юнитов к стороне стимула — значимо {int(sig.sum())} из {len(unit_df)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "ai_unit_fdr.png"
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path


def plot_pca(pcs, y):
    """PCA популяционного отклика: облако испытаний двух классов стимула."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color, label in [(0, "#2563eb", TARGET_LABELS["stimulus"][0]),
                              (1, "#ea580c", TARGET_LABELS["stimulus"][1])]:
        m = y == cls
        ax.scatter(pcs[m, 0], pcs[m, 1], s=26, alpha=0.75, c=color, label=label, edgecolors="none")
    ax.set_xlabel("Главная компонента 1 (PCA)")
    ax.set_ylabel("Главная компонента 2 (PCA)")
    ax.set_title("PCA популяционного отклика: 778 нейронов на испытание\n(стимул слева vs справа)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "ai_pca_population.png"
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path


# ======================================================================
# Оркестрация
# ======================================================================
def main():
    print("=" * 72)
    print("NeuroMouse — эксперименты и ИИ (расширенный анализ декодирования)")
    print("=" * 72)

    session = load_session()
    print(f"\nСессия: {session['identifier']} | животное: {session['subject']}")
    print(f"Юнитов: {session['n_units']} | испытаний: {len(session['trials'])}")
    uniq, cnt = np.unique(session["regions"], return_counts=True)
    print("Структуры мозга:", {u: int(c) for u, c in zip(uniq, cnt)})

    spike_times = session["spike_times"]
    trials = session["trials"]
    onsets = trials["start_time"].to_numpy(dtype=float)

    targets = build_targets(trials)
    for tname, tgt in targets.items():
        col = ALIGNMENT_COLUMNS[tname]
        if col in trials.columns:
            tgt_onset = trials[col].to_numpy(dtype=float)
            tgt["mask"] = tgt["mask"] & ~np.isnan(tgt_onset)
        else:
            tgt_onset = onsets
        tgt["onsets"] = tgt_onset
        y = tgt["y"][tgt["mask"]]
        print(f"Целевая переменная «{tname}»: {len(y)} испытаний, "
              f"баланс классов {np.bincount(y).tolist()}")

    summary = {
        "session": session["identifier"], "subject": session["subject"],
        "n_units": int(session["n_units"]), "n_trials": int(len(trials)),
        "main_window_ms": int(MAIN_WINDOW_S * 1000), "n_splits": N_SPLITS,
        "regions": {str(u): int(c) for u, c in zip(uniq, cnt)},
    }

    # --- e2 ---
    print("\n[A] Сравнение моделей (5-кратная CV) ...")
    cmp_df = experiment_model_comparison(targets, spike_times)
    cmp_df.to_csv(OUTPUT_DIR / "ai_model_comparison.csv", index=False)
    plot_model_comparison(cmp_df)
    best = cmp_df.loc[cmp_df["accuracy_mean"].idxmax()]
    summary["best_model_accuracy"] = {
        "target": best["target"], "model": best["model"],
        "accuracy": float(best["accuracy_mean"]),
    }

    # --- e3 ---
    print("\n[B] Гиперпараметры и отбор признаков (цель — сторона стимула) ...")
    m = targets["stimulus"]["mask"]
    X_stim = spike_count_matrix(spike_times, targets["stimulus"]["onsets"][m], MAIN_WINDOW_S)
    y_stim = targets["stimulus"]["y"][m]
    hp_df = experiment_hyperparameters(X_stim, y_stim)
    hp_df.to_csv(OUTPUT_DIR / "ai_hyperparam_search.csv", index=False)
    fs_df, scores_df = experiment_feature_selection(X_stim, y_stim)
    fs_df.to_csv(OUTPUT_DIR / "ai_feature_selection.csv", index=False)
    scores_df.to_csv(OUTPUT_DIR / "ai_feature_scores.csv", index=False)
    plot_feature_selection(fs_df)
    summary["best_hyperparameters"] = hp_df.set_index("model")["best_params"].to_dict()

    # --- e4 ---
    print("\n[C] Свип окна и целевой переменной ...")
    ws_df = experiment_window_sweep(targets, spike_times)
    ws_df.to_csv(OUTPUT_DIR / "ai_window_sweep.csv", index=False)
    plot_window_sweep(ws_df)
    for tname in targets:
        sub = ws_df[ws_df["target"] == tname]
        b = sub.loc[sub["accuracy_mean"].idxmax()]
        summary[f"best_window_{tname}"] = {
            "window_ms": int(b["window_ms"]), "model": b["model"],
            "accuracy": float(b["accuracy_mean"]),
        }

    # --- e5 ---
    print("\n[D] Углублённый анализ юнитов (FDR, структуры, PCA) ...")
    unit_df, region_df, pcs, y_pca = experiment_unit_analysis(
        spike_times, session["regions"], targets, MAIN_WINDOW_S)
    unit_df.to_csv(OUTPUT_DIR / "ai_unit_stats.csv", index=False)
    region_df.to_csv(OUTPUT_DIR / "ai_region_stats.csv", index=False)
    plot_unit_fdr(unit_df)
    n_sig = int(unit_df["significant_fdr"].sum())
    plot_pca(pcs, y_pca)
    summary["unit_analysis"] = {
        "n_units": int(len(unit_df)), "n_significant_fdr": n_sig,
        "fraction_significant": float(unit_df["significant_fdr"].mean()),
        "top_region": region_df.iloc[0]["region"] if len(region_df) else None,
    }
    print(f"Значимых юнитов (FDR q<0.05): {n_sig} из {len(unit_df)}")

    with open(OUTPUT_DIR / "ai_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"Готово. Артефакты в: {OUTPUT_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
