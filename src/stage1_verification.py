"""
NeuroMouse — верификация Этапа I (задачи 1.5, 1.8, 1.9, 1.10, 1.15, 1.16).

Скрипт ничего не меняет в конвейере: он читает уже готовые артефакты
(results/stage1/) и проверяет их. Результаты — в results/stage1/.

Запуск из корня проекта в окружении neuromouse:
    python -m src.stage1_verification
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if library_bin.is_dir():
        lib_str = str(library_bin)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_str)
            except Exception:
                pass
        curr_path = os.environ.get("PATH", "")
        if lib_str.lower() not in curr_path.lower():
            os.environ["PATH"] = lib_str + os.pathsep + curr_path

import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from .config import (
    EEG_INPUT_FILE,
    STAGE1_DIR,
    EEG_LOW_CUT_HZ,
    EEG_HIGH_CUT_HZ,
    WINDOW_SECONDS,
    STEP_SECONDS,
)

FILTERED_FIF = STAGE1_DIR / "S001R01_filtered_raw.fif"
FEATURES_CSV = STAGE1_DIR / "features_all_channels.csv"

# Каналы для сравнения PSD (задача 1.8). Имена без завершающих точек:
# реальные имена в EDF записаны как "O1..", "Fz..", "Fc5." и т. п.,
# поэтому сопоставление ведётся после нормализации (снять точки, нижний регистр).
PSD_CHANNELS = ["O1", "O2", "Fz", "Cz", "Pz", "FC5"]

# Имена колонок features_all_channels.csv с указанием единиц (задача 1.15).
FEATURE_UNITS = {
    "window_start_s": "с",
    "window_duration_s": "с",
    "channel": "-",
    "channel_index": "-",
    "power_1_30_hz_v2": "В²",
    "alpha_peak_hz": "Гц",
    "delta_power_v2": "В²",
    "theta_power_v2": "В²",
    "alpha_power_v2": "В²",
    "beta_power_v2": "В²",
    "delta_percent_of_1_30": "%",
    "theta_percent_of_1_30": "%",
    "alpha_percent_of_1_30": "%",
    "beta_percent_of_1_30": "%",
}


def _runs(mask: np.ndarray) -> int:
    """Максимальная длина непрерывной серии True."""
    if not mask.any():
        return 0
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(best)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Границы непрерывных серий True: список (start, stop_exclusive)."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _norm(name: str) -> str:
    return name.rstrip(".").lower()


def _resolve(ch_names: list[str], wanted: list[str]) -> list[str]:
    """Сопоставить желаемые имена каналов с реальными именами в записи."""
    norm_map: dict[str, str] = {}
    for c in ch_names:
        norm_map.setdefault(_norm(c), c)
    return [norm_map[_norm(w)] for w in wanted if _norm(w) in norm_map]


def check_missing_and_artifacts(raw: mne.io.BaseRaw, analysis_end_s: float
                                ) -> tuple[dict, pd.DataFrame]:
    """Задача 1.5: NaN/inf, нулевые участки, плоские каналы, насыщение.

    Нулевые/«полочные» серии ищутся ВНЕ глобальной нулевой забивки
    (отсчёты, где все каналы равны 0 одновременно). Такая забивка — это
    единичное событие записи, а не дефект каждого канала по отдельности.
    """
    data = raw.get_data() * 1e6  # мкВ
    sfreq = float(raw.info["sfreq"])

    # Глобальная нулевая забивка: отсчёты, где ВСЕ каналы равны нулю.
    all_zero = (data == 0).all(axis=0)
    blank_segments = [
        {"start_s": round(s / sfreq, 4), "end_s": round(e / sfreq, 4),
         "duration_s": round((e - s) / sfreq, 4)}
        for s, e in _segments(all_zero)
    ]
    blank_in_analysis = [
        seg for seg in blank_segments
        if seg["start_s"] < analysis_end_s - 1e-9 and seg["end_s"] > 1e-9
    ]
    useful = ~all_zero  # отсчёты вне глобальной забивки

    zero_run_limit = int(0.1 * sfreq)   # >0.1 с — подозрительно
    flat_run_limit = int(0.5 * sfreq)   # >0.5 с — подозрительно

    rows = []
    for i, ch in enumerate(raw.ch_names):
        x = data[i]
        ch_min, ch_max = float(np.min(x)), float(np.max(x))
        if ch_max > ch_min:
            at_extreme = (x == ch_min) | (x == ch_max)
        else:
            at_extreme = np.ones_like(x, dtype=bool)
        zero_runs_useful = (x == 0.0) & useful
        # «Полки» — совпадающие соседние отсчёты, вне глобальной забивки.
        flat = np.zeros_like(x, dtype=bool)
        same = np.diff(x) == 0.0
        flat[:-1] = same & useful[:-1] & useful[1:]
        rows.append({
            "channel": ch,
            "n_nan": int(np.isnan(x).sum()),
            "n_inf": int(np.isinf(x).sum()),
            "zero_samples_total": int((x == 0.0).sum()),
            "zero_samples_in_blank": int(((x == 0.0) & ~useful).sum()),
            "max_zero_run_samples_useful": _runs(zero_runs_useful),
            "std_uv": float(np.std(x)),
            "flat_channel": bool(np.std(x) == 0.0),
            "max_flat_run_samples_useful": _runs(flat),
            "at_extreme_fraction": float(at_extreme.mean()),
        })
    df = pd.DataFrame(rows)

    total_nan = int(df["n_nan"].sum())
    total_inf = int(df["n_inf"].sum())
    flat_channels = df.loc[df["flat_channel"], "channel"].tolist()
    # Насыщение: >5% отсчётов зажаты на минимуме/максимуме канала
    saturated = df.loc[df["at_extreme_fraction"] > 0.05, "channel"].tolist()
    long_flat = df.loc[df["max_flat_run_samples_useful"] > flat_run_limit, "channel"].tolist()
    long_zero = df.loc[df["max_zero_run_samples_useful"] > zero_run_limit, "channel"].tolist()

    summary = {
        "total_nan": total_nan,
        "total_inf": total_inf,
        "flat_channels": flat_channels,
        "saturated_channels": saturated,
        "long_flat_runs_channels": long_flat,
        "long_zero_runs_channels": long_zero,
        "blank_segments": blank_segments,
        "blank_segments_in_analysis_region": blank_in_analysis,
        "analysis_region_end_s": round(analysis_end_s, 4),
        "min_std_uv": float(df["std_uv"].min()),
        "max_at_extreme_fraction": float(df["at_extreme_fraction"].max()),
        "passed": (total_nan == 0 and total_inf == 0 and not flat_channels
                   and not saturated and not long_flat and not long_zero
                   and not blank_in_analysis),
    }
    return summary, df


def check_features_schema(features: pd.DataFrame) -> dict:
    """Задачи 1.15 / 1.3.7: колонки CSV, единицы и отсутствие смешения В² и %."""
    missing = [c for c in FEATURE_UNITS if c not in features.columns]
    extra = [c for c in features.columns if c not in FEATURE_UNITS]
    # Абсолютная мощность (В²) и относительная (%) должны быть разными колонками.
    abs_cols = [c for c in features.columns if c.endswith("_power_v2") or c.endswith("_hz_v2")]
    rel_cols = [c for c in features.columns if c.endswith("_percent_of_1_30")]
    if rel_cols:
        ratio_ok = bool(np.nanmax(features[rel_cols].to_numpy()) <= 100.0 + 1e-6)
    else:
        ratio_ok = True
    return {
        "columns": list(features.columns),
        "units": {k: v for k, v in FEATURE_UNITS.items() if k in features.columns},
        "missing_columns": missing,
        "extra_columns": extra,
        "absolute_columns": abs_cols,
        "relative_columns": rel_cols,
        "relative_within_0_100": bool(ratio_ok),
        "passed": bool(not missing and not extra and ratio_ok),
    }


def check_psd(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw) -> tuple[Path, Path, dict]:
    """Задача 1.8: PSD до/после фильтрации для нескольких каналов."""
    sfreq = float(raw.info["sfreq"])
    names = _resolve(raw.ch_names, PSD_CHANNELS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()
    rows = []
    for ax, ch in zip(axes, names):
        x_raw = raw.get_data(picks=[ch])[0] * 1e6
        x_flt = filtered.get_data(picks=[ch])[0] * 1e6
        f, p_raw = welch(x_raw, fs=sfreq, nperseg=int(4 * sfreq))
        _, p_flt = welch(x_flt, fs=sfreq, nperseg=int(4 * sfreq))
        ax.semilogy(f, p_raw, label="до фильтрации", color="#94a3b8", linewidth=1.0)
        ax.semilogy(f, p_flt, label="после (1–40 Гц)", color="#2563eb", linewidth=1.2)
        ax.axvspan(EEG_LOW_CUT_HZ, EEG_HIGH_CUT_HZ, color="#22c55e", alpha=0.06)
        ax.axvline(EEG_HIGH_CUT_HZ, color="#dc2626", linestyle="--", linewidth=0.8)
        ax.set_title(ch)
        ax.set_xlim(0, 80)
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("PSD, мкВ²/Гц")
        ax.grid(True, alpha=0.3)

        band = (f > EEG_HIGH_CUT_HZ)
        p_out_raw = float(np.trapezoid(p_raw[band], f[band])) if band.any() else 0.0
        p_out_flt = float(np.trapezoid(p_flt[band], f[band])) if band.any() else 0.0
        ratio_db = 10 * np.log10((p_out_flt + 1e-30) / (p_out_raw + 1e-30))
        rows.append({
            "channel": ch,
            "power_above_40hz_raw_uv2": p_out_raw,
            "power_above_40hz_filtered_uv2": p_out_flt,
            "suppression_db": float(ratio_db),
        })
    axes[0].legend(fontsize=8)
    fig.suptitle("PSD до и после полосовой фильтрации 1–40 Гц (несколько каналов)", fontsize=13)
    fig.tight_layout()
    plot_path = STAGE1_DIR / "psd_before_after_channels.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    df = pd.DataFrame(rows)
    csv_path = STAGE1_DIR / "psd_suppression.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "channels": names,
        "min_suppression_db": float(df["suppression_db"].min()),
        "mean_suppression_db": float(df["suppression_db"].mean()),
        "passed": bool((df["suppression_db"] < -3).all()),  # >3 дБ подавления вне полосы
    }
    return plot_path, csv_path, summary


def check_edges(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw) -> tuple[Path, dict]:
    """Задача 1.9: артефакты ('звон') на краях записи."""
    sfreq = float(raw.info["sfreq"])
    edge_s = 1.0  # анализируем первые и последние 1 с
    n_edge = int(edge_s * sfreq)
    data = filtered.get_data() * 1e6
    rms_edge = np.sqrt(np.mean(data[:, :n_edge] ** 2, axis=1))
    rms_edge_end = np.sqrt(np.mean(data[:, -n_edge:] ** 2, axis=1))
    rms_mid = np.sqrt(np.mean(data[:, n_edge:-n_edge] ** 2, axis=1))
    ratio_start = rms_edge / np.where(rms_mid > 0, rms_mid, np.nan)
    ratio_end = rms_edge_end / np.where(rms_mid > 0, rms_mid, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t = np.arange(n_edge) / sfreq
    ch = (_resolve(raw.ch_names, ["O1"]) or raw.ch_names[:1])[0]
    axes[0].plot(t, raw.get_data(picks=[ch])[0, :n_edge] * 1e6, label="до", color="#94a3b8")
    axes[0].plot(t, filtered.get_data(picks=[ch])[0, :n_edge] * 1e6, label="после", color="#2563eb")
    axes[0].set_title(f"Начало записи, канал {ch}")
    axes[0].set_xlabel("Время, с")
    axes[0].set_ylabel("мкВ")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bins = np.arange(0, raw.n_times / sfreq, edge_s)
    idx = np.searchsorted(np.arange(raw.n_times) / sfreq, bins)
    rms_time = [np.sqrt(np.mean(data[:, idx[i]:idx[i + 1]] ** 2)) for i in range(len(idx) - 1)]
    axes[1].plot(bins[:-1], rms_time, marker="o", markersize=3, color="#0f766e")
    axes[1].set_title("RMS по секундам (все каналы), видна ли «звонь» на краях")
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("RMS, мкВ")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = STAGE1_DIR / "edge_effects.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "edge_window_s": edge_s,
        "max_rms_ratio_start": float(np.nanmax(ratio_start)),
        "max_rms_ratio_end": float(np.nanmax(ratio_end)),
        # Порог 2.0 — заметный, но не катастрофический «звон»
        "passed": bool(np.nanmax(ratio_start) < 2.0 and np.nanmax(ratio_end) < 2.0),
    }
    return plot_path, summary


def check_event_timing(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw) -> dict:
    """Задача 1.10: не смещаются ли метки событий после фильтрации."""
    ann_raw = raw.annotations
    ann_flt = filtered.annotations
    same_count = len(ann_raw) == len(ann_flt)
    max_onset_diff = 0.0
    if same_count:
        max_onset_diff = float(np.max(np.abs(ann_raw.onset - ann_flt.onset))) if len(ann_raw) else 0.0

    # Эмпирическая проверка задержки: кросс-корреляция сигнала до/после на среднем участке
    sfreq = float(raw.info["sfreq"])
    ch = (_resolve(raw.ch_names, ["O1"]) or raw.ch_names[:1])[0]
    start = int(10 * sfreq)
    stop = int(30 * sfreq)
    a = raw.get_data(picks=[ch])[0, start:stop]
    b = filtered.get_data(picks=[ch])[0, start:stop]
    a = a - a.mean()
    b = b - b.mean()
    corr = np.correlate(b, a, mode="full")
    lag = int(np.argmax(corr) - (len(a) - 1))
    lag_ms = lag / sfreq * 1000.0

    return {
        "n_annotations_raw": int(len(ann_raw)),
        "n_annotations_filtered": int(len(ann_flt)),
        "annotations_identical": bool(same_count and max_onset_diff == 0.0),
        "max_onset_diff_s": max_onset_diff,
        "measured_lag_samples": lag,
        "measured_lag_ms": float(lag_ms),
        "passed": bool(same_count and max_onset_diff == 0.0 and abs(lag) <= 1),
    }


def check_window_bounds(raw: mne.io.BaseRaw, features: pd.DataFrame) -> dict:
    """Задача 1.16: корректность границ временных окон в CSV."""
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    starts = np.sort(features["window_start_s"].unique())
    expected = np.arange(0.0, duration - WINDOW_SECONDS + 1e-9, STEP_SECONDS)
    expected = np.round(expected, 3)
    actual = np.round(starts, 3)
    last_end = float(starts[-1] + WINDOW_SECONDS)
    dur_vals = sorted(features["window_duration_s"].unique().tolist())
    return {
        "recording_duration_s": duration,
        "n_windows_in_csv": int(len(starts)),
        "n_windows_expected": int(len(expected)),
        "starts_match_expected": bool(np.array_equal(actual, expected)),
        "first_window_start_s": float(starts[0]),
        "last_window_start_s": float(starts[-1]),
        "last_window_end_s": last_end,
        "last_window_within_recording": bool(last_end <= duration + 1e-9),
        "window_duration_values": dur_vals,
        "rows_total": int(len(features)),
        "channels": int(features["channel"].nunique()),
        "passed": bool(np.array_equal(actual, expected) and last_end <= duration + 1e-9
                       and len(features) == len(starts) * features["channel"].nunique()),
    }


def main() -> int:
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Stage 1 verification started.")

    raw = mne.io.read_raw_edf(EEG_INPUT_FILE, preload=True, verbose="ERROR")
    if FILTERED_FIF.exists():
        filtered = mne.io.read_raw_fif(FILTERED_FIF, preload=True, verbose="ERROR")
    else:
        filtered = raw.copy().filter(l_freq=EEG_LOW_CUT_HZ, h_freq=EEG_HIGH_CUT_HZ,
                                     method="fir", phase="zero", fir_design="firwin",
                                     verbose="ERROR")
    features = pd.read_csv(FEATURES_CSV)

    # Граница «полезной» области записи: конец последней аннотации, если они есть.
    if len(raw.annotations):
        analysis_end_s = float(np.max(raw.annotations.onset + raw.annotations.duration))
    else:
        analysis_end_s = raw.n_times / float(raw.info["sfreq"])

    q_summary, q_df = check_missing_and_artifacts(raw, analysis_end_s)
    q_df.to_csv(STAGE1_DIR / "quality_metrics_extended.csv", index=False)
    psd_plot, psd_csv, psd_summary = check_psd(raw, filtered)
    edge_plot, edge_summary = check_edges(raw, filtered)
    timing_summary = check_event_timing(raw, filtered)
    window_summary = check_window_bounds(raw, features)
    schema_summary = check_features_schema(features)

    all_passed = (q_summary["passed"] and psd_summary["passed"]
                  and edge_summary["passed"] and timing_summary["passed"]
                  and window_summary["passed"] and schema_summary["passed"])

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "all_checks_passed": bool(all_passed),
        "checks": {
            "1.5_missing_and_artifacts": q_summary,
            "1.8_psd_before_after": psd_summary,
            "1.9_edge_effects": edge_summary,
            "1.10_event_timing": timing_summary,
            "1.15_features_schema_and_units": schema_summary,
            "1.16_window_bounds": window_summary,
        },
    }
    (STAGE1_DIR / "verification_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logging.info("Verification done. all_checks_passed=%s", all_passed)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
