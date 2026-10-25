"""
NeuroMouse — Stage 1 reproducible EEG pipeline
Input:  data/S001R01.edf
Output: results/stage1/

Запуск из корня проекта в окружении neuromouse:
    python -m src.eeg_pipeline
"""

from pathlib import Path
import json
import logging
import os
import sys
from datetime import datetime

# Windows DLL directory fix for Conda environments when invoked directly
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

from .config import (
    PROJECT_ROOT,
    EEG_INPUT_FILE,
    STAGE1_DIR,
    EEG_LOW_CUT_HZ,
    EEG_HIGH_CUT_HZ,
    WINDOW_SECONDS,
    STEP_SECONDS,
    FREQUENCY_BANDS,
)

# ---------- Configuration ----------
INPUT_FILE = EEG_INPUT_FILE
OUTPUT_DIR = STAGE1_DIR

LOW_CUT_HZ = EEG_LOW_CUT_HZ
HIGH_CUT_HZ = EEG_HIGH_CUT_HZ
BANDS = FREQUENCY_BANDS


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integrate PSD (V²/Hz) within [low, high) to estimate power in V²."""
    mask = (freqs >= low) & (freqs < high)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def build_features(raw_filtered: mne.io.BaseRaw) -> pd.DataFrame:
    """Create one row per channel and 4-s window, stepped every 2 s."""
    from scipy.signal import welch

    sfreq = float(raw_filtered.info["sfreq"])
    data = raw_filtered.get_data()  # Volts; shape = channels x samples
    window_size = int(round(WINDOW_SECONDS * sfreq))
    step_size = int(round(STEP_SECONDS * sfreq))
    rows = []

    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window and step sizes must be positive.")

    for start in range(0, data.shape[1] - window_size + 1, step_size):
        stop = start + window_size
        segment_start_s = start / sfreq
        for ch_idx, ch_name in enumerate(raw_filtered.ch_names):
            segment = data[ch_idx, start:stop]
            freqs, psd = welch(
                segment,
                fs=sfreq,
                nperseg=min(window_size, len(segment)),
            )
            powers = {
                name: band_power(freqs, psd, low, high)
                for name, (low, high) in BANDS.items()
            }
            total_1_30 = float(np.nansum(list(powers.values())))
            alpha_mask = (freqs >= BANDS["alpha"][0]) & (freqs < BANDS["alpha"][1])
            if alpha_mask.any():
                alpha_peak_hz = float(freqs[alpha_mask][np.argmax(psd[alpha_mask])])
            else:
                alpha_peak_hz = float("nan")

            row = {
                "window_start_s": segment_start_s,
                "window_duration_s": WINDOW_SECONDS,
                "channel": ch_name,
                "channel_index": ch_idx,
                "power_1_30_hz_v2": total_1_30,
                "alpha_peak_hz": alpha_peak_hz,
            }
            for band_name, power in powers.items():
                row[f"{band_name}_power_v2"] = power
                row[f"{band_name}_percent_of_1_30"] = (
                    100.0 * power / total_1_30
                    if total_1_30 > 0 and np.isfinite(power)
                    else float("nan")
                )
            rows.append(row)

    return pd.DataFrame(rows)


def make_plots(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw, features: pd.DataFrame,
               output_dir: Path) -> None:
    sfreq = float(raw.info["sfreq"])
    ch_names = raw.ch_names
    preferred = next((ch for ch in ch_names if ch.startswith("O1")), ch_names[0])
    start_s, end_s = 34.0, 42.0
    start = int(start_s * sfreq)
    stop = min(int(end_s * sfreq), raw.n_times)
    times = np.arange(start, stop) / sfreq

    x = raw.get_data(picks=[preferred], start=start, stop=stop)[0] * 1e6
    y = filtered.get_data(picks=[preferred], start=start, stop=stop)[0] * 1e6

    # 1. Сравнение сигнала канала O1 до и после фильтрации
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(times, x, linewidth=0.8)
    axes[0].set_title(f"Исходный сигнал {preferred}, {start_s:.0f}–{end_s:.0f} с")
    axes[1].plot(times, y, linewidth=0.8)
    axes[1].set_title(f"После фильтрации {LOW_CUT_HZ:g}–{HIGH_CUT_HZ:g} Гц")
    for ax in axes:
        ax.set_ylabel("Амплитуда, мкВ")
        ax.grid(True)
    axes[-1].set_xlabel("Время, с")
    fig.tight_layout()
    fig.savefig(output_dir / "o1_filter_comparison_34_42s.png", dpi=160)
    plt.close(fig)

    # 2. Многоканальная карта RMS
    data_uv = raw.get_data() * 1e6
    win_s = 0.5
    win_size = int(round(win_s * sfreq))
    n_windows = data_uv.shape[1] // win_size
    rms_map = np.empty((data_uv.shape[0], n_windows), dtype=float)
    for i in range(n_windows):
        seg = data_uv[:, i * win_size:(i + 1) * win_size]
        rms_map[:, i] = np.sqrt(np.mean(seg**2, axis=1))
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        rms_map, aspect="auto", origin="lower", interpolation="nearest",
        extent=[0, n_windows * win_s, 0, data_uv.shape[0]],
    )
    fig.colorbar(im, ax=ax, label="RMS, мкВ")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Индекс канала (порядок EDF)")
    ax.set_title("Многоканальная карта RMS — исходная ЭЭГ")
    fig.tight_layout()
    fig.savefig(output_dir / "multichannel_rms_map.png", dpi=160)
    plt.close(fig)

    # 3. RMS по окнам для канала O1
    raw_data = raw.get_data(picks=[preferred])[0]
    filtered_data = filtered.get_data(picks=[preferred])[0]
    window_size = int(round(WINDOW_SECONDS * sfreq))
    step_size = int(round(STEP_SECONDS * sfreq))
    rms_rows = []
    for start in range(0, len(raw_data) - window_size + 1, step_size):
        a = raw_data[start:start + window_size] * 1e6
        b = filtered_data[start:start + window_size] * 1e6
        rms_rows.append({
            "window_start_s": start / sfreq,
            "rms_raw_uv": np.sqrt(np.mean(a**2)),
            "rms_filtered_uv": np.sqrt(np.mean(b**2)),
            "rms_difference_uv": np.sqrt(np.mean((a-b)**2)),
        })
    rms_df = pd.DataFrame(rms_rows)
    rms_df.to_csv(output_dir / "o1_rms_by_window.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rms_df["window_start_s"], rms_df["rms_raw_uv"], label="До фильтрации")
    ax.plot(rms_df["window_start_s"], rms_df["rms_filtered_uv"], label="После фильтрации")
    ax.plot(rms_df["window_start_s"], rms_df["rms_difference_uv"], label="Разность")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("RMS, мкВ")
    ax.set_title(f"RMS по окнам — канал {preferred}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "o1_rms_by_window.png", dpi=160)
    plt.close(fig)

    # 4. Средняя мощность частотных диапазонов по окнам
    power_cols = [f"{band}_power_v2" for band in BANDS]
    mean_features = features.groupby("window_start_s")[power_cols].mean().reset_index()
    mean_features.to_csv(output_dir / "mean_band_power_by_window.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for band in BANDS:
        ax.plot(
            mean_features["window_start_s"],
            mean_features[f"{band}_power_v2"],
            marker="o", markersize=2, label=band
        )
    ax.set_xlabel("Начало окна, с")
    ax.set_ylabel("Средняя мощность по каналам, В²")
    ax.set_title("Средняя мощность частотных диапазонов по окнам")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mean_band_power_by_window.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    logging.info("NeuroMouse Stage 1 pipeline started.")
    logging.info("Input file: %s", INPUT_FILE)

    if not INPUT_FILE.exists():
        logging.error("EDF not found: %s", INPUT_FILE)
        logging.error("Check that S001R01.edf exists in %s", PROJECT_ROOT / "data")
        return 1

    try:
        raw = mne.io.read_raw_edf(INPUT_FILE, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        duration_s = raw.n_times / sfreq
        logging.info("Loaded: %s", raw)
        logging.info("Channels=%d; sfreq=%.3f Hz; duration=%.3f s",
                     len(raw.ch_names), sfreq, duration_s)

        # Сохраняем исходные данные без изменений; фильтруем копию
        logging.info("CHECKPOINT 1: creating a copy of the raw recording.")
        filtered = raw.copy()

        logging.info("CHECKPOINT 2: starting 1-40 Hz FIR filtering.")
        filtered.filter(
            l_freq=LOW_CUT_HZ,
            h_freq=HIGH_CUT_HZ,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose="INFO",
        )
        logging.info("CHECKPOINT 3: filtering completed.")

        # Сохранение отфильтрованного сигнала и метаданных
        logging.info("CHECKPOINT 3A: saving filtered FIF.")
        filtered.save(OUTPUT_DIR / "S001R01_filtered_raw.fif", overwrite=True, verbose="ERROR")
        logging.info("CHECKPOINT 3B: filtered FIF saved.")
        logging.info("CHECKPOINT 3C: writing channel inventory.")
        pd.DataFrame({
            "channel_index": range(len(raw.ch_names)),
            "channel": raw.ch_names,
            "channel_type": [raw.get_channel_types(picks=[ch])[0] for ch in raw.ch_names],
        }).to_csv(OUTPUT_DIR / "channel_inventory.csv", index=False)
        logging.info("CHECKPOINT 3D: channel inventory saved.")

        logging.info("CHECKPOINT 4A: starting feature calculation.")
        features = build_features(filtered)
        logging.info("CHECKPOINT 4B: feature calculation completed; rows=%d", len(features))
        features.to_csv(OUTPUT_DIR / "features_all_channels.csv", index=False)
        logging.info("CHECKPOINT 4C: features CSV saved.")

        # Метрики качества каналов по всей записи
        raw_data_uv = raw.get_data() * 1e6
        filt_data_uv = filtered.get_data() * 1e6
        quality_rows = []
        for i, ch in enumerate(raw.ch_names):
            x = raw_data_uv[i]
            y = filt_data_uv[i]
            quality_rows.append({
                "channel": ch,
                "rms_raw_uv": float(np.sqrt(np.mean(x**2))),
                "rms_filtered_uv": float(np.sqrt(np.mean(y**2))),
                "rms_difference_uv": float(np.sqrt(np.mean((x-y)**2))),
                "min_raw_uv": float(np.min(x)),
                "max_raw_uv": float(np.max(x)),
            })
        pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "channel_quality_metrics.csv", index=False)

        make_plots(raw, filtered, features, OUTPUT_DIR)
        logging.info("CHECKPOINT 4: plots completed.")

        metadata = {
            "project": "NeuroMouse",
            "stage": 1,
            "input_file": str(INPUT_FILE),
            "mne_version": mne.__version__,
            "sampling_frequency_hz": float(sfreq),
            "n_channels": int(len(raw.ch_names)),
            "n_samples": int(raw.n_times),
            "duration_seconds": float(duration_s),
            "filter": {
                "type": "FIR band-pass",
                "low_cut_hz": LOW_CUT_HZ,
                "high_cut_hz": HIGH_CUT_HZ,
                "phase": "zero",
                "fir_design": "firwin",
            },
            "features": {
                "window_seconds": WINDOW_SECONDS,
                "step_seconds": STEP_SECONDS,
                "bands_hz": BANDS,
                "rows_in_features_csv": int(len(features)),
                "note": "Band powers are Welch PSD integrals; percentages are relative to the sum of delta/theta/alpha/beta power from 1–30 Hz.",
            },
            "limitations": [
                "This is a technical preprocessing prototype, not a diagnostic system.",
                "One recording from one participant is insufficient to establish generalization.",
                "High RMS is a quality-control flag, not proof of an artifact or pathology.",
                "Channel positions are not used for the RMS map; its vertical axis is EDF channel order.",
            ],
            "generated_at": datetime.now().astimezone().isoformat(),
        }
        (OUTPUT_DIR / "run_metadata.json").write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                default=lambda o: int(o) if isinstance(o, (np.integer, int)) else float(o) if isinstance(o, (np.floating, float)) else str(o),
            ),
            encoding="utf-8",
        )
        logging.info("CHECKPOINT 5: metadata saved.")

        summary = [
            "# NeuroMouse — отчёт первого этапа",
            "",
            "## Входные данные",
            f"- Файл: `{INPUT_FILE}`",
            f"- Каналов: {len(raw.ch_names)}",
            f"- Частота дискретизации: {sfreq:.3f} Гц",
            f"- Длительность: {duration_s:.3f} с",
            "",
            "## Обработка",
            f"- Полосовой FIR-фильтр: {LOW_CUT_HZ:g}–{HIGH_CUT_HZ:g} Гц",
            f"- Окно признаков: {WINDOW_SECONDS:g} с; шаг: {STEP_SECONDS:g} с",
            "- Признаки: мощность delta/theta/alpha/beta и частота альфа-пика по каналам и окнам",
            "",
            "## Выходные файлы",
            "- `S001R01_filtered_raw.fif` — обработанная запись",
            "- `features_all_channels.csv` — признаки всех каналов и окон",
            "- `channel_quality_metrics.csv` — RMS и диапазон амплитуды по каналам",
            "- `channel_inventory.csv` — перечень каналов",
            "- `o1_filter_comparison_34_42s.png` — сравнение O1 до/после фильтрации",
            "- `multichannel_rms_map.png` — карта RMS всех каналов",
            "- `o1_rms_by_window.png` и `o1_rms_by_window.csv` — RMS по окнам",
            "- `mean_band_power_by_window.png` и `.csv` — средняя мощность диапазонов",
            "- `run_metadata.json` — параметры и ограничения",
            "- `pipeline.log` — журнал выполнения",
            "",
            "## Ограничения",
            "Это технический прототип обработки, а не диагностическая модель. Одна запись не позволяет обучить или валидировать обобщающую модель. Высокая амплитуда сама по себе не доказывает артефакт или патологическую активность.",
        ]
        (OUTPUT_DIR / "stage1_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        logging.info("CHECKPOINT 6: report saved.")

        logging.info("Pipeline completed successfully.")
        logging.info("Output directory: %s", OUTPUT_DIR)
        logging.info("Feature rows: %d", len(features))
        return 0

    except Exception:
        logging.exception("Pipeline failed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
