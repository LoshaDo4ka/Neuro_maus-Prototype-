"""
NeuroMouse — Mouse Neuropixels LFP analysis
Dataset: Allen Brain Observatory (Neuropixels, CA1 + VISl)
Задача: загрузить внеклеточный LFP, сравнить зрительную кору и гиппокамп.

Запуск из корня проекта в окружении neuromouse:
    python -m src.mouse_lfp_analysis
"""

import pynwb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from .config import MOUSE_ALLEN_NWB, MOUSE_STAGE_DIR

NWB_FILE = MOUSE_ALLEN_NWB
OUTPUT_DIR = MOUSE_STAGE_DIR


def main() -> int:
    if not NWB_FILE.exists():
        print(f"File not found: {NWB_FILE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading mouse Neuropixels NWB recording...")
    with pynwb.NWBHDF5IO(str(NWB_FILE), "r", load_namespaces=True) as io:
        nwb = io.read()

        # Electrodes
        elec_df = nwb.electrodes.to_dataframe()
        print(f"Total Neuropixels electrodes: {len(elec_df)}")

        # Acquisition
        lfp_series = nwb.acquisition["probe_729445654_lfp_data"]

        # Calculate sampling rate from first 1000 samples
        if hasattr(lfp_series, "rate") and lfp_series.rate is not None:
            rate = float(lfp_series.rate)
        elif lfp_series.timestamps is not None:
            t_sample = lfp_series.timestamps[:1000]
            dt = np.median(np.diff(t_sample))
            rate = 1.0 / dt
        else:
            rate = 1250.0
        print(f"Calculated LFP sampling rate: {rate:.1f} Hz")

        # Pick representative channels: Visual Cortex (VISl) and Hippocampus (CA1)
        vis_matches = np.where(elec_df["location"].values == "VISl")[0]
        ca1_matches = np.where(elec_df["location"].values == "CA1")[0]

        vis_idx = vis_matches[0]
        ca1_idx = ca1_matches[0]

        print(f"Selected channel indices: VISl (col {vis_idx}, id {elec_df.index[vis_idx]}), CA1 (col {ca1_idx}, id {elec_df.index[ca1_idx]})")

        # Read 10 seconds of data (from t=10s to t=20s)
        start_sample = int(10 * rate)
        n_samples = int(10 * rate)

        # Note: ElectricalSeries conversion factor is in volts, convert to uV
        conversion = lfp_series.conversion if hasattr(lfp_series, "conversion") else 1.0
        vis_signal = lfp_series.data[start_sample : start_sample + n_samples, vis_idx] * conversion * 1e6
        ca1_signal = lfp_series.data[start_sample : start_sample + n_samples, ca1_idx] * conversion * 1e6
        time_vec = np.arange(n_samples) / rate

        # 1. Plot raw traces (Visual Cortex vs Hippocampus)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(time_vec, vis_signal, color="#2563eb", linewidth=0.8)
        axes[0].set_title(f"Мышь: Зрительная кора (VISl, электрод {vis_idx}) — внеклеточный LFP", fontsize=12)
        axes[0].set_ylabel("Амплитуда (мкВ)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_vec, ca1_signal, color="#dc2626", linewidth=0.8)
        axes[1].set_title(f"Мышь: Гиппокамп (CA1, электрод {ca1_idx}) — тета-ритм", fontsize=12)
        axes[1].set_ylabel("Амплитуда (мкВ)")
        axes[1].set_xlabel("Время (секунды)")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        trace_path = OUTPUT_DIR / "mouse_lfp_traces_VISl_CA1.png"
        fig.savefig(trace_path, dpi=160)
        plt.close(fig)
        print(f"Saved traces plot to: {trace_path}")

        # 2. Plot Power Spectral Density (PSD)
        f_vis, psd_vis = welch(vis_signal, fs=rate, nperseg=int(2*rate))
        f_ca1, psd_ca1 = welch(ca1_signal, fs=rate, nperseg=int(2*rate))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(f_vis, psd_vis, label="Зрительная кора (VISl)", color="#2563eb", linewidth=1.5)
        ax.semilogy(f_ca1, psd_ca1, label="Гиппокамп (CA1)", color="#dc2626", linewidth=1.5)
        ax.set_xlim(0, 100)
        ax.set_title("Спектральная плотность мощности (PSD) мозга мыши: Зрительная кора vs CA1")
        ax.set_xlabel("Частота (Гц)")
        ax.set_ylabel("Мощность (мкВ²/Гц)")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        psd_path = OUTPUT_DIR / "mouse_lfp_psd_comparison.png"
        fig.savefig(psd_path, dpi=160)
        plt.close(fig)
        print(f"Saved PSD plot to: {psd_path}")

        # 3. Export channel inventory of mouse brain structures
        elec_summary = elec_df[["location", "probe_vertical_position", "probe_horizontal_position"]].copy()
        elec_summary.to_csv(OUTPUT_DIR / "mouse_electrodes_inventory.csv")
        print(f"Saved electrode inventory to: {OUTPUT_DIR / 'mouse_electrodes_inventory.csv'}")

    print("\nMouse analysis pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
