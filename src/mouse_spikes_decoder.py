"""
NeuroMouse — Mouse Neuropixels Spike Decoder
Dataset: Steinmetz et al. (2019) / Neuropixels in behaving mice
Target: Decode visual stimulus condition from population neural spike counts.

Запуск из корня проекта в окружении neuromouse:
    python -m src.mouse_spikes_decoder
"""

import pynwb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import joblib

from .config import (
    MOUSE_STEINMETZ_NWB,
    MOUSE_STAGE_DIR,
    MOUSE_DECODER_MODEL,
    RANDOM_STATE,
)

DATA_FILE = MOUSE_STEINMETZ_NWB
OUTPUT_DIR = MOUSE_STAGE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return 1

    print(f"Loading mouse Neuropixels session: {DATA_FILE.name}...")
    with pynwb.NWBHDF5IO(str(DATA_FILE), "r", load_namespaces=True) as io:
        nwb = io.read()
        
        print(f"Session identifier: {nwb.identifier}")
        print(f"Experimenter: {nwb.experimenter}")
        if nwb.subject:
            print(f"Subject species: {nwb.subject.species}, description: {nwb.subject.description}")

        # 1. Units (Neurons)
        units = nwb.units
        n_units = len(units)
        print(f"\nTotal sorted neurons (units): {n_units}")
        
        units_df = units.to_dataframe()
        print("Units summary:")
        if "location" in units_df.columns:
            print("Recorded brain structures:")
            print(units_df["location"].value_counts().head(10))

        # 2. Trials (Visual Stimuli & Behavior)
        trials = nwb.trials
        n_trials = len(trials)
        print(f"\nTotal behavioral trials: {n_trials}")
        trials_df = trials.to_dataframe()
        print("Trial table columns:", list(trials_df.columns))

        # Determine visual stimulus columns
        # In Steinmetz NWB, stimulus columns typically are:
        # visual_stimulus_left_contrast, visual_stimulus_right_contrast
        # or visual_stimulus_type
        left_col = [c for c in trials_df.columns if "left" in c.lower() and "contrast" in c.lower()]
        right_col = [c for c in trials_df.columns if "right" in c.lower() and "contrast" in c.lower()]

        print(f"Found stimulus contrast columns: left={left_col}, right={right_col}")
        
        # 3. Extract Spike Times for each neuron
        print("\nExtracting spike times per neuron...")
        spike_times_list = []
        for unit_idx in range(n_units):
            spikes = units["spike_times"][unit_idx]
            spike_times_list.append(spikes)

        # 4. Raster Plot for first 50 neurons during first 30 seconds
        print("Generating mouse population raster plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(min(50, n_units)):
            spks = spike_times_list[i]
            spks_win = spks[(spks >= 0) & (spks <= 30)]
            ax.vlines(spks_win, i - 0.4, i + 0.4, color="#1e293b", linewidth=0.7)
            
        ax.set_xlabel("Время (секунды)", fontsize=11)
        ax.set_ylabel("Индекс нейрона (единица)", fontsize=11)
        ax.set_title("Спайковая активность популяции нейронов мыши (Neuropixels, первые 30 с)", fontsize=12)
        ax.set_xlim(0, 30)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        raster_path = OUTPUT_DIR / "mouse_population_raster.png"
        fig.savefig(str(raster_path), dpi=100)
        plt.close(fig)
        print(f"Raster plot saved to: {raster_path}")

        # 5. Build Population Spike Count Feature Matrix X for each trial
        # Window: [start_time, start_time + 0.4s] after stimulus onset
        print("\nBuilding trial feature matrix (spike counts per neuron)...")
        start_times = trials_df["start_time"].values
        window_duration = 0.4  # 400 ms post-stimulus

        X = np.zeros((n_trials, n_units), dtype=np.float32)
        for i in range(n_units):
            spks = spike_times_list[i]
            for tr in range(n_trials):
                t0 = start_times[tr]
                t1 = t0 + window_duration
                # Count spikes in window
                X[tr, i] = np.sum((spks >= t0) & (spks < t1))

        # 6. Formulate Target y: Visual Stimulus Condition
        # If left contrast != right contrast, target = 0 (left stronger) vs 1 (right stronger)
        if left_col and right_col:
            c_left = trials_df[left_col[0]].values
            c_right = trials_df[right_col[0]].values
            
            # Filter trials where there was a stimulus difference
            valid_mask = (c_left != c_right) & ~np.isnan(c_left) & ~np.isnan(c_right)
            X_dec = X[valid_mask]
            y = (c_right[valid_mask] > c_left[valid_mask]).astype(int)
            class_names = ["Стимул Слева", "Стимул Справа"]
            task_name = "Декодирование положения стимула (Левый vs Правый)"
        else:
            # Fallback: choice / response decoding
            response_col = [c for c in trials_df.columns if "response" in c.lower() or "choice" in c.lower()][0]
            y_raw = trials_df[response_col].values
            valid_mask = ~np.isnan(y_raw)
            X_dec = X[valid_mask]
            y = (y_raw[valid_mask] > 0).astype(int)
            class_names = ["Класс 0", "Класс 1"]
            task_name = f"Декодирование {response_col}"

        print(f"\nTask: {task_name}")
        print(f"Samples for decoding: {len(y)}, Neurons (features): {X_dec.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")

        # 7. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_dec, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )

        # 8. Train Baseline Models
        print("\nTraining Logistic Regression decoder...")
        lr = LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        b_acc_lr = balanced_accuracy_score(y_test, y_pred_lr)
        print(f"Logistic Regression — Accuracy: {acc_lr*100:.2f}%, Balanced Accuracy: {b_acc_lr*100:.2f}%")

        print("Training LDA decoder...")
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        b_acc_lda = balanced_accuracy_score(y_test, y_pred_lda)
        print(f"LDA — Accuracy: {acc_lda*100:.2f}%, Balanced Accuracy: {b_acc_lda*100:.2f}%")

        # 9. Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_lr)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Предсказание модели ИИ")
        ax.set_ylabel("Реальный стимул мыши")
        ax.set_title(f"Матрица ошибок ИИ-декодера\nТочность: {acc_lr*100:.1f}%")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="black" if cm[r, c] < cm.max()/2 else "white", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        cm_path = OUTPUT_DIR / "mouse_decoder_confusion_matrix.png"
        fig.savefig(str(cm_path), dpi=100)
        plt.close(fig)
        print(f"Confusion matrix saved to: {cm_path}")

        # 10. Save trained model & metadata
        model_save_path = MOUSE_DECODER_MODEL
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, model_save_path)
        print(f"Saved trained decoder model to: {model_save_path}")

        # Save metrics summary
        metrics_df = pd.DataFrame([{
            "model": "LogisticRegression",
            "accuracy": acc_lr,
            "balanced_accuracy": b_acc_lr,
            "n_neurons": n_units,
            "n_trials": len(y),
            "test_samples": len(y_test),
        }, {
            "model": "LDA",
            "accuracy": acc_lda,
            "balanced_accuracy": b_acc_lda,
            "n_neurons": n_units,
            "n_trials": len(y),
            "test_samples": len(y_test),
        }])
        metrics_df.to_csv(str(OUTPUT_DIR / "mouse_decoder_metrics.csv"), index=False)
        print(f"Saved metrics to: {OUTPUT_DIR / 'mouse_decoder_metrics.csv'}")

    print("\nMouse spike decoding pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    main()
