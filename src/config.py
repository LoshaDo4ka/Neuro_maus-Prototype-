"""NeuroMouse — единая конфигурация проекта.

Все пути и параметры обработки собраны здесь. Рабочие скрипты
(src/eeg_pipeline.py, src/mouse_lfp_analysis.py, src/mouse_spikes_decoder.py)
не содержат жёстко прописанных путей и импортируют всё из этого модуля.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Директории ----------
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

STAGE1_DIR = RESULTS_DIR / "stage1"          # конвейер ЭЭГ (Этап I)
MOUSE_STAGE_DIR = RESULTS_DIR / "mouse_stage"  # анализ и декодер мыши (Этапы II–III)

RANDOM_STATE = 42

# ---------- Этап I: конвейер обработки ЭЭГ ----------
EEG_INPUT_FILE = DATA_DIR / "S001R01.edf"
EEG_LOW_CUT_HZ = 1.0
EEG_HIGH_CUT_HZ = 40.0
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0

# Частотные диапазоны, Гц. Используются и в ЭЭГ-конвейере, и в извлечении признаков.
FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# ---------- Этапы II–III: электрофизиология мыши (Neuropixels) ----------
MOUSE_ALLEN_NWB = DATA_DIR / "mouse_allen_probe729445654.nwb"        # Allen (LFP: CA1 + VISl)
MOUSE_STEINMETZ_NWB = DATA_DIR / "mouse_steinmetz_ses20171031.nwb"    # Steinmetz (спайки)
MOUSE_DECODER_MODEL = MODELS_DIR / "mouse_lr_decoder.joblib"          # обученный декодер

# ---------- Этап IV: spike sorting и одиночные юниты ----------
# Локальные NWB не содержат сырого AP, поэтому spike sorting идёт на
# синтетической записи с известным ground truth (SpikeInterface).
STAGE4_DIR = MOUSE_STAGE_DIR / "stage4"
SYNTHETIC_SFREQ_HZ = 30000.0        # частота AP-полосы Neuropixels
SYNTHETIC_DURATION_S = 30.0
SYNTHETIC_N_CHANNELS = 16
SYNTHETIC_N_UNITS = 12
SYNTHETIC_FIRING_RATE_HZ = 8.0
SYNTHETIC_NOISE_LEVEL = 8.0
SYNTHETIC_SEED = 42
SORTER_NAME = "spykingcircus2"      # доступен в [full] и работает на Python 3.14

# Не считать метки событий заданными заранее:
# их нужно проверить для каждого набора данных.
