"""Feature extraction for EEG machine learning."""

def extract_bandpower(raw, channel, bands=None):
    """Calculate Welch power in predefined frequency bands."""
    import numpy as np
    from scipy.signal import welch
    from .config import FREQUENCY_BANDS

    bands = bands or FREQUENCY_BANDS

    if channel not in raw.ch_names:
        raise ValueError(
            f"Unknown channel {channel!r}. "
            f"Available channels include: {raw.ch_names[:10]}"
        )

    signal = raw.get_data(picks=[channel])[0]
    sfreq = raw.info["sfreq"]

    frequencies, psd = welch(signal, fs=sfreq, nperseg=min(
        int(sfreq * 2), len(signal)
    ))

    result = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        result[name] = (
            float(np.trapezoid(psd[mask], frequencies[mask]))
            if mask.sum() >= 2 else 0.0
        )

    return result
