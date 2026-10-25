"""Signal loading and preprocessing."""

def load_recording(path):
    """Load an EEG recording using MNE."""
    from pathlib import Path
    import mne

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")

    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")


def bandpass_filter(raw, low_hz=1.0, high_hz=40.0):
    """Return a filtered copy; preserve the original recording."""
    filtered = raw.copy()
    filtered.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        verbose="ERROR",
    )
    return filtered
