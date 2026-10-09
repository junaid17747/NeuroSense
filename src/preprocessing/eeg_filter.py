import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, sampling_rate=256, lowcut=1.0, highcut=40.0):
    """
    Band-pass filter for EEG data.
    Keeps frequencies between lowcut and highcut.
    """

    nyquist = 0.5 * sampling_rate

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        N=4,
        Wn=[low, high],
        btype="band"
    )

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def notch_filter(signal, sampling_rate=256, notch_freq=50.0, quality_factor=30.0):
    """
    Remove 50 Hz mains interference from EEG.
    """
    signal = np.asarray(signal, dtype=float)

    nyquist = 0.5 * sampling_rate
    normalized_freq = notch_freq / nyquist

    b, a = iirnotch(
        normalized_freq,
        quality_factor
    )

    return filtfilt(b, a, signal)


def preprocess_eeg(signal, sampling_rate=256):
    """
    NeuroSense EEG preprocessing V1:
    1. 50 Hz notch filter
    2. 1-40 Hz band-pass filter
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("EEG signal must be 1-D.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("EEG contains non-finite values.")

    notched = notch_filter(
        signal,
        sampling_rate=sampling_rate
    )

    cleaned = bandpass_filter(
        notched,
        sampling_rate=sampling_rate,
        lowcut=1.0,
        highcut=40.0
    )

    return cleaned


if __name__ == "__main__":
    from sys import path
    from pathlib import Path

    project_src = Path(__file__).resolve().parents[1]
    path.append(str(project_src))

    from acquisition.eeg_reader import EEGReader

    reader = EEGReader()

    time, raw_eeg = reader.generate_eeg()

    filtered_eeg = bandpass_filter(raw_eeg)

    print("Raw EEG samples:", len(raw_eeg))
    print("Filtered EEG samples:", len(filtered_eeg))

    print("\nFirst 10 raw samples:")
    print(raw_eeg[:10])

    print("\nFirst 10 filtered samples:")
    print(filtered_eeg[:10])