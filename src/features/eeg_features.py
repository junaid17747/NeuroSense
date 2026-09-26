import numpy as np
from scipy.signal import welch


def band_power(signal, sampling_rate, low_freq, high_freq):
    """
    Calculate EEG power inside a frequency band.
    """

    frequencies, psd = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(256, len(signal))
    )

    mask = (frequencies >= low_freq) & (frequencies <= high_freq)

    if not np.any(mask):
        return 0.0

    return np.trapezoid(psd[mask], frequencies[mask])


def extract_eeg_features(signal, sampling_rate=256):

    theta_power = band_power(signal, sampling_rate, 4, 8)
    alpha_power = band_power(signal, sampling_rate, 8, 13)
    beta_power = band_power(signal, sampling_rate, 13, 30)

    return {
        "theta": theta_power,
        "alpha": alpha_power,
        "beta": beta_power
    }


def calculate_attention_index(features):
    """
    Simple experimental attention index.

    Higher beta relative to alpha + theta
    is treated as stronger task engagement.

    This is a prototype metric, not a medical diagnostic score.
    """

    theta = features["theta"]
    alpha = features["alpha"]
    beta = features["beta"]

    denominator = alpha + theta

    if denominator == 0:
        return 0.0

    return beta / denominator


if __name__ == "__main__":

    from pathlib import Path
    import sys

    src_path = Path(__file__).resolve().parents[1]
    sys.path.append(str(src_path))

    from acquisition.eeg_reader import EEGReader
    from preprocessing.eeg_filter import bandpass_filter

    reader = EEGReader()

    _, raw_eeg = reader.generate_eeg()

    filtered_eeg = bandpass_filter(raw_eeg)

    features = extract_eeg_features(filtered_eeg)

    attention = calculate_attention_index(features)

    print("Theta Power:", round(features["theta"], 2))
    print("Alpha Power:", round(features["alpha"], 2))
    print("Beta Power:", round(features["beta"], 2))
    print("Attention Index:", round(attention, 3))