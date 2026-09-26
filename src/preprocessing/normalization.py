import numpy as np


def normalize_signal(signal):
    """
    Standardize EEG signal using Z-score normalization.
    """

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal

    return (signal - mean) / std


if __name__ == "__main__":
    sample_signal = np.array([10, 12, 15, 11, 17, 14])

    normalized = normalize_signal(sample_signal)

    print("Original:", sample_signal)
    print("Normalized:", normalized)