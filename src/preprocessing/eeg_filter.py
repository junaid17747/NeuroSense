import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, sampling_rate=256, lowcut=1.0, highcut=45.0):
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