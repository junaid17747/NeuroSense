from preprocessing.eeg_filter import bandpass_filter


def remove_noise(signal, sampling_rate=256):
    """
    Remove basic EEG noise using a 1-45 Hz band-pass filter.
    """
    return bandpass_filter(
        signal,
        sampling_rate=sampling_rate,
        lowcut=1.0,
        highcut=45.0
    )