import numpy as np


class EEGReader:
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate

    def generate_eeg(self, duration=2):
        """
        Generate simulated EEG signal.
        Alpha + Beta waves + small random noise.
        """

        t = np.linspace(
            0,
            duration,
            self.sampling_rate * duration,
            endpoint=False
        )

        alpha = 20 * np.sin(2 * np.pi * 10 * t)
        beta = 10 * np.sin(2 * np.pi * 20 * t)

        noise = np.random.normal(0, 5, len(t))

        eeg_signal = alpha + beta + noise

        return t, eeg_signal


if __name__ == "__main__":
    reader = EEGReader()

    time, eeg = reader.generate_eeg()

    print("EEG samples:", len(eeg))
    print("First 10 samples:")
    print(eeg[:10])