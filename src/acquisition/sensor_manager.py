from eeg_reader import EEGReader
from heart_rate_reader import HeartRateReader


class SensorManager:
    def __init__(self):
        self.eeg_sensor = EEGReader()
        self.heart_rate_sensor = HeartRateReader()

    def collect_data(self):
        time, eeg = self.eeg_sensor.generate_eeg()

        heart_rate = self.heart_rate_sensor.read_heart_rate()

        return {
            "time": time,
            "eeg": eeg,
            "heart_rate": heart_rate
        }


if __name__ == "__main__":
    manager = SensorManager()

    data = manager.collect_data()

    print("EEG Samples:", len(data["eeg"]))
    print("Heart Rate:", data["heart_rate"], "BPM")