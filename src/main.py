from acquisition.eeg_reader import EEGReader
from acquisition.heart_rate_reader import HeartRateReader


def main():
    print("=" * 55)
    print("NeuroSense")
    print("Multi-Sensor Brain Attention Monitoring System")
    print("=" * 55)

    eeg_sensor = EEGReader()
    heart_sensor = HeartRateReader()

    time, eeg = eeg_sensor.generate_eeg()
    heart_rate = heart_sensor.read_heart_rate()

    print("\nSystem Status: Running")
    print("-" * 30)
    print("EEG Samples:", len(eeg))
    print("Heart Rate:", heart_rate, "BPM")
    print("\nNeuroSense data acquisition successful.")


if __name__ == "__main__":
    main()