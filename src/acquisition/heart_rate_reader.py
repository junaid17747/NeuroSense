import random


class HeartRateReader:
    def read_heart_rate(self):
        """
        Simulate heart-rate sensor reading.
        """

        bpm = random.randint(65, 95)

        return bpm


if __name__ == "__main__":
    sensor = HeartRateReader()

    print("Heart Rate:", sensor.read_heart_rate(), "BPM")