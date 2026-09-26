import random


class MotionReader:
    """
    Simulated motion sensor.

    0.0 = very little movement
    1.0 = high movement
    """

    def read_motion(self):
        return round(random.uniform(0.0, 1.0), 2)


if __name__ == "__main__":
    sensor = MotionReader()
    print("Motion Level:", sensor.read_motion())
