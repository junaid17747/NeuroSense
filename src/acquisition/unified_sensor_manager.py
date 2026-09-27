import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from acquisition.config import MODE
from acquisition.eeg_reader import EEGReader
from acquisition.heart_rate_reader import HeartRateReader
from acquisition.motion_reader import MotionReader
from acquisition.replay_reader import ReplayReader


class UnifiedSensorManager:
    def __init__(self):
        self.mode = MODE.lower()

        self.eeg_reader = EEGReader()
        self.heart_reader = HeartRateReader()
        self.motion_reader = MotionReader()

        self.replay_reader = None

        if self.mode == "replay":
            self.replay_reader = ReplayReader()

    def read_all(self):

        if self.mode == "simulation":
            time_axis, eeg = self.eeg_reader.generate_eeg()

            heart_rate = self.heart_reader.read_heart_rate()
            motion_level = self.motion_reader.read_motion()

            return {
                "mode": "simulation",
                "time": time_axis,
                "eeg": eeg,
                "heart_rate": heart_rate,
                "motion_level": motion_level
            }

        elif self.mode == "replay":
            return self.replay_reader.read_next()

        elif self.mode == "real":
            raise NotImplementedError(
                "Real sensor mode is not connected yet."
            )

        else:
            raise ValueError(
                f"Unknown NeuroSense mode: {self.mode}"
            )


if __name__ == "__main__":
    manager = UnifiedSensorManager()

    data = manager.read_all()

    print("=" * 55)
    print("NeuroSense Unified Sensor Manager")
    print("=" * 55)

    print("Mode:", data["mode"])
    print("EEG Samples:", len(data["eeg"]))
    print("Heart Rate:", data["heart_rate"], "BPM")
    print("Motion Level:", data["motion_level"])
