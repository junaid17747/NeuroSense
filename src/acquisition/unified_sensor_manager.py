import sys
from pathlib import Path

import numpy as np

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

        # Blueprint:
        # 4-second window with 2-second overlap
        self.sampling_rate = 256
        self.window_seconds = 4
        self.step_seconds = 2

        self.window_samples = (
            self.sampling_rate * self.window_seconds
        )

        self.step_samples = (
            self.sampling_rate * self.step_seconds
        )

        self.eeg_buffer = None

    def _update_eeg_window(self, new_eeg):

        new_eeg = np.asarray(
            new_eeg,
            dtype=float
        )

        # First reading: create full 4-second window
        if self.eeg_buffer is None:

            if len(new_eeg) >= self.window_samples:

                self.eeg_buffer = new_eeg[
                    -self.window_samples:
                ]

            else:

                repeats = int(
                    np.ceil(
                        self.window_samples / len(new_eeg)
                    )
                )

                self.eeg_buffer = np.tile(
                    new_eeg,
                    repeats
                )[-self.window_samples:]

        else:

            # Keep the last 2 seconds
            overlap = self.eeg_buffer[
                -self.step_samples:
            ]

            # Add the newest 2 seconds
            latest = new_eeg[
                -self.step_samples:
            ]

            self.eeg_buffer = np.concatenate(
                [
                    overlap,
                    latest
                ]
            )

        return self.eeg_buffer.copy()

    def read_all(self):

        if self.mode == "simulation":

            # Generate a new 2-second chunk.
            # The manager creates the required
            # 4-second rolling window with
            # 2-second overlap.
            time_axis, new_eeg = (
                self.eeg_reader.generate_eeg(
                    duration=self.step_seconds
                )
            )

            eeg = self._update_eeg_window(
                new_eeg
            )

            time_axis = np.arange(
                len(eeg)
            ) / self.sampling_rate

            heart_rate = (
                self.heart_reader.read_heart_rate()
            )

            motion_level = (
                self.motion_reader.read_motion()
            )

            return {
                "mode": "simulation",
                "time": time_axis,
                "eeg": eeg,
                "heart_rate": heart_rate,
                "motion_level": motion_level
            }

        elif self.mode == "replay":

            data = self.replay_reader.read_next()

            eeg = self._update_eeg_window(
                data["eeg"]
            )

            time_axis = np.arange(
                len(eeg)
            ) / self.sampling_rate

            return {
                "mode": "replay",
                "timestamp": data["timestamp"],
                "time": time_axis,
                "eeg": eeg,
                "heart_rate": data["heart_rate"],
                "motion_level": data["motion_level"]
            }

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

    print("=" * 60)
    print("NeuroSense Rolling Sensor Test")
    print("=" * 60)

    for i in range(3):

        data = manager.read_all()

        print(
            f"\nReading {i + 1}"
        )

        print(
            "Mode:",
            data["mode"]
        )

        print(
            "EEG window:",
            len(data["eeg"]),
            "samples"
        )

        print(
            "Window duration:",
            len(data["eeg"]) / 256,
            "seconds"
        )

        print(
            "Heart rate:",
            data["heart_rate"]
        )

        print(
            "Motion:",
            data["motion_level"]
        )
