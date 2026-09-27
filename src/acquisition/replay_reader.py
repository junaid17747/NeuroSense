import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPLAY_FILE = ROOT / "data" / "sample" / "replay_session.jsonl"


class ReplayReader:
    def __init__(self, path=REPLAY_FILE):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Replay file not found: {self.path}"
            )

        with self.path.open() as f:
            self.records = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        self.index = 0

    def read_next(self):
        if not self.records:
            raise RuntimeError("Replay file contains no records.")

        record = self.records[self.index]

        self.index = (
            self.index + 1
        ) % len(self.records)

        eeg = np.asarray(
            record["eeg"],
            dtype=float
        )

        time_axis = np.arange(
            len(eeg)
        ) / 256.0

        return {
            "mode": "replay",
            "timestamp": record["timestamp"],
            "time": time_axis,
            "eeg": eeg,
            "heart_rate": record["heart_rate"],
            "motion_level": record["motion_level"]
        }


if __name__ == "__main__":
    reader = ReplayReader()

    data = reader.read_next()

    print("Mode:", data["mode"])
    print("EEG samples:", len(data["eeg"]))
    print("Heart rate:", data["heart_rate"])
    print("Motion:", data["motion_level"])
