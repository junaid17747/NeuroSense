import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "sample" / "replay_session.jsonl"


def generate_replay(seconds=30, sampling_rate=256):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w") as f:
        start = time.time()

        for second in range(seconds):
            t = np.linspace(
                second,
                second + 1,
                sampling_rate,
                endpoint=False
            )

            eeg = (
                20 * np.sin(2 * np.pi * 10 * t)
                + 10 * np.sin(2 * np.pi * 20 * t)
                + np.random.normal(0, 4, len(t))
            )

            packet = {
                "timestamp": start + second,
                "heart_rate": int(np.random.randint(65, 95)),
                "motion_level": float(np.random.uniform(0.0, 1.0)),
                "eeg": eeg.tolist()
            }

            f.write(json.dumps(packet) + "\n")

    print("Replay file created:")
    print(OUTPUT)


if __name__ == "__main__":
    generate_replay()
