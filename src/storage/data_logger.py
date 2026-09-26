import csv
from pathlib import Path
from datetime import datetime


LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "logs" / "neurosense_readings.csv"


def log_reading(heart_rate, motion_level, eeg_score, fusion_score, state):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "heart_rate",
                "motion_level",
                "eeg_score",
                "fusion_score",
                "state"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            heart_rate,
            motion_level,
            eeg_score,
            fusion_score,
            state
        ])
