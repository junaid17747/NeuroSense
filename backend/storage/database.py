from pathlib import Path
import sqlite3

DB_PATH = Path("data/database/neurosense.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sampling_rate INTEGER NOT NULL,
                eeg_json TEXT NOT NULL,
                heart_rate REAL NOT NULL,
                motion_level REAL NOT NULL,
                prediction_allowed INTEGER,
                signal_status TEXT,
                state TEXT,
                confidence REAL,
                attention_score REAL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_session
            ON sensor_records(session_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_subject
            ON sensor_records(subject_id)
        """)

        conn.commit()


if __name__ == "__main__":
    initialize_database()
    print("Database ready:", DB_PATH)
