import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data/database/neurosense.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )
        """)

        columns = {row[1] for row in conn.execute("PRAGMA table_info(sensor_records)")}
        if "user_id" not in columns:
            conn.execute(
                "ALTER TABLE sensor_records ADD COLUMN user_id INTEGER REFERENCES users(id)"
            )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                revoked_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS eeg_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                subject_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_user
            ON sensor_records(user_id, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_auth_session_user
            ON auth_sessions(user_id, expires_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_eeg_session_user
            ON eeg_sessions(user_id, last_seen_at)
        """)

        conn.commit()


if __name__ == "__main__":
    initialize_database()
    print("Database ready:", DB_PATH)
