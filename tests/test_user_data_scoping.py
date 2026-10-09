from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.schemas.sensor import SensorPacket
from backend.services import auth_service
from backend.storage import database, repository


class UserDataScopingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, "DB_PATH", Path(self.tempdir.name) / "scope.db")
        self.db_patch.start()
        database.initialize_database()

    def tearDown(self):
        self.db_patch.stop()
        self.tempdir.cleanup()

    @staticmethod
    def packet(session_id: str) -> SensorPacket:
        return SensorPacket(
            session_id=session_id,
            subject_id="subject-1",
            device_id="simulation",
            timestamp=datetime.now(timezone.utc),
            eeg=[0.1] * 1024,
            heart_rate=75,
            motion_level=0.2,
        )

    @staticmethod
    def result() -> dict:
        return {
            "prediction_allowed": True,
            "signal_quality": {"status": "good"},
            "prediction": {
                "state": "Focused", "confidence": 0.9,
                "attention_score": 88.0, "model_version": "test",
            },
        }

    def test_records_and_sessions_are_scoped_to_authenticated_user(self):
        first = auth_service.register_user("one@example.com", "SecurePass1", "One")
        second = auth_service.register_user("two@example.com", "SecurePass1", "Two")
        repository.save_sensor_record(self.packet("session-one"), self.result(), user_id=first["id"])
        repository.save_sensor_record(self.packet("session-two"), self.result(), user_id=second["id"])
        self.assertEqual([row["session_id"] for row in repository.list_user_sessions(first["id"])], ["session-one"])
        self.assertEqual([row["session_id"] for row in repository.list_user_records(first["id"])], ["session-one"])
        self.assertFalse(repository.user_has_session(first["id"], "session-two"))
        with self.assertRaises(PermissionError):
            repository.save_sensor_record(self.packet("session-one"), self.result(), user_id=second["id"])

    def test_legacy_records_remain_readable(self):
        record_id = repository.save_sensor_record(self.packet("legacy"), self.result())
        self.assertTrue(record_id)
        with database.get_connection() as conn:
            row = conn.execute("SELECT user_id FROM sensor_records WHERE id = ?", (record_id,)).fetchone()
        self.assertIsNone(row["user_id"])

    def test_legacy_schema_is_migrated_without_rewriting_records(self):
        legacy_path = Path(self.tempdir.name) / "legacy.db"
        database.DB_PATH = legacy_path
        with sqlite3.connect(legacy_path) as conn:
            conn.execute(
                """
                CREATE TABLE sensor_records (
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
                """
            )
            conn.execute(
                """
                INSERT INTO sensor_records (
                    session_id, subject_id, device_id, timestamp, sampling_rate,
                    eeg_json, heart_rate, motion_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("legacy-session", "subject", "simulation", "2026-01-01T00:00:00+00:00", 256, "[]", 70, 0.1),
            )

        database.initialize_database()
        with database.get_connection() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sensor_records)")}
            count = conn.execute("SELECT COUNT(*) FROM sensor_records").fetchone()[0]
        self.assertIn("user_id", columns)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
