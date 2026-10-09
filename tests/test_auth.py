from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from backend.security.auth import digest_session_token
from backend.services import auth_service
from backend.storage import database, repository


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, "DB_PATH", Path(self.tempdir.name) / "auth.db")
        self.db_patch.start()
        database.initialize_database()

    def tearDown(self):
        self.db_patch.stop()
        self.tempdir.cleanup()

    def test_registration_hashes_password_and_normalizes_email(self):
        user = auth_service.register_user("  Researcher@Example.com ", "SecurePass1", "Researcher")
        self.assertEqual(user["email"], "researcher@example.com")
        row = repository.get_user_by_email("researcher@example.com")
        self.assertNotEqual(row["password_hash"], "SecurePass1")
        self.assertTrue(row["password_hash"].startswith("$argon2"))

    def test_duplicate_and_invalid_credentials_are_rejected(self):
        auth_service.register_user("person@example.com", "SecurePass1", "Person")
        with self.assertRaisesRegex(ValueError, "already exists"):
            auth_service.register_user("PERSON@example.com", "SecurePass1", "Person")
        with self.assertRaisesRegex(ValueError, "Invalid email or password"):
            auth_service.login_user("person@example.com", "WrongPass1")

    def test_registration_validates_email_password_and_display_name(self):
        invalid_inputs = (
            ("invalid-email", "SecurePass1", "Researcher"),
            ("person@example.com", "short", "Researcher"),
            ("person@example.com", "NoNumberPass", "Researcher"),
            ("person@example.com", "SecurePass1", "   "),
        )
        for email, password, display_name in invalid_inputs:
            with self.subTest(email=email, password=password, display_name=display_name):
                with self.assertRaises(ValueError):
                    auth_service.register_user(email, password, display_name)

    def test_session_token_is_stored_as_a_digest(self):
        auth_service.register_user("person@example.com", "SecurePass1", "Person")
        result = auth_service.login_user("person@example.com", "SecurePass1")
        with database.get_connection() as conn:
            row = conn.execute("SELECT token_hash FROM auth_sessions").fetchone()
        self.assertIsNotNone(row)
        self.assertNotEqual(row["token_hash"], result["access_token"])
        self.assertEqual(len(row["token_hash"]), 64)

    def test_session_logout_and_expiration(self):
        auth_service.register_user("person@example.com", "SecurePass1", "Person")
        result = auth_service.login_user("person@example.com", "SecurePass1")
        self.assertEqual(auth_service.current_user(result["access_token"])["email"], "person@example.com")
        auth_service.logout_user(result["access_token"])
        self.assertIsNone(auth_service.current_user(result["access_token"]))

        row = repository.get_user_by_email("person@example.com")
        expired = "expired-token"
        repository.create_auth_session(
            digest_session_token(expired), row["id"], datetime.now(timezone.utc) - timedelta(minutes=1)
        )
        self.assertIsNone(auth_service.current_user(expired))


if __name__ == "__main__":
    unittest.main()
