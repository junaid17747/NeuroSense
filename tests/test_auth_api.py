from __future__ import annotations

import importlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.schemas.auth import LoginRequest, RegisterRequest
from backend.schemas.sensor import SensorPacket
from backend.storage import repository
from backend.storage import database


class AuthApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, "DB_PATH", Path(self.tempdir.name) / "api.db")
        self.db_patch.start()
        database.initialize_database()
        module = importlib.import_module("backend.api.main")
        self.api = importlib.reload(module)

    def tearDown(self):
        self.db_patch.stop()
        self.tempdir.cleanup()

    def test_registration_login_authorization_and_logout(self):
        response = self.api.register(
            RegisterRequest(email="user@example.com", password="SecurePass1", display_name="User")
        )
        self.assertEqual(response.user.email, "user@example.com")
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=response.access_token)

        with self.assertRaises(HTTPException) as unauthenticated:
            self.api.get_current_user(None)
        self.assertEqual(unauthenticated.exception.status_code, 401)
        self.assertEqual(self.api.me(self.api.get_current_user(credentials)).email, "user@example.com")
        self.assertEqual(self.api.sessions(self.api.get_current_user(credentials)), [])

        with self.assertRaises(HTTPException) as duplicate:
            self.api.register(RegisterRequest(email="USER@example.com", password="SecurePass1", display_name="Other"))
        self.assertEqual(duplicate.exception.status_code, 400)
        with self.assertRaises(HTTPException) as invalid:
            self.api.login(LoginRequest(email="user@example.com", password="WrongPass1"))
        self.assertEqual(invalid.exception.status_code, 401)

        self.assertIsNone(self.api.logout(credentials, self.api.get_current_user(credentials)))
        with self.assertRaises(HTTPException):
            self.api.get_current_user(credentials)

    def test_registration_rejects_invalid_input(self):
        invalid_requests = (
            RegisterRequest(email="not-an-email", password="SecurePass1", display_name="User"),
            RegisterRequest(email="user@example.com", password="short", display_name="User"),
            RegisterRequest(email="user@example.com", password="NoNumberPass", display_name="User"),
            RegisterRequest(email="user@example.com", password="SecurePass1", display_name=" "),
        )
        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(HTTPException) as error:
                    self.api.register(request)
                self.assertEqual(error.exception.status_code, 400)

    def test_prediction_endpoints_require_authentication(self):
        with self.assertRaises(HTTPException) as error:
            self.api.get_current_user(None)
        self.assertEqual(error.exception.status_code, 401)

    def test_session_api_cannot_cross_account_boundaries(self):
        first = self.api.register(
            RegisterRequest(email="first@example.com", password="SecurePass1", display_name="First")
        )
        second = self.api.register(
            RegisterRequest(email="second@example.com", password="SecurePass1", display_name="Second")
        )
        packet = SensorPacket(
            session_id="private-session", subject_id="subject-1", device_id="simulation",
            timestamp=datetime.now(timezone.utc), eeg=[0.1] * 1024,
            heart_rate=75, motion_level=0.2,
        )
        repository.save_sensor_record(
            packet,
            {"prediction_allowed": True, "signal_quality": {"status": "good"},
             "prediction": {"state": "Focused", "confidence": 0.9,
                            "attention_score": 88, "model_version": "test"}},
            user_id=first.user.id,
        )
        first_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=first.access_token)
        second_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=second.access_token)
        self.assertEqual(len(self.api.sessions(self.api.get_current_user(first_creds))), 1)
        with self.assertRaises(HTTPException) as error:
            self.api.session_records("private-session", self.api.get_current_user(second_creds))
        self.assertEqual(error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
