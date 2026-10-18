"""Exercise the real dashboard router with isolated auth storage and sensor data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from backend.storage import database
from dashboard import auth_client


APP_SCRIPT = """
import importlib
import streamlit as st
from unittest.mock import patch
from dashboard import app

# AppTest creates a fresh component registry for each test runtime, while
# Python keeps imported modules cached. Register the component in this runtime.
if "test_components_registered" not in st.session_state:
    from dashboard import webcam
    importlib.reload(webcam)
    st.session_state.test_components_registered = True

snapshot = {
    "heart_rate": 72, "motion_level": 0.2, "fusion_score": 80,
    "state": "Focused", "sensor_mode": "simulation", "heart_rate_source": "SIMULATED_HEART_RATE",
    "eeg_quality": {"quality": 0.95, "status": "good"},
    "signal_status": {"status": "good", "allow_prediction": True},
}
with patch.object(app, "_sensor_snapshot", return_value=snapshot):
    app.main()
"""


class WebcamNavigationTests(unittest.TestCase):
    def setUp(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        db_patch = patch.object(database, "DB_PATH", Path(tempdir.name) / "dashboard.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        database.initialize_database()
        self.auth = auth_client.register("webcam@example.com", "SecurePass1", "Camera User")
        self.app = AppTest.from_string(APP_SCRIPT, default_timeout=10)

    def sign_in(self):
        self.app.session_state.auth_token = self.auth["access_token"]
        self.app.session_state.auth_user = self.auth["user"]
        self.app.session_state.attention_history = [
            {"Attention Score": 80, "Heart Rate": 72, "Motion Level": 0.2},
        ]
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_overview_button_opens_webcam_page_and_back_preserves_history(self):
        self.sign_in()
        self.assertEqual(self.app.title[0].value, "Overview")
        self.assertEqual(self.app.subheader[0].value, "EEG Seizure Research")
        self.assertTrue(any("SIMULATED_HEART_RATE" in metric.label for metric in self.app.metric))
        elements = list(self.app.main)
        research_button = self.app.button(key="open_eeg_seizure_research")
        button = self.app.button(key="start_webcam_monitoring")
        attention_metrics = [m for m in self.app.metric if "Attention" in m.label]
        self.assertLess(elements.index(self.app.subheader[0]), elements.index(research_button))
        for metric in attention_metrics:
            self.assertLess(elements.index(research_button), elements.index(metric))
        for metric in attention_metrics:
            self.assertLess(elements.index(metric), elements.index(button))
        self.assertEqual(len(self.app.get("vega_lite_chart")), 1)

        button.click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.title[0].value, "Webcam Monitoring")
        self.assertEqual(self.app.radio[0].value, "Webcam Monitoring")
        self.assertEqual(len(self.app.get("bidi_component")), 1)
        self.assertNotIn("start_webcam_monitoring", [b.key for b in self.app.button])

        self.app.button(key="back_to_overview").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.title[0].value, "Overview")
        self.assertEqual(self.app.radio[0].value, "Overview")
        self.assertFalse(self.app.get("bidi_component"))
        self.assertEqual(self.app.session_state.attention_history[0]["Attention Score"], 80)
        self.assertEqual(self.app.session_state.auth_token, self.auth["access_token"])
        self.assertEqual(len(self.app.get("vega_lite_chart")), 1)

    def test_overview_research_card_opens_authenticated_seizure_workspace(self):
        self.sign_in()
        self.app.button(key="open_eeg_seizure_research").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.title[0].value, "EEG Seizure Research")
        self.assertEqual(self.app.radio(key="workspace_page").value, "EEG Seizure Research")
        self.assertTrue(all(item.value == "Not evaluated" for item in self.app.metric))
        self.assertTrue(any(button.key == "synthetic_eeg_player_start" for button in self.app.button))
        self.assertTrue(any(button.key == "synthetic_eeg_player_pause" for button in self.app.button))
        self.assertTrue(any(button.key == "synthetic_eeg_player_resume" for button in self.app.button))
        self.assertTrue(any(button.key == "synthetic_eeg_player_stop" for button in self.app.button))
        self.assertTrue(any("Live Synthetic EEG" in item.value
                            for item in self.app.caption))
        self.assertGreaterEqual(len(self.app.get("vega_lite_chart")), 1)
        self.assertNotIn("Bounded waveform viewer", [item.value for item in self.app.subheader])
        self.assertFalse(any("Synthetic display duration" in item.label for item in self.app.slider))
        self.assertTrue(any(button.key == "seizure_status_refresh" for button in self.app.button))
        self.assertFalse(self.app.get("bidi_component"))
        elements = list(self.app.main)
        chart = self.app.get("vega_lite_chart")[0]
        status = next(item for item in self.app.markdown if "**Research status:**" in item.value)
        details = self.app.expander[0]
        self.assertEqual(details.label, "Research Details")
        self.assertFalse(details.proto.expanded)
        self.assertEqual(len(self.app.metric), 4)
        self.assertLess(elements.index(self.app.title[0]), elements.index(chart))
        for card in self.app.metric:
            self.assertLess(elements.index(chart), elements.index(card))
            self.assertLess(elements.index(card), elements.index(status))
        self.assertLess(elements.index(status), elements.index(details))

    def test_sidebar_can_open_webcam_page(self):
        self.sign_in()
        self.app.radio[0].set_value("Webcam Monitoring").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.title[0].value, "Webcam Monitoring")
        self.assertEqual(len(self.app.get("bidi_component")), 1)

    def test_streamlit_rerun_keeps_the_same_keyed_webcam_component_mounted(self):
        self.sign_in()
        self.app.radio[0].set_value("Webcam Monitoring").run()
        before = self.app.get("bidi_component")
        self.assertEqual(len(before), 1)
        self.app.run()
        self.assertFalse(self.app.exception)
        after = self.app.get("bidi_component")
        self.assertEqual(len(after), 1)
        self.assertEqual(before[0].key, after[0].key)
        self.assertEqual(after[0].key, "webcam_preview")

    def test_webcam_selection_does_not_bypass_login(self):
        self.app.session_state.workspace_page = "Webcam Monitoring"
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader[0].value, "Sign in to NeuroSense")
        self.assertFalse(self.app.get("bidi_component"))
        self.assertFalse(self.app.radio)

    def test_revoked_session_cannot_open_webcam_page(self):
        self.sign_in()
        auth_client.logout(self.auth["access_token"])
        self.app.button(key="start_webcam_monitoring").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader[0].value, "Sign in to NeuroSense")
        self.assertFalse(self.app.get("bidi_component"))
        self.assertNotIn("auth_token", self.app.session_state)

    def test_logout_removes_webcam_and_resets_navigation(self):
        self.sign_in()
        self.app.session_state.synthetic_eeg_player_mode = "playing"
        self.app.session_state.approved_edf_player_elapsed = 12.5
        self.app.button(key="start_webcam_monitoring").click().run()
        next(b for b in self.app.button if b.label == "Log out").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader[0].value, "Sign in to NeuroSense")
        self.assertFalse(self.app.get("bidi_component"))
        self.assertNotIn("workspace_page", self.app.session_state)
        self.assertNotIn("attention_history", self.app.session_state)
        self.assertNotIn("synthetic_eeg_player_mode", self.app.session_state)
        self.assertNotIn("approved_edf_player_elapsed", self.app.session_state)
        self.assertIsNone(auth_client.current_user(self.auth["access_token"]))


if __name__ == "__main__":
    unittest.main()
