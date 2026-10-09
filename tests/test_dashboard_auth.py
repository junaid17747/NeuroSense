import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardAuthTests(unittest.TestCase):
    def test_dashboard_has_auth_navigation_and_theme(self):
        app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        css = (ROOT / "dashboard" / "static" / "neurosense.css").read_text(encoding="utf-8")
        for label in ("Login", "Sign up", "Overview", "Live EEG", "Sessions", "Analytics", "Profile", "Log out"):
            self.assertIn(label, app)
        for color in ("--ns-navy", "--ns-cyan", "--ns-violet"):
            self.assertIn(color, css)

    def test_logout_clears_user_specific_workspace_state(self):
        app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        self.assertIn("def _reset_user_workspace", app)
        for key in ("eeg_session_id", "attention_history", "smoothed_attention_score"):
            self.assertIn(f'"{key}"', app)


if __name__ == "__main__":
    unittest.main()
