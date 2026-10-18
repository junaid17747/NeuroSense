"""Synthetic-only checks for the authenticated seizure research dashboard."""

import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import matplotlib.pyplot as plt
from streamlit.testing.v1 import AppTest

from dashboard.seizure_lab import (
    age_group_status, lab_empty_states, load_real_prediction_artifacts,
    plot_waveform, read_approved_real_waveform, synthetic_waveform,
    synthetic_scrolling_waveform, _render_playback_controls,
    _elapsed_acquisition_seconds,
    validate_partitions,
)
from dashboard.research_status import (clear_dashboard_status_cache,
                                       load_dashboard_snapshot,
                                       status_fingerprint,
                                       validate_test_evaluation_artifact)
from seizure_research.splitting import digest


class SeizureLabTests(unittest.TestCase):
    def test_completed_bundle_changes_invalidate_status_and_artifact_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"
            run.mkdir()
            def fingerprint():
                return status_fingerprint(archive_root=root / "archive", approval_path=root / "approval",
                                          review_path=root / "review", pilot_root=None, run_root=run,
                                          test_artifact=None)
            previous = fingerprint()
            for relative in ("complete.json", "evaluation_details.json", "test_freeze.json", "split_proposal.json",
                             "review_packet.json", "run_config.json", "features/test/test.parquet"):
                path = run / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("TEST_ONLY")
                current = fingerprint()
                self.assertNotEqual(current, previous, relative)
                previous = current
            (run / "complete.json").unlink()
            self.assertNotEqual(fingerprint(), previous)

    def test_empty_states_are_explicit_and_metrics_have_no_fake_values(self):
        states = lab_empty_states()
        self.assertIn("No approved real recordings", states["data"])
        self.assertIn("No real-data", states["model"])
        self.assertIn("has not been supplied", states["approvals"])
        self.assertIn("No real prediction", states["predictions"])

    def test_frozen_partitions_are_patient_disjoint_and_alias_safe(self):
        validate_partitions()
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_partitions({"train": ("chb02",), "validation": ("chb02",),
                                 "test": ("chb05", "chb06", "chb08", "chb09", "chb10")})
        with self.assertRaisesRegex(ValueError, "alias"):
            validate_partitions({"train": ("chb01", "chb05", "chb08"),
                                 "validation": ("chb09",), "test": ("chb06", "chb10")})

    def test_child_is_only_pilot_age_target_and_other_groups_are_unavailable(self):
        self.assertTrue(age_group_status("child")["supported"])
        for age_group in ("neonatal", "infant", "adolescent", "adult", "older_adult"):
            with self.subTest(age_group=age_group):
                self.assertFalse(age_group_status(age_group)["supported"])

    def test_synthetic_waveform_is_bounded_and_deterministic(self):
        time, values = synthetic_waveform(duration_seconds=5, sampling_rate_hz=64)
        self.assertEqual(values.shape, (320, 2))
        np.testing.assert_array_equal(time, np.arange(320) / 64)
        with self.assertRaisesRegex(ValueError, "duration_out_of_bounds"):
            synthetic_waveform(duration_seconds=61)

    def test_scrolling_waveform_changes_with_time_and_is_bounded(self):
        first_time, first = synthetic_scrolling_waveform(1.0)
        next_time, next_frame = synthetic_scrolling_waveform(2.0)
        self.assertGreater(next_time[-1], first_time[-1])
        self.assertFalse(np.array_equal(first, next_frame))
        self.assertLessEqual(len(next_time), 10 * 128)

    def test_acquisition_clock_advances_freezes_resumes_and_resets(self):
        self.assertEqual(_elapsed_acquisition_seconds("playing", 3.0, 10.0, 12.5), 5.5)
        self.assertEqual(_elapsed_acquisition_seconds("paused", 5.5, 10.0, 40.0), 5.5)
        self.assertEqual(_elapsed_acquisition_seconds("playing", 5.5, 50.0, 52.0), 7.5)
        self.assertEqual(_elapsed_acquisition_seconds("stopped", 0.0, 10.0, 100.0), 0.0)

    def test_playback_controls_start_pause_resume_stop_seek_and_speed(self):
        script = """
import streamlit as st
from dashboard.seizure_lab import _render_playback_controls, synthetic_waveform
axis, signal = synthetic_waveform(duration_seconds=20)
_render_playback_controls(st, source="GENERATED TEST EDF", time_seconds=axis,
                         samples_uv=signal, channel_labels=("A", "B"),
                         state_key="playback_test", annotations=((4.0, 6.0),))
"""
        app = AppTest.from_string(script, default_timeout=15).run()
        self.assertFalse(app.exception)
        app.button(key="playback_test_start").click().run()
        self.assertEqual(app.session_state["playback_test_mode"], "playing")
        app.button(key="playback_test_pause").click().run()
        self.assertEqual(app.session_state["playback_test_mode"], "paused")
        app.slider(key="playback_test_seek").set_value(5.0).run()
        self.assertEqual(app.session_state["playback_test_elapsed"], 5.0)
        app.selectbox(key="playback_test_speed_widget").select(2.0).run()
        self.assertEqual(app.session_state["playback_test_speed"], 2.0)
        app.button(key="playback_test_resume").click().run()
        self.assertEqual(app.session_state["playback_test_mode"], "playing")
        app.button(key="playback_test_stop").click().run()
        self.assertEqual(app.session_state["playback_test_mode"], "stopped")
        self.assertEqual(app.session_state["playback_test_elapsed"], 0.0)
        self.assertTrue(app.get("vega_lite_chart"))
        self.assertFalse(app.exception)
        from dashboard.seizure_lab import playback_position
        self.assertEqual(playback_position("playing", 2, 10, 12, 2, 20), 6)
        self.assertEqual(playback_position("paused", 5, 10, 100, 4, 20), 5)
        self.assertEqual(playback_position("playing", 19, 10, 100, 4, 20), 20)

    @staticmethod
    def _test_metrics():
        return {"detected_events": 1, "reference_events": 2, "missed_events": 1,
                "false_events": 3, "matched_events": 1, "evaluable_hours": 2.0,
                "event_sensitivity": 0.5, "false_alarms_per_hour": 1.5,
                "median_detection_latency_seconds": 4.25}

    def _write_test_artifact(self, path):
        payload = {
            "schema_version": "neurosense-dashboard-test-evaluation-v1",
            "artifact_kind": "SYNTHETIC_TEST_ONLY",
            "clinical_validation": False,
            "research_performance_claim": False,
            "evaluated_at_utc": "2026-10-13T00:00:00+00:00",
            "metrics": self._test_metrics(),
            "artifact_sha256": None,
        }
        payload["artifact_sha256"] = digest(payload)
        path.write_text(json.dumps(payload))

    def test_valid_test_metric_artifact_updates_snapshot_and_removal_reverts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "test-evaluation.json"
            kwargs = {"archive_root": root / "archive", "approval_path": root / "approval.json",
                      "review_path": root / "review.json", "test_artifact": evaluation}
            clear_dashboard_status_cache()
            initial = load_dashboard_snapshot(**kwargs)
            self.assertIsNone(initial["metrics"])
            self.assertIsNone(initial["test_metrics"])
            self._write_test_artifact(evaluation)
            arrived = load_dashboard_snapshot(**kwargs)
            self.assertIsNone(arrived["metrics"], "test metrics must never enter real result field")
            self.assertEqual(arrived["test_metrics"]["event_sensitivity"], 0.5)
            self.assertIn("SYNTHETIC TEST ONLY", arrived["test_artifact_label"])
            evaluation.unlink()
            removed = load_dashboard_snapshot(**kwargs)
            self.assertIsNone(removed["test_metrics"])

    def test_invalid_test_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "evaluation.json"
            self._write_test_artifact(artifact)
            payload = json.loads(artifact.read_text())
            payload["metrics"]["event_sensitivity"] = 0.99
            artifact.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash_invalid"):
                validate_test_evaluation_artifact(artifact)

    def test_archive_folder_does_not_grant_approval_and_status_refreshes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            kwargs = {"archive_root": archive, "approval_path": root / "missing-approval.json",
                      "review_path": root / "missing-review.json"}
            clear_dashboard_status_cache()
            before = load_dashboard_snapshot(**kwargs)
            self.assertEqual(before["archive"], "not staged")
            (archive / "index.json").write_text(json.dumps({"complete": True}))
            after = load_dashboard_snapshot(**kwargs)
            self.assertEqual(after["archive"], "files present; approval or verification unavailable")
            self.assertEqual(after["data_use_staging"], "not verified")
            self.assertEqual(after["archive_integrity"], "not verified")

    def test_rendered_ui_keeps_research_metrics_honest_and_player_first(self):
        script = """
import streamlit as st
from dashboard.seizure_lab import render_seizure_lab
render_seizure_lab(st)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            artifact = root / "evaluation.json"
            env = {
                "NEUROSENSE_CHBMIT_ARCHIVE_ROOT": str(archive),
                "NEUROSENSE_CHBMIT_TEMPORARY_ROOT": str(root / "temporary"),
                "NEUROSENSE_CHBMIT_ARCHIVE_APPROVAL": str(root / "no-approval.json"),
                "NEUROSENSE_CHBMIT_REVIEW_PACKET": str(root / "no-review.json"),
                "NEUROSENSE_SEIZURE_TEST_ARTIFACT": str(artifact),
            }
            with patch.dict("os.environ", env, clear=True):
                app = AppTest.from_string(script, default_timeout=15).run()
                self.assertFalse(app.exception)
                metric = {item.label: item.value for item in app.metric}
                self.assertEqual(metric, {label: "Not evaluated" for label in (
                    "Event sensitivity", "Missed seizures", "False alarms/hour", "Detection latency")})
                details = app.expander[0]
                self.assertEqual(details.label, "Research Details")
                self.assertFalse(details.proto.expanded)
                elements = list(app.main)
                chart = app.get("vega_lite_chart")[0]
                summary = next(item for item in app.markdown if "**Research status:**" in item.value)
                self.assertLess(elements.index(app.title[0]), elements.index(chart))
                for card in app.metric:
                    self.assertLess(elements.index(chart), elements.index(card))
                    self.assertLess(elements.index(card), elements.index(summary))
                self.assertLess(elements.index(summary), elements.index(details))
                self.assertTrue(all(item in list(details) for item in app.subheader))
                channels = app.multiselect(key="synthetic_eeg_player_channels")
                self.assertEqual(channels.value, ["Synthetic Fp1-F7", "Synthetic F7-T7"])
                channels.set_value(["Synthetic Fp1-F7"]).run()
                self.assertFalse(app.exception)
                self.assertTrue(app.get("vega_lite_chart"))

                self._write_test_artifact(artifact)
                app.button(key="seizure_status_refresh").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.get("vega_lite_chart"), "live synthetic waveform uses Streamlit's interactive chart")
                self.assertTrue(any("Live Synthetic EEG" in item.value for item in app.caption))
                metric = {item.label: item.value for item in app.metric}
                self.assertEqual(metric["Event sensitivity"], "Not evaluated")
                self.assertTrue(any("SYNTHETIC TEST ONLY" in item.value for item in app.warning))
                self.assertTrue(all(item in list(app.expander[0]) for item in app.warning))

                app.button(key="synthetic_eeg_player_start").click().run()
                self.assertTrue(any("Status: Acquiring" in item.value for item in app.caption))
                app.button(key="synthetic_eeg_player_pause").click().run()
                self.assertTrue(any("Status: Paused" in item.value for item in app.caption))
                app.button(key="synthetic_eeg_player_resume").click().run()
                self.assertTrue(any("Status: Acquiring" in item.value for item in app.caption))
                app.button(key="synthetic_eeg_player_stop").click().run()
                self.assertTrue(any("Status: Stopped / reset" in item.value for item in app.caption))

                artifact.unlink()
                app.button(key="seizure_status_refresh").click().run()
                metric = {item.label: item.value for item in app.metric}
                self.assertEqual(metric["Event sensitivity"], "Not evaluated")
                self.assertFalse(any("SYNTHETIC TEST ONLY" in item.value for item in app.warning))
                app.multiselect(key="synthetic_eeg_player_channels").set_value([]).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.get("vega_lite_chart"))
                self.assertTrue(any("Select at least one channel" in item.value for item in app.info))
                app.selectbox(key="seizure_age_group").select("adult").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("outside the child pilot scope" in item.value for item in app.caption))

    def test_annotation_and_experimental_prediction_overlays_are_aligned_separately(self):
        time, values = synthetic_waveform(duration_seconds=10)
        figure = plot_waveform(
            time, values, ("A", "B"), annotations=((2, 4),), predictions=((6, 8),))
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes[0].lines), 2)
        self.assertEqual(len(figure.axes[0].patches), 2)
        self.assertEqual(figure.axes[0].patches[0].get_x(), 2)
        self.assertEqual(figure.axes[0].patches[0].get_width(), 2)
        self.assertEqual(figure.axes[0].patches[1].get_x(), 6)
        self.assertEqual(figure.axes[0].patches[1].get_width(), 2)

    def test_synthetic_dashboard_render_never_opens_real_edf(self):
        fake = Mock()
        def columns(count):
            result = [Mock() for _ in range(count)]
            for column in result:
                column.button.return_value = False
                column.slider.return_value = 10
                column.multiselect.return_value = ["Synthetic Fp1-F7", "Synthetic F7-T7"]
            return result
        fake.columns.side_effect = columns
        fake.selectbox.return_value = "child"
        fake.radio.return_value = "Live Synthetic EEG"
        fake.button.return_value = False
        fake.slider.return_value = 10
        fake.session_state = {}
        fake.fragment.side_effect = lambda **_kwargs: (lambda function: function)
        fake.expander.side_effect = lambda *_args, **_kwargs: nullcontext()
        with patch.dict("os.environ", {}, clear=True), \
                patch("seizure_research.waveform.read_edf",
                      side_effect=AssertionError("synthetic dashboard attempted EDF read")):
            from dashboard.seizure_lab import render_seizure_lab
            render_seizure_lab(fake)
        fake.line_chart.assert_called_once()
        self.assertFalse(any(call.args[0] == "Synthetic display duration (seconds)"
                             for call in fake.slider.call_args_list))
        fake.button.assert_any_call("Refresh now", key="seizure_status_refresh",
                                    on_click=clear_dashboard_status_cache)

    def test_real_waveform_gate_blocks_before_edf_access(self):
        with patch("seizure_research.waveform.read_edf",
                   side_effect=AssertionError("EDF should not be opened")) as reader:
            with self.assertRaisesRegex(PermissionError, "readiness_missing"):
                read_approved_real_waveform(
                    "/tmp/not-used", "chb02/chb02_16.edf", channel_indices=(0,),
                    start_seconds=0, end_seconds=5, readiness_report={}, approval_record={})
        reader.assert_not_called()

    def test_real_prediction_gate_requires_readiness_and_processing_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(PermissionError, "readiness_missing"):
                load_real_prediction_artifacts(root, readiness_report={}, approval_record={})
            with self.assertRaisesRegex(PermissionError, "approval_missing"):
                load_real_prediction_artifacts(
                    root, readiness_report={"ready_for_real_data_processing": True},
                    approval_record={"processing_status": "pending"})
            with self.assertRaises(FileNotFoundError):
                load_real_prediction_artifacts(
                    root, readiness_report={"ready_for_real_data_processing": True},
                    approval_record={"processing_status": "approved",
                                     "processing_approver": "unit-test fixture",
                                     "processing_approved_on": "2026-10-12",
                                     "processing_approval_ref": "TEST_ONLY"})

    def test_self_hashed_artifacts_without_reviewed_cohort_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            model_path = model_dir / "xgboost.json"
            model_path.write_bytes(b"unit-test-only model bytes")
            model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            split_hash = "a" * 64
            base_provenance = {
                "model_sha256": model_hash, "cohort_manifest_sha256": "b" * 64,
                "split_manifest_sha256": split_hash, "validation_threshold": 0.5,
                "threshold_selected_on": "validation", "evaluation_partition": "test",
                "clinical_validation": False,
            }
            (model_dir / "provenance.json").write_text(json.dumps({
                "model_sha256": model_hash, "split_manifest_sha256": split_hash,
                "selected_validation_threshold": 0.5, "clinical_validation": False,
            }))
            windows = [{"recording_id": "chb06/chb06_01.edf", "patient_group_id": "test-patient",
                        "start_seconds": 0.0, "end_seconds": 5.0, "probability": 0.25,
                        "split": "test"}]
            candidates = []
            metrics = {"event_sensitivity": None, "missed_events": 0,
                       "false_alarms_per_hour": None,
                       "median_detection_latency_seconds": None, "per_patient": {}}
            evaluation = {
                "schema_version": "seizure-event-evaluation-v1",
                "dataset": "CHB-MIT Scalp EEG Database v1.0.0",
                "provenance": base_provenance, "metrics": metrics,
                "clinical_validation": False,
            }
            evaluation_hash = digest(evaluation)
            evaluation["evaluation_artifact_sha256"] = evaluation_hash
            (root / "test_evaluation.json").write_text(json.dumps(evaluation))
            prediction_provenance = {**base_provenance,
                                     "evaluation_artifact_sha256": evaluation_hash}
            prediction_provenance["prediction_artifact_sha256"] = digest({
                "candidate_events": candidates, "windows": windows})
            (root / "test_predictions.json").write_text(json.dumps({
                "schema_version": "seizure-window-predictions-v1",
                "dataset": "CHB-MIT Scalp EEG Database v1.0.0", "split": "test",
                "provenance": prediction_provenance,
                "candidate_events": candidates, "windows": windows,
            }))
            approvals = {"processing_status": "approved",
                         "processing_approver": "unit-test fixture",
                         "processing_approved_on": "2026-10-12",
                         "processing_approval_ref": "TEST_ONLY"}
            with self.assertRaises(FileNotFoundError):
                load_real_prediction_artifacts(
                    root, readiness_report={"ready_for_real_data_processing": True},
                    approval_record=approvals)



if __name__ == "__main__":
    unittest.main()
