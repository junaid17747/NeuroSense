"""D6: real pipeline execution on generated EDFs and fictitious annotations only."""

import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np

from dashboard.seizure_lab import (clip_overlays, load_real_prediction_artifacts,
                                   render_seizure_lab)
from seizure_research import training
from seizure_research.readiness import ApprovalRecord, validate_approved_storage
from seizure_research.smoke import run_smoke
from seizure_research.splitting import digest
from seizure_research.staging import preflight_storage
from seizure_research.training import DiskBudget, MAX_PILOT_BYTES, MIN_FREE_BYTES, PilotRunConfig
from seizure_research.waveform import EDFStreamReader
from seizure_synthetic_fixture import SyntheticPilot, synthetic_environment, write_json


class SyntheticEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="neurosense-D6-TEST_ONLY-")
        cls.addClassCleanup(temporary.cleanup)
        environment = synthetic_environment()
        cls.addClassCleanup(environment.close)
        network = patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden in D6"))
        network.start()
        cls.addClassCleanup(network.stop)
        cls.fixture = SyntheticPilot(Path(temporary.name))
        cls.staged, cls.transfers = cls.fixture.stage()
        cls.fixture.review()
        cls.report = cls.fixture.readiness()
        if not cls.report["ready_for_real_data_processing"]:
            raise AssertionError(cls.report)
        cls.smoke = run_smoke(**cls.fixture.arguments(), relative_path="chb02/chb02_16.edf", authorized=True)
        cls.output = cls.fixture.root / "runs" / "TEST_ONLY"
        cls.loaded_partitions = []
        cls.bounded_reads = []
        cls.feature_tables = {}
        original_load = training.load_feature_table
        original_read = EDFStreamReader.read_range

        def observed_load(path, **kwargs):
            split = kwargs["split"]
            if split == "test" and not (cls.output / "model" / "provenance.json").is_file():
                raise AssertionError("Test features loaded before model/threshold freeze")
            cls.loaded_partitions.append(split)
            table = original_load(path, **kwargs)
            cls.feature_tables.setdefault(split, []).append(table)
            return table

        def observed_read(reader, start_seconds, end_seconds, **kwargs):
            decoded = original_read(reader, start_seconds, end_seconds, **kwargs)
            total = sum(len(item.samples_uv) for item in decoded.channels)
            if total > kwargs["max_samples"]:
                raise AssertionError("Bounded decoder exceeded its allocation limit")
            cls.bounded_reads.append((start_seconds, end_seconds, total))
            return decoded

        with patch.object(training, "load_feature_table", side_effect=observed_load), \
                patch.object(EDFStreamReader, "read_range", new=observed_read):
            cls.pilot_run = training.run_approved_pilot(
                **cls.fixture.arguments(), manifest=cls.fixture.cohort, config=cls.fixture.config,
                annotation_evidence=cls.fixture.evidence, output_dir=cls.output, authorized=True,
                run_config=PilotRunConfig(n_estimators=12, early_stopping_rounds=4,
                                          n_jobs=1, chunk_seconds=30, halo_seconds=30))

    def bundle(self):
        return load_real_prediction_artifacts(self.output, readiness_report=self.report,
                                             approval_record=self.fixture.approval)

    def render(self, age="child", run_dir=None):
        fake = Mock()
        fake.columns.return_value = [Mock() for _ in range(4)]
        for column in fake.columns.return_value:
            column.button.return_value = False
        fake.selectbox.side_effect = lambda label, *_args, **_kwargs: (
            age if label == "Age group" else 1.0 if label == "Playback speed" else "chb06" if label == "Pilot patient" else "chb06/chb06_01.edf")
        fake.multiselect.return_value = list(self.fixture.config.channels[:2][i].target_label
                                             for i in range(2))
        fake.number_input.return_value = 0
        fake.slider.return_value = 20
        fake.radio.return_value = "Verified CHB-MIT EDF playback"
        fake.button.side_effect = lambda _label, key=None, **_kwargs: key == "seizure_lab_load_real"
        fake.session_state = {"seizure_age_group": age}
        fake.fragment.side_effect = lambda **_kwargs: (lambda function: function)
        fake.expander.side_effect = lambda *_args, **_kwargs: nullcontext()
        env = {"NEUROSENSE_CHBMIT_ROOT": str(self.fixture.root),
               "NEUROSENSE_SEIZURE_RUN_DIR": str(run_dir or self.output),
               "NEUROSENSE_CHBMIT_FILE_MANIFEST": str(self.fixture.file_manifest)}
        with patch.dict("os.environ", env), patch("dashboard.seizure_lab._approved_lab_assets", return_value=(
                self.report, self.fixture.approval, self.fixture.cohort, self.fixture.config)) as assets:
            render_seizure_lab(fake)
            if fake.session_state.get("approved_edf_player_segment"):
                fake.button.side_effect = lambda *_args, **_kwargs: False
                fake.columns.return_value[0].button.side_effect = (
                    lambda _label, *, key, **_kwargs: key == "approved_edf_player_start")
                with patch("dashboard.seizure_lab.read_approved_real_waveform") as decoder:
                    render_seizure_lab(fake)
                decoder.assert_not_called()
                self.assertEqual(fake.session_state["approved_edf_player_mode"], "playing")
                self.assertEqual(fake.altair_chart.call_count, 2)
        plt.close("all")
        return fake, assets

    def test_generated_edf_staging_checksums_readiness_smoke_and_resume(self):
        self.assertEqual((self.staged["file_count"], self.transfers), (25, 25))
        self.assertEqual(len(self.report["checks"]["official_hashes"]["files"]), 24)
        self.assertTrue(all(item["status"] == "match" for item in self.report["checks"]["official_hashes"]["files"]))
        self.assertEqual((self.smoke["duration_seconds"], self.smoke["channels_decoded"],
                          self.smoke["samples_per_channel"]), (5, 18, 1280))
        self.assertFalse(self.smoke["writes_artifacts"])
        result, calls = self.fixture.stage()
        self.assertEqual(calls, 0)
        self.assertEqual(result["status"], "staged_and_verified")

    def test_bounded_features_patient_split_train_only_scaler_and_validation_freeze(self):
        self.assertEqual(self.loaded_partitions, ["train"] * 6 + ["validation"] * 2 + ["test"] * 4)
        self.assertGreaterEqual(len(self.bounded_reads), 36)
        self.assertTrue(all(end - start <= 90 and samples <= 2_000_000
                            for start, end, samples in self.bounded_reads))
        patients = {split: {patient for table in tables for patient in table.patient_ids}
                    for split, tables in self.feature_tables.items()}
        self.assertEqual([len(patients[name]) for name in ("train", "validation", "test")], [3, 1, 2])
        self.assertFalse(patients["train"] & patients["test"] | patients["train"] & patients["validation"])
        train_rows = np.concatenate([table.values[table.labels >= 0] for table in self.feature_tables["train"]])
        np.testing.assert_allclose(self.pilot_run["model"]["scaler"].mean_, train_rows.mean(axis=0, dtype=np.float64))
        self.assertEqual(train_rows.shape[1], 378)
        for tables in self.feature_tables.values():
            for table in tables:
                self.assertEqual(len(table.window_ids), len(set(table.window_ids)))
        self.assertIn(self.pilot_run["model"]["threshold"], self.pilot_run["model"]["validation_curve"])

    def test_actual_written_results_load_and_render_with_canonical_recording_ids(self):
        bundle = self.bundle()
        self.assertTrue(all(row["recording_id"].startswith("chb_mit:1.0.0:")
                            for row in bundle["predictions"]["windows"]))
        self.assertEqual(len(bundle["evaluation"]["metrics"]["per_patient"]), 2)
        self.assertFalse(bundle["evaluation"]["clinical_validation"])
        fake, _ = self.render()
        fake.error.assert_not_called()
        self.assertEqual(fake.altair_chart.call_count, 2)
        expected_metric = bundle["evaluation"]["metrics"]
        self.assertEqual(fake.columns.return_value[0].metric.call_args.args[1],
                         f"{expected_metric['detected_events']}/{expected_metric['reference_events']} · "
                         f"{100 * expected_metric['event_sensitivity']:.1f}%",
                         msg=f"dashboard status messages: {fake.info.call_args_list}; {fake.warning.call_args_list}; {fake.write.call_args_list}")
        self.assertTrue(any("not a diagnostic tool" in str(call) for call in fake.write.call_args_list))

    def test_saved_model_and_scaler_reload_reproduce_test_probabilities(self):
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        model = xgb.XGBClassifier()
        model.load_model(self.output / "model/xgboost.json")
        scaler = StandardScaler()
        with np.load(self.output / "model/scaler.npz", allow_pickle=False) as state:
            scaler.mean_, scaler.scale_, scaler.var_ = state["mean"], state["scale"], state["var"]
            scaler.n_features_in_ = int(state["n_features_in_"])
        tables = self.feature_tables["test"]
        values = np.concatenate([item.values for item in tables])
        actual = model.predict_proba(scaler.transform(values), iteration_range=(
            0, self.pilot_run["model"]["best_iteration"] + 1))[:, 1]
        expected = [item["probability"] for item in self.bundle()["predictions"]["windows"]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_documented_training_command_parses_with_the_dashboard_approval_location(self):
        document = (Path(__file__).resolve().parents[1] / "docs/checkpoint_d6_readiness.md").read_text()
        command = next(block for block in re.findall(r"```bash\n(.*?)```", document, re.S)
                       if "seizure_research.training run-pilot" in block)
        command = command.replace("\\\n", " ").replace("/mnt/approved/neurosense/chbmit-pilot", str(self.fixture.root))
        tokens = shlex.split(command)
        argv = tokens[tokens.index("seizure_research.training") + 1:]
        with patch.object(training, "run_approved_pilot") as runner:
            self.assertEqual(training.main(argv), 0)
        kwargs = runner.call_args.kwargs
        self.assertEqual(Path(kwargs["approval_record_path"]), self.fixture.root / "faculty-approval.json")
        self.assertEqual(kwargs["cohort_manifest_path"], str(self.fixture.cohort_path))
        self.assertTrue(kwargs["authorized"])

    def test_unsupported_age_blocks_all_real_access_even_with_valid_artifacts(self):
        fake, assets = self.render(age="adult")
        assets.assert_not_called()
        self.assertFalse(any(call.kwargs.get("key") == "seizure_lab_load_real"
                             for call in fake.button.call_args_list))
        for column in fake.columns.return_value:
            self.assertEqual(column.metric.call_args.args[1], "Not evaluated")

    def test_dashboard_missing_approval_and_data_states_do_not_decode(self):
        for state in ("missing_approval", "missing_data"):
            fake = Mock()
            fake.columns.return_value = [Mock() for _ in range(4)]
            fake.selectbox.return_value = "child"
            fake.slider.return_value = 5
            fake.radio.return_value = "Verified CHB-MIT EDF playback"
            fake.selectbox.side_effect = lambda label, *_args, **_kwargs: (
                "child" if label == "Age group" else "chb06/chb06_01.edf")
            fake.multiselect.return_value = [self.fixture.config.channels[0].target_label]
            fake.number_input.return_value = 0
            fake.slider.return_value = 20
            fake.button.side_effect = lambda _label, key=None, **_kwargs: key == "seizure_lab_load_real"
            fake.session_state = {"seizure_age_group": "child"}
            fake.fragment.side_effect = lambda **_kwargs: (lambda function: function)
            fake.expander.side_effect = lambda *_args, **_kwargs: nullcontext()
            report = {"ready_for_real_data_processing": False, "missing_approvals": [state],
                      "blockers": [state]}
            with self.subTest(state=state), patch.dict("os.environ", {
                    "NEUROSENSE_CHBMIT_ROOT": str(self.fixture.root)}), \
                    patch("dashboard.seizure_lab._approved_lab_assets", return_value=(report, {}, None, None)), \
                    patch("dashboard.seizure_lab.load_dashboard_snapshot", return_value={
                        "errors": [], "metrics": None, "test_metrics": None, "provenance": {},
                        "inventory": "verified official metadata inventory", "archive": "not staged",
                        "archive_integrity": "not verified", "recordings_verified": 0,
                        "recordings_available": 45, "annotation_review": "not reviewed",
                        "data_use_staging": "not approved", "processing_authorization": "not approved",
                        "preprocessing": "not ready", "training_readiness": "not ready",
                        "model": "unavailable", "training_status": "not trained or validated",
                        "model_eligibility": "not established", "evaluation": "unavailable",
                        "dashboard_artifacts": "unavailable"}), \
                    patch("dashboard.seizure_lab.read_approved_real_waveform") as decoder:
                render_seizure_lab(fake)
                decoder.assert_not_called()
                fake.error.assert_called_once()
                for column in fake.columns.return_value:
                    self.assertEqual(column.metric.call_args.args[1], "Not evaluated")
            plt.close("all")

    def test_missing_model_and_malformed_results_render_unavailable_without_crashing(self):
        fake, _ = self.render(run_dir=self.fixture.root / "missing-run")
        fake.error.assert_not_called()
        for column in fake.columns.return_value:
            self.assertEqual(column.metric.call_args.args[1], "Not evaluated")
        path = self.output / "test_predictions.json"
        original = path.read_bytes()
        try:
            for malformed in (b"[]", b'{"windows":NaN}', b'{"x":1,"x":2}', b'{"windows":null}'):
                with self.subTest(malformed=malformed):
                    path.write_bytes(malformed)
                    fake, _ = self.render()
                    fake.error.assert_not_called()
                    for column in fake.columns.return_value:
                        self.assertEqual(column.metric.call_args.args[1], "Not evaluated")
        finally:
            path.write_bytes(original)

    def test_corruption_wrong_patients_metrics_and_approvals_fail_closed(self):
        originals = {relative: (self.output / relative).read_bytes() for relative in (
            "model/xgboost.json", "model/scaler.npz", "model/provenance.json",
            "split_manifest.json", "test_predictions.json", "test_evaluation.json")}
        mutations = []
        for path in ("model/xgboost.json", "model/scaler.npz"):
            mutations.append((path, originals[path] + b"corrupt"))
        provenance = json.loads(originals["model/provenance.json"])
        provenance["cohort_manifest_sha256"] = "0" * 64
        mutations.append(("model/provenance.json", json.dumps(provenance).encode()))
        for path, payload in mutations:
            with self.subTest(path=path):
                (self.output / path).write_bytes(payload)
                try:
                    with self.assertRaises(ValueError):
                        self.bundle()
                finally:
                    (self.output / path).write_bytes(originals[path])
        for changed in ("train_recording", "wrong_patient", "bad_probability", "out_of_bounds", "empty"):
            with self.subTest(changed=changed):
                payload = json.loads(originals["test_predictions.json"])
                row = payload["windows"][0]
                if changed == "train_recording":
                    row["recording_id"] = "chb_mit:1.0.0:chb02_01.edf"
                elif changed == "wrong_patient":
                    row["patient_group_id"] = "TEST_ONLY_chb02"
                elif changed == "bad_probability":
                    row["probability"] = "invalid"
                elif changed == "out_of_bounds":
                    row["end_seconds"] = 999
                else:
                    payload["windows"] = []
                payload["provenance"]["prediction_artifact_sha256"] = digest({
                    "windows": payload["windows"], "candidate_events": payload["candidate_events"]})
                write_json(self.output / "test_predictions.json", payload)
                try:
                    with self.assertRaises(ValueError):
                        self.bundle()
                finally:
                    (self.output / "test_predictions.json").write_bytes(originals["test_predictions.json"])
        # Rehashed, internally consistent JSON must not turn invented metrics into results.
        evaluation = json.loads(originals["test_evaluation.json"])
        evaluation["metrics"]["missed_events"] += 1
        evaluation["evaluation_artifact_sha256"] = digest({key: value for key, value in evaluation.items()
                                                          if key != "evaluation_artifact_sha256"})
        prediction = json.loads(originals["test_predictions.json"])
        prediction["provenance"]["evaluation_artifact_sha256"] = evaluation["evaluation_artifact_sha256"]
        try:
            write_json(self.output / "test_predictions.json", prediction)
            write_json(self.output / "test_evaluation.json", evaluation)
            with self.assertRaisesRegex(ValueError, "metrics_disagree"):
                self.bundle()
        finally:
            for relative in ("test_predictions.json", "test_evaluation.json"):
                (self.output / relative).write_bytes(originals[relative])
        with self.assertRaisesRegex(PermissionError, "approval_missing"):
            load_real_prediction_artifacts(self.output, readiness_report=self.report, approval_record={})

    def test_source_annotation_and_approval_failures_block_before_decode_or_training(self):
        targets = (self.fixture.root / "chb02/chb02_16.edf", self.fixture.evidence_path, self.fixture.approval_path)
        for path in targets:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"corrupt")
                    report = self.fixture.readiness()
                    self.assertFalse(report["ready_for_real_data_processing"])
                    with patch.object(EDFStreamReader, "__enter__", side_effect=AssertionError("gate decoded EEG")):
                        with self.assertRaises(PermissionError):
                            run_smoke(**self.fixture.arguments(), relative_path="chb02/chb02_16.edf", authorized=True)
                        with self.assertRaises(PermissionError):
                            training.run_approved_pilot(
                                **self.fixture.arguments(), manifest=self.fixture.cohort,
                                config=self.fixture.config, annotation_evidence=self.fixture.evidence,
                                output_dir=self.fixture.root / "blocked", authorized=True)
                    self.assertFalse((self.fixture.root / "blocked").exists())
                finally:
                    path.write_bytes(original)

    def test_partial_output_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "new_directory"):
            training.run_approved_pilot(
                **self.fixture.arguments(), manifest=self.fixture.cohort, config=self.fixture.config,
                annotation_evidence=self.fixture.evidence, output_dir=self.output, authorized=True)


class StorageAndViewerBoundaryTests(unittest.TestCase):
    def test_persistent_location_requires_custodian_attestation_and_matching_root(self):
        with self.assertRaisesRegex(ValueError, "temporary_root"):
            validate_approved_storage("/tmp/not-for-patient-data", ApprovalRecord())
        with self.assertRaisesRegex(PermissionError, "custodian_review"):
            validate_approved_storage("/mnt/approved/neurosense/chbmit-pilot", ApprovalRecord())
        approval = ApprovalRecord(staging_root="/mnt/different", storage_persistence_confirmed=True,
                                  storage_policy_ref="TEST_ONLY")
        with self.assertRaisesRegex(PermissionError, "storage_root_mismatch"):
            validate_approved_storage("/mnt/approved/neurosense/chbmit-pilot", approval)

    def test_exact_five_gib_cap_and_twenty_gib_reserve_boundaries(self):
        self.assertEqual((MAX_PILOT_BYTES, MIN_FREE_BYTES), (5_368_709_120, 21_474_836_480))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "existing").write_bytes(b"x")
            from types import SimpleNamespace
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=MIN_FREE_BYTES + MAX_PILOT_BYTES)):
                preflight_storage(root, additional_bytes=MAX_PILOT_BYTES - 1)
                DiskBudget(root).check(MAX_PILOT_BYTES - 1)
                for check in (lambda: preflight_storage(root, additional_bytes=MAX_PILOT_BYTES),
                              lambda: DiskBudget(root).check(MAX_PILOT_BYTES)):
                    with self.assertRaisesRegex(OSError, "cap_exceeded"):
                        check()
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=MIN_FREE_BYTES + 1)):
                preflight_storage(root, additional_bytes=1)
                DiskBudget(root).check(1)
                for check in (lambda: preflight_storage(root, additional_bytes=2),
                              lambda: DiskBudget(root).check(2)):
                    with self.assertRaisesRegex(OSError, "reserve"):
                        check()

    def test_long_annotations_are_clipped_and_offscreen_events_omitted(self):
        self.assertEqual(clip_overlays(((5, 30), (50, 90)), 0, 20),
                         ({"start_seconds": 5, "end_seconds": 20},))
        self.assertEqual(clip_overlays(((5, 30),), 10, 20),
                         ({"start_seconds": 10, "end_seconds": 20},))


if __name__ == "__main__":
    unittest.main()
