"""Synthetic-only tests for the D1 seizure feature and training path."""

import hashlib
import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

import test_seizure_processing as fixtures
from seizure_research.events import (
    ScoredWindow, evaluate_events, select_validation_threshold, windows_to_events,
)
from seizure_research.features import (
    FEATURE_VERSION, WindowExample, channel_features, feature_names, iter_processed_windows,
    window_features, write_feature_rows,
)
from seizure_research.ingestion import ingest_cohort
from seizure_research.preprocessing import (
    ChannelMapping, make_age_config, preprocess_recording,
)
from seizure_research.quality import QualityThresholds
from seizure_research.schemas import AgeGroup
from seizure_research.splitting import build_explicit_split_manifest, digest, validate_split_manifest
from seizure_research.training import (
    AnnotationEventEvidence, DiskBudget, FeatureTable, PilotRunConfig,
    RecordingAnnotationEvidence, patient_sample_weights, train_xgboost,
    build_test_artifact_payloads, run_approved_pilot, main as training_main,
    validate_recording_annotation_evidence,
)
from seizure_research.waveform import EDFStreamReader


class SeizureFeatureTests(unittest.TestCase):
    def test_feature_schema_and_spectral_features_are_fixed_finite(self):
        fs = 128
        time = np.arange(5 * fs) / fs
        ten_hz = np.sin(2 * np.pi * 10 * time)
        louder = 2 * ten_hz
        twenty_hz = np.sin(2 * np.pi * 20 * time)
        a = channel_features(ten_hz, fs)
        b = channel_features(louder, fs)
        c = channel_features(twenty_hz, fs)
        self.assertEqual(a.shape, (18,))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(b[0], a[0])
        self.assertGreater(c[17], a[17])
        self.assertEqual(len(feature_names(18)), 378)
        self.assertEqual(len(window_features(np.column_stack([ten_hz] * 18), fs)), 378)
        self.assertEqual(FEATURE_VERSION, "seizure-eeg-features-v1")

    def test_bad_or_degenerate_feature_window_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            channel_features(np.zeros(5 * 128), 128)
        with self.assertRaisesRegex(ValueError, "shape_or_rate"):
            channel_features(np.full((640, 1), np.nan), 128)

    def test_window_label_policy_keeps_ambiguous_and_quality_windows_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            recording = manifest.records[0].recording
            decoded = fixtures.read_edf(root, manifest.records[0].signal.relative_path,
                                        channel_indices=(0, 1), start_seconds=0, end_seconds=60)
            mappings = tuple(ChannelMapping(source_index=i, source_label=label, target_label=label,
                                            expected_source_rate_hz=128)
                             for i, label in enumerate(recording.montage.channels))
            config = make_age_config(AgeGroup.CHILD, mappings, target_sampling_rate_hz=128,
                                     research_review_ref="SYNTHETIC_TEST", line_noise_hz=None,
                                     window_seconds=5, step_seconds=1)
            config = config.model_copy(update={"quality_thresholds": QualityThresholds(
                window_seconds=5, step_seconds=1)})
            processed = preprocess_recording(recording, decoded, config, approved_manifest=manifest)
            rows = tuple(iter_processed_windows(processed, recording, core_start_seconds=0,
                                                core_end_seconds=60, window_seconds=5, step_seconds=1))
            labels = {row.label for row in rows}
            self.assertEqual(len(rows), 56)
            self.assertTrue({"positive", "negative", "ambiguous", "excluded"} <= labels)
            ambiguous = next(row for row in rows if row.label == "ambiguous")
            self.assertIsNotNone(ambiguous.features)
            excluded = next(row for row in rows if row.label == "excluded")
            self.assertIsNone(excluded.features)

    def test_parquet_writer_flushes_incrementally_and_separates_excluded_rows(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.parquet"
            names = feature_names(18)
            rows = (WindowExample(
                window_id=f"w{i}", recording_id="r", patient_group_id="p",
                start_seconds=float(i), end_seconds=float(i + 5),
                label="excluded" if i == 4 else "ambiguous" if i == 3 else "negative",
                event_ids=(), reasons=(),
                features=None if i == 4 else np.full(378, i, dtype=np.float32))
                for i in range(7))
            result = write_feature_rows(rows, path, feature_schema=names, row_group_size=2)
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 6)
            self.assertEqual(result["excluded_windows"], 1)
            self.assertEqual(result["ambiguous_windows"], 1)  # ambiguous scores remain available
            self.assertGreaterEqual(pq.ParquetFile(path).num_row_groups, 3)


class SeizureEventTests(unittest.TestCase):
    def test_validation_threshold_rejects_test_scores(self):
        bad = [ScoredWindow("r", "p", 0, 5, 0.8, split="test")]
        with self.assertRaisesRegex(ValueError, "validation_only"):
            select_validation_threshold([], bad)

    def test_missing_probability_resets_event_state_and_no_cross_gap_merge(self):
        values = []
        for start in range(0, 50):
            score = 0.95 if start < 10 or start > 18 else None
            values.append(ScoredWindow("r", "p", float(start), float(start + 5), score,
                                       split="validation"))
        events = windows_to_events(values, threshold_on=0.5, merge_gap_seconds=5)
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(all(event.coverage_segment >= 0 for event in events))

    def test_duplicate_candidates_count_false_and_metrics_are_per_patient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            recording = manifest.records[0].recording
            # The review fixture has one event; a trigger inside it matches once.
            from seizure_research.events import CandidateEvent
            candidate = CandidateEvent("c1", recording.recording_id, recording.patient_group_id,
                                       10, 16, 12, ("a",), 0)
            duplicate = CandidateEvent("c2", recording.recording_id, recording.patient_group_id,
                                       10, 16, 13, ("b",), 0)
            windows = [ScoredWindow(recording.recording_id, recording.patient_group_id,
                                    float(i), float(i + 5), 0.1, split="test")
                       for i in range(10)]
            metrics = evaluate_events([recording], windows, [candidate, duplicate])
            self.assertEqual(metrics.detected_events, 1)
            self.assertEqual(metrics.false_events, 1)
            self.assertEqual(metrics.missed_events, 0)
            self.assertEqual(metrics.per_patient[recording.patient_group_id].false_events, 1)
            self.assertAlmostEqual(metrics.evaluable_hours, 8 / 3600)


class SeizureSafetyTests(unittest.TestCase):
    def test_streaming_edf_reader_reuses_verified_source_for_bounded_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            entry = manifest.records[0]
            with EDFStreamReader(root, entry.signal.relative_path) as reader:
                first = reader.read_range(5, 6, channel_indices=(0, 1), max_samples=256)
                second = reader.read_range(30, 31, channel_indices=(1,), max_samples=128)
                self.assertEqual(reader.metadata.sha256, entry.signal.sha256)
                self.assertEqual(first.channels[0].timestamps_seconds[0], 5)
                self.assertEqual(second.channels[0].timestamps_seconds[0], 30)
                self.assertEqual(len(first.channels[0].samples_uv), 128)
                self.assertEqual(len(second.channels[0].samples_uv), 128)
                with self.assertRaisesRegex(ValueError, "sample_limit"):
                    reader.read_range(0, 2, channel_indices=(0, 1), max_samples=256)

    def test_annotation_sidecar_mismatch_and_timing_mismatch_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            entry = manifest.records[0]
            recording = entry.recording
            reviewed = manifest.review_plan.recordings[0]
            sidecar_path = entry.signal.relative_path + ".seizures"
            sidecar = root / sidecar_path
            sidecar.write_bytes(b"synthetic test sidecar\n")
            evidence = RecordingAnnotationEvidence(
                recording_id=recording.recording_id, policy="summary_sidecar_reconciled",
                summary_sha256=entry.annotations.sha256, sidecar_path=sidecar_path,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                coverage_start_seconds=0, coverage_end_seconds=recording.duration_seconds,
                events=tuple(AnnotationEventEvidence(start_seconds=item.start_seconds,
                                                     end_seconds=item.end_seconds)
                             for item in recording.annotations),
                reviewer="synthetic-reviewer", reviewed_on="2026-10-12",
                evidence_ref=reviewed.annotation_review.evidence_ref,
                reconciliation_ref=recording.provenance.annotation_provenance_ref,
                complete_coverage_attested=True)
            validate_recording_annotation_evidence(recording, entry, evidence, reviewed, root)
            changed = evidence.model_copy(update={"events": (
                AnnotationEventEvidence(start_seconds=12, end_seconds=20),)})
            with self.assertRaisesRegex(ValueError, "timing_mismatch"):
                validate_recording_annotation_evidence(recording, entry, changed, reviewed, root)
            sidecar.write_bytes(b"mutated sidecar")
            with self.assertRaisesRegex(ValueError, "sidecar_hash_mismatch"):
                validate_recording_annotation_evidence(recording, entry, evidence, reviewed, root)

    def test_disk_budget_enforces_cap_and_system_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "existing.bin").write_bytes(b"1234567890")
            budget = DiskBudget(root, cap_bytes=10, reserve_bytes=0)
            with self.assertRaisesRegex(OSError, "storage_cap"):
                budget.check(1)
            reserve_budget = DiskBudget(root, cap_bytes=1000, reserve_bytes=20)
            fake_usage = type("Usage", (), {"free": 100})()
            with patch("seizure_research.training.shutil.disk_usage", return_value=fake_usage):
                with self.assertRaisesRegex(OSError, "reserve"):
                    reserve_budget.check(81)

    def test_patient_weighting_uses_training_labels_and_equalizes_patient_mass(self):
        weights, pos_ratio = patient_sample_weights(
            ["p1", "p1", "p1", "p2", "p2"], np.asarray([0, 0, 1, 0, 1]))
        self.assertAlmostEqual(weights[:3].sum(), weights[3:].sum())
        self.assertGreater(pos_ratio, 0)
        with self.assertRaisesRegex(ValueError, "both_classes"):
            patient_sample_weights(["p1", "p1"], np.asarray([0, 0]))

    def test_explicit_split_rejects_patient_reassignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            base = manifest.records[0].recording
            records = []
            for name in ("train", "validation", "test"):
                provenance = base.provenance.model_copy(update={
                    "signal_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "source_recording_id": name,
                })
                records.append(base.model_copy(update={
                    "recording_id": f"synthetic:{name}", "subject_id": f"synthetic_{name}",
                    "patient_group_id": f"patient_{name}", "provenance": provenance,
                }))
            mapping = {f"patient_{name}": name for name in ("train", "validation", "test")}
            split = build_explicit_split_manifest(records, mapping)
            validate_split_manifest(records, split)
            assignments = list(split.assignments)
            # Make the train recording claim the validation patient's identity;
            # the manifest must reject this before any feature/model work.
            assignments[0] = assignments[0].model_copy(update={
                "patient_group_id": "patient_validation",
            })
            altered = split.model_copy(update={"assignments": tuple(assignments)})
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                validate_split_manifest(records, altered)


def _synthetic_table(split, patient, recording_id, rows=56):
    labels = []
    starts = np.arange(rows, dtype=float)
    values = np.zeros((rows, 378), dtype=np.float32)
    events = []
    for index, start in enumerate(starts):
        overlap = max(0.0, min(start + 5, 20) - max(start, 10))
        label = 1 if overlap >= 2.5 else 0 if overlap == 0 else -1
        labels.append(label)
        values[index, 0] = 1.0 if label == 1 else 0.0
        events.append(("seizure-1",) if overlap else ())
    return FeatureTable(
        split=split, feature_names=feature_names(18), values=values,
        labels=np.asarray(labels, dtype=np.int8), patient_ids=(patient,) * rows,
        recording_ids=(recording_id,) * rows,
        window_ids=tuple(f"{recording_id}:{index}" for index in range(rows)),
        starts=starts, ends=starts + 5, event_ids=tuple(events), sha256="a" * 64)


class SyntheticTrainingTests(unittest.TestCase):
    def test_test_artifacts_bind_predictions_metrics_model_and_split(self):
        provenance = {"model_sha256": "a" * 64, "cohort_manifest_sha256": "b" * 64,
                      "split_manifest_sha256": "c" * 64, "validation_threshold": 0.5,
                      "threshold_selected_on": "validation", "evaluation_partition": "test",
                      "clinical_validation": False}
        result = {"provenance": provenance, "prediction_windows": [{
                      "recording_id": "chb06/chb06_01.edf", "patient_group_id": "p6",
                      "start_seconds": 0.0, "end_seconds": 5.0,
                      "probability": 0.25, "split": "test"}],
                  "candidate_events": [], "metrics": {"event_sensitivity": None}}
        predictions, evaluation = build_test_artifact_payloads(result)
        self.assertEqual(evaluation["evaluation_artifact_sha256"],
                         digest({key: value for key, value in evaluation.items()
                                 if key != "evaluation_artifact_sha256"}))
        self.assertEqual(predictions["provenance"]["evaluation_artifact_sha256"],
                         evaluation["evaluation_artifact_sha256"])
        self.assertNotIn("evaluation_artifact_sha256", provenance)
        self.assertEqual(predictions["windows"][0]["split"], "test")
        self.assertTrue(evaluation["evaluated_at_utc"].endswith("+00:00"))

    def test_training_dry_run_plan_does_not_read_or_create_patient_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            output = io.StringIO()
            with patch.object(EDFStreamReader, "__enter__",
                              side_effect=AssertionError("dry-run opened EDF")), \
                    redirect_stdout(output):
                result = training_main(["plan-pilot", "--root", str(root)])
            plan = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertFalse(root.exists())
            self.assertFalse(plan["reads_patient_eeg"])
            self.assertFalse(plan["creates_files"])
            self.assertEqual(plan["recordings"], 12)
            self.assertEqual(plan["patients"]["validation"], ["chb09"])

    def test_end_to_end_training_requires_readiness_inputs_and_human_record(self):
        with self.assertRaisesRegex(PermissionError, "explicit_real_data_pilot_authorization"):
            run_approved_pilot(None, None, None, None, None)
        with self.assertRaisesRegex(PermissionError, "readiness_and_human_approval_inputs"):
            run_approved_pilot(None, None, None, None, None, authorized=True)

    def test_trainer_rejects_test_shard_as_validation_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            base = manifest.records[0].recording
            records = []
            for name in ("train", "validation", "test"):
                provenance = base.provenance.model_copy(update={
                    "signal_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "source_recording_id": name,
                })
                records.append(base.model_copy(update={
                    "recording_id": f"synthetic:{name}", "subject_id": f"synthetic_{name}",
                    "patient_group_id": f"patient_{name}", "provenance": provenance,
                }))
            split = build_explicit_split_manifest(
                records, {f"patient_{name}": name for name in ("train", "validation", "test")})
            train = _synthetic_table("train", "patient_train", "synthetic:train")
            wrong_validation = _synthetic_table("test", "patient_test", "synthetic:test")
            with self.assertRaisesRegex(ValueError, "only_train_and_validation"):
                train_xgboost(train, wrong_validation, recordings=records,
                              split_manifest=split, cohort_sha256="a" * 64,
                              annotation_evidence_sha256="b" * 64,
                              output_dir=root / "model", authorized=True)

    def test_synthetic_fit_scaler_is_training_only_and_threshold_is_validation_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = ingest_cohort(root, fixtures._fixture(root)).manifest
            base = manifest.records[0].recording
            records = []
            for name in ("train", "validation", "test"):
                provenance = base.provenance.model_copy(update={
                    "signal_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "source_recording_id": name,
                })
                records.append(base.model_copy(update={
                    "recording_id": f"synthetic:{name}", "subject_id": f"synthetic_{name}",
                    "patient_group_id": f"patient_{name}", "provenance": provenance,
                }))
            split = build_explicit_split_manifest(
                records, {f"patient_{name}": name for name in ("train", "validation", "test")})
            train = _synthetic_table("train", "patient_train", "synthetic:train")
            validation = _synthetic_table("validation", "patient_validation", "synthetic:validation")
            validation.values.setflags(write=True)
            validation.values[:, 1] = 1000.0  # should not affect fitted scaler state
            validation.values.setflags(write=False)
            run = train_xgboost(
                train, validation, recordings=records, split_manifest=split,
                cohort_sha256="a" * 64, annotation_evidence_sha256="b" * 64,
                output_dir=root / "model", config=PilotRunConfig(n_estimators=12, early_stopping_rounds=4,
                                                                  n_jobs=1), authorized=True)
            self.assertAlmostEqual(run["scaler"].mean_[1], float(train.values[:, 1].mean()))
            self.assertIn(run["threshold"], run["validation_curve"])
            self.assertFalse(run["provenance"]["clinical_validation"])
            self.assertTrue(run["provenance"]["trained_at_utc"].endswith("+00:00"))
            self.assertTrue((root / "model" / "xgboost.json").is_file())


if __name__ == "__main__":
    unittest.main()
