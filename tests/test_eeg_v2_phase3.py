"""Phase 3 calibration, held-out evaluation and frozen inference tests."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from test_eeg_v2_phase1 import fixture
from src.model.calibrate_eeg_v2_phase3 import (
    OUTPUT_DIR, Partition, assert_partition, booster_digest, fit_sigmoid,
    load_calibrated_model, run_phase3, select_partition,
)
from src.model.eeg_v2_data import (
    CLASSES, FEATURE_NAMES, build_manifest, split_row_indices,
)
from src.model.eeg_v2_evaluation import (
    checked_probabilities, cluster_bootstrap, point_metrics, reliability_bins,
    subject_statistics,
)


class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.fingerprint = {"path": "fixture.csv", "size_bytes": 123, "sha256": "a" * 64}
        cls.manifest = build_manifest(cls.df, cls.fingerprint, samples_per_subject=3)
        cls.indices = split_row_indices(cls.df, {
            name: split["subjects"] for name, split in cls.manifest["splits"].items()
        })
        cls.mapping = {name: i for i, name in enumerate(CLASSES)}
        train = cls.df.iloc[cls.indices["train"]]
        cls.base = XGBClassifier(
            n_estimators=3, max_depth=2, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", n_jobs=1, random_state=42,
        )
        cls.base.fit(train[list(FEATURE_NAMES)], train["label"].map(cls.mapping), verbose=False)

    def partition(self, name):
        return select_partition(self.df, self.indices, self.manifest, name)

    def test_calibration_and_test_subjects_are_disjoint(self):
        calibration, test = self.partition("calibration"), self.partition("test")
        calibration_subjects = set(calibration.X.index.get_level_values("subject_id"))
        test_subjects = set(test.X.index.get_level_values("subject_id"))
        self.assertEqual(len(calibration_subjects), 100)
        self.assertEqual(len(test_subjects), 100)
        self.assertFalse(calibration_subjects & test_subjects)
        with self.assertRaisesRegex(ValueError, "only selects calibration or test"):
            select_partition(self.df, self.indices, self.manifest, "train")

    def test_calibrator_uses_frozen_base_and_preserves_booster_bytes(self):
        calibration = self.partition("calibration")
        before = booster_digest(self.base)
        frozen, duration, reported_digest = fit_sigmoid(self.base, calibration, self.manifest)
        self.assertGreaterEqual(duration, 0)
        self.assertEqual(reported_digest, before)
        self.assertEqual(booster_digest(self.base), before)
        self.assertEqual(type(frozen).__name__, "FrozenEstimator")
        self.assertEqual(type(frozen.estimator).__name__, "CalibratedClassifierCV")
        self.assertEqual(frozen.estimator.method, "sigmoid")
        self.assertFalse(frozen.estimator.ensemble)
        self.assertEqual(len(frozen.estimator.calibrated_classifiers_), 1)
        self.assertEqual(type(frozen.estimator.calibrated_classifiers_[0].estimator).__name__, "FrozenEstimator")

    def test_calibration_output_is_normalized_and_reproducible(self):
        calibration = self.partition("calibration")
        frozen, _, _ = fit_sigmoid(self.base, calibration, self.manifest)
        first = checked_probabilities(frozen.predict_proba(calibration.X), len(calibration.X))
        second = checked_probabilities(frozen.predict_proba(calibration.X), len(calibration.X))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.sum(axis=1), 1, rtol=0, atol=1e-12)
        self.assertTrue((first >= 0).all())
        with self.assertRaisesRegex(ValueError, "sum to one"):
            checked_probabilities(np.full((2, 3), 0.2), 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            checked_probabilities(np.array([[np.nan, 0.5, 0.5]]), 1)

    def test_partition_schema_mismatch_rejected(self):
        calibration = self.partition("calibration")
        for matrix in (calibration.X[list(reversed(FEATURE_NAMES))],
                       calibration.X.drop(columns="rms"),
                       calibration.X.assign(subject_id=1)):
            bad = Partition("calibration", matrix, calibration.y)
            with self.subTest(columns=list(matrix.columns)), self.assertRaisesRegex(ValueError, "20 features in order"):
                assert_partition(bad, self.manifest, "calibration")

    def test_partition_provenance_and_membership_tampering_rejected(self):
        calibration = self.partition("calibration")
        keys = calibration.X.index.tolist()
        keys[0] = (self.manifest["splits"]["test"]["subjects"][0], 1)
        bad_index = pd.MultiIndex.from_tuples(keys, names=["subject_id", "sample_id"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_partition(Partition("calibration", calibration.X.set_axis(bad_index), calibration.y), self.manifest, "calibration")
        with self.assertRaisesRegex(ValueError, "Only the test"):
            assert_partition(calibration, self.manifest, "test")

    def test_calibration_cannot_change_when_test_values_are_poisoned(self):
        calibration = self.partition("calibration")
        poisoned = self.df.copy(deep=True)
        poisoned.loc[self.indices["test"], list(FEATURE_NAMES)] = np.nan
        poisoned.loc[self.indices["test"], "label"] = "POISON"
        poisoned_calibration = select_partition(poisoned, self.indices, self.manifest, "calibration")
        clean_model = XGBClassifier(
            n_estimators=3, max_depth=2, learning_rate=0.1, objective="multi:softprob",
            eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=42,
        )
        train = self.df.iloc[self.indices["train"]]
        clean_model.fit(train[list(FEATURE_NAMES)], train["label"].map(self.mapping), verbose=False)
        clean, _, _ = fit_sigmoid(clean_model, calibration, self.manifest)
        poisoned_calibrated, _, _ = fit_sigmoid(clean_model, poisoned_calibration, self.manifest)
        np.testing.assert_array_equal(clean.predict_proba(calibration.X), poisoned_calibrated.predict_proba(calibration.X))

    def test_frozen_estimator_fit_cannot_refit_base_model(self):
        calibration = self.partition("calibration")
        frozen, _, _ = fit_sigmoid(self.base, calibration, self.manifest)
        before = booster_digest(self.base)
        frozen.fit(calibration.X, calibration.y)
        self.assertEqual(booster_digest(self.base), before)

    def test_metrics_include_required_fields_and_per_class_values(self):
        calibration = self.partition("calibration")
        frozen, _, _ = fit_sigmoid(self.base, calibration, self.manifest)
        probabilities = checked_probabilities(frozen.predict_proba(calibration.X), len(calibration.X))
        stats = subject_statistics(calibration.y, probabilities, calibration.X.index.get_level_values("subject_id"))
        metrics = point_metrics(stats)
        for name in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier_score"):
            self.assertIn(name, metrics)
            self.assertTrue(np.isfinite(metrics[name]))
        self.assertEqual(set(metrics["per_class"]), set(CLASSES))
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).sum(), len(calibration.X))

    def test_reliability_bins_are_deterministic_and_cover_every_row(self):
        calibration = self.partition("calibration")
        frozen, _, _ = fit_sigmoid(self.base, calibration, self.manifest)
        probabilities = frozen.predict_proba(calibration.X)
        first = reliability_bins(calibration.y, probabilities)
        second = reliability_bins(calibration.y, probabilities)
        self.assertEqual(first, second)
        for series in first["series"].values():
            self.assertEqual(sum(bin_["count"] for bin_ in series["bins"]), len(calibration.X))

    def test_subject_bootstrap_is_paired_and_reproducible(self):
        calibration = self.partition("calibration")
        frozen, _, _ = fit_sigmoid(self.base, calibration, self.manifest)
        probabilities = frozen.predict_proba(calibration.X)
        subjects = calibration.X.index.get_level_values("subject_id")
        stats = subject_statistics(calibration.y, probabilities, subjects)
        report1, weights1 = cluster_bootstrap({"calibrated": stats, "uncalibrated": stats}, repeats=20, seed=42)
        report2, weights2 = cluster_bootstrap({"calibrated": stats, "uncalibrated": stats}, repeats=20, seed=42)
        self.assertEqual(report1, report2)
        np.testing.assert_array_equal(weights1, weights2)
        self.assertEqual(report1["paired_calibrated_minus_uncalibrated"]["log_loss"]["estimate"], 0)
        self.assertEqual(report1["models"]["calibrated"]["accuracy"]["estimate"], point_metrics(stats)["accuracy"])


class PublishedPhase3Tests(unittest.TestCase):
    @unittest.skipUnless((OUTPUT_DIR / "evaluation_report.json").exists(), "Run Phase 3 first.")
    def test_published_bundle_checksums_and_single_test_ledger(self):
        directory = OUTPUT_DIR
        loaded = load_calibrated_model(directory)
        self.assertEqual(tuple(loaded.model.feature_names_in_), FEATURE_NAMES)
        self.assertEqual(tuple(loaded.model.classes_), (0, 1, 2))
        report = json.loads((directory / "evaluation_report.json").read_text())
        self.assertEqual(report["test_rows"], 100000)
        self.assertEqual(report["test_subjects"], 100)
        self.assertEqual(report["test_inference_calls"], {"calibrated": 1, "uncalibrated": 1})
        self.assertFalse(report["further_tuning_performed"])
        completed = json.loads((directory / "test_inference_completed.json").read_text())
        self.assertEqual(completed["calibrated_test_predict_calls"], 1)
        self.assertEqual(completed["uncalibrated_test_predict_calls"], 1)
        ledger = Path(directory.parent / "phase3_test_evaluation.lock.json")
        self.assertTrue(ledger.exists())
        checksums = json.loads((directory / "artifact_checksums.json").read_text())
        for name, fingerprint in checksums.items():
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            self.assertEqual(digest, fingerprint["sha256"], name)
        self.assertEqual(json.loads((directory / "class_mapping.json").read_text()),
                         {str(i): label for i, label in enumerate(CLASSES)})
        schema = json.loads((directory / "feature_schema.json").read_text())
        self.assertEqual(schema["feature_names"], list(FEATURE_NAMES))
        with np.load(directory / "test_predictions.npz", allow_pickle=False) as predictions:
            self.assertEqual(predictions["labels"].shape, (100000,))
            self.assertEqual(predictions["calibrated"].shape, (100000, 3))
            np.testing.assert_allclose(predictions["calibrated"].sum(axis=1), 1, atol=1e-12)

    def test_completed_phase3_refuses_any_rerun_or_overwrite(self):
        with self.assertRaisesRegex(FileExistsError, "already started"):
            run_phase3(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
