"""Phase 2 fit-boundary, early-stopping, serialization and integration checks."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_eeg_v2_phase1 import fixture
from src.model.eeg_v2_data import (
    CLASSES, DATA_FILE, FEATURE_NAMES, ROOT, build_manifest,
    dataset_fingerprint, split_row_indices,
)
from src.model.train_eeg_v2_phase2 import (
    MANIFEST_FILE, MODEL_ROOT, OUTPUT_DIR, TrainingConfig, assert_feature_matrix,
    assert_training_boundary, fit_training_data, load_phase2_model,
    prepare_training_data, reserve_run_directory, save_learning_curves,
    save_model_new, train, write_json_new,
)


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.manifest = build_manifest(
            cls.df, {"path": "fixture.csv", "sha256": "a" * 64, "size_bytes": 123},
            samples_per_subject=3,
        )
        cls.indices = split_row_indices(cls.df, {
            name: split["subjects"] for name, split in cls.manifest["splits"].items()
        })
        cls.data = prepare_training_data(cls.df, cls.indices, cls.manifest)

    def test_train_validation_have_exact_frozen_features_and_subject_counts(self):
        assert_training_boundary(self.data)
        for matrix, subjects, rows in (
            (self.data.X_train, 700, 2100), (self.data.X_validation, 100, 300),
        ):
            self.assertEqual(tuple(matrix.columns), FEATURE_NAMES)
            self.assertEqual(matrix.index.get_level_values("subject_id").nunique(), subjects)
            self.assertEqual(len(matrix), rows)
            self.assertFalse({"subject_id", "sample_id", "label"} & set(matrix.columns))

    def test_fit_interface_contains_no_calibration_or_test_matrix(self):
        self.assertEqual(set(vars(self.data)), {
            "X_train", "y_train", "X_validation", "y_validation", "classes", "manifest",
        })

    def test_estimator_and_eval_set_receive_only_train_and_validation(self):
        with patch("src.model.train_eeg_v2_phase2.XGBClassifier") as estimator:
            model, duration = fit_training_data(self.data, verbose=False)
        arguments = estimator.call_args.kwargs
        self.assertEqual(arguments["n_estimators"], 1000)
        self.assertEqual(arguments["random_state"], 42)
        self.assertEqual(arguments["eval_metric"], "mlogloss")
        self.assertEqual(arguments["device"], "cpu")
        callback = arguments["callbacks"][0]
        self.assertEqual(callback.rounds, 50)
        self.assertEqual(callback.data, "validation_1")
        self.assertEqual(callback.metric_name, "mlogloss")
        self.assertFalse(callback.maximize)
        self.assertTrue(callback.save_best)
        positional = model.fit.call_args.args
        self.assertIs(positional[0], self.data.X_train)
        self.assertIs(positional[1], self.data.y_train)
        evaluation = model.fit.call_args.kwargs["eval_set"]
        self.assertEqual(len(evaluation), 2)
        self.assertIs(evaluation[0][0], self.data.X_train)
        self.assertIs(evaluation[0][1], self.data.y_train)
        self.assertIs(evaluation[1][0], self.data.X_validation)
        self.assertIs(evaluation[1][1], self.data.y_validation)
        self.assertGreaterEqual(duration, 0)

    def test_label_encoder_is_fit_only_on_training_labels(self):
        from sklearn.preprocessing import LabelEncoder
        original_fit = LabelEncoder.fit
        observed = []

        def spy(encoder, labels):
            observed.append(labels.index.tolist())
            return original_fit(encoder, labels)

        with patch.object(LabelEncoder, "fit", spy):
            prepare_training_data(self.df, self.indices, self.manifest)
        self.assertEqual(observed, [self.indices["train"].tolist()])

    def test_changing_heldout_features_and_labels_cannot_change_training_inputs(self):
        poisoned = self.df.copy(deep=True)
        heldout_rows = np.concatenate([self.indices["calibration"], self.indices["test"]])
        poisoned.loc[heldout_rows, list(FEATURE_NAMES)] = np.nan
        poisoned.loc[heldout_rows, "label"] = "DO_NOT_USE"
        prepared = prepare_training_data(poisoned, self.indices, self.manifest)
        pd.testing.assert_frame_equal(prepared.X_train, self.data.X_train)
        pd.testing.assert_frame_equal(prepared.X_validation, self.data.X_validation)
        np.testing.assert_array_equal(prepared.y_train, self.data.y_train)
        np.testing.assert_array_equal(prepared.y_validation, self.data.y_validation)

    def test_injected_heldout_indices_rejected_before_fit(self):
        for heldout in ("calibration", "test"):
            for destination in ("train", "validation"):
                indices = {name: rows.copy() for name, rows in self.indices.items()}
                indices[destination][0] = indices[heldout][0]
                with self.subTest(heldout=heldout, destination=destination), self.assertRaisesRegex(ValueError, "indices"):
                    prepare_training_data(self.df, indices, self.manifest)

    def test_heldout_subject_provenance_rejected_at_fit_boundary(self):
        for heldout in ("calibration", "test"):
            for matrix_name in ("X_train", "X_validation"):
                matrix = getattr(self.data, matrix_name).copy()
                keys = matrix.index.tolist()
                keys[0] = (self.manifest["splits"][heldout]["subjects"][0], 1)
                matrix.index = pd.MultiIndex.from_tuples(keys, names=["subject_id", "sample_id"])
                data = replace(self.data, **{matrix_name: matrix})
                with self.subTest(heldout=heldout, matrix=matrix_name):
                    with patch("src.model.train_eeg_v2_phase2.XGBClassifier") as estimator:
                        with self.assertRaisesRegex(ValueError, "leakage"):
                            fit_training_data(data)
                        estimator.assert_not_called()

    def test_train_validation_overlap_rejected(self):
        matrix = self.data.X_validation.copy()
        keys = matrix.index.tolist()
        keys[0] = self.data.X_train.index[0]
        matrix.index = pd.MultiIndex.from_tuples(keys, names=["subject_id", "sample_id"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_training_boundary(replace(self.data, X_validation=matrix))

    def test_bad_manifest_membership_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["splits"]["validation"]["subjects"][0] = manifest["splits"]["train"]["subjects"][0]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            prepare_training_data(self.df, self.indices, manifest)

    def test_incomplete_partition_map_rejected(self):
        indices = dict(self.indices)
        del indices["test"]
        with self.assertRaisesRegex(ValueError, "four"):
            prepare_training_data(self.df, indices, self.manifest)

    def test_reordered_input_columns_are_selected_in_frozen_order(self):
        frame = self.df[list(reversed(self.df.columns))]
        data = prepare_training_data(frame, self.indices, self.manifest)
        pd.testing.assert_frame_equal(data.X_train, self.data.X_train)

    def test_reordered_or_extra_model_features_rejected(self):
        for matrix in (self.data.X_train[list(reversed(FEATURE_NAMES))],
                       self.data.X_train.assign(subject_id=1), self.data.X_train.drop(columns="rms")):
            with self.subTest(columns=list(matrix.columns)), self.assertRaisesRegex(ValueError, "20 features in order"):
                assert_feature_matrix(matrix)

    def test_missing_provenance_and_duplicate_windows_rejected(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            assert_training_boundary(replace(self.data, X_train=self.data.X_train.reset_index(drop=True)))
        matrix = self.data.X_train.copy()
        keys = matrix.index.tolist()
        keys[1] = keys[0]
        matrix.index = pd.MultiIndex.from_tuples(keys, names=matrix.index.names)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assert_training_boundary(replace(self.data, X_train=matrix))

    def test_nonfinite_features_and_invalid_labels_rejected(self):
        matrix = self.data.X_train.copy()
        matrix.iloc[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            assert_training_boundary(replace(self.data, X_train=matrix))
        with self.assertRaisesRegex(ValueError, "labels"):
            assert_training_boundary(replace(self.data, y_train=self.data.y_train[:-1]))
        with self.assertRaisesRegex(ValueError, "class order"):
            assert_training_boundary(replace(self.data, classes=tuple(reversed(CLASSES))))

    def test_training_limits_and_seed_are_enforced(self):
        for kwargs in ({"max_rounds": 1001}, {"max_rounds": 0}, {"max_rounds": True},
                       {"patience": 49}, {"seed": 43}, {"n_jobs": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs).validate()

    def test_output_cannot_target_v1_or_leave_project(self):
        for path in (ROOT / "models", MODEL_ROOT, ROOT.parent / "outside-models"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "strictly inside"):
                reserve_run_directory(path)

    def test_existing_run_is_refused_before_dataset_loading_or_fitting(self):
        MODEL_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=MODEL_ROOT) as directory:
            marker = Path(directory) / "existing.txt"
            marker.write_text("preserve")
            with patch("src.model.train_eeg_v2_phase2.load_manifest_dataset") as loader:
                with patch("src.model.train_eeg_v2_phase2.fit_training_data") as fit:
                    with self.assertRaises(FileExistsError):
                        train(output_dir=directory)
                    loader.assert_not_called()
                    fit.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")

    def test_json_artifacts_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "report.json"
            write_json_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_json_new(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_native_model_roundtrip_checksums_and_feature_order(self):
        model, _ = fit_training_data(self.data, TrainingConfig(max_rounds=3, n_jobs=1), verbose=False)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            model_path = directory / "xgboost.ubj"
            save_model_new(model, model_path)
            before = model_path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_model_new(model, model_path)
            self.assertEqual(before, model_path.read_bytes())
            write_json_new(directory / "class_mapping.json", {str(i): name for i, name in enumerate(CLASSES)})
            write_json_new(directory / "split_manifest.json", self.manifest)
            report = {
                "feature_names": list(FEATURE_NAMES), "classes": list(CLASSES),
                "configuration": {"n_jobs": 1}, "best_iteration_zero_based": model.best_iteration,
                "artifacts": {p.name: dataset_fingerprint(p) for p in directory.iterdir()},
            }
            write_json_new(directory / "training_report.json", report)
            restored = load_phase2_model(directory)
            np.testing.assert_array_equal(
                model.predict_proba(self.data.X_validation), restored.predict_proba(self.data.X_validation),
            )
            self.assertEqual(restored.classes, CLASSES)
            self.assertEqual(model.get_booster().num_boosted_rounds(), model.best_iteration + 1)
            with self.assertRaisesRegex(ValueError, "20 features in order"):
                restored.predict_proba(self.data.X_validation[list(reversed(FEATURE_NAMES))])
            with model_path.open("ab") as stream:
                stream.write(b"tampering")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_phase2_model(directory)

    def test_bad_saved_schema_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            write_json_new(Path(directory) / "training_report.json", {
                "feature_names": list(reversed(FEATURE_NAMES)), "classes": list(CLASSES),
            })
            with self.assertRaisesRegex(ValueError, "schema"):
                load_phase2_model(directory)

    def test_curves_record_every_round_including_patience_rounds(self):
        history = {"validation_0": {"mlogloss": [1.0, 0.8, 0.7]},
                   "validation_1": {"mlogloss": [1.0, 0.9, 0.95]}}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            save_learning_curves(Path(directory), history, best_iteration=1)
            report = json.loads((Path(directory) / "learning_curves.json").read_text())
            self.assertEqual(report["train_mlogloss"], [1.0, 0.8, 0.7])
            self.assertEqual(report["validation_mlogloss"], [1.0, 0.9, 0.95])
            self.assertEqual(report["best_iteration"], 1)
            self.assertTrue((Path(directory) / "learning_curves.png").read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(len(pd.read_csv(Path(directory) / "learning_curves.csv")), 3)


class PublishedPhase2Tests(unittest.TestCase):
    @unittest.skipUnless((OUTPUT_DIR / "reload_verification.json").exists(), "Run Phase 2 training first.")
    def test_published_model_and_full_run_contract(self):
        restored = load_phase2_model(OUTPUT_DIR)
        report = restored.report
        curves = json.loads((OUTPUT_DIR / "learning_curves.json").read_text())
        manifest_bytes = MANIFEST_FILE.read_bytes()
        self.assertEqual((OUTPUT_DIR / "split_manifest.json").read_bytes(), manifest_bytes)
        self.assertEqual(report["manifest"]["sha256"], hashlib.sha256(manifest_bytes).hexdigest())
        self.assertEqual(report["dataset"], dataset_fingerprint(DATA_FILE))
        self.assertEqual(report["configuration"]["max_rounds"], 1000)
        self.assertEqual(report["configuration"]["patience"], 50)
        self.assertLessEqual(report["rounds_executed"], 1000)
        self.assertEqual(report["best_iteration_zero_based"], int(np.argmin(curves["validation_mlogloss"])))
        self.assertEqual(report["best_validation_mlogloss"], min(curves["validation_mlogloss"]))
        self.assertEqual(len(curves["train_mlogloss"]), report["rounds_executed"])
        self.assertEqual(len(curves["validation_mlogloss"]), report["rounds_executed"])
        if report["stopping_reason"] == "early_stopping":
            self.assertEqual(report["rounds_executed"] - report["best_round_one_based"], 50)
        self.assertEqual(report["data_usage"]["fit"], ["train"])
        self.assertEqual(report["data_usage"]["early_stopping"], ["validation"])
        self.assertEqual(report["data_usage"]["metrics"], ["validation"])
        self.assertFalse(report["data_usage"]["calibration_performed"])
        self.assertFalse(report["data_usage"]["test_evaluated"])
        self.assertFalse(report["data_usage"]["hyperparameter_search_performed"])
        self.assertEqual({k: v["rows"] for k, v in report["split_summary"].items()},
                         {"train": 700000, "validation": 100000, "calibration": 100000, "test": 100000})
        for name, fingerprint in report["artifacts"].items():
            self.assertEqual(dataset_fingerprint(OUTPUT_DIR / name), fingerprint)
        reload_report = json.loads((OUTPUT_DIR / "reload_verification.json").read_text())
        self.assertEqual(reload_report["partition"], "validation")
        self.assertEqual(reload_report["rows"], 100000)
        self.assertTrue(reload_report["probabilities_exactly_equal"])
        self.assertEqual(reload_report["maximum_absolute_difference"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
