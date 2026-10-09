"""Phase 1 integrity and leakage regression tests; no estimator is imported or fit."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.model.audit_eeg_v2 import json_text, run_audit, write_new_or_identical
from src.model.eeg_v2_data import (
    CLASSES, COLUMNS, DATA_FILE, FEATURE_NAMES, ROOT, SPLIT_COUNTS,
    build_manifest, dataset_fingerprint, feature_audit, load_manifest_dataset,
    model_features, require_valid_structure, split_row_indices,
    structural_audit, subject_partition, validate_manifest,
)


def fixture():
    """1000 subjects, each with three distinct windows and all three classes."""
    rng = np.random.default_rng(7)
    rows = 3000
    powers = rng.uniform(1, 3, (rows, 5))
    total = powers.sum(axis=1)
    data = {"subject_id": np.repeat([f"S{i:04d}" for i in range(1, 1001)], 3),
            "sample_id": np.tile([1, 2, 3], 1000)}
    for i, name in enumerate(FEATURE_NAMES[:5]):
        data[name] = powers[:, i]
    for i, name in enumerate(FEATURE_NAMES[5:10]):
        data[name] = powers[:, i] / total
    data.update({
        "theta_beta_ratio": powers[:, 1] / powers[:, 3],
        "alpha_beta_ratio": powers[:, 2] / powers[:, 3],
        "spectral_entropy": np.full(rows, 3.0), "hjorth_activity": total,
        "hjorth_mobility": np.full(rows, 0.5), "hjorth_complexity": np.full(rows, 2.0),
        "rms": np.sqrt(total), "variance": total, "kurtosis": np.zeros(rows),
        "line_length": np.sqrt(total) * 420, "label": np.tile(CLASSES, 1000),
    })
    return pd.DataFrame(data, columns=COLUMNS)


class Phase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.fingerprint = {"path": "fixture.csv", "size_bytes": 123, "sha256": "a" * 64}
        cls.partitions = subject_partition(cls.original["subject_id"].unique())
        cls.manifest = build_manifest(cls.original, cls.fingerprint, samples_per_subject=3)

    def setUp(self):
        self.df = self.original.copy(deep=True)

    def assert_invalid(self, df, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            require_valid_structure(df, samples_per_subject=3)

    def test_exact_counts_disjoint_subjects_and_complete_row_coverage(self):
        indices = split_row_indices(self.df, self.partitions)
        seen_subjects, seen_rows = set(), set()
        for name, count in SPLIT_COUNTS.items():
            self.assertEqual(len(self.partitions[name]), count)
            subjects = set(self.df.iloc[indices[name]]["subject_id"])
            self.assertEqual(subjects, set(self.partitions[name]))
            self.assertFalse(subjects & seen_subjects)
            self.assertFalse(set(indices[name]) & seen_rows)
            seen_subjects.update(subjects)
            seen_rows.update(indices[name])
            self.assertEqual(len(indices[name]), count * 3)
            self.assertEqual(set(self.df.iloc[indices[name]]["label"]), set(CLASSES))
        self.assertEqual(seen_subjects, set(self.df["subject_id"]))
        self.assertEqual(seen_rows, set(range(len(self.df))))

    def test_repeated_sample_ids_across_subjects_are_valid(self):
        self.assertEqual(self.df["sample_id"].nunique(), 3)
        self.assertFalse(structural_audit(self.df, 3)["errors"])

    def test_repeat_runs_and_row_reordering_produce_identical_manifest(self):
        shuffled = self.df.sample(frac=1, random_state=99)
        shuffled.index = np.zeros(len(shuffled), dtype=int)
        rebuilt = build_manifest(shuffled, self.fingerprint, samples_per_subject=3)
        self.assertEqual(json_text(rebuilt), json_text(self.manifest))
        indices = split_row_indices(shuffled, self.partitions)
        for name, rows in indices.items():
            self.assertEqual(set(shuffled.iloc[rows]["subject_id"]), set(self.partitions[name]))

    def test_assignment_is_independent_of_subject_input_order(self):
        subjects = list(self.df["subject_id"].unique())
        self.assertEqual(subject_partition(subjects), subject_partition(subjects[::-1]))

    def test_seed_changes_membership_reproducibly(self):
        subjects = self.df["subject_id"].unique()
        self.assertNotEqual(subject_partition(subjects, 43), self.partitions)
        self.assertEqual(subject_partition(subjects, 43), subject_partition(subjects, 43))

    def test_invalid_seeds_rejected(self):
        for seed in (-1, 1.5, "42", True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                subject_partition(self.df["subject_id"].unique(), seed)

    def test_subject_assignment_does_not_depend_on_labels_or_features(self):
        self.df["label"] = self.df["label"].iloc[::-1].to_numpy()
        self.df["rms"] *= 2
        manifest = build_manifest(self.df, self.fingerprint, samples_per_subject=3)
        for name in SPLIT_COUNTS:
            self.assertEqual(manifest["splits"][name]["subjects"], self.partitions[name])

    def test_schema_order_missing_and_extra_columns_rejected(self):
        for df in (self.df[list(reversed(COLUMNS))], self.df.drop(columns="rms"),
                   self.df.assign(leaked_target=1)):
            with self.subTest(columns=list(df.columns)):
                self.assert_invalid(df, "schema")

    def test_nan_infinity_and_text_features_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), "not-a-number"):
            frame = self.df.copy()
            frame["rms"] = frame["rms"].astype(object)
            frame.loc[0, "rms"] = bad
            with self.subTest(value=bad):
                self.assert_invalid(frame, "numeric and finite")

    def test_invalid_subject_ids_rejected(self):
        for bad in ("", " S0001", "S0001 ", None, 17):
            frame = self.df.copy()
            frame["subject_id"] = frame["subject_id"].astype(object)
            frame.loc[0, "subject_id"] = bad
            with self.subTest(value=bad):
                self.assert_invalid(frame, "Subject IDs")

    def test_invalid_sample_ids_rejected(self):
        for bad in (0, -1, 1.5, 4, float("inf"), float("nan"), "bad"):
            frame = self.df.copy()
            frame["sample_id"] = frame["sample_id"].astype(object)
            frame.loc[0, "sample_id"] = bad
            with self.subTest(value=bad):
                self.assert_invalid(frame, "Sample IDs")

    def test_duplicate_subject_sample_key_rejected(self):
        self.df.loc[1, "sample_id"] = 1
        self.assert_invalid(self.df, "Duplicate.*windows")

    def test_missing_subject_or_window_rejected(self):
        self.assert_invalid(self.df.iloc[3:], "Exactly 1000")
        self.assert_invalid(self.df.iloc[1:], "exactly 3 windows")

    def test_wrong_or_missing_labels_rejected(self):
        for bad in ("Unknown", None):
            frame = self.df.copy()
            frame.loc[0, "label"] = bad
            with self.subTest(value=bad):
                self.assert_invalid(frame, "Missing values|Labels")
        self.df["label"] = "Focused"
        self.assert_invalid(self.df, "Labels")

    def test_empty_dataset_rejected(self):
        self.assert_invalid(self.df.iloc[:0], "not be empty")

    def test_subject_overlap_rejected(self):
        partitions = copy.deepcopy(self.partitions)
        partitions["test"][0] = partitions["train"][0]
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            split_row_indices(self.df, partitions)

    def test_missing_partition_or_duplicate_member_rejected(self):
        partitions = copy.deepcopy(self.partitions)
        del partitions["validation"]
        with self.assertRaisesRegex(ValueError, "Exactly train"):
            split_row_indices(self.df, partitions)
        partitions = copy.deepcopy(self.partitions)
        partitions["train"][0] = partitions["train"][1]
        with self.assertRaisesRegex(ValueError, "duplicate subject count"):
            split_row_indices(self.df, partitions)

    def test_unknown_subject_and_unassigned_rows_rejected(self):
        partitions = copy.deepcopy(self.partitions)
        partitions["test"][0] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "cover every dataset subject"):
            split_row_indices(self.df, partitions)

    def test_each_partition_requires_all_classes(self):
        for name in SPLIT_COUNTS:
            frame = self.df.copy()
            frame.loc[frame["subject_id"].isin(self.partitions[name]), "label"] = "Focused"
            with self.subTest(split=name), self.assertRaisesRegex(ValueError, name):
                split_row_indices(frame, self.partitions)

    def test_identical_features_across_partitions_rejected_even_with_different_label(self):
        source = self.df.index[self.df["subject_id"].eq(self.partitions["train"][0])][0]
        target = self.df.index[self.df["subject_id"].eq(self.partitions["test"][0])][1]
        self.df.loc[target, list(FEATURE_NAMES)] = self.df.loc[source, list(FEATURE_NAMES)].to_numpy()
        with self.assertRaisesRegex(ValueError, "Identical feature vectors cross"):
            split_row_indices(self.df, self.partitions)

    def test_duplicate_features_within_partition_reported_without_false_leakage(self):
        self.df.loc[1, list(FEATURE_NAMES)] = self.df.loc[0, list(FEATURE_NAMES)].to_numpy()
        self.assertEqual(structural_audit(self.df, 3)["duplicate_feature_vectors"], 1)
        split_row_indices(self.df, self.partitions)

    def test_feature_allowlist_excludes_identifiers_labels_and_added_metadata(self):
        self.df["partition"] = "train"
        self.df["encoded_label"] = 1
        matrix = model_features(self.df)
        self.assertEqual(list(matrix.columns), list(FEATURE_NAMES))
        matrix.iloc[0, 0] = -1
        self.assertGreater(self.df.iloc[0]["delta_power"], 0)

    def test_tampered_manifest_fields_rejected(self):
        mutations = (
            lambda m: m.update(seed=43), lambda m: m.update(algorithm="row-random"),
            lambda m: m.update(rows=1), lambda m: m["feature_names"].reverse(),
            lambda m: m["splits"]["test"].update(rows=1),
            lambda m: m["splits"]["train"]["class_counts"].update(Focused=0),
        )
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with self.subTest(manifest=manifest["seed"]), self.assertRaisesRegex(ValueError, "contract"):
                validate_manifest(self.df, manifest, self.fingerprint, samples_per_subject=3)

    def test_disjoint_but_reassigned_manifest_subjects_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        train, test = manifest["splits"]["train"]["subjects"], manifest["splits"]["test"]["subjects"]
        train[0], test[0] = test[0], train[0]
        with self.assertRaisesRegex(ValueError, "contract"):
            validate_manifest(self.df, manifest, self.fingerprint, samples_per_subject=3)

    def test_stale_dataset_fingerprint_rejected(self):
        for change in ({"sha256": "b" * 64}, {"size_bytes": 124}):
            fingerprint = {**self.fingerprint, **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_manifest(self.df, self.manifest, fingerprint, samples_per_subject=3)

    def test_manifest_roundtrip_and_file_byte_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv = Path(directory) / "fixture.csv"
            manifest_path = Path(directory) / "split.json"
            self.df.to_csv(csv, index=False)
            fingerprint = dataset_fingerprint(csv)
            manifest = build_manifest(self.df, fingerprint, samples_per_subject=3)
            write_new_or_identical(manifest_path, manifest)
            loaded, indices = load_manifest_dataset(csv, manifest_path, samples_per_subject=3)
            self.assertEqual(len(loaded), 3000)
            self.assertEqual(len(indices["validation"]), 300)
            with csv.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_manifest_dataset(csv, manifest_path, samples_per_subject=3)

    def test_feature_inconsistencies_are_measured_without_mutating_data(self):
        self.df.loc[0, "rms"] = 0.1
        self.df.loc[1, "kurtosis"] = -2.5
        before = self.df.copy(deep=True)
        report = feature_audit(self.df)
        self.assertEqual(report["counts"]["rms_squared_below_variance"], 1)
        self.assertEqual(report["counts"]["kurtosis_below_unbiased_1024_sample_bound"], 1)
        self.assertEqual(report["counts"]["hjorth_activity_variance_exact_duplicates"], len(self.df))
        pd.testing.assert_frame_equal(self.df, before)
        self.assertFalse(structural_audit(self.df, 3)["errors"])

    def test_bias_corrected_kurtosis_is_not_incorrectly_rejected_at_minus_two(self):
        self.df.loc[0, "kurtosis"] = -2.001
        counts = feature_audit(self.df)["counts"]
        self.assertEqual(counts["kurtosis_below_minus_two"], 1)
        self.assertEqual(counts["kurtosis_below_unbiased_1024_sample_bound"], 0)

    def test_spectral_identity_and_range_mismatches_are_recorded(self):
        self.df.loc[0, "relative_delta"] = 2
        self.df.loc[1, "theta_beta_ratio"] = 999
        self.df.loc[2, "alpha_beta_ratio"] = 999
        self.df.loc[3, "hjorth_activity"] = -1
        counts = feature_audit(self.df)["counts"]
        for name in ("relative_power_outside_unit_interval", "relative_power_sum_mismatch",
                     "relative_power_formula_mismatch", "theta_beta_ratio_mismatch",
                     "alpha_beta_ratio_mismatch", "hjorth_activity_variance_mismatch",
                     "negative_non_kurtosis_features"):
            with self.subTest(check=name):
                self.assertEqual(counts[name], 1)

    def test_existing_artifacts_are_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "manifest.json"
            write_new_or_identical(path, self.manifest)
            before = path.stat().st_mtime_ns
            write_new_or_identical(path, self.manifest)
            self.assertEqual(path.stat().st_mtime_ns, before)
            with self.assertRaises(FileExistsError):
                write_new_or_identical(path, {"different": True})
            self.assertEqual(json.loads(path.read_text()), self.manifest)

    def test_invalid_audit_writes_failure_report_but_no_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv = Path(directory) / "invalid.csv"
            self.df.drop(columns="rms").to_csv(csv, index=False)
            output = Path(directory) / "reports"
            with patch("builtins.print"):
                report = run_audit(csv, output)
            self.assertEqual(report["status"], "failed")
            self.assertTrue((output / "dataset_audit.json").exists())
            self.assertFalse((output / "split_manifest.json").exists())

    def test_audit_paths_cannot_escape_neurosense(self):
        with self.assertRaisesRegex(ValueError, "inside the NeuroSense"):
            run_audit(DATA_FILE, ROOT.parent / "outside-report")


class PublishedDatasetTests(unittest.TestCase):
    @unittest.skipUnless(
        DATA_FILE.exists() and (ROOT / "reports/eeg_v2_phase1/split_manifest.json").exists(),
        "Run the Phase 1 audit against the local EEG V2 dataset first.",
    )
    def test_published_manifest_matches_all_one_million_windows(self):
        directory = ROOT / "reports/eeg_v2_phase1"
        df, indices = load_manifest_dataset(DATA_FILE, directory / "split_manifest.json")
        audit = json.loads((directory / "dataset_audit.json").read_text())
        self.assertEqual(audit["dataset"], dataset_fingerprint(DATA_FILE))
        self.assertEqual(audit["integrity"]["errors"], [])
        self.assertTrue(audit["dataset_unchanged"])
        self.assertEqual(len(df), 1_000_000)
        for name, subjects in SPLIT_COUNTS.items():
            self.assertEqual(len(indices[name]), subjects * 1000)
            self.assertEqual(df.iloc[indices[name]]["subject_id"].nunique(), subjects)
        self.assertEqual(sum(map(len, indices.values())), len(df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
