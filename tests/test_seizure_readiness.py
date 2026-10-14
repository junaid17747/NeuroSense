"""Readiness and staging tests use synthetic bytes only; no EEG is fetched."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_seizure_processing as synthetic_fixture
from seizure_research.ingestion import ingest_cohort
from seizure_research.readiness import (
    _check_file_manifest, _parse_official_sums, check_readiness,
)
from seizure_research.staging import build_staging_plan, preflight_storage, stage_approved_pilot
from seizure_research.preprocessing import ChannelMapping, make_age_config
from seizure_research.schemas import AgeGroup
from seizure_research.smoke import decode_smoke_window, run_smoke
from test_seizure_processing import _edf


REPOSITORY = Path(__file__).resolve().parents[1]
FILE_MANIFEST = REPOSITORY / "docs" / "chbmit_pilot_file_manifest.json"


class ReadinessAuditTests(unittest.TestCase):
    def test_manifest_pins_exact_cohort_hashes_sizes_and_patient_split(self):
        manifest, file_hash = _check_file_manifest(FILE_MANIFEST)
        self.assertEqual(len(manifest["recordings"]), 12)
        self.assertEqual(len(manifest["patient_documents"]), 6)
        self.assertEqual(len(file_hash), 64)
        self.assertTrue(all(len(item["expected_sha256"]) == 64 for item in manifest["recordings"]))
        self.assertEqual(manifest["checksum_manifest"]["path"], "SHA256SUMS.txt")
        self.assertEqual(manifest["expected_edf_bytes_total"], 1_004_143_360)
        self.assertEqual(manifest["expected_staged_bytes_total"], 1_004_239_275)
        self.assertEqual(manifest["split"]["test"], ["chb06", "chb10"])
        self.assertEqual(set(manifest["split"]["alias_safeguard"]), {"chb01", "chb21"})

    def test_official_checksums_preserve_relative_paths_and_reject_traversal(self):
        checksum = hashlib.sha256(b"synthetic only").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS.txt"
            path.write_text(f"{checksum}  ./chb02/sample.edf\n", encoding="utf-8")
            self.assertEqual(_parse_official_sums(path)["chb02/sample.edf"], checksum)
            path.write_text(f"{checksum}  ../escape.edf\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path_invalid"):
                _parse_official_sums(path)
            path.write_text(f"{checksum}  sample.edf\nnot a checksum\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed"):
                _parse_official_sums(path)

    def test_staged_files_must_match_pinned_official_global_checksum_entries(self):
        manifest, _ = _check_file_manifest(FILE_MANIFEST)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_hashes = {}
            for item in manifest["recordings"]:
                expected_hashes[item["path"]] = item["expected_sha256"]
                if "sidecar_path" in item:
                    expected_hashes[item["sidecar_path"]] = item["sidecar_expected_sha256"]
            for item in manifest["patient_documents"]:
                expected_hashes[item["summary_path"]] = item["expected_sha256"]
            checksum_lines = []
            for relative, checksum in expected_hashes.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"synthetic content does not match pinned source")
                checksum_lines.append(f"{checksum}  {relative}")
            checksum_path = root / "SHA256SUMS.txt"
            checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
            checksum_path.write_bytes(checksum_path.read_bytes().ljust(73298, b" "))
            with patch("seizure_research.training.shutil.disk_usage",
                       return_value=SimpleNamespace(free=100 * 1024**3)):
                report = check_readiness(
                    root=root, cohort_manifest_path=root / "missing-cohort.json",
                    preprocessing_config_path=root / "missing-config.json",
                    annotation_evidence_path=root / "missing-evidence.json",
                    file_manifest_path=FILE_MANIFEST,
                )
            self.assertIn("official_checksum_mismatch", report["blockers"])

    def test_missing_files_and_approvals_are_reported_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "empty-external-root"
            root.mkdir()
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            report = check_readiness(
                root=root,
                cohort_manifest_path=root / "cohort.json",
                preprocessing_config_path=root / "preprocessing.json",
                annotation_evidence_path=root / "annotations.json",
                approval_record_path=None,
                file_manifest_path=FILE_MANIFEST,
            )
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            self.assertFalse(report["ready_for_real_data_processing"])
            self.assertEqual(report["checks"]["file_manifest"]["status"], "valid")
            self.assertEqual(report["checks"]["file_manifest"]["file_count"], 25)
            self.assertEqual(report["checks"]["local_files"]["expected_count"], 25)
            self.assertIn("human_approval_record_missing", report["blockers"])
            self.assertIn("faculty_seizure_research_guidance_and_cohort_review", report["missing_approvals"])
            self.assertIn("authorization_to_stage_real_recordings", report["missing_approvals"])
            self.assertIn("required_local_source_files_missing_or_unsafe", report["blockers"])
            self.assertEqual(before, after)

    def test_staged_approval_cannot_be_used_for_another_cohort_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = ingest_cohort(root, synthetic_fixture._fixture(root)).manifest
            cohort_path = root / "cohort.json"
            cohort_path.write_text(cohort.model_dump_json(), encoding="utf-8")
            approval_path = root / "approvals.json"
            approval_path.write_text(json.dumps({"cohort_manifest_sha256": "a" * 64}), encoding="utf-8")
            report = check_readiness(
                root=root, cohort_manifest_path=cohort_path,
                preprocessing_config_path=root / "missing-preprocessing.json",
                annotation_evidence_path=root / "missing-annotations.json",
                approval_record_path=approval_path, file_manifest_path=FILE_MANIFEST,
            )
            self.assertIn("approval_record_cohort_hash_mismatch", report["blockers"])
            self.assertFalse(report["ready_for_real_data_processing"])

    def test_malformed_file_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({"schema_version": "1", "dataset_version": "1.0.0",
                                        "recordings": [], "patient_documents": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _check_file_manifest(path)

    def test_pending_decisions_and_storage_failures_remain_blockers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps({"cohort_manifest_sha256": "b" * 64}), encoding="utf-8")
            fake_budget = Mock()
            fake_budget.used_bytes.return_value = 5 * 1024**3 + 1
            fake_budget.cap_bytes = 5 * 1024**3
            fake_budget.reserve_bytes = 20 * 1024**3
            fake_budget.check.side_effect = OSError("pilot_storage_cap_exceeded")
            with patch("seizure_research.readiness.DiskBudget", return_value=fake_budget):
                report = check_readiness(
                    root=root, cohort_manifest_path=root / "cohort.json",
                    preprocessing_config_path=root / "preprocessing.json",
                    annotation_evidence_path=root / "annotations.json",
                    approval_record_path=approval_path, file_manifest_path=FILE_MANIFEST,
                )
            self.assertIn("storage_cap_or_free_space_check_failed", report["blockers"])
            self.assertIn("dataset_terms_review", report["missing_approvals"])
            self.assertIn("institutional_ethics_determination", report["missing_approvals"])
            self.assertIn("authorization_to_stage_real_recordings", report["missing_approvals"])
            self.assertIn("explicit_real_data_processing_and_training_authorization", report["missing_approvals"])
            self.assertFalse(report["ready_for_real_data_processing"])

    def test_read_only_plan_lists_exact_allowlist_and_does_not_create_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "external-pilot"
            with patch("seizure_research.staging.shutil.disk_usage",
                       return_value=SimpleNamespace(free=45 * 1024**3)):
                plan = build_staging_plan(root, FILE_MANIFEST)
            self.assertTrue(plan["planning_only"])
            self.assertEqual(plan["file_count"], 25)
            self.assertEqual(plan["total_bytes"], 1_004_239_275)
            self.assertEqual(plan["storage"]["free_after_stage_bytes"], 45 * 1024**3 - plan["total_bytes"])
            self.assertFalse(root.exists())

    def test_staging_storage_preflight_enforces_cap_and_free_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "external"
            root.mkdir()
            with patch("seizure_research.staging.shutil.disk_usage",
                       return_value=SimpleNamespace(free=100 * 1024**3)):
                with self.assertRaisesRegex(OSError, "cap_exceeded"):
                    preflight_storage(root, additional_bytes=6 * 1024**3)
            with patch("seizure_research.staging.shutil.disk_usage",
                       return_value=SimpleNamespace(free=21 * 1024**3)):
                with self.assertRaisesRegex(OSError, "reserve_would_be_violated"):
                    preflight_storage(root, additional_bytes=2 * 1024**3)

    def test_staging_plan_rejects_edf_outside_the_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "external"
            root.mkdir()
            extra = root / "unapproved" / "other.edf"
            extra.parent.mkdir()
            extra.write_bytes(b"synthetic only")
            with patch("seizure_research.staging.shutil.disk_usage",
                       return_value=SimpleNamespace(free=45 * 1024**3)):
                with self.assertRaisesRegex(ValueError, "outside_frozen_cohort"):
                    build_staging_plan(root, FILE_MANIFEST)

    def test_staging_refuses_pending_human_approval_before_network_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "external"
            approval = Path(temporary) / "pending-approval.json"
            approval.write_text("{}", encoding="utf-8")
            with patch("seizure_research.staging.urllib.request.urlopen") as open_url:
                with self.assertRaisesRegex(PermissionError, "explicit_staging_authorization_missing"):
                    stage_approved_pilot(root, FILE_MANIFEST, approval)
                open_url.assert_not_called()
            self.assertFalse(root.exists())

    def test_real_data_smoke_requires_authorization_and_decodes_only_a_bounded_window(self):
        with self.assertRaisesRegex(PermissionError, "explicit_real_data_smoke_test_authorization"):
            run_smoke(root="/tmp/unused", relative_path="chb02/chb02_01.edf",
                      cohort_manifest_path="/tmp/cohort.json",
                      preprocessing_config_path="/tmp/preprocessing.json",
                      annotation_evidence_path="/tmp/annotations.json",
                      approval_record_path="/tmp/approval.json", file_manifest_path=FILE_MANIFEST,
                      authorized=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "chb02" / "chb02_01.edf"
            target.parent.mkdir()
            target.write_bytes(_edf(duration=60))
            mapping = tuple(ChannelMapping(source_index=index, source_label=label,
                                           target_label=label, expected_source_rate_hz=128)
                            for index, label in enumerate(("FP1-F7", "F7-T7")))
            config = make_age_config(AgeGroup.CHILD, mapping, target_sampling_rate_hz=128,
                                     research_review_ref="synthetic-test")
            from seizure_research.edf import inspect_edf
            expected_hash = inspect_edf(target).sha256
            result = decode_smoke_window(root, "chb02/chb02_01.edf", config, expected_hash)
            self.assertEqual(result["status"], "decoded")
            self.assertEqual(result["samples_per_channel"], 640)
            self.assertFalse(result["writes_artifacts"])
            with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
                decode_smoke_window(root, "chb02/chb02_01.edf", config, "0" * 64)


if __name__ == "__main__":
    unittest.main()
