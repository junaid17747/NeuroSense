"""Readiness audit tests use missing/temporary paths only; no EEG is fetched."""

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


REPOSITORY = Path(__file__).resolve().parents[1]
FILE_MANIFEST = REPOSITORY / "docs" / "chbmit_pilot_file_manifest.json"


class ReadinessAuditTests(unittest.TestCase):
    def test_static_manifest_matches_exact_frozen_twelve_file_cohort(self):
        manifest, file_hash = _check_file_manifest(FILE_MANIFEST)
        self.assertEqual(len(manifest["recordings"]), 12)
        self.assertEqual(len(manifest["patient_documents"]), 6)
        self.assertEqual(len(file_hash), 64)
        self.assertTrue(all(item["expected_sha256"] is None for item in manifest["recordings"]))

    def test_official_checksums_parse_and_conflicts_fail_closed(self):
        checksum = hashlib.sha256(b"synthetic only").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS"
            path.write_text(f"{checksum}  ./sample.edf\n", encoding="utf-8")
            self.assertEqual(_parse_official_sums(path)["sample.edf"], checksum)
            path.write_text(f"{checksum}  sample.edf\nnot a checksum\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed"):
                _parse_official_sums(path)

    def test_staged_file_hash_is_compared_to_official_patient_checksum(self):
        manifest, _ = _check_file_manifest(FILE_MANIFEST)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files_by_patient = {}
            for item in manifest["recordings"]:
                relative = item["path"]
                files_by_patient.setdefault(item["patient"], {})[Path(relative).name] = b"synthetic EDF bytes"
                if "sidecar_path" in item:
                    files_by_patient[item["patient"]][Path(item["sidecar_path"]).name] = b"synthetic annotations"
            for item in manifest["patient_documents"]:
                files_by_patient.setdefault(item["patient"], {})[Path(item["summary_path"]).name] = b"synthetic summary"
            altered_patient, altered_name = "chb10", "chb10_01.edf"
            for patient, files in files_by_patient.items():
                directory = root / patient
                directory.mkdir()
                checksum_lines = []
                for name, content in files.items():
                    (directory / name).write_bytes(content)
                    checksum = hashlib.sha256(content).hexdigest()
                    if patient == altered_patient and name == altered_name:
                        checksum = hashlib.sha256(b"different bytes").hexdigest()
                    checksum_lines.append(f"{checksum}  {name}")
                (directory / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
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
            self.assertEqual(report["checks"]["local_files"]["expected_count"], 30)
            self.assertIn("human_approval_record_missing", report["blockers"])
            self.assertIn("faculty_cohort_and_protocol_review", report["missing_approvals"])
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

    def test_pending_human_decisions_and_storage_failures_remain_blockers(self):
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
            self.assertIn("explicit_real_data_processing_authorization", report["missing_approvals"])
            self.assertFalse(report["ready_for_real_data_processing"])


if __name__ == "__main__":
    unittest.main()
