"""Read-only local readiness audit for the proposed CHB-MIT pilot.

This command performs no network access, file creation, EEG decoding, feature
extraction, or model fitting. It only reads review artifacts and hashes files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import StrictBool

from .preprocessing import PreprocessingConfig
from .splitting import digest
from .training import (
    AnnotationEvidenceManifest, DiskBudget, PILOT_FILES, PILOT_SEIZURE_FILES,
    PILOT_SPLITS, validate_pilot_cohort, validate_pilot_preprocessing,
)
from .ingestion import CohortManifest
from .schemas import ResearchSchema, SHA256, Text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE_MANIFEST = REPOSITORY_ROOT / "docs" / "chbmit_pilot_file_manifest.json"
SUBJECTS = tuple(sorted(PILOT_SPLITS))
SUMMARY_PATHS = {f"{subject}/{subject}-summary.txt" for subject in SUBJECTS}
EPHEMERAL_ROOTS = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"), Path("/run"))


class ApprovalRecord(ResearchSchema):
    """Human-authored decisions bound to one reviewed cohort manifest."""

    schema_version: Literal["1"] = "1"
    cohort_manifest_sha256: SHA256 | None = None
    dataset_terms_status: Literal["pending", "approved"] = "pending"
    dataset_terms_reviewer: Text | None = None
    dataset_terms_reviewed_on: date | None = None
    dataset_terms_evidence_ref: Text | None = None
    institutional_determination: Literal["pending", "approved", "exempt", "not_required"] = "pending"
    institutional_reviewer: Text | None = None
    institutional_reviewed_on: date | None = None
    institutional_evidence_ref: Text | None = None
    faculty_status: Literal["pending", "approved"] = "pending"
    faculty_reviewer: Text | None = None
    faculty_reviewed_on: date | None = None
    faculty_evidence_ref: Text | None = None
    staging_manifest_sha256: SHA256 | None = None
    staging_status: Literal["pending", "approved"] = "pending"
    staging_approver: Text | None = None
    staging_approved_on: date | None = None
    staging_approval_ref: Text | None = None
    staging_root: Text | None = None
    storage_persistence_confirmed: StrictBool = False
    storage_policy_ref: Text | None = None
    processing_status: Literal["pending", "approved"] = "pending"
    processing_approver: Text | None = None
    processing_approved_on: date | None = None
    processing_approval_ref: Text | None = None


def validate_approved_storage(root: Path | str, approval: ApprovalRecord) -> Path:
    """Bind staging/processing to the custodian's named persistent location."""
    requested = Path(root).expanduser().absolute()
    for component in (requested, *requested.parents):
        if component.is_symlink():
            raise ValueError("storage_path_symlink_not_allowed")
    resolved = requested.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("research_data_must_be_outside_repository")
    if any(resolved.is_relative_to(path) for path in EPHEMERAL_ROOTS):
        raise ValueError("persistent_storage_required_temporary_root_rejected")
    if not approval.storage_persistence_confirmed or not approval.storage_policy_ref:
        raise PermissionError("persistent_storage_custodian_review_missing")
    if (not approval.staging_root or not Path(approval.staging_root).is_absolute()
            or Path(approval.staging_root).resolve() != resolved):
        raise PermissionError("staging_authorization_storage_root_mismatch")
    return resolved


def _read_json(path: Path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def _sha256(path: Path) -> str:
    digestor = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digestor.update(chunk)
    return digestor.hexdigest()


def _safe_existing_file(root: Path, relative: str) -> Path | None:
    candidate = root / relative
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None
    cursor = candidate.parent
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            return None
        cursor = cursor.parent
    return resolved


def _parse_official_sums(path: Path) -> dict[str, str]:
    """Parse sha256sum-compatible official checksum text, rejecting conflicts."""
    checksums: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = pattern.fullmatch(line)
        if not match:
            raise ValueError("official_checksum_file_malformed")
        checksum, name = match.groups()
        name = name.removeprefix("./")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("official_checksum_path_invalid")
        if name in checksums and checksums[name] != checksum.lower():
            raise ValueError("official_checksum_entry_conflict")
        checksums[name] = checksum.lower()
    if not checksums:
        raise ValueError("official_checksum_file_empty")
    return checksums


def _check_file_manifest(path: Path) -> tuple[dict, str]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != "2" or payload.get("dataset_version") != "1.0.0":
        raise ValueError("pilot_file_manifest_version_invalid")
    base = "https://physionet.org/files/chbmit/1.0.0"
    checksum_manifest = payload.get("checksum_manifest")
    if (not isinstance(checksum_manifest, dict)
            or checksum_manifest.get("path") != "SHA256SUMS.txt"
            or checksum_manifest.get("url") != f"{base}/SHA256SUMS.txt"
            or type(checksum_manifest.get("expected_bytes")) is not int
            or checksum_manifest["expected_bytes"] <= 0):
        raise ValueError("official_dataset_checksum_manifest_invalid")
    records = payload.get("recordings")
    patients = payload.get("patient_documents")
    if not isinstance(records, list) or not isinstance(patients, list):
        raise ValueError("pilot_file_manifest_entries_invalid")
    record_paths = [entry.get("path") for entry in records if isinstance(entry, dict)]
    patient_ids = [entry.get("patient") for entry in patients if isinstance(entry, dict)]
    if len(record_paths) != len(records) or set(record_paths) != PILOT_FILES or len(record_paths) != len(set(record_paths)):
        raise ValueError("pilot_recording_file_manifest_mismatch")
    if set(patient_ids) != set(PILOT_SPLITS) or len(patient_ids) != 6:
        raise ValueError("pilot_patient_document_manifest_mismatch")
    for entry in records:
        relative_path = entry.get("path")
        should_have_sidecar = relative_path in PILOT_SEIZURE_FILES
        if (entry.get("url") != f"{base}/{entry['path']}"
                or entry.get("patient") != relative_path.split("/")[0]
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("expected_sha256", "")))
                or type(entry.get("expected_bytes")) is not int or entry["expected_bytes"] <= 0
                or entry.get("partition") != PILOT_SPLITS[entry.get("patient")]
                or entry.get("annotation") != ("seizure" if should_have_sidecar else "background_review_required")
                or (should_have_sidecar and (
                    entry.get("sidecar_path") != relative_path + ".seizures"
                    or entry.get("sidecar_url") != f"{base}/{relative_path}.seizures"
                    or type(entry.get("sidecar_expected_bytes")) is not int
                    or entry["sidecar_expected_bytes"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sidecar_expected_sha256", "")))))
                or (not should_have_sidecar and ("sidecar_path" in entry or "sidecar_url" in entry))):
            raise ValueError("pilot_recording_manifest_entry_invalid")
    support_paths = []
    for entry in patients:
        patient = entry["patient"]
        if (entry.get("summary_path") != f"{patient}/{patient}-summary.txt"
                or entry.get("summary_url") != f"{base}/{patient}/{patient}-summary.txt"
                or type(entry.get("expected_bytes")) is not int or entry["expected_bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("expected_sha256", "")))):
            raise ValueError("pilot_patient_document_entry_invalid")
        support_paths.append(entry["summary_path"])
    sidecar_paths = [item + ".seizures" for item in PILOT_SEIZURE_FILES]
    paths = record_paths + support_paths + sidecar_paths + [checksum_manifest["path"]]
    expected = PILOT_FILES | SUMMARY_PATHS | {"SHA256SUMS.txt"} | {
        relative + ".seizures" for relative in PILOT_SEIZURE_FILES
    }
    if set(paths) != expected or len(paths) != len(set(paths)):
        raise ValueError("pilot_file_manifest_does_not_match_frozen_cohort")
    expected_edf_bytes = sum(item["expected_bytes"] for item in records)
    expected_staged_bytes = (expected_edf_bytes + checksum_manifest["expected_bytes"]
                             + sum(item["expected_bytes"] for item in patients)
                             + sum(item["sidecar_expected_bytes"] for item in records
                                   if "sidecar_path" in item))
    if (payload.get("expected_edf_bytes_total") != expected_edf_bytes
            or payload.get("expected_staged_bytes_total") != expected_staged_bytes):
        raise ValueError("pilot_manifest_size_totals_invalid")
    split = payload.get("split")
    if (not isinstance(split, dict)
            or split.get("train") != ["chb02", "chb05", "chb08"]
            or split.get("validation") != ["chb09"]
            or split.get("test") != ["chb06", "chb10"]
            or set(split.get("alias_safeguard", [])) != {"chb01", "chb21"}
            or ({item["patient"] for item in records} & {"chb01", "chb21"})):
        raise ValueError("pilot_patient_split_or_alias_safeguard_invalid")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def check_readiness(
    *, root: Path | str, cohort_manifest_path: Path | str,
    preprocessing_config_path: Path | str, annotation_evidence_path: Path | str,
    approval_record_path: Path | str | None = None,
    file_manifest_path: Path | str = DEFAULT_FILE_MANIFEST,
) -> dict:
    """Audit local files and review records without changing the filesystem."""
    report = {
        "schema_version": "chbmit-pilot-readiness-v1",
        "ready_for_real_data_processing": False,
        "patient_split": {split: sorted(patient for patient, name in PILOT_SPLITS.items()
                                        if name == split)
                          for split in ("train", "validation", "test")},
        "alias_safeguard": {"canonical_aliases": ["chb01", "chb21"],
                             "cohort_excludes_both": True},
        "checks": {}, "missing_approvals": [], "blockers": [],
    }

    def blocker(code: str):
        if code not in report["blockers"]:
            report["blockers"].append(code)

    file_manifest_file = Path(file_manifest_path)
    if not file_manifest_file.is_file():
        report["checks"]["file_manifest"] = {"status": "missing"}
        blocker("pilot_file_manifest_missing")
        file_manifest = None
    else:
        try:
            file_manifest, file_manifest_hash = _check_file_manifest(file_manifest_file)
            report["checks"]["file_manifest"] = {
                "status": "valid", "sha256": file_manifest_hash,
                "file_count": len(file_manifest["recordings"])
                + len(file_manifest["patient_documents"])
                + sum("sidecar_path" in entry for entry in file_manifest["recordings"])
                + 1,
            }
        except (OSError, ValueError, TypeError, KeyError):
            file_manifest = None
            report["checks"]["file_manifest"] = {"status": "invalid"}
            blocker("pilot_file_manifest_invalid")

    root_path = Path(root)
    root_ok = root_path.is_dir() and not root_path.is_symlink()
    if not root_ok:
        report["checks"]["storage"] = {"status": "root_missing_or_unsafe"}
        blocker("storage_root_missing_or_unsafe")
        root_resolved = None
    else:
        root_resolved = root_path.resolve(strict=True)
        if root_resolved.is_relative_to(REPOSITORY_ROOT):
            report["checks"]["storage"] = {"status": "root_inside_repository"}
            blocker("research_data_must_be_outside_repository")
            root_resolved = None

    cohort_file = Path(cohort_manifest_path)
    preprocessing_file = Path(preprocessing_config_path)
    evidence_file = Path(annotation_evidence_path)
    for name, path in (("cohort_manifest", cohort_file),
                       ("preprocessing_config", preprocessing_file),
                       ("annotation_evidence", evidence_file)):
        if not path.is_file() or path.is_symlink():
            report["checks"][name] = {"status": "missing"}
            blocker(f"{name}_missing")

    approval: ApprovalRecord | None = None
    if approval_record_path is None or not Path(approval_record_path).is_file():
        report["checks"]["approval_record"] = {"status": "missing"}
        report["missing_approvals"].extend((
            "dataset_terms_review", "institutional_ethics_determination",
            "faculty_seizure_research_guidance_and_cohort_review",
            "authorization_to_stage_real_recordings",
            "explicit_real_data_processing_and_training_authorization",
        ))
        blocker("human_approval_record_missing")
    else:
        approval_file = Path(approval_record_path)
        try:
            approval = ApprovalRecord.model_validate(_read_json(approval_file))
            report["checks"]["approval_record"] = {"status": "parsed"}
        except (OSError, ValueError, TypeError):
            report["checks"]["approval_record"] = {"status": "invalid"}
            blocker("human_approval_record_invalid")

    local_file_results = []
    hash_results = []
    verified_sums: dict[str, str] = {}
    if root_resolved is not None and file_manifest is not None:
        checksum_path = file_manifest["checksum_manifest"]["path"]
        expected_paths = PILOT_FILES | SUMMARY_PATHS | {checksum_path} | {
            item + ".seizures" for item in PILOT_SEIZURE_FILES}
        for relative in sorted(expected_paths):
            local = _safe_existing_file(root_resolved, relative)
            local_file_results.append({"path": relative, "status": "present" if local else "missing",
                                       "bytes": local.stat().st_size if local else None})
            if local is None:
                blocker("required_local_source_files_missing_or_unsafe")
        sums_file = _safe_existing_file(root_resolved, checksum_path)
        if sums_file is not None:
            try:
                verified_sums = _parse_official_sums(sums_file)
                if sums_file.stat().st_size != file_manifest["checksum_manifest"]["expected_bytes"]:
                    blocker("official_checksum_manifest_size_mismatch")
            except (OSError, ValueError):
                blocker("official_checksum_file_invalid")
        expected_hashes = {entry["path"]: entry["expected_sha256"]
                           for entry in file_manifest["recordings"]}
        expected_hashes.update({entry["summary_path"]: entry["expected_sha256"]
                                for entry in file_manifest["patient_documents"]})
        expected_hashes.update({entry["sidecar_path"]: entry["sidecar_expected_sha256"]
                                for entry in file_manifest["recordings"]
                                if "sidecar_path" in entry})
        for relative in sorted(expected_hashes):
            local = _safe_existing_file(root_resolved, relative)
            if local is None:
                continue
            expected_hash = expected_hashes[relative]
            official_hash = verified_sums.get(relative)
            if official_hash is None:
                hash_results.append({"path": relative, "status": "official_checksum_missing"})
                blocker("official_checksum_entry_missing")
                continue
            if official_hash != expected_hash:
                hash_results.append({"path": relative, "status": "official_manifest_conflicts_with_reviewed_cohort"})
                blocker("official_manifest_conflicts_with_reviewed_cohort")
                continue
            actual_hash = _sha256(local)
            matched = actual_hash == expected_hash
            hash_results.append({"path": relative, "status": "match" if matched else "mismatch",
                                 "sha256": actual_hash})
            if not matched:
                blocker("official_checksum_mismatch")
        expected_sizes = {entry["path"]: entry["expected_bytes"]
                          for entry in file_manifest["recordings"]}
        expected_sizes.update({entry["summary_path"]: entry["expected_bytes"]
                               for entry in file_manifest["patient_documents"]})
        expected_sizes.update({entry["sidecar_path"]: entry["sidecar_expected_bytes"]
                               for entry in file_manifest["recordings"]
                               if "sidecar_path" in entry})
        expected_sizes[checksum_path] = file_manifest["checksum_manifest"]["expected_bytes"]
        for relative, expected_size in expected_sizes.items():
            local = _safe_existing_file(root_resolved, relative)
            if local is not None and local.stat().st_size != expected_size:
                blocker("official_file_size_mismatch")
        allowed_eeg_files = set(PILOT_FILES) | {
            item + ".seizures" for item in PILOT_SEIZURE_FILES
        }
        unexpected_eeg_files = []
        for candidate in root_resolved.rglob("*"):
            if candidate.is_file() and candidate.name.lower().endswith((".edf", ".bdf", ".seizures")):
                relative = candidate.relative_to(root_resolved).as_posix()
                if relative not in allowed_eeg_files:
                    unexpected_eeg_files.append(relative)
        report["checks"]["unexpected_eeg_files"] = {
            "status": "clear" if not unexpected_eeg_files else "blocked",
            "files": sorted(unexpected_eeg_files),
        }
        if unexpected_eeg_files:
            blocker("unexpected_eeg_file_outside_frozen_cohort")
        report["checks"]["local_files"] = {
            "status": "checked", "expected_count": len(local_file_results),
            "files": local_file_results,
        }
        report["checks"]["official_hashes"] = {
            "status": "checked", "files": hash_results,
        }
        try:
            budget = DiskBudget(root_resolved)
            usage = budget.used_bytes()
            free = shutil.disk_usage(root_resolved).free
            budget.check()
            report["checks"]["storage"] = {
                "status": "within_limits", "used_bytes": usage,
                "cap_bytes": budget.cap_bytes, "free_bytes": free,
                "minimum_free_bytes": budget.reserve_bytes,
                "expected_source_bytes": file_manifest["expected_staged_bytes_total"],
            }
        except (OSError, ValueError):
            report["checks"].setdefault("storage", {"status": "limit_or_symlink_failure"})
            blocker("storage_cap_or_free_space_check_failed")

    manifest: CohortManifest | None = None
    config: PreprocessingConfig | None = None
    annotations: AnnotationEvidenceManifest | None = None
    for label, file, model in (
        ("cohort_manifest", cohort_file, CohortManifest),
        ("preprocessing_config", preprocessing_file, PreprocessingConfig),
        ("annotation_evidence", evidence_file, AnnotationEvidenceManifest),
    ):
        if not file.is_file() or file.is_symlink():
            continue
        try:
            parsed = model.model_validate(_read_json(file))
            if label == "cohort_manifest":
                manifest = parsed
                report["checks"][label] = {"status": "parsed", "sha256": manifest.sha256}
            elif label == "preprocessing_config":
                config = parsed
                validate_pilot_preprocessing(config)
                report["checks"][label] = {"status": "valid"}
            else:
                annotations = parsed
                report["checks"][label] = {"status": "parsed", "recordings": len(annotations.recordings)}
        except Exception:
            report["checks"][label] = {"status": "invalid_or_inconsistent"}
            blocker(f"{label}_invalid_or_inconsistent")

    if manifest is not None:
        review_plan = manifest.review_plan
        if review_plan.permissions.status != "approved":
            report["missing_approvals"].append("dataset_use_permission")
        if not review_plan.dataset_review.verified:
            report["missing_approvals"].append("dataset_scope_review")
        if not review_plan.identities.review.verified:
            report["missing_approvals"].append("patient_identity_and_alias_review")
        for item in review_plan.recordings:
            if not item.metadata_review.verified:
                report["missing_approvals"].append(f"recording_metadata_quality_review:{item.edf_path}")
            if not item.annotation_review.verified:
                report["missing_approvals"].append(f"annotation_review:{item.edf_path}")
        if approval is not None and approval.cohort_manifest_sha256 != manifest.sha256:
            blocker("approval_record_cohort_hash_mismatch")
        if root_resolved is None or annotations is None:
            blocker("cohort_annotation_validation_inputs_missing")
        elif root_resolved is not None:
            try:
                validate_pilot_cohort(manifest, annotations, root_resolved)
                report["checks"]["annotation_and_manifest_consistency"] = {
                    "status": "valid", "recordings": 12, "patients": 6,
                }
            except Exception:
                report["checks"]["annotation_and_manifest_consistency"] = {"status": "blocked"}
                blocker("cohort_hash_identity_permission_or_annotation_gate_failed")

    if config is None:
        blocker("approved_preprocessing_profile_missing")
    if annotations is None:
        report["missing_approvals"].append("annotation_provenance_and_coverage_evidence")

    if approval is not None:
        try:
            approved_root = validate_approved_storage(root, approval)
            report["checks"]["approved_storage"] = {"status": "approved_persistent_location",
                                                    "root": str(approved_root)}
        except (OSError, ValueError, PermissionError) as exc:
            report["checks"]["approved_storage"] = {"status": "blocked", "reason": str(exc)}
            blocker(str(exc))
        required_approvals = (
            ("dataset_terms_review", approval.dataset_terms_status == "approved"
             and all((approval.dataset_terms_reviewer, approval.dataset_terms_reviewed_on,
                      approval.dataset_terms_evidence_ref))),
            ("institutional_ethics_determination", approval.institutional_determination != "pending"
             and all((approval.institutional_reviewer, approval.institutional_reviewed_on,
                      approval.institutional_evidence_ref))),
            ("faculty_seizure_research_guidance_and_cohort_review", approval.faculty_status == "approved"
             and all((approval.faculty_reviewer, approval.faculty_reviewed_on,
                      approval.faculty_evidence_ref))),
            ("authorization_to_stage_real_recordings", approval.staging_status == "approved"
             and all((approval.staging_approver, approval.staging_approved_on,
                      approval.staging_approval_ref))),
            ("explicit_real_data_processing_and_training_authorization", approval.processing_status == "approved"
             and all((approval.processing_approver, approval.processing_approved_on,
                      approval.processing_approval_ref))),
        )
        for name, complete in required_approvals:
            if not complete:
                report["missing_approvals"].append(name)
                blocker(f"approval_missing:{name}")
        if (approval.staging_status == "approved" and file_manifest is not None
                and approval.staging_manifest_sha256 != report["checks"]["file_manifest"].get("sha256")):
            blocker("staging_authorization_manifest_hash_mismatch")
        if (approval.institutional_determination == "not_required"
                and not approval.institutional_evidence_ref):
            blocker("institutional_not_required_decision_needs_evidence")

    report["missing_approvals"] = sorted(set(report["missing_approvals"]))
    report["blockers"] = sorted(set(report["blockers"]))
    report["ready_for_real_data_processing"] = not report["blockers"] and not report["missing_approvals"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Read-only local pilot readiness audit")
    check.add_argument("--root", required=True)
    check.add_argument("--cohort-manifest", required=True)
    check.add_argument("--preprocessing-config", required=True)
    check.add_argument("--annotation-evidence", required=True)
    check.add_argument("--approval-record")
    check.add_argument("--file-manifest", default=str(DEFAULT_FILE_MANIFEST))
    args = parser.parse_args(argv)
    report = check_readiness(
        root=args.root, cohort_manifest_path=args.cohort_manifest,
        preprocessing_config_path=args.preprocessing_config,
        annotation_evidence_path=args.annotation_evidence,
        approval_record_path=args.approval_record, file_manifest_path=args.file_manifest,
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["ready_for_real_data_processing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
