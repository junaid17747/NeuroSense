"""Approved, allowlisted CHB-MIT pilot staging. No command runs implicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from .readiness import (ApprovalRecord, _check_file_manifest, _parse_official_sums,
                        _read_json, validate_approved_storage)
from .training import MAX_PILOT_BYTES, MIN_FREE_BYTES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024


def _download_items(manifest: dict) -> list[dict]:
    items = [{"path": manifest["checksum_manifest"]["path"],
              "url": manifest["checksum_manifest"]["url"],
              "expected_bytes": manifest["checksum_manifest"]["expected_bytes"],
              "expected_sha256": None, "kind": "official_checksum_manifest"}]
    for item in manifest["patient_documents"]:
        items.append({"path": item["summary_path"], "url": item["summary_url"],
                      "expected_bytes": item["expected_bytes"],
                      "expected_sha256": item["expected_sha256"], "kind": "summary"})
    for item in manifest["recordings"]:
        items.append({"path": item["path"], "url": item["url"],
                      "expected_bytes": item["expected_bytes"],
                      "expected_sha256": item["expected_sha256"], "kind": "edf"})
        if "sidecar_path" in item:
            items.append({"path": item["sidecar_path"], "url": item["sidecar_url"],
                          "expected_bytes": item["sidecar_expected_bytes"],
                          "expected_sha256": item["sidecar_expected_sha256"],
                          "kind": "seizure_sidecar"})
    return items


def _safe_destination(root: Path) -> tuple[Path, Path]:
    """Return canonical root and existing filesystem anchor; reject repo/symlinks."""
    root = root.expanduser().absolute()
    if root.is_symlink():
        raise ValueError("staging_root_symlink_not_allowed")
    current = root
    while not current.exists():
        current = current.parent
    if not current.is_dir():
        raise ValueError("staging_root_parent_not_directory")
    cursor = root
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("staging_path_symlink_not_allowed")
        cursor = cursor.parent
    resolved_root = root.resolve(strict=False)
    resolved_anchor = current.resolve(strict=True)
    if resolved_root.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("research_data_must_be_outside_repository")
    return resolved_root, resolved_anchor


def _existing_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("staging_root_contains_symlink")
        if item.is_file():
            total += item.stat().st_size
    return total


def _assert_only_allowlisted_eeg(root: Path, manifest: dict) -> None:
    allowed = {item["path"] for item in manifest["recordings"]}
    allowed.update(item["sidecar_path"] for item in manifest["recordings"]
                   if "sidecar_path" in item)
    if not root.exists():
        return
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("staging_root_contains_symlink")
        if item.is_file() and item.name.lower().endswith((".edf", ".bdf", ".seizures")):
            relative = item.relative_to(root).as_posix()
            if relative not in allowed:
                raise ValueError("unexpected_eeg_file_outside_frozen_cohort")


def preflight_storage(root: Path | str, *, additional_bytes: int,
                      cap_bytes: int = MAX_PILOT_BYTES,
                      reserve_bytes: int = MIN_FREE_BYTES) -> dict:
    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ValueError("staging_size_invalid")
    resolved, anchor = _safe_destination(Path(root))
    used = _existing_bytes(resolved)
    free = shutil.disk_usage(anchor).free
    if used + additional_bytes > cap_bytes:
        raise OSError("pilot_storage_cap_exceeded")
    if free - additional_bytes < reserve_bytes:
        raise OSError("minimum_system_free_space_reserve_would_be_violated")
    return {"root": str(resolved), "existing_bytes": used,
            "planned_bytes": additional_bytes, "after_stage_bytes": used + additional_bytes,
            "cap_bytes": cap_bytes, "free_bytes": free,
            "free_after_stage_bytes": free - additional_bytes,
            "minimum_free_bytes": reserve_bytes}


def build_staging_plan(root: Path | str, manifest_path: Path | str) -> dict:
    manifest, manifest_sha256 = _check_file_manifest(Path(manifest_path))
    destination, _ = _safe_destination(Path(root))
    _assert_only_allowlisted_eeg(destination, manifest)
    items = _download_items(manifest)
    total = sum(item["expected_bytes"] for item in items)
    if total != manifest["expected_staged_bytes_total"]:
        raise ValueError("staging_plan_total_mismatch")
    # Existing verified files need no second allocation on a resumed staging run.
    remaining = sum(item["expected_bytes"] for item in items
                    if not (destination / item["path"]).is_file())
    storage = preflight_storage(root, additional_bytes=remaining)
    return {"schema_version": "chbmit-staging-plan-v1", "planning_only": True,
            "manifest_sha256": manifest_sha256,
            "dataset_page": manifest["dataset_page"], "files": items,
            "file_count": len(items), "edf_bytes": manifest["expected_edf_bytes_total"],
            "total_bytes": total, "storage": storage,
            "split": manifest["split"],
            "note": "No network requests or filesystem writes are performed by plan."}


def _approved_to_stage(approval_path: Path, manifest_sha256: str) -> ApprovalRecord:
    approval = ApprovalRecord.model_validate(_read_json(approval_path))
    if approval.staging_status != "approved" or not all((
            approval.staging_approver, approval.staging_approved_on,
            approval.staging_approval_ref)):
        raise PermissionError("explicit_staging_authorization_missing")
    if approval.dataset_terms_status != "approved" or not all((
            approval.dataset_terms_reviewer, approval.dataset_terms_reviewed_on,
            approval.dataset_terms_evidence_ref)):
        raise PermissionError("dataset_use_permission_missing")
    if approval.institutional_determination == "pending" or not all((
            approval.institutional_reviewer, approval.institutional_reviewed_on,
            approval.institutional_evidence_ref)):
        raise PermissionError("institutional_ethics_determination_missing")
    if approval.faculty_status != "approved" or not all((
            approval.faculty_reviewer, approval.faculty_reviewed_on,
            approval.faculty_evidence_ref)):
        raise PermissionError("faculty_cohort_review_missing")
    if getattr(approval, "staging_manifest_sha256", None) != manifest_sha256:
        raise PermissionError("staging_authorization_manifest_hash_mismatch")
    return approval


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _verify_existing(path: Path, item: dict, official_sums: dict[str, str]) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError("staging_destination_unsafe")
    if item["kind"] == "official_checksum_manifest":
        if path.stat().st_size != item["expected_bytes"]:
            raise ValueError("existing_staged_file_mismatch")
        _parse_official_sums(path)
        return True
    expected = item["expected_sha256"] or official_sums.get(item["path"])
    if expected is None:
        raise ValueError("official_checksum_entry_missing")
    if path.stat().st_size != item["expected_bytes"] or _sha256_file(path) != expected:
        raise ValueError("existing_staged_file_mismatch")
    return True


def _fetch_one(root: Path, item: dict) -> None:
    """Fetch one allowlisted file in bounded chunks after a fresh disk check."""
    preflight_storage(root, additional_bytes=item["expected_bytes"])
    destination = root / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("staging_destination_unsafe")
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ValueError("partial_download_already_exists_remove_after_review")
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(item["url"], headers={"User-Agent": "NeuroSense-research-staging/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
            if response.status != 200:
                raise OSError("source_http_status_invalid")
            length = response.headers.get("Content-Length")
            if length is None or int(length) != item["expected_bytes"]:
                raise OSError("source_content_length_mismatch")
            while block := response.read(CHUNK_BYTES):
                received += len(block)
                if received > item["expected_bytes"]:
                    raise OSError("source_exceeded_manifest_size")
                output.flush()
                preflight_storage(root, additional_bytes=len(block))
                output.write(block)
                digest.update(block)
        if received != item["expected_bytes"]:
            raise OSError("source_truncated")
        if item["expected_sha256"] and digest.hexdigest() != item["expected_sha256"]:
            raise OSError("source_sha256_mismatch")
        os.replace(partial, destination)
    except Exception:
        if partial.exists() and not partial.is_symlink():
            partial.unlink()
        raise


def stage_approved_pilot(root: Path | str, manifest_path: Path | str,
                         approval_path: Path | str) -> dict:
    manifest, manifest_sha256 = _check_file_manifest(Path(manifest_path))
    approval = _approved_to_stage(Path(approval_path), manifest_sha256)
    validate_approved_storage(root, approval)
    destination, _ = _safe_destination(Path(root))
    _assert_only_allowlisted_eeg(destination, manifest)
    plan = build_staging_plan(root, manifest_path)
    destination = Path(plan["storage"]["root"])
    destination.mkdir(parents=True, exist_ok=True)

    items = _download_items(manifest)
    checksum_item = items[0]
    checksum_path = destination / checksum_item["path"]
    if not _verify_existing(checksum_path, checksum_item, {}):
        _fetch_one(destination, checksum_item)
    official_sums = _parse_official_sums(checksum_path)

    expected = {item["path"]: item["expected_sha256"] for item in items[1:]}
    if any(official_sums.get(path) != checksum for path, checksum in expected.items()):
        raise ValueError("official_checksum_manifest_conflicts_with_frozen_cohort")
    results = []
    for item in items[1:]:
        target = destination / item["path"]
        already_present = _verify_existing(target, item, official_sums)
        if not already_present:
            _fetch_one(destination, item)
        results.append({"path": item["path"], "status": "verified_existing" if already_present else "downloaded_and_verified"})
    return {"status": "staged_and_verified", "manifest_sha256": manifest_sha256,
            "files": results, "file_count": len(results) + 1,
            "staged_bytes": _existing_bytes(destination)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Read-only storage and allowlist preflight")
    plan.add_argument("--root", required=True)
    plan.add_argument("--file-manifest", default=str(Path(__file__).resolve().parents[1] / "docs" / "chbmit_pilot_file_manifest.json"))
    stage = subparsers.add_parser("stage", help="Download only after recorded human authorization")
    stage.add_argument("--root", required=True)
    stage.add_argument("--file-manifest", default=str(Path(__file__).resolve().parents[1] / "docs" / "chbmit_pilot_file_manifest.json"))
    stage.add_argument("--approval-record", required=True)
    stage.add_argument("--confirm-authorized-staging", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = build_staging_plan(args.root, args.file_manifest)
        else:
            if not args.confirm_authorized_staging:
                raise PermissionError("explicit_confirm_authorized_staging_flag_required")
            result = stage_approved_pilot(args.root, args.file_manifest, args.approval_record)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
