"""Authorized, resumable lossless CHB-MIT byte archiving. No training occurs."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool

from .archive_selection import ARCHIVE_LIMIT, ZIP_OVERHEAD_LIMIT, validate_selection
from .inventory import BASE, safe_relative
from .readiness import ApprovalRecord, validate_approved_storage
from .schemas import SHA256, Text
from .splitting import digest

CHUNK = 1024 * 1024
MIN_SYSTEM_FREE = 20 * 1024**3
REPOSITORY = Path(__file__).resolve().parents[1]


class ArchiveApproval(ApprovalRecord):
    """Real human evidence only. Archiving bytes is distinct from interpreting EEG."""
    schema_version: Literal["chbmit-archive-approval-v1"] = "chbmit-archive-approval-v1"
    inventory_sha256: SHA256 | None = None
    selection_sha256: SHA256 | None = None
    temporary_root: Text | None = None
    temporary_byte_limit: int = Field(default=1024**3, strict=True, gt=0)
    temporary_cleanup_permitted: StrictBool = False
    archive_status: Literal["pending", "approved"] = "pending"
    archive_approval_ref: Text | None = None
    training_status: Literal["pending", "approved"] = "pending"
    training_approver: Text | None = None
    training_approved_on: Text | None = None
    training_approval_ref: Text | None = None
    training_plan_sha256: SHA256 | None = None
    training_split_sha256: SHA256 | None = None
    training_preprocessing_sha256: SHA256 | None = None
    training_review_packet_sha256: SHA256 | None = None
    training_run_config_sha256: SHA256 | None = None
    training_runs_root: Text | None = None


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_archive_json_key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite_archive_json")
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("archive_metadata_size_limit")
    result = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("archive_metadata_object_required")
    return result


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def safe_root(path: Path | str) -> Path:
    path = Path(path).expanduser().absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("archive_storage_symlink_rejected")
    path = path.resolve()
    if path.is_relative_to(REPOSITORY) or REPOSITORY.is_relative_to(path):
        raise ValueError("archive_storage_must_be_external_to_repository")
    if path.exists() and not path.is_dir():
        raise ValueError("archive_storage_not_directory")
    return path


def prepare_local_storage(base: Path | None = None) -> dict[str, Path]:
    """Idempotently create the user's private local CHB-MIT directory tree."""
    base = Path.home() / "NeuroSense-EEG-Data" if base is None else Path(base)
    base = base.expanduser().absolute()
    if any(item.is_symlink() for item in (base, *base.parents)):
        raise ValueError("local_storage_symlink_rejected")
    base = safe_root(base)
    if base == Path.home().resolve() or base.parent == Path("/"):
        raise ValueError("local_storage_location_invalid")
    uid = os.getuid() if hasattr(os, "getuid") else None
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = {name: base / name for name in ("archive", "temporary", "manifests", "reports", "runs")}
    for path in (base, *paths.values()):
        path.mkdir(mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)
        if uid is not None and path.stat().st_uid != uid:
            raise PermissionError("local_storage_directory_not_owned_by_current_user")
    return paths


def safe_file(root: Path, relative: str) -> Path:
    relative = safe_relative(relative)
    path = root / relative
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.resolve().is_relative_to(root):
        raise ValueError("archive_file_symlink_or_escape")
    return path


def directory_bytes(root: Path) -> int:
    total = 0
    for item in root.rglob("*") if root.exists() else ():
        if item.is_symlink():
            raise ValueError("archive_storage_contains_symlink")
        if item.is_file():
            total += item.stat().st_size
    return total


def anchor(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


class ArchiveBudget:
    """All retained bytes count. Temporary quota and system reserve are separate."""
    def __init__(self, root, temporary_root, *, limit=ARCHIVE_LIMIT, temporary_limit=1024**3):
        self.root, self.temporary_root = safe_root(root), safe_root(temporary_root)
        if self.root.is_relative_to(self.temporary_root) or self.temporary_root.is_relative_to(self.root):
            raise ValueError("archive_and_temporary_roots_must_be_disjoint")
        if type(limit) is not int or not 0 < limit <= ARCHIVE_LIMIT or type(temporary_limit) is not int or temporary_limit <= 0:
            raise ValueError("archive_budget_invalid")
        self.limit, self.temporary_limit = limit, temporary_limit

    def check(self, retained_add=0, temporary_add=0) -> dict:
        if any(type(value) is not int or value < 0 for value in (retained_add, temporary_add)):
            raise ValueError("archive_byte_estimate_invalid")
        retained, temporary = directory_bytes(self.root), directory_bytes(self.temporary_root)
        if retained + retained_add > self.limit:
            raise OSError("retained_archive_2gib_limit_exceeded")
        if temporary + temporary_add > self.temporary_limit:
            raise OSError("temporary_storage_limit_exceeded")
        system = Path("/")
        system_device = system.stat().st_dev
        devices = {system_device: [system, 0]}
        for root, additional in ((self.root, retained_add), (self.temporary_root, temporary_add)):
            location = anchor(root)
            device = location.stat().st_dev
            devices.setdefault(device, [location, 0])[1] += additional
        for device, (location, additional) in devices.items():
            reserve = MIN_SYSTEM_FREE if device == system_device else 0
            if shutil.disk_usage(location).free - additional < reserve:
                raise OSError("system_20gib_reserve_or_volume_space_insufficient")
        return {"archive_limit_bytes": self.limit, "archive_used_bytes": retained,
                "archive_remaining_bytes": self.limit - retained,
                "temporary_used_bytes": temporary, "temporary_limit_bytes": self.temporary_limit,
                "system_free_bytes": shutil.disk_usage(system).free,
                "system_reserve_bytes": MIN_SYSTEM_FREE}


def atomic_json(path: Path, payload: dict, budget: ArchiveBudget) -> None:
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    budget.check(retained_add=len(data))
    pending = safe_file(budget.root, path.relative_to(budget.root).as_posix() + ".pending")
    # A stale metadata transaction is disposable; it is never an original EDF.
    with pending.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def authorize(root, temporary_root, approval_path, inventory, selection, *, processing=False) -> ArchiveApproval:
    approval = ArchiveApproval.model_validate(read_json(Path(approval_path)))
    if approval.temporary_byte_limit > 1024**3:
        raise PermissionError("temporary_quota_must_not_exceed_one_gib")
    if (approval.staging_status != "approved" or approval.archive_status != "approved"
            or not all((approval.staging_approver, approval.staging_approved_on, approval.staging_approval_ref,
                        approval.archive_approval_ref))):
        raise PermissionError("archive_staging_authorization_missing")
    if (approval.dataset_terms_status != "approved" or not all((approval.dataset_terms_reviewer,
            approval.dataset_terms_reviewed_on, approval.dataset_terms_evidence_ref))):
        raise PermissionError("archive_data_use_review_missing")
    if approval.institutional_determination == "pending" or not all((approval.institutional_reviewer,
            approval.institutional_reviewed_on, approval.institutional_evidence_ref)):
        raise PermissionError("archive_institutional_determination_missing")
    if approval.faculty_status != "approved" or not all((approval.faculty_reviewer,
            approval.faculty_reviewed_on, approval.faculty_evidence_ref)):
        raise PermissionError("archive_faculty_cohort_review_missing")
    if (approval.inventory_sha256 != inventory["inventory_sha256"]
            or approval.selection_sha256 != selection["selection_sha256"]
            or approval.staging_manifest_sha256 != selection["selection_sha256"]):
        raise PermissionError("archive_approval_manifest_binding_mismatch")
    validate_approved_storage(root, approval)
    if not approval.temporary_root or safe_root(temporary_root) != safe_root(approval.temporary_root):
        raise PermissionError("archive_temporary_location_not_approved")
    if not approval.temporary_cleanup_permitted:
        raise PermissionError("verified_temporary_copy_cleanup_policy_missing")
    if processing and (approval.processing_status != "approved" or not all((approval.processing_approver,
            approval.processing_approved_on, approval.processing_approval_ref))):
        raise PermissionError("separate_real_eeg_processing_authorization_missing")
    return approval


def verify_original(path: Path, item: dict) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
        raise ValueError("original_checksum_or_size_mismatch:" + item["path"])


def verify_zip(path: Path, item: dict) -> dict:
    """Full exact decompression to a bounded hash stream, without materializing EEG."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("archive_file_missing_or_unsafe")
    h, size = hashlib.sha256(), 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != item["path"]
                or members[0].file_size != item["bytes"] or members[0].flag_bits & 1
                or members[0].compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError("archive_member_inventory_invalid")
        with archive.open(members[0]) as stream:
            while block := stream.read(CHUNK):
                size += len(block)
                if size > item["bytes"]:
                    raise ValueError("archive_decompressed_size_exceeded")
                h.update(block)
        method = members[0].compress_type
    if size != item["bytes"] or h.hexdigest() != item["sha256"]:
        raise ValueError("archive_decompression_checksum_mismatch")
    return {"original_path": item["path"], "original_sha256": h.hexdigest(), "original_bytes": size,
            "archive_sha256": file_hash(path), "archive_bytes": path.stat().st_size,
            "compression": "zip-deflate" if method == zipfile.ZIP_DEFLATED else "zip-stored",
            "compression_ratio": size / path.stat().st_size, "exact_decompression_verified": True}


class BudgetedOutput:
    def __init__(self, stream, budget):
        self.stream, self.budget = stream, budget

    def write(self, data):
        self.stream.flush()
        self.budget.check(temporary_add=len(data))
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def compress_recording(source: Path, destination: Path, item: dict, budget: ArchiveBudget) -> dict:
    verify_original(source, item)
    if destination.exists() or destination.is_symlink():
        raise ValueError("compression_destination_exists")
    bound = item["bytes"] + ZIP_OVERHEAD_LIMIT
    budget.check(temporary_add=bound)
    try:
        for method in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            with destination.open("xb") as stream:
                writer = BudgetedOutput(stream, budget)
                with zipfile.ZipFile(writer, "w", compression=method, compresslevel=6 if method == zipfile.ZIP_DEFLATED else None) as archive:
                    info = zipfile.ZipInfo(item["path"], date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = method
                    info.external_attr = 0o600 << 16
                    with source.open("rb") as raw, archive.open(info, "w", force_zip64=True) as packed:
                        while block := raw.read(CHUNK):
                            packed.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if destination.stat().st_size <= bound:
                return verify_zip(destination, item)
            destination.unlink()  # Only this newly generated compressed candidate, never the original.
        raise OSError("archive_exceeds_admitted_byte_bound")
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise


def recover_temporary_archive(packed: Path, original: Path, item: dict) -> bool:
    """Keep a complete orphan ZIP; discard only corrupt derivative bytes safely."""
    if not packed.exists():
        return False
    try:
        verify_zip(packed, item)
        return True
    except (ValueError, OSError, zipfile.BadZipFile):
        if original.exists():
            verify_original(original, item)
        packed.unlink()
        return False


def download_original(item: dict, temporary_root: Path, budget: ArchiveBudget, reauthorize, *, opener=None) -> Path:
    """Resume Range transfers; preserve incomplete/corrupt bytes for explicit review."""
    reauthorize(item["path"])
    if item["url"] != BASE + safe_relative(item["path"]):
        raise ValueError("download_url_not_official_selected_source")
    partial = safe_file(temporary_root, item["sha256"] + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["bytes"]:
        raise ValueError("partial_exceeds_expected_size")
    if offset == item["bytes"]:
        verify_original(partial, item)
        return partial
    budget.check(temporary_add=item["bytes"] - offset)
    request = urllib.request.Request(item["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
    with (opener or urllib.request.urlopen)(request, timeout=60) as response:
        if response.geturl() != item["url"]:
            raise ValueError("download_redirect_rejected")
        if offset and response.status == 200:
            # Only an incomplete transfer may be restarted; authorization already
            # confirms temporary cleanup. Full originals take the hash branch above.
            offset = 0
            budget.check(temporary_add=item["bytes"] - partial.stat().st_size)
        elif offset:
            expected = f"bytes {offset}-{item['bytes'] - 1}/{item['bytes']}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("resume_content_range_invalid")
        elif response.status != 200:
            raise ValueError("download_http_status_invalid")
        if int(response.headers.get("Content-Length", -1)) != item["bytes"] - offset:
            raise ValueError("download_content_length_mismatch")
        with partial.open("ab" if offset else "wb") as stream:
            received = offset
            while block := response.read(CHUNK):
                received += len(block)
                if received > item["bytes"]:
                    raise ValueError("download_exceeds_official_size")
                stream.flush()
                budget.check(temporary_add=len(block))
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if received != item["bytes"]:
            raise OSError("download_interrupted_or_truncated_resume_available")
    verify_original(partial, item)
    return partial


def promote_verified(source: Path, destination: Path, budget: ArchiveBudget) -> None:
    """Copy across filesystems atomically; all .pending retained bytes are counted."""
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = safe_file(budget.root, destination.relative_to(budget.root).as_posix() + ".pending")
    if pending.exists():
        pending.unlink()  # incomplete retained copy; verified temporary original is preserved
    budget.check(retained_add=source.stat().st_size)
    with source.open("rb") as src, pending.open("xb") as target:
        while block := src.read(CHUNK):
            target.flush()
            budget.check(retained_add=len(block))
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
    if file_hash(pending) != file_hash(source):
        raise ValueError("archive_promotion_hash_mismatch")
    os.replace(pending, destination)


def _inputs(inventory, selection):
    inventory = read_json(Path(inventory)) if not isinstance(inventory, dict) else inventory
    selection = read_json(Path(selection)) if not isinstance(selection, dict) else selection
    validate_selection(selection, inventory)
    return inventory, selection


def archive_plan(root, temporary_root, inventory, selection, *, temporary_limit=1024**3) -> dict:
    inventory, selection = _inputs(inventory, selection)
    budget = ArchiveBudget(root, temporary_root, limit=selection["budget_bytes"], temporary_limit=temporary_limit)
    status = budget.check()
    # Estimate the final selected archive and the largest download+compression
    # overlap together. On a shared device their needs must be summed against
    # the free-space reserve before staging begins.
    root, temporary_root = budget.root, budget.temporary_root
    retained_needed = max(0, selection["total_retained_byte_bound"] - status["archive_used_bytes"])
    pending_recordings = [item for item in selection["recordings"]
                          if not safe_file(root, "recordings/" + item["path"] + ".zip").exists()]
    missing_support = [item for item in selection["support_files"]
                       if not safe_file(root, "sources/" + item["path"]).exists()]
    max_edf = max((item["bytes"] for item in pending_recordings), default=0)
    max_support = max((item["bytes"] for item in missing_support), default=0)
    temporary_peak = max(2 * max_edf + ZIP_OVERHEAD_LIMIT if max_edf else 0,
                         2 * max_support if max_support else 0)
    temporary_needed = max(0, temporary_peak - status["temporary_used_bytes"])
    budget.check(retained_add=retained_needed, temporary_add=temporary_needed)
    return {"status": "dry_run_no_downloads", "selection_sha256": selection["selection_sha256"],
            "selected_recordings": selection["selected_recording_count"], "storage": status,
            "proposed_retained_upper_bound": selection["total_retained_byte_bound"],
            "minimum_peak_temporary_estimate": temporary_peak,
            "projected_retained_additional_bytes": retained_needed,
            "projected_temporary_additional_bytes": temporary_needed,
            "combined_disk_reserve_preflight": "passed",
            "processing_authorized": False, "training_performed": False}


def run_archive(root, temporary_root, inventory, selection, approval_path, *, confirmed=False, progress=None):
    if not confirmed:
        raise PermissionError("explicit_archive_execution_confirmation_required")
    inventory, selection = _inputs(inventory, selection)
    approval = authorize(root, temporary_root, approval_path, inventory, selection)
    budget = ArchiveBudget(root, temporary_root, limit=selection["budget_bytes"], temporary_limit=approval.temporary_byte_limit)
    root, temporary_root = budget.root, budget.temporary_root
    budget.check()
    archive_plan(root, temporary_root, inventory, selection,
                 temporary_limit=approval.temporary_byte_limit)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    allowed = {item["path"] for item in selection["recordings"] + selection["support_files"]}
    def gate(relative):
        if relative not in allowed:
            raise PermissionError("source_not_in_approved_selection")
        return authorize(root, temporary_root, approval_path, inventory, selection)
    def emit(event, **values):
        if progress:
            progress({"event": event, **values})
    with safe_file(root, ".archive.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another_archive_writer_is_active") from exc
        index_path = safe_file(root, "index.json")
        for name, payload in (("inventory.json", inventory), ("selection.json", selection)):
            path = safe_file(root, name)
            if path.exists():
                if read_json(path) != payload:
                    raise ValueError("archive_existing_manifest_conflict")
            else:
                atomic_json(path, payload, budget)
        index = read_json(index_path) if index_path.exists() else {
            "schema_version": "chbmit-lossless-archive-index-v1", "selection_sha256": selection["selection_sha256"],
            "inventory_sha256": inventory["inventory_sha256"], "recordings": {}, "support_files": {},
            "complete": False, "clinical_validation": False, "training_performed": False}
        if index["selection_sha256"] != selection["selection_sha256"]:
            raise ValueError("archive_index_selection_mismatch")
        current = None
        try:
            # Checksums/summaries are preserved byte-for-byte in the retained archive.
            for item in selection["support_files"]:
                current = item["path"]
                gate(current)
                destination = safe_file(root, "sources/" + current)
                if not destination.exists():
                    raw = download_original(item, temporary_root, budget, gate)
                    promote_verified(raw, destination, budget)
                    verify_original(destination, item)
                    raw.unlink()
                verify_original(destination, item)
                index["support_files"][current] = {"sha256": item["sha256"], "bytes": item["bytes"]}
                atomic_json(index_path, index, budget)
                emit("support_verified", path=current)
            for item in selection["recordings"]:
                current = item["path"]
                gate(current)  # fresh human record check before each EDF and on resume
                destination = safe_file(root, "recordings/" + current + ".zip")
                raw = safe_file(temporary_root, item["sha256"] + ".part")
                packed = safe_file(temporary_root, item["sha256"] + ".zip.part")
                if not destination.exists():
                    # Recover a verified orphan ZIP. A truncated temporary ZIP
                    # is disposable only because its source is independently
                    # verified (or can be re-fetched) and cleanup was approved.
                    if packed.exists() and not recover_temporary_archive(packed, raw, item):
                        emit("discarded_corrupt_temporary_archive", path=current)
                    if not packed.exists():
                        remaining_raw = item["bytes"] - (raw.stat().st_size if raw.exists() else 0)
                        budget.check(temporary_add=max(0, remaining_raw) + item["archive_byte_bound"])
                        # Existing .part files may be interrupted transfers.
                        # The downloader validates/resumes them before compression.
                        raw = download_original(item, temporary_root, budget, gate)
                        compress_recording(raw, packed, item, budget)
                    verify_zip(packed, item)
                    promote_verified(packed, destination, budget)
                verified = verify_zip(destination, item)
                prior = index["recordings"].get(current)
                if prior and (prior["archive_sha256"] != verified["archive_sha256"] or prior["original_sha256"] != item["sha256"]):
                    raise ValueError("indexed_archive_hash_changed")
                verified["archive_path"] = destination.relative_to(root).as_posix()
                verified["canonical_patient_id"] = item["canonical_patient_id"]
                verified["annotation_review"] = item["annotation_review"]
                index["recordings"][current] = verified
                atomic_json(index_path, index, budget)
                # Original source deletion only after official hash + full decompression
                # checks, durable index and the explicit retention-policy gate.
                if raw.exists():
                    verify_original(raw, item)
                    raw.unlink()
                if packed.exists():
                    verify_zip(packed, item)
                    packed.unlink()
                emit("recording_archived_and_verified", path=current, **verified)
                budget.check()
            index["complete"] = True
            atomic_json(index_path, index, budget)
            result = verify_archive(root, inventory, selection)
            result["storage"] = budget.check()
            emit("complete", **result)
            return result
        except Exception as exc:
            failure = {"status": "stopped", "path": current, "error_type": type(exc).__name__,
                       "reason": str(exc), "at_utc": datetime.now(timezone.utc).isoformat(),
                       "resume": "rerun the identical approved command; corrupt originals require custodian review"}
            # Log to stderr/progress even when the quota cannot admit a failure file.
            emit("failed", **failure)
            try:
                atomic_json(safe_file(root, "last_failure.json"), failure, budget)
            except OSError:
                pass
            raise


def verify_archive(root, inventory, selection) -> dict:
    inventory, selection = _inputs(inventory, selection)
    root = safe_root(root)
    if directory_bytes(root) > selection["budget_bytes"]:
        raise OSError("retained_archive_2gib_limit_exceeded")
    if read_json(safe_file(root, "inventory.json")) != inventory or read_json(safe_file(root, "selection.json")) != selection:
        raise ValueError("retained_manifest_mismatch")
    index = read_json(safe_file(root, "index.json"))
    if index.get("schema_version") != "chbmit-lossless-archive-index-v1" or index.get("selection_sha256") != selection["selection_sha256"]:
        raise ValueError("archive_index_invalid")
    results = []
    missing = []
    for item in selection["support_files"]:
        path = safe_file(root, "sources/" + item["path"])
        if not path.exists():
            missing.append(item["path"])
        else:
            verify_original(path, item)
            if index["support_files"].get(item["path"]) != {"sha256": item["sha256"], "bytes": item["bytes"]}:
                raise ValueError("support_index_mismatch")
    for item in selection["recordings"]:
        path = safe_file(root, "recordings/" + item["path"] + ".zip")
        if not path.exists() or item["path"] not in index["recordings"]:
            missing.append(item["path"])
            continue
        result = verify_zip(path, item)
        if any(index["recordings"][item["path"]].get(k) != v for k, v in result.items()):
            raise ValueError("archive_index_checksum_mismatch")
        results.append(result)
    actual = sum(item["archive_bytes"] for item in results)
    original = sum(item["original_bytes"] for item in results)
    return {"status": "complete_verified" if not missing and index.get("complete") is True else "incomplete",
            "verified_recordings": len(results), "missing": missing,
            "actual_compressed_eeg_bytes": actual, "original_verified_edf_bytes": original,
            "actual_compression_ratio": original / actual if actual else None,
            "total_retained_bytes": directory_bytes(root),
            "budget_remaining_bytes": selection["budget_bytes"] - directory_bytes(root),
            "training_performed": False}


@contextlib.contextmanager
def extracted_recording(root, temporary_root, inventory, selection, approval_path, relative_path):
    """Verified temporary EDF for a separately authorized future processing caller."""
    inventory, selection = _inputs(inventory, selection)
    approval = authorize(root, temporary_root, approval_path, inventory, selection, processing=True)
    budget = ArchiveBudget(root, temporary_root, limit=selection["budget_bytes"], temporary_limit=approval.temporary_byte_limit)
    item = next((item for item in selection["recordings"] if item["path"] == relative_path), None)
    if item is None:
        raise PermissionError("recording_not_in_selected_archive")
    archive_path = safe_file(budget.root, "recordings/" + item["path"] + ".zip")
    verified = verify_zip(archive_path, item)
    index = read_json(safe_file(budget.root, "index.json"))
    if (index.get("selection_sha256") != selection["selection_sha256"]
            or index.get("recordings", {}).get(relative_path, {}).get("archive_sha256") != verified["archive_sha256"]):
        raise ValueError("archive_reader_index_mismatch")
    budget.check(temporary_add=item["bytes"])
    with tempfile.TemporaryDirectory(prefix="verified-reader-", dir=budget.temporary_root) as directory:
        target = Path(directory) / relative_path
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with zipfile.ZipFile(archive_path) as archive, archive.open(relative_path) as src, target.open("xb") as dst:
            while block := src.read(CHUNK):
                dst.flush()
                budget.check(temporary_add=len(block))
                dst.write(block)
        verify_original(target, item)
        yield target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "stage", "verify", "smoke"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--approval-record", type=Path)
    parser.add_argument("--recording")
    parser.add_argument("--confirm-authorized-archive", action="store_true")
    parser.add_argument("--confirm-authorized-processing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_archive(args.root, args.inventory, args.selection)
        elif args.temporary_root is None:
            raise ValueError("separate_temporary_root_required")
        elif args.command == "plan":
            result = archive_plan(args.root, args.temporary_root, args.inventory, args.selection)
        elif args.approval_record is None:
            raise PermissionError("archive_approval_record_required")
        elif args.command == "stage":
            result = run_archive(args.root, args.temporary_root, args.inventory, args.selection,
                                 args.approval_record, confirmed=args.confirm_authorized_archive,
                                 progress=lambda event: print(json.dumps(event, sort_keys=True), flush=True))
        else:
            if not args.confirm_authorized_processing or not args.recording:
                raise PermissionError("separate_processing_confirmation_and_recording_required")
            with extracted_recording(args.root, args.temporary_root, args.inventory, args.selection,
                                     args.approval_record, args.recording) as edf:
                from .waveform import read_edf
                data = read_edf(edf.parent, edf.name, channel_indices=(0,), start_seconds=0,
                                end_seconds=5, max_samples=256 * 30)
                result = {"status": "bounded_decode_verified", "samples": len(data.channels[0].samples_uv),
                          "training_performed": False}
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result.get("status") != "incomplete" else 2
    except (OSError, ValueError, PermissionError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
