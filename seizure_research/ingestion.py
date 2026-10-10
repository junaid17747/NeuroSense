"""Deterministic cohort approval and metadata-only audit artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import pydantic
from pydantic import ValidationError

from .adapters.base import AdaptedRecording, DatasetAdapter, IngestionError, IngestionPlan
from .adapters.chb_mit import CHBMITAdapter
from .schemas import AgeGroup, ResearchSchema, SHA256, Text
from .splitting import canonical_json, checked_recordings, digest

PIPELINE_VERSION = "2.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS: dict[str, DatasetAdapter] = {"chb_mit": CHBMITAdapter()}


class AuditEntry(ResearchSchema):
    source_recording: Text
    status: Literal["validated", "rejected"]
    reasons: tuple[Text, ...] = ()


class CohortManifest(ResearchSchema):
    schema_version: Literal["1"] = "1"
    pipeline_version: Literal["2.0.0"] = PIPELINE_VERSION
    adapter_id: Text
    adapter_version: Text
    plan_sha256: SHA256
    review_plan: IngestionPlan
    records: tuple[AdaptedRecording, ...]

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class IngestionAudit(ResearchSchema):
    schema_version: Literal["1"] = "1"
    pipeline_version: Literal["2.0.0"] = PIPELINE_VERSION
    python_version: Text
    pydantic_version: Text
    plan_sha256: SHA256
    cohort_approved: bool
    manifest_sha256: SHA256 | None
    entries: tuple[AuditEntry, ...]
    cohort_errors: tuple[Text, ...]
    validated_recordings: int
    rejected_recordings: int
    validated_patients: int
    validated_seizure_free_recordings: int
    validated_recordings_per_age: dict[AgeGroup, int]


class IngestionResult(ResearchSchema):
    audit: IngestionAudit
    manifest: CohortManifest | None


def normalized_plan(plan: IngestionPlan) -> IngestionPlan:
    """Canonicalize set-like input order without reordering physical channels."""
    plan = IngestionPlan.model_validate(plan)
    data = plan.model_dump(mode="json")
    data["recordings"].sort(key=lambda item: item["edf_path"])
    data["identities"]["entries"].sort(key=lambda item: item["subject_id"])
    return IngestionPlan.model_validate(data)


def ingest_cohort(root: Path | str, plan: IngestionPlan) -> IngestionResult:
    """Approve all requested recordings or withhold the entire cohort manifest.

    A validated row in a rejected batch is an audit finding, not a partial
    approved cohort. No files or network connections are created by this call.
    """
    plan = normalized_plan(plan)
    plan_hash = digest(plan.model_dump(mode="json"))
    adapter = ADAPTERS[plan.adapter_id]
    entries, records = [], []
    for item in plan.recordings:
        try:
            result = adapter.ingest(Path(root), item, plan)
            records.append(result)
            entries.append(AuditEntry(source_recording=item.edf_path, status="validated"))
        except IngestionError as exc:
            entries.append(AuditEntry(source_recording=item.edf_path, status="rejected", reasons=(str(exc),)))
        except OSError:
            entries.append(AuditEntry(source_recording=item.edf_path, status="rejected", reasons=("source_unavailable",)))
        except (ValidationError, ValueError):
            entries.append(AuditEntry(source_recording=item.edf_path, status="rejected",
                                      reasons=("recording_contract_or_quality_invalid",)))
    cohort_errors = []
    if records:
        try:
            checked_recordings(record.recording for record in records)
        except ValueError:
            cohort_errors.append("cohort_duplicate_or_identity_conflict")
    rejected = sum(entry.status == "rejected" for entry in entries)
    manifest = None
    if not rejected and not cohort_errors:
        manifest = CohortManifest(adapter_id=adapter.adapter_id, adapter_version=adapter.adapter_version,
                                  plan_sha256=plan_hash, review_plan=plan, records=tuple(records))
    audit = IngestionAudit(
        python_version=".".join(map(str, sys.version_info[:3])), pydantic_version=pydantic.__version__,
        plan_sha256=plan_hash, cohort_approved=manifest is not None,
        manifest_sha256=manifest.sha256 if manifest else None,
        entries=tuple(entries), cohort_errors=tuple(cohort_errors), validated_recordings=len(records),
        rejected_recordings=rejected,
        validated_patients=len({entry.recording.patient_group_id for entry in records}),
        validated_seizure_free_recordings=sum(not entry.recording.annotations for entry in records),
        validated_recordings_per_age={age: sum(entry.recording.age.group == age for entry in records)
                                     for age in AgeGroup},
    )
    return IngestionResult(audit=audit, manifest=manifest)


def validate_manifest(root: Path | str, manifest: CohortManifest) -> None:
    """Re-read source bytes, reapply reviews, and reject stale/edited manifests."""
    manifest = CohortManifest.model_validate(manifest)
    rebuilt = ingest_cohort(root, manifest.review_plan)
    if rebuilt.manifest is None or rebuilt.manifest != manifest:
        raise IngestionError("cohort_manifest_or_sources_changed")


def artifact_bytes(result: IngestionResult) -> dict[str, bytes]:
    result = IngestionResult.model_validate(result)
    documents = {"audit.json": result.audit.model_dump(mode="json")}
    if result.manifest is not None:
        if not result.audit.cohort_approved or result.audit.manifest_sha256 != result.manifest.sha256:
            raise IngestionError("inconsistent_audit_and_manifest")
        documents["cohort_manifest.json"] = result.manifest.model_dump(mode="json")
    elif result.audit.cohort_approved:
        raise IngestionError("approved_audit_requires_manifest")
    return {name: (canonical_json(value) + "\n").encode("utf-8") for name, value in documents.items()}


def write_artifacts(result: IngestionResult, output_dir: Path | str) -> None:
    """Write JSON only outside this repository; refuse changed or stale bundles.

    A newly created directory is private. Interrupted writes are detectable:
    missing/extra files fail the exact-bundle check on a rerun. Never overwrite.
    """
    path = Path(output_dir)
    if path.is_symlink():
        raise IngestionError("artifact_symlink_not_allowed")
    path = path.resolve()
    if path.is_relative_to(REPOSITORY_ROOT):
        raise IngestionError("research_artifacts_must_be_outside_repository")
    files = artifact_bytes(result)
    if path.exists():
        if not path.is_dir() or {entry.name for entry in path.iterdir()} != set(files):
            raise IngestionError("artifact_bundle_conflict_use_new_directory")
        for name, payload in files.items():
            target = path / name
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise IngestionError("artifact_bundle_conflict_use_new_directory")
        return
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name, payload in files.items():
        descriptor = os.open(path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)


def load_plan(path: Path | str) -> IngestionPlan:
    """Reject repeated JSON keys instead of accepting a last-value override."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise IngestionError("duplicate_review_json_key")
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique_pairs)
    return IngestionPlan.model_validate(value)
