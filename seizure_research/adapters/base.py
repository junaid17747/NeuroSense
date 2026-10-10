"""Adapter boundary and explicit human-reviewed ingestion inputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol

from pydantic import Field, StrictBool, StringConstraints, field_validator, model_validator

from ..schemas import (
    AgeAtRecording, ChannelMontage, EEGRecording, NonnegativeInt, PositiveNumber,
    RecordingDevice, RecordingQuality, ResearchPermissions, ResearchSchema, SHA256, Text,
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}$")]


class IngestionError(ValueError):
    """Stable code only; never include EDF identity fields or raw exception text."""


class HumanReview(ResearchSchema):
    verified: StrictBool = False
    reviewer: Identifier | None = None
    reviewed_on: date | None = None
    evidence_ref: Text | None = None

    @model_validator(mode="after")
    def require_evidence(self):
        if self.verified and not all((self.reviewer, self.reviewed_on, self.evidence_ref)):
            raise ValueError("Verified review requires a reviewer, date, and evidence reference.")
        return self


class PatientIdentity(ResearchSchema):
    subject_id: Identifier
    canonical_patient_id: Identifier


class IdentityMap(ResearchSchema):
    review: HumanReview
    entries: tuple[PatientIdentity, ...]

    @model_validator(mode="after")
    def reject_ambiguous_aliases(self):
        if len({item.subject_id for item in self.entries}) != len(self.entries):
            raise ValueError("Each source subject must occur exactly once in the identity map.")
        return self

    def resolve(self, subject_id: str) -> str:
        if not self.review.verified:
            raise IngestionError("identity_review_required")
        matches = [item.canonical_patient_id for item in self.entries if item.subject_id == subject_id]
        if len(matches) != 1:
            raise IngestionError("canonical_identity_missing_or_ambiguous")
        return matches[0]


class ChannelSelection(ResearchSchema):
    """Account for every source channel by index; never silently drop a channel."""

    source_index: NonnegativeInt
    source_label: Text
    action: Literal["keep", "exclude"]
    output_label: Text | None = None
    reason: Text | None = None

    @model_validator(mode="after")
    def check_action(self):
        if self.action == "keep" and self.output_label is None:
            raise ValueError("Kept channels require an explicit unique output label.")
        if self.action == "exclude" and (self.reason is None or self.output_label is not None):
            raise ValueError("Excluded channels need a reason and cannot have an output label.")
        if self.action == "keep" and self.output_label != self.source_label and self.reason is None:
            raise ValueError("Channel label aliases require an explicit review reason.")
        return self


class RecordingInput(ResearchSchema):
    edf_path: Text
    summary_path: Text
    subject_id: Identifier
    age: AgeAtRecording
    device: RecordingDevice
    montage: ChannelMontage
    channels: Annotated[tuple[ChannelSelection, ...], Field(min_length=1)]
    expected_sampling_rate_hz: PositiveNumber
    # Reviewed hashes bind all decisions below to exact source files.
    expected_signal_sha256: SHA256
    expected_summary_sha256: SHA256
    metadata_review: HumanReview
    annotation_review: HumanReview
    annotation_coverage: Literal["complete", "partial", "unknown"]
    annotation_provenance_ref: Text
    annotation_protocol_ref: Text
    annotator_ref: Text
    quality: RecordingQuality

    @field_validator("edf_path", "summary_path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        parts = value.split("/")
        if (PurePosixPath(value).is_absolute() or "\\" in value
                or any(part in ("", ".", "..") for part in parts)
                or any(ord(char) < 32 for char in value)):
            raise ValueError("Use a normalized relative POSIX path inside the local dataset root.")
        return value


class IngestionPlan(ResearchSchema):
    schema_version: Literal["1"] = "1"
    adapter_id: Literal["chb_mit"]
    dataset_version: Literal["1.0.0"]
    dataset_review: HumanReview
    permissions: ResearchPermissions
    identities: IdentityMap
    recordings: Annotated[tuple[RecordingInput, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_inputs(self):
        if len({item.edf_path for item in self.recordings}) != len(self.recordings):
            raise ValueError("Duplicate input recording paths.")
        return self


class SourceArtifact(ResearchSchema):
    relative_path: Text
    sha256: SHA256
    size_bytes: NonnegativeInt


class AdaptedRecording(ResearchSchema):
    recording: EEGRecording
    signal: SourceArtifact
    annotations: SourceArtifact
    recording_review_sha256: SHA256


class DatasetAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def ingest(self, root: Path, item: RecordingInput, plan: IngestionPlan) -> AdaptedRecording:
        """Validate local sources and return an approved Phase 1 recording or reject."""
        ...


def local_file(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise IngestionError("source_path_outside_root_or_not_regular")
    return path
