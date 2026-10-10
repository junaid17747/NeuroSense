"""Deterministic whole-patient splitting of approved recording metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from .schemas import (
    AgeGroup, EEGRecording, ResearchSchema, SHA256, Text, require_research_ready,
)

SplitName = Literal["train", "validation", "test"]
ALGORITHM = "sha256-json-seed-patient-v1"
Fraction = Annotated[float, Field(strict=True, gt=0, lt=1, allow_inf_nan=False)]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class SplitAssignment(ResearchSchema):
    recording_id: Text
    patient_group_id: Text
    age_group: AgeGroup
    metadata_sha256: SHA256
    split: SplitName


class SplitManifest(ResearchSchema):
    schema_version: Literal["1"] = "1"
    algorithm: Literal["sha256-json-seed-patient-v1"] = ALGORITHM
    seed: Annotated[int, Field(strict=True)] = 42
    train_fraction: Fraction = 0.6
    validation_fraction: Fraction = 0.2
    cohort_sha256: SHA256
    assignments: Annotated[tuple[SplitAssignment, ...], Field(min_length=3)]

    @model_validator(mode="after")
    def check_membership(self):
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Train and validation fractions must leave a test partition.")
        if len({item.recording_id for item in self.assignments}) != len(self.assignments):
            raise ValueError("Duplicate recording assignments.")
        membership = {}
        for item in self.assignments:
            if item.patient_group_id in membership and membership[item.patient_group_id] != item.split:
                raise ValueError("Patient leakage across splits.")
            membership[item.patient_group_id] = item.split
        if set(membership.values()) != {"train", "validation", "test"}:
            raise ValueError("All three patient partitions must be nonempty.")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


def checked_recordings(recordings: Iterable[EEGRecording]) -> tuple[EEGRecording, ...]:
    records = tuple(sorted((EEGRecording.model_validate(item) for item in recordings),
                           key=lambda item: item.recording_id))
    if not records:
        raise ValueError("An explicit cohort of recordings is required.")
    if len({item.recording_id for item in records}) != len(records):
        raise ValueError("Duplicate recording IDs; use globally unique recording IDs.")
    if len({item.provenance.signal_sha256 for item in records}) != len(records):
        raise ValueError("Duplicate signal files; resolve repeated recordings before splitting/scoring.")
    source_keys = {(item.provenance.dataset_id, item.provenance.source_recording_id) for item in records}
    if len(source_keys) != len(records):
        raise ValueError("Repeated source recordings across releases must be resolved before splitting/scoring.")
    subject_groups = {}
    for item in records:
        subject_key = (item.provenance.dataset_id, item.subject_id)
        if subject_key in subject_groups and subject_groups[subject_key] != item.patient_group_id:
            raise ValueError("One dataset subject cannot have conflicting patient group IDs.")
        subject_groups[subject_key] = item.patient_group_id
        require_research_ready(item)
    return records


def build_split_manifest(
    recordings: Iterable[EEGRecording], *, seed: int = 42,
    train_fraction: float = 0.6, validation_fraction: float = 0.2,
) -> SplitManifest:
    records = checked_recordings(recordings)
    # Validate scalar inputs before using them to calculate slice bounds.
    seed = TypeAdapter(Annotated[int, Field(strict=True)]).validate_python(seed)
    train_fraction = TypeAdapter(Fraction).validate_python(train_fraction)
    validation_fraction = TypeAdapter(Fraction).validate_python(validation_fraction)
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition.")
    patients = sorted({item.patient_group_id for item in records},
                      key=lambda patient: (digest([seed, patient]), patient))
    if len(patients) < 3:
        raise ValueError("At least three distinct patients are required.")
    # Floor requested counts, bounded to reserve >=1 patient for every split.
    n_train = min(max(1, int(len(patients) * train_fraction)), len(patients) - 2)
    n_validation = min(max(1, int(len(patients) * validation_fraction)), len(patients) - n_train - 1)
    membership = {}
    for index, patient in enumerate(patients):
        membership[patient] = ("train" if index < n_train else
                               "validation" if index < n_train + n_validation else "test")
    assignments = tuple(SplitAssignment(
        recording_id=item.recording_id, patient_group_id=item.patient_group_id,
        age_group=item.age.group, metadata_sha256=digest(item.model_dump(mode="json")),
        split=membership[item.patient_group_id],
    ) for item in records)
    return SplitManifest(
        seed=seed, train_fraction=train_fraction, validation_fraction=validation_fraction,
        cohort_sha256=digest([item.model_dump(mode="json") for item in records]),
        assignments=assignments,
    )


def validate_split_manifest(recordings: Iterable[EEGRecording], manifest: SplitManifest) -> None:
    """Reject stale metadata, altered assignments, omissions, and patient leakage."""
    manifest = SplitManifest.model_validate(manifest)
    expected = build_split_manifest(recordings, seed=manifest.seed,
                                    train_fraction=manifest.train_fraction,
                                    validation_fraction=manifest.validation_fraction)
    if manifest != expected:
        raise ValueError("Split manifest does not match the complete cohort and deterministic assignments.")
