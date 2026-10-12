"""Deterministic whole-patient splitting of approved recording metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from .schemas import (
    AgeGroup, EEGRecording, ResearchSchema, SHA256, Text, require_research_ready,
)

SplitName = Literal["train", "validation", "test"]
ALGORITHM = "sha256-json-seed-patient-v1"
EXPLICIT_ALGORITHM = "reviewed-explicit-patient-v1"
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
    algorithm: Literal["sha256-json-seed-patient-v1", "reviewed-explicit-patient-v1"] = ALGORITHM
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


class WindowSplitInput(ResearchSchema):
    """A processed window inherits its recording's patient partition."""

    window_id: Text
    recording_id: Text
    patient_group_id: Text
    start_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    end_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def check_interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Window end must follow start.")
        return self


class WindowSplitAssignment(WindowSplitInput):
    split: SplitName


class WindowSplitManifest(ResearchSchema):
    schema_version: Literal["1"] = "1"
    split_manifest_sha256: SHA256
    assignments: tuple[WindowSplitAssignment, ...]

    @model_validator(mode="after")
    def check_membership(self):
        if len({item.window_id for item in self.assignments}) != len(self.assignments):
            raise ValueError("Duplicate window IDs.")
        patients, recordings = {}, {}
        for item in self.assignments:
            if item.patient_group_id in patients and patients[item.patient_group_id] != item.split:
                raise ValueError("Patient window leakage across splits.")
            if item.recording_id in recordings and recordings[item.recording_id] != (item.patient_group_id, item.split):
                raise ValueError("Recording window identity or split conflict.")
            patients[item.patient_group_id] = item.split
            recordings[item.recording_id] = (item.patient_group_id, item.split)
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


def assign_windows_to_splits(
    windows: Sequence[WindowSplitInput], manifest: SplitManifest,
) -> WindowSplitManifest:
    """Assign every window solely from its recording's canonical patient split.

    Overlapping windows are permitted within a patient partition. A recording
    absent from the split, an identity mismatch, or a duplicate window ID fails.
    """
    manifest = SplitManifest.model_validate(manifest)
    assignments = {item.recording_id: item for item in manifest.assignments}
    checked = tuple(WindowSplitInput.model_validate(item) for item in windows)
    if len({item.window_id for item in checked}) != len(checked):
        raise ValueError("Duplicate window IDs.")
    result = []
    for window in sorted(checked, key=lambda item: (item.recording_id, item.start_seconds, item.end_seconds, item.window_id)):
        recording = assignments.get(window.recording_id)
        if recording is None:
            raise ValueError("Window recording is absent from the patient split manifest.")
        if recording.patient_group_id != window.patient_group_id:
            raise ValueError("Window canonical patient identity conflicts with recording split.")
        result.append(WindowSplitAssignment(**window.model_dump(), split=recording.split))
    return WindowSplitManifest(split_manifest_sha256=manifest.sha256, assignments=tuple(result))


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
    alias_ids = {subject_groups[("chb_mit", subject)] for subject in ("chb01", "chb21")
                 if ("chb_mit", subject) in subject_groups}
    if len(alias_ids) > 1:
        raise ValueError("Known CHB-MIT patient aliases must share one canonical patient identity.")
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
    if manifest.algorithm == EXPLICIT_ALGORITHM:
        records = checked_recordings(recordings)
        if len({item.patient_group_id for item in records}) < 3:
            raise ValueError("At least three distinct patients are required.")
        by_id = {item.recording_id: item for item in records}
        expected_cohort = digest([item.model_dump(mode="json") for item in records])
        if manifest.cohort_sha256 != expected_cohort or set(by_id) != {
                item.recording_id for item in manifest.assignments}:
            raise ValueError("Explicit split manifest does not cover the exact frozen cohort.")
        for assignment in manifest.assignments:
            record = by_id[assignment.recording_id]
            if (assignment.patient_group_id != record.patient_group_id
                    or assignment.age_group != record.age.group
                    or assignment.metadata_sha256 != digest(record.model_dump(mode="json"))):
                raise ValueError("Explicit split assignment differs from reviewed recording metadata.")
        patient_splits = {}
        for assignment in manifest.assignments:
            prior = patient_splits.setdefault(assignment.patient_group_id, assignment.split)
            if prior != assignment.split:
                raise ValueError("Patient leakage across explicit partitions.")
        return
    expected = build_split_manifest(recordings, seed=manifest.seed,
                                    train_fraction=manifest.train_fraction,
                                    validation_fraction=manifest.validation_fraction)
    if manifest != expected:
        raise ValueError("Split manifest does not match the complete cohort and deterministic assignments.")


def build_explicit_split_manifest(
    recordings: Iterable[EEGRecording], patient_splits: dict[str, SplitName], *, seed: int = 42,
) -> SplitManifest:
    """Freeze a reviewer-declared patient partition, retaining full validation."""
    records = checked_recordings(recordings)
    patient_ids = {item.patient_group_id for item in records}
    if set(patient_splits) != patient_ids or set(patient_splits.values()) != {"train", "validation", "test"}:
        raise ValueError("Explicit split must assign every patient exactly once to all three partitions.")
    assignments = tuple(SplitAssignment(
        recording_id=item.recording_id, patient_group_id=item.patient_group_id,
        age_group=item.age.group, metadata_sha256=digest(item.model_dump(mode="json")),
        split=patient_splits[item.patient_group_id],
    ) for item in records)
    result = SplitManifest(
        algorithm=EXPLICIT_ALGORITHM, seed=seed,
        cohort_sha256=digest([item.model_dump(mode="json") for item in records]),
        assignments=assignments,
    )
    validate_split_manifest(records, result)
    return result
