"""Versioned, conservative metadata contracts for de-identified clinical EEG."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel, ConfigDict, Field, HttpUrl, StrictBool, StringConstraints,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Seconds = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class ResearchSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class AgeGroup(str, Enum):
    NEONATAL = "neonatal"
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"


class AgeAtRecording(ResearchSchema):
    """Chronological age, never inferred from cohort name or surrogate EDF dates.

    Under one calendar year, completed_days is mandatory. At/after the first
    birthday only completed_years is stored. This avoids a 365-day year shortcut.
    """

    completed_years: NonnegativeInt
    completed_days: NonnegativeInt | None = None
    source: Text
    gestational_age_at_birth_weeks: PositiveNumber | None = None
    postmenstrual_age_at_recording_weeks: PositiveNumber | None = None

    @model_validator(mode="after")
    def check_precision(self) -> Self:
        if self.completed_years == 0:
            if self.completed_days is None or self.completed_days > 365:
                raise ValueError("Under-one age needs 0..365 completed days and verified calendar age.")
        elif self.completed_days is not None:
            raise ValueError("Store completed_days only for age under one calendar year.")
        return self

    @property
    def group(self) -> AgeGroup:
        if self.completed_years == 0:
            return AgeGroup.NEONATAL if self.completed_days <= 28 else AgeGroup.INFANT
        if self.completed_years <= 12:
            return AgeGroup.CHILD
        if self.completed_years <= 17:
            return AgeGroup.ADOLESCENT
        if self.completed_years <= 64:
            return AgeGroup.ADULT
        return AgeGroup.OLDER_ADULT

    @classmethod
    def from_dates(cls, birth_date: date, recording_date: date, *, source: str) -> Self:
        """Use verified dates transiently; never persist dates of birth here.

        For Feb 29 births the non-leap-year anniversary is March 1, by explicit
        research convention. Do not call this on de-identified surrogate dates.
        """
        if type(birth_date) is not date or type(recording_date) is not date:
            raise ValueError("Expected calendar dates, not datetimes or strings.")
        if recording_date < birth_date:
            raise ValueError("Recording precedes birth.")
        years = recording_date.year - birth_date.year - (
            (recording_date.month, recording_date.day) < (birth_date.month, birth_date.day)
        )
        return cls(completed_years=years,
                   completed_days=(recording_date - birth_date).days if years == 0 else None,
                   source=source)


class TimeInterval(ResearchSchema):
    """Half-open [start_seconds, end_seconds), relative to recording start."""

    start_seconds: Seconds
    end_seconds: PositiveNumber

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Interval end must follow start.")
        return self


def nonoverlapping(intervals: tuple[TimeInterval, ...]) -> bool:
    ordered = sorted(intervals, key=lambda interval: interval.start_seconds)
    return all(left.end_seconds <= right.start_seconds for left, right in zip(ordered, ordered[1:]))


def overlaps(left: TimeInterval, right: TimeInterval) -> bool:
    return max(left.start_seconds, right.start_seconds) < min(left.end_seconds, right.end_seconds)


class ChannelMontage(ResearchSchema):
    name: Text
    kind: Literal["bipolar", "referential", "average_reference", "other"]
    channels: Annotated[tuple[Text, ...], Field(min_length=1)]
    reference: Text
    units: Literal["uV", "mV", "V"]

    @model_validator(mode="after")
    def unique_channels(self) -> Self:
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Channel labels must be unique and ordered.")
        return self


class RecordingDevice(ResearchSchema):
    manufacturer: Text
    model: Text
    acquisition_software_version: Text
    clinical_eeg: StrictBool


class DatasetProvenance(ResearchSchema):
    dataset_id: Text
    version: Text
    source_url: HttpUrl
    source_recording_id: Text
    signal_sha256: SHA256
    annotation_sha256: SHA256
    annotation_source: Text


class SeizureAnnotation(TimeInterval):
    event_id: Text
    seizure_type: Text
    annotator_ref: Text
    clinical_review_ref: Text


class RecordingQuality(ResearchSchema):
    status: Literal["unreviewed", "acceptable", "unusable"]
    review_ref: Text | None = None
    excluded_intervals: tuple[TimeInterval, ...] = ()
    notes: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def check_review(self) -> Self:
        if self.status == "acceptable" and self.review_ref is None:
            raise ValueError("Acceptable quality requires a documented review.")
        if not nonoverlapping(self.excluded_intervals):
            raise ValueError("Quality exclusions must not overlap.")
        return self


class ResearchPermissions(ResearchSchema):
    """Recorded governance decisions, not an access-control or legal authority."""

    status: Literal["unknown", "denied", "approved"] = "unknown"
    purpose: Literal["offline_seizure_research"] = "offline_seizure_research"
    deidentified: StrictBool = False
    terms_review_ref: Text | None = None
    ethics_review_ref: Text | None = None
    access_review_ref: Text | None = None
    reviewed_by: Text | None = None
    reviewed_on: date | None = None

    @model_validator(mode="after")
    def check_approval(self) -> Self:
        if self.status == "approved" and not all((
            self.deidentified, self.terms_review_ref, self.ethics_review_ref,
            self.access_review_ref, self.reviewed_by, self.reviewed_on,
        )):
            raise ValueError("Approval needs de-identification and terms, ethics, and access review references.")
        return self


class EEGRecording(ResearchSchema):
    schema_version: Literal["1"] = "1"
    recording_id: Text
    subject_id: Text
    # Curated global pseudonym: all aliases/sessions/datasets for one person share it.
    patient_group_id: Text
    patient_identity_review_ref: Text
    age: AgeAtRecording
    montage: ChannelMontage
    sampling_rate_hz: PositiveNumber
    device: RecordingDevice
    preprocessing_id: Text
    provenance: DatasetProvenance
    duration_seconds: PositiveNumber
    annotation_coverage: Literal["complete", "partial", "unknown"]
    annotation_protocol_ref: Text
    annotations: tuple[SeizureAnnotation, ...]
    quality: RecordingQuality
    permissions: ResearchPermissions

    @model_validator(mode="after")
    def check_timeline(self) -> Self:
        events = self.annotations
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("Duplicate annotation event IDs.")
        if not nonoverlapping(events):
            raise ValueError("Adjudicate overlapping annotations into nonoverlapping reference events.")
        intervals = (*events, *self.quality.excluded_intervals)
        if any(interval.end_seconds > self.duration_seconds for interval in intervals):
            raise ValueError("Annotations and exclusions must be inside the recording.")
        return self

    @property
    def usable_seconds(self) -> float:
        return self.duration_seconds - sum(
            gap.end_seconds - gap.start_seconds for gap in self.quality.excluded_intervals
        )


def require_research_ready(recording: EEGRecording) -> None:
    """Common offline analysis gate; does not constitute model eligibility."""
    if recording.permissions.status != "approved":
        raise ValueError("Research permissions are not approved.")
    if not recording.device.clinical_eeg:
        raise ValueError("A clinical EEG acquisition device is required.")
    if recording.quality.status != "acceptable" or recording.usable_seconds <= 0:
        raise ValueError("Recording quality must be reviewed with positive usable duration.")
    if recording.annotation_coverage != "complete":
        raise ValueError("Complete clinical annotation coverage is required; missing labels are not negatives.")
    if any(overlaps(event, gap) for event in recording.annotations
           for gap in recording.quality.excluded_intervals):
        raise ValueError("Reference events intersect exclusions; adjudicate the evaluation domain first.")
