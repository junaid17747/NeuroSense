"""Score externally supplied offline decision times; never generate predictions.

Protocol v1: one decision in [reference onset, reference offset) detects one
event. The first decision wins; extra decisions are false alarms. No tolerance,
merging, or refractory suppression is silently applied.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean, median
from typing import Annotated, Literal

from pydantic import Field

from .schemas import (
    AgeGroup, EEGRecording, ResearchSchema, Seconds, SHA256, Text,
)
from .splitting import (
    SplitManifest, checked_recordings, digest, validate_split_manifest,
)


class OfflineDetection(ResearchSchema):
    detection_id: Text
    # Time when the decision was available, including buffering/processing delay.
    decision_seconds: Seconds


class EvaluationProvenance(ResearchSchema):
    model_id: Text
    model_version: Text
    model_sha256: SHA256
    split_manifest_sha256: SHA256
    prediction_artifact_sha256: SHA256
    frozen_protocol_ref: Text
    training_patient_ids: Annotated[tuple[Text, ...], Field(min_length=1)]
    selection_patient_ids: Annotated[tuple[Text, ...], Field(min_length=1)]


class EventMetrics(ResearchSchema):
    patient_count: int
    recording_count: int
    reference_events: int
    detected_events: int
    missed_events: int
    false_alarms: int
    evaluated_hours: float
    excluded_hours: float
    event_sensitivity: float | None
    false_alarms_per_hour: float | None
    detection_latencies_seconds: tuple[float, ...]
    median_detection_latency_seconds: float | None


class EvaluationReport(ResearchSchema):
    protocol: Literal["decision-within-reference-v1"] = "decision-within-reference-v1"
    provenance: EvaluationProvenance
    detections_sha256: SHA256
    overall: EventMetrics
    per_patient: dict[str, EventMetrics]
    per_age_group: dict[AgeGroup, EventMetrics]
    macro_patient_event_sensitivity: float | None
    patients_with_reference_events: int


def _score_recording(recording: EEGRecording, detections: Sequence[OfflineDetection]) -> dict:
    detections = tuple(OfflineDetection.model_validate(item) for item in detections)
    if len({item.detection_id for item in detections}) != len(detections):
        raise ValueError("Duplicate detection IDs within a recording.")
    for item in detections:
        time = item.decision_seconds
        if time >= recording.duration_seconds:
            raise ValueError("Detection decision must lie inside the recording.")
        if any(gap.start_seconds <= time < gap.end_seconds for gap in recording.quality.excluded_intervals):
            raise ValueError("Detection is in an excluded interval; adjudicate the evaluation domain first.")

    matched = set()
    latencies = []
    false_alarms = 0
    for item in sorted(detections, key=lambda item: (item.decision_seconds, item.detection_id)):
        event = next((event for event in recording.annotations
                      if event.start_seconds <= item.decision_seconds < event.end_seconds), None)
        if event is None or event.event_id in matched:
            false_alarms += 1
        else:
            matched.add(event.event_id)
            latencies.append(item.decision_seconds - event.start_seconds)
    return {
        "patient": recording.patient_group_id, "age": recording.age.group,
        "events": len(recording.annotations), "detected": len(matched),
        "false_alarms": false_alarms, "latencies": latencies,
        "seconds": recording.usable_seconds,
        "excluded_seconds": recording.duration_seconds - recording.usable_seconds,
    }


def _aggregate(rows: Sequence[dict]) -> EventMetrics:
    events = sum(row["events"] for row in rows)
    detected = sum(row["detected"] for row in rows)
    false_alarms = sum(row["false_alarms"] for row in rows)
    hours = sum(row["seconds"] for row in rows) / 3600
    latencies = tuple(value for row in rows for value in row["latencies"])
    return EventMetrics(
        patient_count=len({row["patient"] for row in rows}), recording_count=len(rows),
        reference_events=events, detected_events=detected, missed_events=events - detected,
        false_alarms=false_alarms, evaluated_hours=hours,
        excluded_hours=sum(row["excluded_seconds"] for row in rows) / 3600,
        event_sensitivity=detected / events if events else None,
        false_alarms_per_hour=false_alarms / hours if hours else None,
        detection_latencies_seconds=latencies,
        median_detection_latency_seconds=median(latencies) if latencies else None,
    )


def evaluate_test_set(
    recordings: Iterable[EEGRecording], manifest: SplitManifest,
    detections: Mapping[str, Sequence[OfflineDetection]], provenance: EvaluationProvenance,
) -> EvaluationReport:
    """Require complete test coverage and an unchanged patient split.

    `recordings` is the full metadata cohort (no signal arrays). `detections`
    must explicitly contain every test recording, including empty sequences.
    Training/selection provenance is an auditable declaration, not proof of how
    an external artifact was trained. Eligibility is a separate gate.
    """
    records = checked_recordings(recordings)
    manifest = SplitManifest.model_validate(manifest)
    validate_split_manifest(records, manifest)
    provenance = EvaluationProvenance.model_validate(provenance)
    if provenance.split_manifest_sha256 != manifest.sha256:
        raise ValueError("Evaluation provenance names a different split manifest.")
    patients = {name: {item.patient_group_id for item in manifest.assignments if item.split == name}
                for name in ("train", "validation", "test")}
    for supplied, name in ((provenance.training_patient_ids, "train"),
                           (provenance.selection_patient_ids, "validation")):
        if len(supplied) != len(set(supplied)) or not set(supplied) <= patients[name]:
            raise ValueError("Training/selection patient provenance violates the split.")
    test_ids = {item.recording_id for item in manifest.assignments if item.split == "test"}
    if set(detections) != test_ids:
        raise ValueError("Provide exactly all test recordings, including explicit empty detection lists.")
    checked_detections = {
        key: tuple(sorted((OfflineDetection.model_validate(item) for item in value),
                          key=lambda item: (item.decision_seconds, item.detection_id)))
        for key, value in detections.items()
    }
    rows = [_score_recording(item, checked_detections[item.recording_id])
            for item in records if item.recording_id in test_ids]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient"]].append(row)
    per_patient = {patient: _aggregate(items) for patient, items in sorted(grouped.items())}
    sensitivities = [item.event_sensitivity for item in per_patient.values()
                     if item.event_sensitivity is not None]
    return EvaluationReport(
        provenance=provenance,
        detections_sha256=digest({key: [item.model_dump(mode="json") for item in value]
                                  for key, value in checked_detections.items()}),
        overall=_aggregate(rows), per_patient=per_patient,
        # Empty subgroups remain visible with zero counts and null metrics.
        per_age_group={age: _aggregate([row for row in rows if row["age"] == age]) for age in AgeGroup},
        macro_patient_event_sensitivity=mean(sensitivities) if sensitivities else None,
        patients_with_reference_events=len(sensitivities),
    )
