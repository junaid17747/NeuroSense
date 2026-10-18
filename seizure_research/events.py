"""Window-score to candidate-event conversion and retrospective event metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from .schemas import EEGRecording, TimeInterval, overlaps, require_research_ready


@dataclass(frozen=True)
class ScoredWindow:
    recording_id: str
    patient_group_id: str
    start_seconds: float
    end_seconds: float
    probability: float | None
    split: str


@dataclass(frozen=True)
class CandidateEvent:
    event_id: str
    recording_id: str
    patient_group_id: str
    start_seconds: float
    end_seconds: float
    trigger_seconds: float
    constituent_ids: tuple[str, ...]
    coverage_segment: int
    censored: bool = False


def windows_to_events(windows: Sequence[ScoredWindow], *, threshold_on: float,
                      median_width: int = 3, open_count: int = 2, close_count: int = 3,
                      hysteresis: float = 0.10, merge_gap_seconds: float = 5.0) -> tuple[CandidateEvent, ...]:
    """Convert sequential scores to events; missing windows reset all state.

    Trigger times are the right edge of the second qualifying window. Times are
    EEG-recording time and therefore retrospective when upstream filtering is
    zero-phase; this function makes no real-time latency claim.
    """
    if (not math.isfinite(threshold_on) or not 0 <= threshold_on <= 1
            or median_width < 1 or open_count < 1 or close_count < 1
            or not 0 <= hysteresis < 1 or merge_gap_seconds < 0):
        raise ValueError("event_parameters_invalid")
    ordered = tuple(sorted(windows,
                           key=lambda item: (item.recording_id, item.start_seconds, item.end_seconds)))
    if any(not (0 <= item.start_seconds < item.end_seconds)
           or not math.isfinite(item.start_seconds + item.end_seconds)
           or (item.probability is not None and
               (not math.isfinite(item.probability) or not 0 <= item.probability <= 1))
           for item in ordered):
        raise ValueError("scored_window_invalid")
    threshold_off = max(0.01, threshold_on - hysteresis)
    raw_events: list[CandidateEvent] = []
    by_record: dict[str, list[ScoredWindow]] = defaultdict(list)
    for item in ordered:
        by_record[item.recording_id].append(item)
    for recording_id, group in by_record.items():
        if len({item.patient_group_id for item in group}) != 1:
            raise ValueError("recording_patient_identity_conflict")
        queue: list[ScoredWindow] = []
        active = False
        above = below = 0
        candidate_start = trigger = last_active_end = 0.0
        event_number = 0
        previous: ScoredWindow | None = None
        coverage_segment = 0

        def close(censored: bool = False):
            nonlocal active, below, event_number
            if active:
                event_number += 1
                raw_events.append(CandidateEvent(
                    event_id=f"{recording_id}:candidate:{event_number}", recording_id=recording_id,
                    patient_group_id=group[0].patient_group_id, start_seconds=candidate_start,
                    end_seconds=max(candidate_start, last_active_end), trigger_seconds=trigger,
                    constituent_ids=(f"{recording_id}:episode:{event_number}",),
                    coverage_segment=coverage_segment, censored=censored))
            active = False
            below = 0

        for item in group:
            if previous is not None and (item.start_seconds > previous.start_seconds + 1.0 + 1e-9
                                         or item.start_seconds <= previous.start_seconds):
                close(censored=True)
                queue.clear()
                above = 0
                coverage_segment += 1
            if item.probability is None:
                close(censored=True)
                queue.clear()
                above = 0
                previous = item
                coverage_segment += 1
                continue
            queue.append(item)
            queue = queue[-median_width:]
            if len(queue) < median_width:
                previous = item
                continue
            score = median([float(value.probability) for value in queue])
            if not active:
                above = above + 1 if score >= threshold_on else 0
                if above >= open_count:
                    active = True
                    candidate_start = queue[-open_count].start_seconds
                    trigger = item.end_seconds
                    last_active_end = item.end_seconds
                    below = 0
            elif score < threshold_off:
                below += 1
                if below >= close_count:
                    # End support at the first window in the closing run.
                    last_active_end = queue[-close_count].start_seconds
                    close()
            else:
                below = 0
                last_active_end = item.end_seconds
            previous = item
        close(censored=True)

    # Merge only intervals separated by <= limit and represented by contiguous
    # scored coverage. Missing-score gaps were already state resets above.
    merged: list[CandidateEvent] = []
    for item in sorted(raw_events, key=lambda value: (value.recording_id, value.start_seconds, value.event_id)):
        if merged:
            prior = merged[-1]
            same_record = prior.recording_id == item.recording_id
            gap = item.start_seconds - prior.end_seconds
            if (same_record and prior.coverage_segment == item.coverage_segment
                    and 0 <= gap <= merge_gap_seconds):
                merged[-1] = CandidateEvent(
                    event_id=prior.event_id, recording_id=prior.recording_id,
                    patient_group_id=prior.patient_group_id,
                    start_seconds=prior.start_seconds, end_seconds=item.end_seconds,
                    trigger_seconds=prior.trigger_seconds,
                    constituent_ids=prior.constituent_ids + item.constituent_ids,
                    coverage_segment=prior.coverage_segment,
                    censored=prior.censored or item.censored)
                continue
        merged.append(item)
    return tuple(merged)


@dataclass(frozen=True)
class PatientEventMetrics:
    patient_group_id: str
    reference_events: int
    detected_events: int
    missed_event_ids: tuple[str, ...]
    false_events: int
    evaluable_hours: float
    false_alarms_per_hour: float | None
    sensitivity: float | None
    trigger_latencies_seconds: tuple[float, ...]
    excluded_reference_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class EventEvaluation:
    reference_events: int
    detected_events: int
    missed_events: int
    false_events: int
    evaluable_hours: float
    false_alarms_per_hour: float | None
    event_sensitivity: float | None
    macro_patient_sensitivity: float | None
    trigger_latencies_seconds: tuple[float, ...]
    median_detection_latency_seconds: float | None
    per_patient: Mapping[str, PatientEventMetrics]


def _evaluable_windows(windows: Sequence[ScoredWindow], warmup_windows: int = 2) -> tuple[ScoredWindow, ...]:
    ordered = sorted((item for item in windows if item.probability is not None),
                     key=lambda item: (item.recording_id, item.start_seconds))
    result: list[ScoredWindow] = []
    run: list[ScoredWindow] = []
    for item in ordered:
        if run and (item.recording_id != run[-1].recording_id
                    or item.start_seconds - run[-1].start_seconds > 1.0 + 1e-9):
            result.extend(run[warmup_windows:])
            run = []
        run.append(item)
    result.extend(run[warmup_windows:])
    return tuple(result)


def _union_seconds(windows: Sequence[ScoredWindow],
                   exclude_intervals: Sequence[TimeInterval] = ()) -> float:
    intervals = sorted((item.start_seconds, item.end_seconds) for item in _evaluable_windows(windows))
    if not intervals:
        return 0.0
    merged: list[list[float]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    excluded = sorted((item.start_seconds, item.end_seconds) for item in exclude_intervals)
    total = 0.0
    for left, right in merged:
        removed = sum(max(0.0, min(right, end) - max(left, start))
                      for start, end in excluded if start < right and end > left)
        total += max(0.0, right - left - removed)
    return total


def evaluate_events(recordings: Iterable[EEGRecording], windows: Sequence[ScoredWindow],
                    candidates: Sequence[CandidateEvent]) -> EventEvaluation:
    """One-to-one event matching with full reference inventory and explicit abstention."""
    records = tuple(EEGRecording.model_validate(item) for item in recordings)
    by_id = {item.recording_id: item for item in records}
    if len(by_id) != len(records) or not records:
        raise ValueError("evaluation_recording_inventory_invalid")
    for rec in records:
        require_research_ready(rec)
    if {item.recording_id for item in windows} - set(by_id):
        raise ValueError("evaluation_window_recording_unknown")
    if {item.recording_id for item in candidates} - set(by_id):
        raise ValueError("candidate_recording_unknown")
    grouped_candidates: dict[str, list[CandidateEvent]] = defaultdict(list)
    candidate_ids: set[tuple[str, str]] = set()
    for item in candidates:
        rec = by_id[item.recording_id]
        if (item.patient_group_id != rec.patient_group_id
                or not all(math.isfinite(v) for v in (item.start_seconds, item.end_seconds, item.trigger_seconds))
                or not 0 <= item.start_seconds < item.end_seconds <= rec.duration_seconds
                or not 0 <= item.trigger_seconds <= rec.duration_seconds):
            raise ValueError("candidate_identity_or_interval_invalid")
        interval = TimeInterval(start_seconds=item.start_seconds, end_seconds=item.end_seconds)
        if any(overlaps(interval, gap) for gap in rec.quality.excluded_intervals):
            raise ValueError("candidate_overlaps_reviewed_exclusion")
        key = (item.recording_id, item.event_id)
        if key in candidate_ids:
            raise ValueError("duplicate_candidate_event_id")
        candidate_ids.add(key)
        grouped_candidates[item.recording_id].append(item)
    grouped_windows: dict[str, list[ScoredWindow]] = defaultdict(list)
    window_keys: set[tuple[str, float, float]] = set()
    for item in windows:
        rec = by_id[item.recording_id]
        if item.patient_group_id != rec.patient_group_id:
            raise ValueError("window_patient_identity_conflict")
        if (not all(math.isfinite(v) for v in (item.start_seconds, item.end_seconds))
                or not 0 <= item.start_seconds < item.end_seconds <= rec.duration_seconds
                or (item.probability is not None and
                    (not math.isfinite(item.probability) or not 0 <= item.probability <= 1))):
            raise ValueError("evaluation_window_invalid")
        interval = TimeInterval(start_seconds=item.start_seconds, end_seconds=item.end_seconds)
        if item.probability is not None and any(overlaps(interval, gap) for gap in rec.quality.excluded_intervals):
            raise ValueError("scored_window_overlaps_reviewed_exclusion")
        key = (item.recording_id, item.start_seconds, item.end_seconds)
        if key in window_keys:
            raise ValueError("duplicate_scored_window")
        window_keys.add(key)
        grouped_windows[item.recording_id].append(item)
    patient_records: dict[str, list[EEGRecording]] = defaultdict(list)
    for rec in records:
        patient_records[rec.patient_group_id].append(rec)
    result: dict[str, PatientEventMetrics] = {}
    for patient, patient_recs in sorted(patient_records.items()):
        ref_count = detected = false = 0
        missed: list[str] = []
        excluded: list[str] = []
        latencies: list[float] = []
        evaluable = 0.0
        for rec in patient_recs:
            local_windows = grouped_windows[rec.recording_id]
            refs = tuple(rec.annotations)
            # Exclude seizure time from the false-alarm/hour denominator.
            evaluable += _union_seconds(local_windows, refs)
            ref_count += len(refs)
            matched: set[str] = set()
            for candidate in sorted(grouped_candidates[rec.recording_id],
                                    key=lambda item: (item.trigger_seconds, item.event_id)):
                target = next((event for event in refs if event.event_id not in matched
                               and overlaps(TimeInterval(start_seconds=candidate.start_seconds,
                                                         end_seconds=candidate.end_seconds), event)
                               and event.start_seconds <= candidate.trigger_seconds < event.end_seconds), None)
                if target is None:
                    false += 1
                else:
                    matched.add(target.event_id)
                    latencies.append(candidate.trigger_seconds - target.start_seconds)
            detected += len(matched)
            for event in refs:
                if event.event_id in matched:
                    continue
                missed.append(event.event_id)
                interval = TimeInterval(start_seconds=event.start_seconds, end_seconds=event.end_seconds)
                if any(overlaps(interval, gap) for gap in rec.quality.excluded_intervals):
                    excluded.append(event.event_id)
        result[patient] = PatientEventMetrics(
            patient_group_id=patient, reference_events=ref_count, detected_events=detected,
            missed_event_ids=tuple(missed), false_events=false, evaluable_hours=evaluable / 3600,
            false_alarms_per_hour=false / (evaluable / 3600) if evaluable else None,
            sensitivity=detected / ref_count if ref_count else None,
            trigger_latencies_seconds=tuple(latencies), excluded_reference_event_ids=tuple(excluded))
    refs = sum(item.reference_events for item in result.values())
    found = sum(item.detected_events for item in result.values())
    false = sum(item.false_events for item in result.values())
    hours = sum(item.evaluable_hours for item in result.values())
    latencies = tuple(value for item in result.values() for value in item.trigger_latencies_seconds)
    patient_sens = [item.sensitivity for item in result.values() if item.sensitivity is not None]
    return EventEvaluation(
        reference_events=refs, detected_events=found, missed_events=refs - found,
        false_events=false, evaluable_hours=hours,
        false_alarms_per_hour=false / hours if hours else None,
        event_sensitivity=found / refs if refs else None,
        macro_patient_sensitivity=mean(patient_sens) if patient_sens else None,
        trigger_latencies_seconds=latencies,
        median_detection_latency_seconds=median(latencies) if latencies else None,
        per_patient=result)


def select_validation_threshold(recordings: Iterable[EEGRecording], windows: Sequence[ScoredWindow], *,
                                thresholds: Sequence[float] = tuple(i / 100 for i in range(10, 91, 5)),
                                max_false_events_per_hour: float = 1.0,
                                max_alert_occupancy: float = 0.10) -> tuple[float, dict[float, EventEvaluation]]:
    """Select only from explicitly supplied validation patients/windows."""
    records = tuple(recordings)
    patient_ids = {item.patient_group_id for item in records}
    window_patients = {item.patient_group_id for item in windows}
    if (not records or not window_patients <= patient_ids
            or any(item.split != "validation" for item in windows)):
        raise ValueError("validation_threshold_requires_validation_only_inputs")
    curve = {}
    feasible = []
    for threshold in thresholds:
        events = windows_to_events(windows, threshold_on=float(threshold))
        metrics = evaluate_events(records, windows, events)
        curve[float(threshold)] = metrics
        occupancy = sum(max(0.0, item.end_seconds - item.start_seconds) for item in events)
        total = sum(_union_seconds([item for item in windows if item.recording_id == rec.recording_id])
                    for rec in records)
        occupancy = occupancy / total if total else 0.0
        if (metrics.false_alarms_per_hour is not None
                and metrics.false_alarms_per_hour <= max_false_events_per_hour
                and occupancy <= max_alert_occupancy and metrics.event_sensitivity is not None):
            feasible.append((metrics.macro_patient_sensitivity or 0.0,
                             -(metrics.false_alarms_per_hour or 0.0), float(threshold)))
    if not feasible:
        raise ValueError("no_validation_threshold_meets_prespecified_operating_limits")
    return max(feasible)[2], curve
