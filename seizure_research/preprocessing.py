"""Versioned, explicitly age-selected offline EEG preprocessing contracts.

Profiles here are engineering starting points only; none is clinically
validated. No transform is fitted on data, so there is no train/test-fitted
normalizer to leak across partitions in this checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import scipy
from pydantic import Field, model_validator
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, sos2zpk, tf2sos

from .ingestion import CohortManifest, validate_manifest
from .quality import QualityInput, QualityThresholds, SignalQualityReport, assess_signal_quality, merge_intervals
from .schemas import (
    AgeGroup, EEGRecording, PositiveNumber, ResearchSchema, SHA256, Text, TimeInterval,
    SeizureAnnotation, overlaps, require_research_ready,
)
from .splitting import SplitManifest, canonical_json, digest, validate_split_manifest
from .waveform import EDFReadResult, MAX_DECODED_SAMPLES, read_edf

PREPROCESSING_VERSION = "research-eeg-preprocessing-v1"
PROCESSING_CODE_VERSION = "neurosense-seizure-preprocess-1"
QUALITY_ASSESSMENT_VERSION = "signal-quality-v1"
ALIGNMENT_VERSION = "reference-window-alignment-v1"


class ChannelMapping(ResearchSchema):
    source_index: int = Field(ge=0)
    source_label: Text
    target_label: Text
    expected_source_rate_hz: PositiveNumber
    duplicate_label_review_ref: Text | None = None


class PreprocessingConfig(ResearchSchema):
    config_id: Text
    version: Literal["research-eeg-preprocessing-v1"] = PREPROCESSING_VERSION
    age_group: AgeGroup
    research_review_ref: Text
    channels: tuple[ChannelMapping, ...] = Field(min_length=1)
    target_sampling_rate_hz: PositiveNumber
    allow_resampling: bool = False
    highpass_hz: PositiveNumber
    lowpass_hz: PositiveNumber
    line_noise_hz: PositiveNumber | None = None
    filter_order: int = Field(default=4, ge=2, le=8)
    reference_mode: Literal["none", "average"] = "none"
    window_seconds: PositiveNumber = 5.0
    step_seconds: PositiveNumber = 5.0
    quality_thresholds: QualityThresholds = QualityThresholds()
    transient_tolerance: float = Field(default=1e-3, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_profile(self):
        labels = tuple(item.target_label for item in self.channels)
        indices = tuple(item.source_index for item in self.channels)
        if len(set(labels)) != len(labels) or len(set(indices)) != len(indices):
            raise ValueError("Channel mapping labels and source indices must be unique.")
        nyquist = self.target_sampling_rate_hz / 2
        if not 0 < self.highpass_hz < self.lowpass_hz < nyquist:
            raise ValueError("Filter cutoffs must satisfy 0 < high-pass < low-pass < target Nyquist.")
        if self.line_noise_hz is not None and not 0 < self.line_noise_hz < nyquist:
            raise ValueError("Line-noise frequency must be below target Nyquist.")
        if self.step_seconds > self.window_seconds:
            raise ValueError("Window step cannot exceed window duration.")
        if not self.research_review_ref.strip():
            raise ValueError("An explicit research population-profile review reference is required.")
        return self


# Provisional engineering values. They are deliberately population-specific
# profiles, not validated recommendations or universal clinical settings.
AGE_PROFILE_PARAMETERS: dict[AgeGroup, tuple[float, float]] = {
    AgeGroup.NEONATAL: (0.5, 30.0),
    AgeGroup.INFANT: (0.5, 35.0),
    AgeGroup.CHILD: (0.5, 40.0),
    AgeGroup.ADOLESCENT: (1.0, 40.0),
    AgeGroup.ADULT: (1.0, 40.0),
    AgeGroup.OLDER_ADULT: (0.5, 35.0),
}


def make_age_config(
    age_group: AgeGroup,
    channels: Sequence[ChannelMapping],
    *,
    target_sampling_rate_hz: float,
    research_review_ref: str,
    line_noise_hz: float | None = 50.0,
    allow_resampling: bool = False,
    reference_mode: Literal["none", "average"] = "none",
    window_seconds: float = 5.0,
    step_seconds: float = 5.0,
) -> PreprocessingConfig:
    """Build one explicit population config; caller supplies reviewed channels."""
    group = AgeGroup(age_group)
    highpass, lowpass = AGE_PROFILE_PARAMETERS[group]
    return PreprocessingConfig(
        config_id=f"{group.value}-research-v1", age_group=group,
        research_review_ref=research_review_ref, channels=tuple(channels),
        target_sampling_rate_hz=target_sampling_rate_hz, allow_resampling=allow_resampling,
        highpass_hz=highpass, lowpass_hz=lowpass, line_noise_hz=line_noise_hz,
        reference_mode=reference_mode, window_seconds=window_seconds, step_seconds=step_seconds,
    )


class WindowReference(ResearchSchema):
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: PositiveNumber
    label: Literal["positive", "negative", "ambiguous", "excluded"]
    event_ids: tuple[Text, ...] = ()
    reasons: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def interval_positive(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Window end must follow start.")
        return self


class ProcessingReport(ResearchSchema):
    version: Literal["research-eeg-preprocessing-v1"] = PREPROCESSING_VERSION
    processing_code_version: Text = PROCESSING_CODE_VERSION
    quality_version: Text = QUALITY_ASSESSMENT_VERSION
    alignment_version: Text = ALIGNMENT_VERSION
    dataset_id: Text
    dataset_version: Text
    cohort_manifest_sha256: SHA256
    recording_id: Text
    source_sha256: SHA256
    annotation_sha256: SHA256
    annotation_provenance_ref: Text | None = None
    config_sha256: SHA256
    configuration: PreprocessingConfig
    runtime_versions: dict[str, str]
    reference_annotations: tuple[SeizureAnnotation, ...]
    annotation_protocol_ref: Text
    annotation_review_ref: Text
    output_units: Literal["uV"] = "uV"
    usable_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["acceptable", "limited", "unusable"]
    reasons: tuple[Text, ...] = ()
    channel_mapping: tuple[ChannelMapping, ...]
    split_manifest_sha256: SHA256 | None = None
    processed_sha256: SHA256
    start_seconds: float
    end_seconds: float
    coverage_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    coverage_complete: bool
    channel_labels: tuple[Text, ...]
    excluded_intervals: tuple[TimeInterval, ...]
    quality: SignalQualityReport
    windows: tuple[WindowReference, ...]
    processing_timestamp_utc: str | None = None


@dataclass(frozen=True)
class ProcessedEEG:
    channel_labels: tuple[str, ...]
    timestamps_seconds: np.ndarray
    samples_uv: np.ndarray  # shape: (sample, channel); excluded areas are NaN.
    report: ProcessingReport


def _filter_sos(config: PreprocessingConfig) -> np.ndarray:
    sos = butter(config.filter_order,
                 (config.highpass_hz, config.lowpass_hz), btype="bandpass",
                 fs=config.target_sampling_rate_hz, output="sos")
    if config.line_noise_hz is not None:
        b, a = iirnotch(config.line_noise_hz, Q=30.0, fs=config.target_sampling_rate_hz)
        sos = np.vstack((sos, tf2sos(b, a)))
    if not np.isfinite(sos).all() or np.max(np.abs(sos2zpk(sos)[1])) >= 1:
        raise ValueError("filter_is_not_stable")
    return sos


def _resample_channel(times: np.ndarray, values: np.ndarray, source_rate: float,
                      target_rate: float, grid: np.ndarray) -> np.ndarray:
    if len(values) != len(times) or len(values) < 2:
        raise ValueError("sample_count_mismatch")
    ratio = Fraction(target_rate / source_rate).limit_denominator(1000)
    if max(ratio.numerator, ratio.denominator) > 1000:
        raise ValueError("resampling_ratio_limit_exceeded")
    if abs(source_rate * ratio.numerator / ratio.denominator - target_rate) > target_rate * 1e-8:
        raise ValueError("resampling_ratio_not_representable")
    converted = (values if ratio.numerator == ratio.denominator == 1 else
                 resample_poly(values, ratio.numerator, ratio.denominator))
    converted_times = times[0] + np.arange(len(converted), dtype=np.float64) / target_rate
    return np.interp(grid, converted_times, converted, left=np.nan, right=np.nan)


def _window_starts(start: float, end: float, window: float, step: float) -> tuple[tuple[float, float], ...]:
    if (end - start) / step > 100_000:
        raise ValueError("processing_window_limit_exceeded")
    result = []
    cursor = start
    while cursor + window <= end:
        result.append((cursor, cursor + window))
        cursor += step
    return tuple(result)


def align_reference_windows(
    recording: EEGRecording,
    intervals: Sequence[tuple[float, float]],
    *,
    excluded_intervals: Sequence[TimeInterval] = (),
    annotation_coverage: Literal["complete", "partial", "unknown"] | None = None,
) -> tuple[WindowReference, ...]:
    """Align half-open seizure intervals; unknown unlabeled time is never negative."""
    recording = EEGRecording.model_validate(recording)
    coverage = recording.annotation_coverage if annotation_coverage is None else annotation_coverage
    if coverage not in ("complete", "partial", "unknown"):
        raise ValueError("annotation_coverage_invalid")
    if coverage == "complete" and recording.annotation_coverage != "complete":
        raise ValueError("annotation_coverage_cannot_be_upgraded")
    refs = tuple(recording.annotations)
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > recording.duration_seconds:
            raise ValueError("window_interval_invalid")
    exclusions = tuple(TimeInterval.model_validate(item) for item in excluded_intervals) + recording.quality.excluded_intervals
    if any(item.end_seconds > recording.duration_seconds for item in exclusions):
        raise ValueError("exclusion_outside_recording")
    windows = []
    for start, end in intervals:
        window = TimeInterval(start_seconds=start, end_seconds=end)
        if any(overlaps(window, excluded) for excluded in exclusions):
            windows.append(WindowReference(start_seconds=start, end_seconds=end,
                                           label="excluded", reasons=("quality_exclusion_overlap",)))
            continue
        events = tuple(event.event_id for event in refs if overlaps(window, event))
        if events:
            windows.append(WindowReference(start_seconds=start, end_seconds=end,
                                           label="positive", event_ids=events))
        elif coverage == "complete":
            windows.append(WindowReference(start_seconds=start, end_seconds=end, label="negative"))
        else:
            windows.append(WindowReference(start_seconds=start, end_seconds=end,
                                           label="ambiguous", reasons=("annotation_coverage_incomplete",)))
    return tuple(windows)


def preprocess_recording(
    recording: EEGRecording,
    decoded: EDFReadResult,
    config: PreprocessingConfig,
    *,
    approved_manifest: CohortManifest,
    thresholds: QualityThresholds | None = None,
    split_manifest: SplitManifest | None = None,
) -> ProcessedEEG:
    """Process trusted decoder output; use process_approved_recording for disk inputs.

    Native samples are assessed before any transform. Resampling and zero-phase
    filtering run independently inside usable intervals; no fitted state exists.
    """
    recording = EEGRecording.model_validate(recording)
    config = PreprocessingConfig.model_validate(config)
    if thresholds is not None:
        config = PreprocessingConfig.model_validate({**config.model_dump(),
                                                     "quality_thresholds": thresholds})
    approved_manifest = CohortManifest.model_validate(approved_manifest)
    require_research_ready(recording)
    plan = approved_manifest.review_plan
    if (plan.permissions.status != "approved" or not plan.dataset_review.verified
            or not plan.identities.review.verified
            or approved_manifest.plan_sha256 != digest(plan.model_dump(mode="json"))):
        raise ValueError("approved_cohort_manifest_review_gates_missing")
    entries = [item for item in approved_manifest.records if item.recording.recording_id == recording.recording_id]
    if len(entries) != 1 or entries[0].recording != recording:
        raise ValueError("recording_not_bound_to_approved_cohort_manifest")
    entry = entries[0]
    reviews = [item for item in plan.recordings if item.edf_path == entry.signal.relative_path]
    if len(reviews) != 1:
        raise ValueError("recording_review_missing")
    review = reviews[0]
    if (not review.metadata_review.verified or not review.annotation_review.verified
            or entry.recording_review_sha256 != digest(review.model_dump(mode="json"))
            or review.expected_signal_sha256 != entry.signal.sha256
            or review.expected_summary_sha256 != entry.annotations.sha256
            or recording.provenance.annotation_sha256 != entry.annotations.sha256
            or recording.permissions != plan.permissions
            or recording.age != review.age or recording.montage != review.montage
            or recording.device != review.device or recording.quality != review.quality
            or recording.subject_id != review.subject_id
            or recording.annotation_coverage != review.annotation_coverage
            or recording.annotation_protocol_ref != review.annotation_protocol_ref
            or recording.provenance.annotation_provenance_ref != review.annotation_provenance_ref
            or recording.patient_group_id != plan.identities.resolve(review.subject_id)):
        raise ValueError("recording_review_binding_mismatch")
    if (decoded.metadata.sha256 != recording.provenance.signal_sha256
            or decoded.metadata.sha256 != entry.signal.sha256):
        raise ValueError("decoded_signal_hash_does_not_match_approved_recording")
    if (decoded.metadata.duration_seconds != recording.duration_seconds
            or decoded.metadata.size_bytes != entry.signal.size_bytes
            or not 0 <= decoded.start_seconds < decoded.end_seconds <= recording.duration_seconds):
        raise ValueError("decoded_recording_metadata_mismatch")
    split_hash = None
    if split_manifest is not None:
        validate_split_manifest((item.recording for item in approved_manifest.records), split_manifest)
        split_hash = split_manifest.sha256
    if config.age_group != recording.age.group:
        raise ValueError("population_configuration_mismatch")
    if tuple(item.target_label for item in config.channels) != recording.montage.channels:
        raise ValueError("reviewed_montage_mapping_mismatch")
    kept = tuple(item for item in review.channels if item.action == "keep")
    if tuple((item.source_index, item.source_label, item.target_label) for item in config.channels) != tuple(
            (item.source_index, item.source_label, item.output_label) for item in kept):
        raise ValueError("channel_mapping_differs_from_approved_selection")
    decoded_by_index = {channel.metadata.index: channel for channel in decoded.channels}
    if (len(decoded_by_index) != len(decoded.channels)
            or set(decoded_by_index) != {item.source_index for item in config.channels}):
        raise ValueError("selected_channels_must_be_explicitly_mapped")
    source_labels = [channel.label for channel in decoded.metadata.channels]
    mapped = []
    for mapping in config.channels:
        channel = decoded_by_index[mapping.source_index]
        if (channel.metadata != decoded.metadata.channels[mapping.source_index]
                or channel.metadata.label != mapping.source_label
                or channel.metadata.units != recording.montage.units):
            raise ValueError("source_channel_mapping_mismatch")
        fs = channel.metadata.sampling_rate_hz
        if abs(fs - mapping.expected_source_rate_hz) > 1e-8:
            raise ValueError("source_sampling_rate_mismatch")
        if source_labels.count(mapping.source_label) > 1 and not mapping.duplicate_label_review_ref:
            raise ValueError("duplicate_source_label_needs_explicit_review")
        if not config.allow_resampling and abs(fs - config.target_sampling_rate_hz) > 1e-8:
            raise ValueError("resampling_not_authorized_by_config")
        if config.highpass_hz >= fs / 2:
            raise ValueError("source_nyquist_below_highpass")
        # EDF supplies a regular timebase. Reject damaged decoded arrays before
        # interpolating, which would otherwise silently bridge missing samples.
        count = len(channel.samples_uv)
        times = channel.timestamps_seconds
        if (channel.samples_uv.ndim != 1 or times.ndim != 1
                or channel.digital_samples.shape != (count,) or len(times) != count or count < 2
                or not np.isfinite(times).all()
                or not np.allclose(np.diff(times), 1 / fs, rtol=1e-6, atol=1e-10)
                or times[0] < decoded.start_seconds - 1e-9
                or times[0] - decoded.start_seconds >= (1 + 1e-6) / fs
                or times[-1] >= decoded.end_seconds
                or decoded.end_seconds - times[-1] > (1 + 1e-6) / fs):
            raise ValueError("decoded_sample_timebase_or_shape_invalid")
        mapped.append(channel)
    reviewed_rates = recording.channel_sampling_rates_hz or (recording.sampling_rate_hz,) * len(mapped)
    if tuple(item.metadata.sampling_rate_hz for item in mapped) != reviewed_rates:
        raise ValueError("recording_per_channel_sampling_metadata_mismatch")
    if config.reference_mode == "average" and (
            len(mapped) < 2 or recording.montage.kind not in ("referential", "average_reference")):
        raise ValueError("average_reference_requires_reviewed_referential_montage")

    target_rate = config.target_sampling_rate_hz
    latest_start = max(decoded.start_seconds, max(float(item.timestamps_seconds[0]) for item in mapped))
    output_end = decoded.end_seconds
    first_grid_index = math.ceil(latest_start * target_rate - 1e-10)
    final_grid_index = math.ceil(output_end * target_rate - 1e-10)
    count = final_grid_index - first_grid_index
    if count * len(mapped) > MAX_DECODED_SAMPLES:
        raise ValueError("processed_sample_limit_exceeded")
    sos = _filter_sos(config)
    padlen = 3 * (2 * len(sos) + 1)
    pole_radius = float(np.max(np.abs(sos2zpk(sos)[1])))
    # Pole-decay guard is a documented engineering heuristic, not a guarantee
    # of clinical transient removal; it is longer than padding for low cutoffs.
    edge = max(padlen, math.ceil(math.log(config.transient_tolerance) / math.log(pole_radius)))
    if count <= 2 * edge:
        raise ValueError("filter_segment_too_short_after_edge_exclusion")
    timestamps = np.arange(first_grid_index, final_grid_index, dtype=np.float64) / target_rate
    labels = tuple(item.target_label for item in config.channels)
    quality_inputs = tuple(QualityInput(
        label=mapping.target_label, source_index=mapping.source_index,
        samples_uv=channel.samples_uv, timestamps_seconds=channel.timestamps_seconds,
        digital_samples=channel.digital_samples, digital_min=channel.metadata.digital_min,
        digital_max=channel.metadata.digital_max, sampling_rate_hz=channel.metadata.sampling_rate_hz,
    ) for mapping, channel in zip(config.channels, mapped))
    quality = assess_signal_quality(quality_inputs, expected_channels=labels,
                                    start_seconds=float(timestamps[0]), end_seconds=output_end,
                                    thresholds=config.quality_thresholds)
    total_interval = TimeInterval(start_seconds=float(timestamps[0]), end_seconds=output_end)
    exclusions = merge_intervals((*recording.quality.excluded_intervals, *quality.excluded_intervals))
    if quality.status == "unusable":
        exclusions = (total_interval,)
    clipped_exclusions = merge_intervals(tuple(TimeInterval(
        start_seconds=max(total_interval.start_seconds, gap.start_seconds),
        end_seconds=min(total_interval.end_seconds, gap.end_seconds),
    ) for gap in exclusions if gap.end_seconds > total_interval.start_seconds
       and gap.start_seconds < total_interval.end_seconds))
    valid_intervals = []
    cursor = total_interval.start_seconds
    for gap in clipped_exclusions:
        if gap.start_seconds > cursor:
            valid_intervals.append((cursor, gap.start_seconds))
        cursor = gap.end_seconds
    if cursor < output_end:
        valid_intervals.append((cursor, output_end))
    output = np.full((count, len(mapped)), np.nan, dtype=np.float64)
    edge_exclusions = []
    processing_reasons = []
    for segment_start, segment_end in valid_intervals:
        first, stop = np.searchsorted(timestamps, (segment_start, segment_end), side="left")
        columns = []
        resampling_guard = 0
        for channel in mapped:
            fs = channel.metadata.sampling_rate_hz
            left, right = np.searchsorted(channel.timestamps_seconds, (segment_start, segment_end), side="left")
            if fs != target_rate:
                resampling_guard = max(resampling_guard, math.ceil(11 * target_rate / min(fs, target_rate)))
            if right - left < 2:
                columns = []
                break
            columns.append(_resample_channel(channel.timestamps_seconds[left:right],
                                              channel.samples_uv[left:right], fs, target_rate,
                                              timestamps[first:stop]))
        segment_edge = max(edge, resampling_guard)
        if not columns or stop - first <= 2 * segment_edge:
            edge_exclusions.append(TimeInterval(start_seconds=segment_start, end_seconds=segment_end))
            processing_reasons.append("filter_segment_too_short_after_edge_exclusion")
            continue
        segment = np.column_stack(columns)
        # Interpolation may leave an unsupported end sample at a rate transition.
        # Remove that edge before filtering; never fill or interpolate a gap.
        finite = np.isfinite(segment).all(axis=1)
        positions = np.flatnonzero(finite)
        if not len(positions):
            raise ValueError("nonfinite_samples_in_usable_filter_segment")
        begin, finish = int(positions[0]), int(positions[-1]) + 1
        if not finite[begin:finish].all():
            raise ValueError("nonfinite_samples_in_usable_filter_segment")
        segment = segment[begin:finish]
        first += begin
        stop = first + len(segment)
        if len(segment) <= 2 * segment_edge:
            edge_exclusions.append(TimeInterval(start_seconds=segment_start, end_seconds=segment_end))
            processing_reasons.append("filter_segment_too_short_after_edge_exclusion")
            continue
        if config.reference_mode == "average":
            segment = segment - segment.mean(axis=1, keepdims=True)
        filtered = sosfiltfilt(sos, segment, axis=0, padlen=padlen)
        if not np.isfinite(filtered).all():
            raise ValueError("nonfinite_filter_output")
        output[first + segment_edge:stop - segment_edge] = filtered[segment_edge:-segment_edge]
        edge_exclusions.extend((
            TimeInterval(start_seconds=segment_start, end_seconds=float(timestamps[first + segment_edge])),
            TimeInterval(start_seconds=float(timestamps[stop - segment_edge]), end_seconds=segment_end),
        ))
    effective_exclusions = merge_intervals((*clipped_exclusions, *edge_exclusions))
    usable_fraction = max(0.0, 1 - sum(item.end_seconds - item.start_seconds for item in effective_exclusions)
                          / (output_end - float(timestamps[0])))
    if usable_fraction < config.quality_thresholds.min_usable_fraction:
        processing_reasons.append("insufficient_usable_processed_signal")
        output[:] = np.nan
        effective_exclusions = (total_interval,)
        usable_fraction = 0.0
    aligned = align_reference_windows(
        recording, _window_starts(float(timestamps[0]), output_end, config.window_seconds, config.step_seconds),
        excluded_intervals=effective_exclusions)
    report = ProcessingReport(
        dataset_id=recording.provenance.dataset_id, dataset_version=recording.provenance.version,
        cohort_manifest_sha256=approved_manifest.sha256, recording_id=recording.recording_id,
        source_sha256=recording.provenance.signal_sha256,
        annotation_sha256=recording.provenance.annotation_sha256,
        annotation_provenance_ref=recording.provenance.annotation_provenance_ref,
        annotation_protocol_ref=recording.annotation_protocol_ref,
        annotation_review_ref=review.annotation_review.evidence_ref,
        reference_annotations=recording.annotations,
        configuration=config, runtime_versions={"python": platform.python_version(),
                                                "numpy": np.__version__, "scipy": scipy.__version__},
        config_sha256=digest(config.model_dump(mode="json")), channel_mapping=config.channels,
        split_manifest_sha256=split_hash,
        processed_sha256=hashlib.sha256(timestamps.astype("<f8").tobytes() + output.astype("<f8").tobytes()).hexdigest(),
        start_seconds=float(timestamps[0]), end_seconds=output_end,
        coverage_fraction=min(1.0, (output_end - float(timestamps[0])) / recording.duration_seconds),
        coverage_complete=(float(timestamps[0]) <= 1e-9 and abs(output_end - recording.duration_seconds) <= 1e-9),
        channel_labels=labels, excluded_intervals=effective_exclusions, quality=quality, windows=aligned,
        usable_fraction=usable_fraction, status="unusable" if not usable_fraction else "limited" if effective_exclusions else "acceptable",
        reasons=tuple(sorted(set((*quality.reasons, *processing_reasons)))),
    )
    timestamps.setflags(write=False)
    output.setflags(write=False)
    return ProcessedEEG(labels, timestamps, output, report)


def process_approved_recording(
    root: Path | str, manifest: CohortManifest, recording_id: str, config: PreprocessingConfig,
    *, start_seconds: float, end_seconds: float,
    split_manifest: SplitManifest | None = None, max_samples: int = 2_000_000,
) -> ProcessedEEG:
    """Revalidate on-disk cohort/annotations, then decode and process one bounded range.

    No download, artifact deserialization beyond typed JSON contracts, writes,
    training, prediction, or backend exposure occurs. A missing split explicitly
    means exploratory processing, not a training/evaluation-ready artifact.
    """
    manifest = CohortManifest.model_validate(manifest)
    config = PreprocessingConfig.model_validate(config)
    validate_manifest(root, manifest)
    entries = [item for item in manifest.records if item.recording.recording_id == recording_id]
    if len(entries) != 1:
        raise ValueError("recording_not_bound_to_approved_cohort_manifest")
    entry = entries[0]
    decoded = read_edf(root, entry.signal.relative_path,
                       channel_indices=tuple(item.source_index for item in config.channels),
                       start_seconds=start_seconds, end_seconds=end_seconds, max_samples=max_samples)
    return preprocess_recording(entry.recording, decoded, config, approved_manifest=manifest,
                                split_manifest=split_manifest)
