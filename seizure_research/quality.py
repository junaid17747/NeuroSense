"""Configurable, conservative signal checks for offline research windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from pydantic import Field, model_validator

from .schemas import PositiveNumber, ResearchSchema, Text, TimeInterval
from .waveform import ChannelSamples

QUALITY_VERSION = "signal-quality-v1"
MAX_QUALITY_WINDOWS = 100_000


class QualityThresholds(ResearchSchema):
    window_seconds: PositiveNumber = 5.0
    step_seconds: PositiveNumber = 5.0
    flatline_std_uv: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    extreme_amplitude_uv: PositiveNumber = 1000.0
    max_jump_uv: PositiveNumber = 1000.0
    max_saturation_fraction: float = Field(default=0.01, ge=0, le=1, allow_inf_nan=False)
    min_usable_fraction: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    max_rate_relative_error: float = Field(default=1e-6, ge=0, le=0.1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_step(self):
        if self.step_seconds > self.window_seconds:
            raise ValueError("Quality step cannot exceed quality window length.")
        return self


class ChannelQuality(ResearchSchema):
    source_index: int
    label: Text
    status: Literal["good", "limited", "unusable"]
    reasons: tuple[Text, ...] = ()
    nonfinite_samples: int = 0
    flatline_windows: int = 0
    extreme_samples: int = 0
    saturated_samples: int = 0
    corrupt_samples: int = 0
    discontinuities: int = 0


class QualityWindow(ResearchSchema):
    interval: TimeInterval
    status: Literal["good", "excluded"]
    reasons: tuple[Text, ...] = ()


class SignalQualityReport(ResearchSchema):
    version: Literal["signal-quality-v1"] = QUALITY_VERSION
    status: Literal["acceptable", "limited", "unusable"]
    expected_channels: tuple[Text, ...]
    missing_channels: tuple[Text, ...]
    duplicate_labels: tuple[Text, ...]
    unexpected_channels: tuple[Text, ...]
    channel_reports: tuple[ChannelQuality, ...]
    windows: tuple[QualityWindow, ...]
    excluded_intervals: tuple[TimeInterval, ...]
    usable_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    reasons: tuple[Text, ...]


@dataclass(frozen=True)
class QualityInput:
    label: str
    source_index: int
    samples_uv: np.ndarray
    timestamps_seconds: np.ndarray
    digital_samples: np.ndarray | None = None
    digital_min: int | None = None
    digital_max: int | None = None
    sampling_rate_hz: float | None = None


def merge_intervals(intervals: Sequence[TimeInterval]) -> tuple[TimeInterval, ...]:
    """Union overlapping or touching exclusions into sorted half-open ranges."""
    ordered = sorted(intervals, key=lambda item: (item.start_seconds, item.end_seconds))
    if not ordered:
        return ()
    merged: list[list[float]] = [[ordered[0].start_seconds, ordered[0].end_seconds]]
    for interval in ordered[1:]:
        if interval.start_seconds <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval.end_seconds)
        else:
            merged.append([interval.start_seconds, interval.end_seconds])
    return tuple(TimeInterval(start_seconds=start, end_seconds=end) for start, end in merged)


def assess_signal_quality(
    channels: Sequence[QualityInput | ChannelSamples],
    *,
    expected_channels: Sequence[str],
    start_seconds: float,
    end_seconds: float,
    expected_sampling_rate_hz: float | None = None,
    thresholds: QualityThresholds | None = None,
) -> SignalQualityReport:
    """Assess configured channel windows; a failed window is excluded, not negative."""
    thresholds = QualityThresholds() if thresholds is None else QualityThresholds.model_validate(thresholds)
    expected = tuple(expected_channels)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected channels must be nonempty and unique.")
    if expected_sampling_rate_hz is not None and (
            not np.isfinite(expected_sampling_rate_hz) or expected_sampling_rate_hz <= 0):
        raise ValueError("Expected sampling rate must be finite and positive.")
    labels = tuple(item.metadata.label if isinstance(item, ChannelSamples) else item.label for item in channels)
    missing = tuple(name for name in expected if name not in labels)
    duplicates = tuple(sorted({name for name in labels if labels.count(name) > 1}))
    unexpected = tuple(name for name in labels if name not in expected)
    length = end_seconds - start_seconds
    if not np.isfinite([start_seconds, end_seconds]).all() or start_seconds < 0 or length <= 0:
        raise ValueError("Quality interval must be finite and positive.")

    window_specs = []
    start = start_seconds
    if length / thresholds.step_seconds > MAX_QUALITY_WINDOWS:
        raise ValueError("quality_window_limit_exceeded")
    while start < end_seconds - 1e-9:
        window_specs.append((start, min(end_seconds, start + thresholds.window_seconds)))
        if window_specs[-1][1] == end_seconds:
            break
        start += thresholds.step_seconds
    windows = []
    channels_by_label: dict[str, list[QualityInput | ChannelSamples]] = {}
    for item, label in zip(channels, labels):
        channels_by_label.setdefault(label, []).append(item)

    per_channel = []
    channel_window_reasons: dict[str, list[tuple[str, ...]]] = {name: [] for name in expected}
    for label in expected:
        candidates = channels_by_label.get(label, [])
        if len(candidates) != 1:
            per_channel.append(ChannelQuality(source_index=-1, label=label, status="unusable",
                                              reasons=("missing_channel" if not candidates else "duplicate_channel_label",)))
            for _ in window_specs:
                channel_window_reasons[label].append(("missing_channel" if not candidates else "duplicate_channel_label",))
            continue
        item = candidates[0]
        metadata = item.metadata if isinstance(item, ChannelSamples) else None
        values = np.asarray(item.samples_uv, dtype=np.float64)
        times = np.asarray(item.timestamps_seconds, dtype=np.float64)
        source_index = metadata.index if metadata else item.source_index
        fs = metadata.sampling_rate_hz if metadata else item.sampling_rate_hz
        digital = np.asarray(item.digital_samples) if isinstance(item, ChannelSamples) else item.digital_samples
        digital_min = metadata.digital_min if metadata else item.digital_min
        digital_max = metadata.digital_max if metadata else item.digital_max
        if values.ndim != 1 or times.ndim != 1 or len(values) != len(times) or not len(values):
            per_channel.append(ChannelQuality(source_index=source_index, label=label, status="unusable",
                                              reasons=("sample_shape_invalid",)))
            for _ in window_specs:
                channel_window_reasons[label].append(("sample_shape_invalid",))
            continue
        reasons = []
        if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
            reasons.append("timestamps_invalid")
        nonfinite_count = int((~np.isfinite(values)).sum())
        if fs is None or not np.isfinite(fs) or fs <= 0:
            reasons.append("sampling_rate_invalid")
        if expected_sampling_rate_hz is not None and fs is not None:
            rel_error = abs(fs - expected_sampling_rate_hz) / expected_sampling_rate_hz
            if rel_error > thresholds.max_rate_relative_error:
                reasons.append("sampling_rate_mismatch")
        extreme_count = int((np.abs(values[np.isfinite(values)]) > thresholds.extreme_amplitude_uv).sum())
        saturated_count = 0
        corrupt_count = 0
        if digital is not None and digital_min is not None and digital_max is not None:
            digital = np.asarray(digital)
            if digital.shape != values.shape or not np.isfinite(digital).all() or digital_min >= digital_max:
                reasons.append("digital_metadata_mismatch")
                digital = None
        elif digital is not None:
            reasons.append("digital_metadata_mismatch")
            digital = None
        if digital is not None:
            saturated_count = int(((digital == digital_min) | (digital == digital_max)).sum())
            corrupt_count = int(((digital < digital_min) | (digital > digital_max)).sum())
        flatline_count = 0
        discontinuity_count = 0
        for left, right in window_specs:
            mask = (times >= left) & (times < right)
            sample = values[mask] if len(values) == len(times) else np.empty(0)
            local_times = times[mask] if len(values) == len(times) else np.empty(0)
            local_reasons = list(reasons)
            if not len(sample):
                local_reasons.append("window_has_no_samples")
            elif not np.isfinite(sample).all():
                local_reasons.append("nonfinite_samples")
            else:
                if fs is not None and np.isfinite(fs) and fs > 0:
                    intervals = np.diff(local_times)
                    if (np.any(np.abs(intervals * fs - 1) > thresholds.max_rate_relative_error + 1e-8)
                            or local_times[0] - left > (1 + 1e-6) / fs
                            or right - local_times[-1] > (1 + 1e-6) / fs):
                        local_reasons.append("sampling_gap")
                if right - left < thresholds.window_seconds - 1e-9:
                    local_reasons.append("incomplete_quality_window")
                if float(np.std(sample)) < thresholds.flatline_std_uv:
                    local_reasons.append("flatline")
                    flatline_count += 1
                if np.any(np.abs(sample) > thresholds.extreme_amplitude_uv):
                    local_reasons.append("extreme_amplitude")
                jumps = np.abs(np.diff(sample)) > thresholds.max_jump_uv
                if np.any(jumps):
                    local_reasons.append("excessive_discontinuity")
                    discontinuity_count += int(jumps.sum())
                if digital is not None and digital_min is not None and digital_max is not None and len(digital) == len(times):
                    local_digital = digital[mask]
                    fraction = float(((local_digital == digital_min) | (local_digital == digital_max)).mean()) if len(local_digital) else 0
                    if fraction > thresholds.max_saturation_fraction:
                        local_reasons.append("saturation")
                    if np.any((local_digital < digital_min) | (local_digital > digital_max)):
                        local_reasons.append("corrupt_digital_samples")
            channel_window_reasons[label].append(tuple(local_reasons))
        aggregate_reasons = list(reasons) + [reason for row in channel_window_reasons[label] for reason in row]
        if nonfinite_count:
            aggregate_reasons.append("nonfinite_samples")
        if flatline_count:
            aggregate_reasons.append("flatline_windows")
        if extreme_count:
            aggregate_reasons.append("extreme_amplitude")
        if saturated_count:
            aggregate_reasons.append("digital_rail_samples")
        if corrupt_count:
            aggregate_reasons.append("corrupt_digital_samples")
        if discontinuity_count:
            aggregate_reasons.append("excessive_discontinuity")
        bad_windows = sum(bool(row) for row in channel_window_reasons[label])
        status = "unusable" if reasons or bad_windows == len(window_specs) else (
            "limited" if aggregate_reasons else "good")
        per_channel.append(ChannelQuality(
            source_index=source_index, label=label, status=status,
            reasons=tuple(sorted(set(aggregate_reasons))), nonfinite_samples=nonfinite_count,
            flatline_windows=flatline_count, extreme_samples=extreme_count,
            saturated_samples=saturated_count, corrupt_samples=corrupt_count,
            discontinuities=discontinuity_count,
        ))

    for index, (left, right) in enumerate(window_specs):
        reasons = [f"{label}:{reason}" for label in expected for reason in channel_window_reasons[label][index]]
        windows.append(QualityWindow(
            interval=TimeInterval(start_seconds=left, end_seconds=right),
            status="excluded" if reasons else "good", reasons=tuple(reasons),
        ))
    exclusions = merge_intervals(tuple(window.interval for window in windows if window.status == "excluded"))
    global_reasons = []
    if missing:
        global_reasons.append("missing_channels")
    if duplicates:
        global_reasons.append("duplicate_channel_labels")
    if unexpected:
        global_reasons.append("unexpected_channels")
    if not window_specs:
        global_reasons.append("insufficient_quality_window_duration")
    if global_reasons:
        exclusions = (TimeInterval(start_seconds=start_seconds, end_seconds=end_seconds),)
        windows = [QualityWindow(interval=window.interval, status="excluded",
                                 reasons=tuple((*window.reasons, *global_reasons))) for window in windows]
    usable_fraction = max(0.0, 1.0 - sum(gap.end_seconds - gap.start_seconds for gap in exclusions) / length)
    if usable_fraction < thresholds.min_usable_fraction:
        global_reasons.append("insufficient_usable_signal")
    if global_reasons or usable_fraction == 0:
        status = "unusable"
    elif exclusions or any(report.status == "limited" for report in per_channel):
        status = "limited"
    else:
        status = "acceptable"
    return SignalQualityReport(
        status=status, expected_channels=expected, missing_channels=missing,
        duplicate_labels=duplicates, unexpected_channels=unexpected,
        channel_reports=tuple(per_channel), windows=tuple(windows),
        excluded_intervals=exclusions, usable_fraction=usable_fraction,
        reasons=tuple(global_reasons),
    )
