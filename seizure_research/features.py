"""Seizure-specific bounded feature extraction for offline research only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import kurtosis

from .preprocessing import (
    ChannelMapping, PreprocessingConfig, ProcessedEEG, _filter_sos, preprocess_recording,
)
from .ingestion import CohortManifest, validate_manifest
from .schemas import EEGRecording, TimeInterval, overlaps
from .splitting import SplitManifest, digest, validate_split_manifest
from .waveform import EDFStreamReader

FEATURE_VERSION = "seizure-eeg-features-v1"
CHANNEL_FEATURES = (
    "log_variance_uv2", "log_line_length_uv_s", "hjorth_mobility",
    "hjorth_complexity", "excess_kurtosis", "log_delta_power",
    "log_theta_power", "log_alpha_power", "log_beta_power", "log_gamma_power",
    "relative_delta_power", "relative_theta_power", "relative_alpha_power",
    "relative_beta_power", "relative_gamma_power", "spectral_entropy",
    "spectral_median_hz", "spectral_95pct_hz",
)
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0))
FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{channel}__{name}" for channel in range(18) for name in CHANNEL_FEATURES
) + tuple(f"{name}__{summary}" for name in CHANNEL_FEATURES
          for summary in ("median", "maximum", "iqr"))


def feature_names(channel_count: int) -> tuple[str, ...]:
    if type(channel_count) is not int or channel_count < 1:
        raise ValueError("channel_count_invalid")
    core = tuple(f"channel_{channel:02d}__{name}"
                 for channel in range(channel_count) for name in CHANNEL_FEATURES)
    return core + tuple(f"across_channels__{name}__{summary}"
                        for name in CHANNEL_FEATURES
                        for summary in ("median", "maximum", "iqr"))


def _band_power(frequency: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequency >= low) & (frequency <= high)
    if mask.sum() < 2:
        return 0.0
    return float(trapezoid(power[mask], frequency[mask]))


def channel_features(samples_uv: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Return the fixed 18-feature schema for one 5-second channel window."""
    x = np.asarray(samples_uv, dtype=np.float64)
    if (x.ndim != 1 or len(x) < int(5 * sampling_rate_hz) - 1
            or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 80
            or not np.isfinite(x).all()):
        raise ValueError("feature_window_shape_or_rate_invalid")
    if np.std(x) <= 1e-12:
        raise ValueError("feature_window_degenerate")
    variance = float(np.var(x, ddof=0))
    derivative = np.diff(x) * sampling_rate_hz
    var_dx = float(np.var(derivative, ddof=0)) if len(derivative) else 0.0
    mobility = math.sqrt(max(var_dx, 0.0) / variance)
    if len(derivative) > 1 and var_dx > 1e-24:
        ddx = np.diff(derivative) * sampling_rate_hz
        complexity = math.sqrt(max(float(np.var(ddx, ddof=0)), 0.0) / var_dx) / max(mobility, 1e-12)
    else:
        complexity = 0.0
    frequency, power = welch(x, fs=sampling_rate_hz, window="hann", nperseg=int(2 * sampling_rate_hz),
                             noverlap=int(sampling_rate_hz), detrend="constant", scaling="density",
                             average="mean")
    keep = (frequency >= 0.5) & (frequency <= 40.0)
    f, p = frequency[keep], power[keep]
    if len(f) < 2:
        raise ValueError("feature_spectral_support_missing")
    powers = np.asarray([_band_power(f, p, low, high) for low, high in BANDS], dtype=np.float64)
    total = max(float(trapezoid(p, f)), 1e-12)
    relative = powers / total
    probability = p / max(float(p.sum()), 1e-12)
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-15)))
                    / math.log(max(len(probability), 2)))
    cumulative = np.cumsum(p)
    cumulative /= max(float(cumulative[-1]), 1e-12)
    q50 = float(np.interp(0.50, cumulative, f))
    q95 = float(np.interp(0.95, cumulative, f))
    values = [
        math.log1p(variance), math.log1p(float(np.mean(np.abs(np.diff(x))) * sampling_rate_hz)),
        mobility, complexity, float(kurtosis(x, fisher=True, bias=False)),
        *(math.log1p(float(value)) for value in powers), *relative,
        entropy, q50, q95,
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(CHANNEL_FEATURES),) or not np.isfinite(result).all():
        raise ValueError("feature_values_nonfinite")
    return result


def window_features(samples_uv: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Compute channel features and robust cross-channel summaries."""
    samples = np.asarray(samples_uv, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 1 or not np.isfinite(samples).all():
        raise ValueError("feature_window_shape_or_values_invalid")
    per_channel = np.vstack([channel_features(samples[:, index], sampling_rate_hz)
                             for index in range(samples.shape[1])])
    summaries = np.column_stack((np.median(per_channel, axis=0), np.max(per_channel, axis=0),
                                 np.subtract(*np.percentile(per_channel, [75, 25], axis=0))))
    result = np.concatenate((per_channel.ravel(), summaries.ravel())).astype(np.float32)
    if result.shape != (len(feature_names(samples.shape[1]),),) or not np.isfinite(result).all():
        raise ValueError("feature_schema_or_values_invalid")
    return result


@dataclass(frozen=True)
class WindowExample:
    window_id: str
    recording_id: str
    patient_group_id: str
    start_seconds: float
    end_seconds: float
    label: Literal["positive", "negative", "ambiguous", "excluded"]
    event_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    features: np.ndarray | None


def _union_overlap_seconds(start: float, end: float, events: Sequence[TimeInterval]) -> float:
    ranges = sorted((max(start, item.start_seconds), min(end, item.end_seconds))
                    for item in events if overlaps(TimeInterval(start_seconds=start, end_seconds=end), item))
    if not ranges:
        return 0.0
    total = 0.0
    left, right = ranges[0]
    for a, b in ranges[1:]:
        if a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + right - left


def iter_processed_windows(processed: ProcessedEEG, recording: EEGRecording, *,
                           core_start_seconds: float, core_end_seconds: float,
                           window_seconds: float = 5.0, step_seconds: float = 1.0,
                           positive_occupancy: float = 0.5) -> Iterator[WindowExample]:
    """Yield each global-grid window owned by a chunk exactly once."""
    if (not math.isfinite(core_start_seconds) or not math.isfinite(core_end_seconds)
            or core_start_seconds < 0 or core_end_seconds <= core_start_seconds
            or window_seconds <= 0 or step_seconds <= 0 or step_seconds > window_seconds
            or not 0 < positive_occupancy <= 1):
        raise ValueError("feature_window_grid_invalid")
    fs = processed.report.configuration.target_sampling_rate_hz
    timestamps = processed.timestamps_seconds
    samples = processed.samples_uv
    if samples.ndim != 2 or samples.shape != (len(timestamps), len(processed.channel_labels)):
        raise ValueError("processed_waveform_shape_invalid")
    start_index = math.ceil((core_start_seconds - 1e-10) / step_seconds)
    stop_limit = core_end_seconds - window_seconds
    exclusions = processed.report.excluded_intervals
    events = recording.annotations
    for grid_index in range(start_index, math.floor((stop_limit + 1e-10) / step_seconds) + 1):
        start = grid_index * step_seconds
        end = start + window_seconds
        first = int(np.searchsorted(timestamps, start, side="left"))
        stop = int(np.searchsorted(timestamps, end, side="left"))
        if start < core_start_seconds - 1e-9 or end > core_end_seconds + 1e-9:
            continue
        event_ids = tuple(event.event_id for event in events
                          if overlaps(TimeInterval(start_seconds=start, end_seconds=end), event))
        reasons = []
        if any(overlaps(TimeInterval(start_seconds=start, end_seconds=end), item) for item in exclusions):
            reasons.append("quality_or_filter_exclusion")
        if (stop - first != int(round(window_seconds * fs))
                or timestamps[first] > start + 1e-9
                or timestamps[stop - 1] >= end - 1e-9
                or not np.isfinite(samples[first:stop]).all()):
            reasons.append("incomplete_or_nonfinite_samples")
        window_id = f"{recording.recording_id}:{grid_index}"
        if reasons:
            yield WindowExample(window_id, recording.recording_id, recording.patient_group_id,
                                start, end, "excluded", event_ids, tuple(reasons), None)
            continue
        occupancy = _union_overlap_seconds(start, end, events) / window_seconds
        if occupancy == 0:
            label = "negative"
        elif occupancy >= positive_occupancy:
            label = "positive"
        else:
            label = "ambiguous"
        try:
            features = window_features(samples[first:stop], fs)
        except ValueError:
            yield WindowExample(window_id, recording.recording_id, recording.patient_group_id,
                                start, end, "excluded", event_ids,
                                ("feature_invalid_or_degenerate",), None)
            continue
        yield WindowExample(window_id, recording.recording_id, recording.patient_group_id,
                            start, end, label, event_ids, (), features)


def write_feature_rows(rows: Iterable[WindowExample], output_path: Path | str, *,
                       feature_schema: Sequence[str], row_group_size: int = 512,
                       max_output_bytes: int = 5 * 1024**3,
                       space_check: Callable[[int], None] | None = None) -> dict:
    """Write row groups incrementally; do not materialize the full cohort matrix."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(output_path)
    if path.exists() or path.is_symlink():
        raise ValueError("feature_output_exists_refusing_overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(feature_schema)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_schema_invalid")
    schema = pa.schema([
        ("window_id", pa.string()), ("recording_id", pa.string()),
        ("patient_group_id", pa.string()), ("start_seconds", pa.float64()),
        ("end_seconds", pa.float64()), ("label", pa.string()),
        ("event_ids", pa.list_(pa.string())), ("reasons", pa.list_(pa.string())),
        *[(name, pa.float32()) for name in names],
    ], metadata={b"feature_version": FEATURE_VERSION.encode()})
    buffer = []
    accepted = excluded = ambiguous = 0
    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=False)
    try:
        for row in rows:
            ambiguous += int(row.label == "ambiguous")
            if row.features is None:
                excluded += int(row.label == "excluded")
                continue
            values = np.asarray(row.features, dtype=np.float32)
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError("feature_row_schema_or_value_invalid")
            payload = {"window_id": row.window_id, "recording_id": row.recording_id,
                       "patient_group_id": row.patient_group_id, "start_seconds": row.start_seconds,
                       "end_seconds": row.end_seconds, "label": row.label,
                       "event_ids": list(row.event_ids), "reasons": list(row.reasons)}
            payload.update(zip(names, values.tolist()))
            buffer.append(payload)
            accepted += 1
            if len(buffer) >= row_group_size:
                if space_check is not None:
                    space_check(row_group_size * (len(names) * 4 + 512))
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
                if path.stat().st_size > max_output_bytes:
                    raise ValueError("feature_output_size_limit_exceeded")
        if buffer:
            if space_check is not None:
                space_check(len(buffer) * (len(names) * 4 + 512))
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    except Exception:
        writer.close()
        path.unlink(missing_ok=True)
        raise
    writer.close()
    if path.stat().st_size > max_output_bytes:
        path.unlink(missing_ok=True)
        raise ValueError("feature_output_size_limit_exceeded")
    fingerprint = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            fingerprint.update(chunk)
    return {"path": str(path), "rows": accepted, "ambiguous_windows": ambiguous,
            "excluded_windows": excluded, "sha256": fingerprint.hexdigest(),
            "schema_sha256": digest(names)}


def _filter_edge_samples(config: PreprocessingConfig) -> int:
    from scipy.signal import sos2zpk

    sos = _filter_sos(config)
    padlen = 3 * (2 * len(sos) + 1)
    radius = float(np.max(np.abs(sos2zpk(sos)[1])))
    return max(padlen, math.ceil(math.log(config.transient_tolerance) / math.log(radius)))


def iter_approved_recording_windows(
    root: Path | str,
    manifest: CohortManifest,
    recording_id: str,
    config: PreprocessingConfig,
    split_manifest: SplitManifest,
    *,
    chunk_seconds: float = 300.0,
    halo_seconds: float = 60.0,
    max_samples: int = 2_000_000,
    space_check: Callable[[int], None] | None = None,
) -> Iterator[WindowExample]:
    """Yield globally aligned features from bounded chunks after one cohort audit."""
    if (not math.isfinite(chunk_seconds) or chunk_seconds <= 0
            or not math.isfinite(halo_seconds) or halo_seconds <= 0):
        raise ValueError("chunk_parameters_invalid")
    manifest = CohortManifest.model_validate(manifest)
    config = PreprocessingConfig.model_validate(config)
    split_manifest = SplitManifest.model_validate(split_manifest)
    if space_check is not None:
        space_check(0)
    validate_manifest(root, manifest)
    recordings = tuple(entry.recording for entry in manifest.records)
    validate_split_manifest(recordings, split_manifest)
    entries = [item for item in manifest.records if item.recording.recording_id == recording_id]
    if len(entries) != 1:
        raise ValueError("recording_not_in_approved_cohort")
    entry = entries[0]
    recording = entry.recording
    if recording.annotation_coverage != "complete":
        raise ValueError("complete_annotation_coverage_required")
    with EDFStreamReader(root, entry.signal.relative_path) as reader:
        yield from iter_verified_recording_windows(
            reader, manifest, entry, config, split_manifest,
            chunk_seconds=chunk_seconds, halo_seconds=halo_seconds,
            max_samples=max_samples, space_check=space_check)


def iter_verified_recording_windows(
    reader: EDFStreamReader, manifest: CohortManifest, entry,
    config: PreprocessingConfig, split_manifest: SplitManifest, *,
    chunk_seconds: float = 300.0, halo_seconds: float = 60.0,
    max_samples: int = 2_000_000,
    space_check: Callable[[int], None] | None = None,
) -> Iterator[WindowExample]:
    """Shared bounded decoder loop after the source-specific review/hash gate.

    The ordinary EDF entrypoint audits the full local cohort. The archive
    entrypoint verifies the archive and re-ingests the single extracted source
    before using this loop; it never needs to extract the entire cohort.
    """
    if (not math.isfinite(chunk_seconds) or not 0 < chunk_seconds <= 300
            or not math.isfinite(halo_seconds) or not 0 < halo_seconds <= 300):
        raise ValueError("chunk_parameters_invalid")
    recording = entry.recording
    validate_split_manifest((item.recording for item in manifest.records), split_manifest)
    selected = tuple(item.source_index for item in config.channels)
    max_rate = max(item.expected_source_rate_hz for item in config.channels)
    required_samples = math.ceil((chunk_seconds + 2 * halo_seconds) * max_rate * len(selected))
    if required_samples > max_samples:
        raise ValueError("chunk_and_halo_exceed_bounded_sample_limit")
    rate = config.target_sampling_rate_hz
    edge_seconds = (_filter_edge_samples(config) / rate) + 2.0
    if halo_seconds < edge_seconds:
        raise ValueError("chunk_halo_below_filter_guard")
    if (reader.metadata.sha256 != entry.signal.sha256
            or reader.metadata.size_bytes != entry.signal.size_bytes
            or reader.metadata.duration_seconds != recording.duration_seconds):
        raise ValueError("edf_source_changed_after_manifest_validation")
    core_start = 0.0
    while core_start < recording.duration_seconds:
        core_end = min(recording.duration_seconds, core_start + chunk_seconds)
        read_start = max(0.0, core_start - halo_seconds)
        read_end = min(recording.duration_seconds, core_end + halo_seconds)
        if space_check is not None:
            upper_rows = math.ceil((core_end - core_start) / config.step_seconds) + 1
            space_check(upper_rows * (len(feature_names(len(config.channels))) * 4 + 512))
        decoded = reader.read_range(read_start, read_end, channel_indices=selected,
                                    max_samples=max_samples)
        processed = preprocess_recording(
            recording, decoded, config, approved_manifest=manifest,
            split_manifest=split_manifest)
        yield from iter_processed_windows(
            processed, recording, core_start_seconds=core_start,
            core_end_seconds=core_end, window_seconds=config.window_seconds,
            step_seconds=config.step_seconds)
        del decoded, processed
        core_start = core_end


def extract_approved_recording_features(
    root: Path | str,
    manifest: CohortManifest,
    recording_id: str,
    config: PreprocessingConfig,
    split_manifest: SplitManifest,
    output_path: Path | str,
    *,
    chunk_seconds: float = 300.0,
    halo_seconds: float = 60.0,
    max_samples: int = 2_000_000,
    space_check: Callable[[int], None] | None = None,
) -> dict:
    """Extract and write one recording's windows incrementally to Parquet."""
    if space_check is not None:
        space_check(0)
    return write_feature_rows(
        iter_approved_recording_windows(
            root, manifest, recording_id, config, split_manifest,
            chunk_seconds=chunk_seconds, halo_seconds=halo_seconds,
            max_samples=max_samples, space_check=space_check),
        output_path, feature_schema=feature_names(len(config.channels)),
        max_output_bytes=5 * 1024**3, space_check=space_check)
