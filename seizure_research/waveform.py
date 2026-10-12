"""Bounded, metadata-preserving sample decoding for ordinary continuous EDF.

This intentionally does not support EDF+, BDF, discontinuous records, or
unknown record counts. Callers select channels by source index (never by an
ambiguous label) and must request a bounded time interval.
"""

from __future__ import annotations

import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np

from .adapters.base import IngestionError
from .edf import EDFChannel, EDFMetadata, inspect_edf

MAX_DECODED_SAMPLES = 10_000_000
SUPPORTED_ELECTRICAL_UNITS = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1_000.0, "v": 1_000_000.0}


@dataclass(frozen=True)
class ChannelSamples:
    metadata: EDFChannel
    timestamps_seconds: np.ndarray
    digital_samples: np.ndarray
    samples_uv: np.ndarray


@dataclass(frozen=True)
class EDFReadResult:
    metadata: EDFMetadata
    start_seconds: float
    end_seconds: float
    channels: tuple[ChannelSamples, ...]


def _safe_path(root: Path | str, relative_path: str) -> Path:
    """Resolve one regular dataset file without permitting traversal/symlinks."""
    if (not isinstance(relative_path, str) or "\\" in relative_path
            or any(ord(character) < 32 for character in relative_path)):
        raise IngestionError("edf_relative_path_invalid")
    rel = PurePosixPath(relative_path)
    if (rel.is_absolute() or any(part in ("", ".", "..") for part in relative_path.split("/"))
            or any(ord(char) < 32 for char in relative_path)):
        raise IngestionError("edf_relative_path_invalid")
    base = Path(root).resolve(strict=True)
    candidate = base.joinpath(*rel.parts)
    if any(base.joinpath(*rel.parts[:index]).is_symlink() for index in range(1, len(rel.parts) + 1)):
        raise IngestionError("edf_source_outside_root_or_not_regular")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise IngestionError("edf_source_unavailable") from exc
    if not resolved.is_relative_to(base) or candidate.is_symlink() or not resolved.is_file():
        raise IngestionError("edf_source_outside_root_or_not_regular")
    return resolved


def _uv_scale(unit: str) -> float:
    key = unit.strip().casefold().replace(" ", "")
    try:
        return SUPPORTED_ELECTRICAL_UNITS[key]
    except KeyError as exc:
        raise IngestionError("edf_signal_unit_unsupported") from exc


def _range_indices(start: float, end: float, rate: float, sample_count: int) -> tuple[int, int]:
    # Sample timestamps are n/fs. The requested interval is half-open [start,end).
    first = max(0, math.ceil(start * rate - 1e-10))
    stop = min(sample_count, math.ceil(end * rate - 1e-10))
    return first, max(first, stop)


def read_edf(
    root: Path | str,
    relative_path: str,
    *,
    channel_indices: Sequence[int],
    start_seconds: float,
    end_seconds: float,
    max_samples: int = 2_000_000,
) -> EDFReadResult:
    """Decode selected channel samples into microvolts for a bounded time range.

    `max_samples` bounds the aggregate output across selected channels and may
    not exceed a hard safety ceiling. The file is scanned for metadata/hash,
    then only selected record/channel blocks are decoded. No EDF identity/date
    header values are returned or logged.
    """
    with EDFStreamReader(root, relative_path) as reader:
        return reader.read_range(start_seconds, end_seconds, channel_indices=channel_indices,
                                 max_samples=max_samples)


class EDFStreamReader:
    """Hash and verify an EDF once, then expose bounded reads from the same file.

    Use as a context manager for chunked extraction. The open file and path
    signatures are checked around every range read so a replacement or
    mutation during an extraction fails closed.
    """

    def __init__(self, root: Path | str, relative_path: str):
        self.root = Path(root)
        self.relative_path = relative_path
        self.path = None
        self.stream = None
        self.metadata = None
        self._signature = None

    @staticmethod
    def _file_signature(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def __enter__(self):
        self.path = _safe_path(self.root, self.relative_path)
        before = self.path.stat()
        self.metadata = inspect_edf(self.path)
        self.stream = self.path.open("rb")
        opened = os.fstat(self.stream.fileno())
        self._signature = self._file_signature(before)
        if (not stat.S_ISREG(opened.st_mode)
                or self._signature != self._file_signature(opened)
                or self._signature != self._file_signature(self.path.stat())):
            self.stream.close()
            self.stream = None
            raise IngestionError("edf_source_changed_before_decode")
        return self

    def __exit__(self, *_exc):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def _check_unchanged(self):
        if self.stream is None or self.path is None:
            raise IngestionError("edf_stream_not_open")
        opened = os.fstat(self.stream.fileno())
        if (self._file_signature(opened) != self._signature
                or self._file_signature(self.path.stat()) != self._signature):
            raise IngestionError("edf_source_changed_during_decode")

    def read_range(self, start_seconds: float, end_seconds: float, *,
                   channel_indices: Sequence[int], max_samples: int = 2_000_000) -> EDFReadResult:
        if self.metadata is None or self.stream is None:
            raise IngestionError("edf_stream_not_open")
        if (not math.isfinite(start_seconds) or not math.isfinite(end_seconds)
                or start_seconds < 0 or end_seconds <= start_seconds):
            raise IngestionError("edf_time_range_invalid")
        if type(max_samples) is not int or not 1 <= max_samples <= MAX_DECODED_SAMPLES:
            raise IngestionError("edf_sample_limit_invalid")
        indices = tuple(channel_indices)
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise IngestionError("edf_channel_selection_invalid")
        if len(set(indices)) != len(indices):
            raise IngestionError("edf_duplicate_channel_selection")
        metadata = self.metadata
        if end_seconds > metadata.duration_seconds:
            raise IngestionError("edf_time_range_out_of_bounds")
        if any(index >= len(metadata.channels) for index in indices):
            raise IngestionError("edf_channel_index_out_of_bounds")
        selected = tuple(metadata.channels[index] for index in indices)
        scales = tuple(_uv_scale(channel.units) for channel in selected)
        ranges = tuple(_range_indices(start_seconds, end_seconds, channel.sampling_rate_hz,
                                      metadata.data_records * channel.samples_per_record)
                       for channel in selected)
        requested_count = sum(stop - first for first, stop in ranges)
        if any(stop <= first for first, stop in ranges) or requested_count > max_samples:
            raise IngestionError("edf_decoded_sample_limit_exceeded")
        self._check_unchanged()
        bytes_per_record = 2 * sum(channel.samples_per_record for channel in metadata.channels)
        channel_offsets = []
        offset = 0
        for channel in metadata.channels:
            channel_offsets.append(offset)
            offset += 2 * channel.samples_per_record
        decoded = []
        for channel, scale, (first, stop) in zip(selected, scales, ranges):
            digital = np.empty(stop - first, dtype=np.int16)
            destination = 0
            record_samples = channel.samples_per_record
            first_record, first_in_record = divmod(first, record_samples)
            last_record = (stop - 1) // record_samples
            for record_index in range(first_record, last_record + 1):
                local_start = first_in_record if record_index == first_record else 0
                local_stop = min(record_samples, stop - record_index * record_samples)
                count = local_stop - local_start
                position = (metadata.header_bytes + record_index * bytes_per_record
                            + channel_offsets[channel.index] + 2 * local_start)
                self.stream.seek(position)
                payload = self.stream.read(2 * count)
                if len(payload) != 2 * count:
                    raise IngestionError("edf_truncated_signal_data")
                digital[destination:destination + count] = np.frombuffer(payload, dtype="<i2")
                destination += count
            physical = ((digital.astype(np.float64) - channel.digital_min)
                        * ((channel.physical_max - channel.physical_min)
                           / (channel.digital_max - channel.digital_min))
                        + channel.physical_min) * scale
            timestamps = np.arange(first, stop, dtype=np.float64) / channel.sampling_rate_hz
            for array in (digital, physical, timestamps):
                array.setflags(write=False)
            decoded.append(ChannelSamples(channel, timestamps, digital, physical))
        self._check_unchanged()
        return EDFReadResult(metadata, start_seconds, end_seconds, tuple(decoded))


def channels_with_label(metadata: EDFMetadata, label: str) -> tuple[EDFChannel, ...]:
    """Return all matches; callers must choose by source index if ambiguous."""
    return tuple(channel for channel in metadata.channels if channel.label == label)


def duplicate_channel_labels(metadata: EDFMetadata) -> tuple[str, ...]:
    """Report duplicated raw labels without dropping or reordering channels."""
    labels = tuple(channel.label for channel in metadata.channels)
    return tuple(sorted({label for label in labels if labels.count(label) > 1}))
