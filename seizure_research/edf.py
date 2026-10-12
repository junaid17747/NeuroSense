"""Bounded legacy EDF header inspection and streaming SHA-256; no signal decoding.

Only continuous, ordinary 16-bit EDF with a known positive record count is
supported. EDF+, BDF, unknown lengths, and inconsistent file sizes are rejected.
Patient/recording identity and date fields are deliberately not returned.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from pathlib import Path

from .adapters.base import IngestionError
from .schemas import NonnegativeInt, PositiveInt, PositiveNumber, ResearchSchema, SHA256, Text


class EDFChannel(ResearchSchema):
    index: NonnegativeInt
    label: Text
    units: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    samples_per_record: PositiveInt
    sampling_rate_hz: PositiveNumber


class EDFMetadata(ResearchSchema):
    format: str = "EDF"
    header_bytes: PositiveInt
    data_records: PositiveInt
    record_duration_seconds: PositiveNumber
    duration_seconds: PositiveNumber
    channels: tuple[EDFChannel, ...]
    size_bytes: PositiveInt
    sha256: SHA256


def _ascii(value: bytes) -> str:
    if any(byte < 32 or byte > 126 for byte in value):
        raise IngestionError("edf_non_ascii_header")
    return value.decode("ascii").strip()


def _integer(value: bytes) -> int:
    text = _ascii(value)
    if not re.fullmatch(r"-?[0-9]+", text):
        raise IngestionError("edf_invalid_integer")
    return int(text)


def _number(value: bytes) -> float:
    try:
        number = float(_ascii(value))
    except ValueError as exc:
        raise IngestionError("edf_invalid_number") from exc
    if not math.isfinite(number):
        raise IngestionError("edf_nonfinite_number")
    return number


def inspect_edf(path: Path | str) -> EDFMetadata:
    """Inspect headers and hash source bytes in bounded memory, without copying EEG."""
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise IngestionError("edf_not_regular_file")
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise IngestionError("edf_truncated_header")
        if _ascii(fixed[:8]) != "0" or _ascii(fixed[192:236]):
            raise IngestionError("edf_variant_unsupported")
        count = _integer(fixed[252:256])
        header_bytes = _integer(fixed[184:192])
        if not 1 <= count <= 4096 or header_bytes != 256 * (count + 1):
            raise IngestionError("edf_invalid_header_size")
        records = _integer(fixed[236:244])
        seconds = _number(fixed[244:252])
        if records <= 0 or seconds <= 0 or not math.isfinite(records * seconds):
            raise IngestionError("edf_unknown_or_invalid_duration")
        variable = stream.read(256 * count)
        if len(variable) != 256 * count:
            raise IngestionError("edf_truncated_signal_headers")
        fields = []
        offset = 0
        for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
            fields.append([variable[offset + index * width:offset + (index + 1) * width]
                           for index in range(count)])
            offset += width * count
        channels = []
        for index in range(count):
            label = _ascii(fields[0][index])
            units = _ascii(fields[2][index])
            samples = _integer(fields[8][index])
            physical_min, physical_max = (_number(fields[i][index]) for i in (3, 4))
            digital_min, digital_max = (_integer(fields[i][index]) for i in (5, 6))
            if not label or label == "EDF Annotations" or samples <= 0:
                raise IngestionError("edf_invalid_signal")
            if not physical_min < physical_max or not -32768 <= digital_min < digital_max <= 32767:
                raise IngestionError("edf_invalid_calibration")
            rate = samples / seconds
            if not math.isfinite(rate) or rate <= 0:
                raise IngestionError("edf_invalid_sampling_rate")
            channels.append(EDFChannel(index=index, label=label, units=units,
                                       physical_min=physical_min, physical_max=physical_max,
                                       digital_min=digital_min, digital_max=digital_max,
                                       samples_per_record=samples, sampling_rate_hz=rate))
        expected_size = header_bytes + records * 2 * sum(item.samples_per_record for item in channels)
        if 2 * sum(item.samples_per_record for item in channels) > 61_440:
            raise IngestionError("edf_data_record_size_exceeded")
        if before.st_size != expected_size:
            raise IngestionError("edf_payload_size_mismatch")
        fingerprint = hashlib.sha256(fixed + variable)
        while chunk := stream.read(1024 * 1024):
            fingerprint.update(chunk)
        after = os.fstat(stream.fileno())
        current = Path(path).stat()
        signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if signature(before) != signature(after) or signature(after) != signature(current):
            raise IngestionError("source_changed_during_inspection")
        return EDFMetadata(header_bytes=header_bytes, data_records=records,
                           record_duration_seconds=seconds, duration_seconds=records * seconds,
                           channels=tuple(channels), size_bytes=before.st_size, sha256=fingerprint.hexdigest())
