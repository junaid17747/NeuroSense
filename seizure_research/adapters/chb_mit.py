"""CHB-MIT 1.0.0 summary-text adapter with explicit source and patient review.

No age, canonical patient, device, annotation completeness, or channel selection
is inferred from a filename or a dataset's population label.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path, PurePosixPath

from ..edf import inspect_edf
from ..schemas import (
    DatasetProvenance, EEGRecording, NonnegativeInt, PositiveNumber, ResearchSchema,
    SeizureAnnotation, Text, TimeInterval, nonoverlapping, require_research_ready,
)
from ..splitting import digest
from .base import (
    AdaptedRecording, IngestionError, IngestionPlan, RecordingInput, SourceArtifact, local_file,
)

SOURCE_URL = "https://physionet.org/content/chbmit/1.0.0/"
FILE_PATTERN = re.compile(r"(?P<subject>chb[0-9]{2})[a-z]?_[0-9]+\+?\.edf")
MAX_SUMMARY_BYTES = 8 * 1024 * 1024


class SummaryRecording(ResearchSchema):
    filename: Text
    sampling_rate_hz: PositiveNumber
    channels: tuple[Text, ...]
    duration_seconds: PositiveNumber
    declared_seizure_count: NonnegativeInt
    events: tuple[TimeInterval, ...]


def _clock(value: str) -> int:
    if not re.fullmatch(r"[0-9]{1,2}:[0-9]{2}:[0-9]{2}", value):
        raise IngestionError("summary_invalid_clock")
    hours, minutes, seconds = map(int, value.split(":"))
    if hours > 47 or minutes > 59 or seconds > 59:
        raise IngestionError("summary_invalid_clock")
    return hours * 3600 + minutes * 60 + seconds


def parse_summary(text: str, subject_id: str) -> tuple[SummaryRecording, ...]:
    """Strictly parse all blocks. Missing counts/blocks never imply no seizures."""
    if not re.fullmatch(r"chb[0-9]{2}", subject_id):
        raise IngestionError("invalid_chb_subject_id")
    records = []
    names = set()
    channels = []
    sampling_rate = None
    current = None

    def finish():
        nonlocal current
        if current is None:
            return
        if not all(key in current for key in ("start", "end", "count")):
            raise IngestionError("summary_incomplete_recording_block")
        duration = current["end"] - current["start"]
        if duration < 0 and current["start"] < 86400 and current["end"] < 86400:
            duration += 86400
        if not 0 < duration < 86400:
            raise IngestionError("summary_ambiguous_or_invalid_duration")
        annotations = current["events"]
        if (len(annotations) != current["count"]
                or set(annotations) != set(range(1, len(annotations) + 1))):
            raise IngestionError("summary_event_count_mismatch")
        if current["unnumbered"] and current["count"] != 1:
            raise IngestionError("summary_ambiguous_event_numbering")
        events = []
        for index in range(1, current["count"] + 1):
            event = annotations[index]
            if set(event) != {"Start", "End"}:
                raise IngestionError("summary_incomplete_event")
            if not 0 <= event["Start"] < event["End"] <= duration:
                raise IngestionError("summary_event_bounds_invalid")
            events.append(TimeInterval(start_seconds=event["Start"], end_seconds=event["End"]))
        if not nonoverlapping(tuple(events)) or events != sorted(events, key=lambda e: e.start_seconds):
            raise IngestionError("summary_overlapping_or_unordered_events")
        records.append(SummaryRecording(
            filename=current["filename"], sampling_rate_hz=sampling_rate,
            channels=current["channels"], duration_seconds=float(duration),
            declared_seizure_count=current["count"], events=tuple(events),
        ))
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or set(line) == {"*"}:
            continue
        match = re.fullmatch(r"Data Sampling Rate: ([0-9]+(?:\.[0-9]+)?) Hz", line)
        if match:
            if sampling_rate is not None or current is not None or records:
                raise IngestionError("summary_duplicate_sampling_rate")
            sampling_rate = float(match[1])
            if not math.isfinite(sampling_rate) or sampling_rate <= 0:
                raise IngestionError("summary_invalid_sampling_rate")
            continue
        if line in ("Channels in EDF Files:", "Channels changed:"):
            finish()
            channels = []
            continue
        match = re.fullmatch(r"Channel ([0-9]+): (.+)", line)
        if match:
            if current is not None or int(match[1]) != len(channels) + 1 or not match[2].strip():
                raise IngestionError("summary_channel_sequence_invalid")
            channels.append(match[2].strip())
            continue
        match = re.fullmatch(r"File Name: (.+)", line)
        if match:
            finish()
            filename = match[1]
            identifier = FILE_PATTERN.fullmatch(filename)
            if identifier is None or identifier["subject"] != subject_id:
                raise IngestionError("summary_recording_identifier_mismatch")
            if filename in names:
                raise IngestionError("summary_duplicate_recording")
            if sampling_rate is None or not channels:
                raise IngestionError("summary_missing_acquisition_metadata")
            names.add(filename)
            current = {"filename": filename, "channels": tuple(channels), "events": {}, "unnumbered": False}
            continue
        if current is None:
            raise IngestionError("summary_unrecognized_line")
        match = re.fullmatch(r"File (Start|End) Time: (.+)", line)
        if match:
            key = match[1].lower()
            if key in current:
                raise IngestionError("summary_duplicate_clock")
            current[key] = _clock(match[2])
            continue
        match = re.fullmatch(r"Number of Seizures in File: ([0-9]+)", line)
        if match:
            if "count" in current:
                raise IngestionError("summary_duplicate_event_count")
            current["count"] = int(match[1])
            continue
        match = re.fullmatch(r"Seizure(?: ([0-9]+))? (Start|End) Time: ([0-9]+(?:\.[0-9]+)?) seconds", line)
        if match:
            index = int(match[1]) if match[1] else 1
            event = current["events"].setdefault(index, {})
            if match[2] in event or (current["events"] and
                    current.get("numbered") is not None and current["numbered"] != bool(match[1])):
                raise IngestionError("summary_duplicate_or_ambiguous_event")
            current["numbered"] = bool(match[1])
            current["unnumbered"] = not bool(match[1])
            value = float(match[3])
            if not math.isfinite(value):
                raise IngestionError("summary_invalid_event_time")
            event[match[2]] = value
            continue
        raise IngestionError("summary_unrecognized_line")
    finish()
    if not records:
        raise IngestionError("summary_has_no_recordings")
    return tuple(records)


def inspect_summary(path: Path | str, subject_id: str) -> tuple[tuple[SummaryRecording, ...], str, int]:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_SUMMARY_BYTES + 1)
    if len(data) > MAX_SUMMARY_BYTES:
        raise IngestionError("summary_size_limit_exceeded")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError("summary_encoding_invalid") from exc
    return parse_summary(text, subject_id), hashlib.sha256(data).hexdigest(), len(data)


class CHBMITAdapter:
    adapter_id = "chb_mit"
    adapter_version = "1"

    def ingest(self, root: Path, item: RecordingInput, plan: IngestionPlan) -> AdaptedRecording:
        item = RecordingInput.model_validate(item)
        plan = IngestionPlan.model_validate(plan)
        if not plan.dataset_review.verified or plan.permissions.status != "approved":
            raise IngestionError("dataset_permissions_review_required")
        patient = plan.identities.resolve(item.subject_id)
        # Known source alias is a constraint, never an automatically assigned identity.
        if item.subject_id in ("chb01", "chb21"):
            if plan.identities.resolve("chb01") != plan.identities.resolve("chb21"):
                raise IngestionError("known_patient_alias_conflict")
        if not item.metadata_review.verified:
            raise IngestionError("recording_metadata_review_required")
        if not item.annotation_review.verified:
            raise IngestionError("annotation_provenance_review_required")
        if item.annotation_coverage != "complete":
            raise IngestionError("annotation_completeness_required")
        filename = PurePosixPath(item.edf_path).name
        identifier = FILE_PATTERN.fullmatch(filename)
        if (identifier is None or identifier["subject"] != item.subject_id
                or item.edf_path != f"{item.subject_id}/{filename}"
                or item.summary_path != f"{item.subject_id}/{item.subject_id}-summary.txt"):
            raise IngestionError("source_recording_identifier_mismatch")

        summary, summary_sha, summary_size = inspect_summary(local_file(root, item.summary_path), item.subject_id)
        if summary_sha != item.expected_summary_sha256:
            raise IngestionError("reviewed_summary_hash_mismatch")
        matches = [entry for entry in summary if entry.filename == filename]
        if len(matches) != 1:
            raise IngestionError("recording_absent_from_summary")
        annotation = matches[0]
        edf = inspect_edf(local_file(root, item.edf_path))
        if edf.sha256 != item.expected_signal_sha256:
            raise IngestionError("reviewed_signal_hash_mismatch")
        if not math.isclose(edf.duration_seconds, annotation.duration_seconds, rel_tol=0, abs_tol=1e-6):
            raise IngestionError("edf_summary_duration_mismatch")
        if tuple(channel.label for channel in edf.channels) != annotation.channels:
            raise IngestionError("edf_summary_channel_mismatch")
        if tuple(channel.source_index for channel in item.channels) != tuple(range(len(edf.channels))):
            raise IngestionError("channel_selection_must_account_for_all_indices_in_order")
        kept = []
        for channel, selection in zip(edf.channels, item.channels):
            if channel.label != selection.source_label:
                raise IngestionError("reviewed_channel_label_mismatch")
            if selection.action == "keep":
                if channel.label == "-":
                    raise IngestionError("dummy_channel_cannot_be_kept")
                if channel.units != item.montage.units:
                    raise IngestionError("channel_units_mismatch")
                if not channel.sampling_rate_hz == annotation.sampling_rate_hz == item.expected_sampling_rate_hz:
                    raise IngestionError("sampling_rate_mismatch")
                kept.append(selection.output_label)
        if tuple(kept) != item.montage.channels:
            raise IngestionError("reviewed_montage_channel_mismatch")
        recording = EEGRecording(
            recording_id=f"chb_mit:1.0.0:{filename}", subject_id=item.subject_id,
            patient_group_id=patient, patient_identity_review_ref=plan.identities.review.evidence_ref,
            age=item.age, montage=item.montage, sampling_rate_hz=item.expected_sampling_rate_hz,
            device=item.device, preprocessing_id="chb_mit_raw_channel_selection_v1",
            provenance=DatasetProvenance(
                dataset_id="chb_mit", version=plan.dataset_version, source_url=SOURCE_URL,
                source_recording_id=item.edf_path, signal_sha256=edf.sha256,
                annotation_sha256=summary_sha, annotation_source=item.summary_path,
                annotation_provenance_ref=item.annotation_provenance_ref,
            ),
            duration_seconds=edf.duration_seconds, annotation_coverage=item.annotation_coverage,
            annotation_protocol_ref=item.annotation_protocol_ref,
            annotations=tuple(SeizureAnnotation(
                event_id=f"{filename}:seizure:{index}", start_seconds=event.start_seconds,
                end_seconds=event.end_seconds, seizure_type="unspecified_in_source",
                annotator_ref=item.annotator_ref, clinical_review_ref=item.annotation_review.evidence_ref,
            ) for index, event in enumerate(annotation.events, 1)),
            quality=item.quality, permissions=plan.permissions,
        )
        require_research_ready(recording)
        return AdaptedRecording(
            recording=recording,
            signal=SourceArtifact(relative_path=item.edf_path, sha256=edf.sha256, size_bytes=edf.size_bytes),
            annotations=SourceArtifact(relative_path=item.summary_path, sha256=summary_sha, size_bytes=summary_size),
            recording_review_sha256=digest(item.model_dump(mode="json")),
        )
