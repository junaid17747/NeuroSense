"""Create an unreviewed human annotation checklist from public metadata only."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from .archive_selection import validate_selection
from .inventory import safe_relative
from .splitting import digest

REPOSITORY = Path(__file__).resolve().parents[1]
EPHEMERAL = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"), Path("/run"))


def parse_seizure_sidecar(data: bytes, *, sampling_rate_hz: float,
                           duration_seconds: float) -> list[dict]:
    """Read CHB-MIT's WFDB MIT-format '['/']' event markers, not EEG.

    Format: https://physionet.org/physiotools/wag/annot-5.htm
    Codes 32/33 are VFON/VFOFF in WFDB's ecgcodes.h; CHB-MIT uses
    their '['/']' symbols for seizure start/end. Unknown formats fail closed.
    Parsing is consistency evidence, never a reviewer attestation.
    """
    if (not data or len(data) > 1024**2 or len(data) % 2
            or not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0
            or not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("sidecar_bounds_invalid")
    offset = sample = 0
    start = None
    events = []
    terminated = False
    while offset + 2 <= len(data):
        word = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2
        code, delta = word >> 10, word & 1023
        if word == 0:
            if any(data[offset:]):
                raise ValueError("sidecar_trailing_data")
            terminated = True
            break
        if code == 59:
            if delta or offset + 4 > len(data):
                raise ValueError("sidecar_skip_invalid")
            high = int.from_bytes(data[offset:offset + 2], "little", signed=True)
            low = int.from_bytes(data[offset + 2:offset + 4], "little")
            sample += (high << 16) + low
            offset += 4
            if sample < 0:
                raise ValueError("sidecar_negative_time")
            continue
        if code in (60, 61, 62):
            continue
        if code == 63:
            offset += delta + delta % 2
            if offset > len(data):
                raise ValueError("sidecar_aux_truncated")
            continue
        sample += delta
        seconds = sample / sampling_rate_hz
        if seconds > duration_seconds:
            raise ValueError("sidecar_event_outside_recording")
        if code == 32:
            if start is not None:
                raise ValueError("sidecar_nested_seizure_start")
            start = seconds
        elif code == 33:
            if start is None or seconds <= start:
                raise ValueError("sidecar_unpaired_seizure_end")
            events.append({"start_seconds": start, "end_seconds": seconds})
            start = None
        elif code != 22:
            raise ValueError("sidecar_unsupported_annotation_code")
    if not terminated or start is not None or not events:
        raise ValueError("sidecar_missing_terminator_or_complete_events")
    return events


def build_review_packet(inventory: dict, selection: dict) -> dict:
    validate_selection(selection, inventory)
    file_index = {row["path"]: row for row in inventory["files"]}
    rows = []
    for item in selection["recordings"]:
        summary_path = item.get("summary_path")
        sidecar_path = item.get("sidecar_path")
        summary = item.get("summary") or {}
        rows.append({
            "recording_id": item["path"],
            "case_id": item["case_id"],
            "canonical_patient_id": item["canonical_patient_id"],
            "original_edf_sha256": item["sha256"],
            "summary_path": summary_path,
            "summary_sha256": file_index.get(summary_path, {}).get("sha256") if summary_path else None,
            "summary_declared_event_count": summary.get("declared_seizures"),
            "summary_declared_intervals_seconds": summary.get("summary_intervals", []),
            "sidecar_path": sidecar_path,
            "sidecar_sha256": file_index.get(sidecar_path, {}).get("sha256") if sidecar_path else None,
            "published_age": item.get("age"),
            "source_inventory_gaps": item.get("annotation_gaps", []),
            "recording_duration_seconds": None,
            "review_status": "unreviewed",
            "reviewer": None,
            "reviewed_on": None,
            "evidence_ref": None,
            "sidecar_parsed_and_reconciled": False,
            "reconciliation_notes": None,
            "reviewed_event_intervals_seconds": None,
            "complete_record_coverage_attested": False,
            "verified_seizure_free": False,
            "quality_and_excluded_intervals_reviewed": False,
            "training_labels_created": False,
            "eligibility_status": "pending",
            "exclusion_reason": None,
            "reviewed_seizure_free_intervals_seconds": None,
            "excluded_intervals_seconds": None,
        })
    packet = {
        "schema_version": "chbmit-annotation-human-review-packet-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required_no_labels_created",
        "inventory_sha256": inventory["inventory_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "selected_recording_count": len(rows),
        "selected_canonical_patient_count": selection["selected_patient_count"],
        "declared_event_count_unreconciled": selection["selected_declared_seizure_events"],
        "selected_seizure_positive_recordings_unreviewed": selection["selected_official_seizure_positive_recordings"],
        "selected_summary_zero_candidates_unverified": selection["selected_summary_zero_candidates"],
        "verified_background_recordings": 0,
        "inventory_annotation_issues": inventory["annotation_issues"],
        "review_requirements": [
            "verify EDF duration, header, labels, channels, rates and original checksum",
            "parse the exact official seizure sidecar where present and reconcile event count/times with the patient summary",
            "review every recording interval for seizure and ambiguous/unreviewed periods",
            "for summary-zero candidates, attest full-record review and the exact covered duration before any negative label",
            "record signal-quality exclusions and confirm reference intervals align to recording-relative time",
            "preserve reviewer, date, evidence reference and source hashes; do not edit source metadata",
        ],
        "training_labels_created": False,
        "recordings": rows,
        "packet_sha256": None,
    }
    packet["packet_sha256"] = digest(packet)
    return packet


def seal_review_packet(packet: dict) -> dict:
    """Recompute the packet checksum after a reviewer records their attestations.

    This digest detects accidental edits; it is not a reviewer signature or
    institutional approval.
    """
    sealed = dict(packet)
    sealed["packet_sha256"] = None
    sealed["packet_sha256"] = digest(sealed)
    return sealed


def audit_review_packet(inventory: dict, selection: dict, packet: dict) -> dict:
    """Fail-closed audit of source bindings and human-completed review fields."""
    validate_selection(selection, inventory)
    blockers: list[dict[str, str]] = []

    def block(recording_id: str, reason: str) -> None:
        blockers.append({"recording_id": recording_id, "reason": reason})

    if packet.get("schema_version") != "chbmit-annotation-human-review-packet-v1":
        block("", "review_packet_schema_mismatch")
    if packet.get("inventory_sha256") != inventory.get("inventory_sha256"):
        block("", "review_packet_inventory_hash_mismatch")
    if packet.get("selection_sha256") != selection.get("selection_sha256"):
        block("", "review_packet_selection_hash_mismatch")
    if packet.get("training_labels_created") is not False:
        block("", "review_packet_must_not_create_training_labels")
    sealed = dict(packet)
    recorded_seal = sealed.pop("packet_sha256", None)
    if recorded_seal != digest({**sealed, "packet_sha256": None}):
        block("", "review_packet_checksum_mismatch")

    inventory_files = {row["path"]: row for row in inventory["files"]}
    expected = {row["path"]: row for row in selection["recordings"]}
    rows = packet.get("recordings")
    if not isinstance(rows, list):
        rows = []
        block("", "review_packet_recordings_invalid")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("recording_id"), str):
            block("", "review_packet_recording_row_invalid")
            continue
        recording_id = row["recording_id"]
        if recording_id in by_id:
            block(recording_id, "duplicate_review_recording")
            continue
        by_id[recording_id] = row
    if set(by_id) != set(expected):
        block("", "review_packet_recording_coverage_mismatch")

    verified_background = 0
    reviewed_events = 0
    for recording_id, selected in expected.items():
        row = by_id.get(recording_id)
        if row is None:
            continue
        summary_path, sidecar_path = selected.get("summary_path"), selected.get("sidecar_path")
        immutable = {
            "case_id": selected.get("case_id"),
            "canonical_patient_id": selected.get("canonical_patient_id"),
            "original_edf_sha256": selected.get("sha256"),
            "summary_path": summary_path,
            "summary_sha256": inventory_files.get(summary_path, {}).get("sha256") if summary_path else None,
            "summary_declared_event_count": (selected.get("summary") or {}).get("declared_seizures"),
            "sidecar_path": sidecar_path,
            "sidecar_sha256": inventory_files.get(sidecar_path, {}).get("sha256") if sidecar_path else None,
        }
        for field, value in immutable.items():
            if row.get(field) != value:
                block(recording_id, "review_source_binding_mismatch:" + field)
        if row.get("review_status") != "reviewed":
            block(recording_id, "human_annotation_review_incomplete")
        for field in ("reviewer", "reviewed_on", "evidence_ref"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                block(recording_id, "reviewer_evidence_missing:" + field)
        if row.get("eligibility_status") == "excluded":
            if not isinstance(row.get("exclusion_reason"), str) or not row["exclusion_reason"].strip():
                block(recording_id, "excluded_recording_reason_missing")
            continue
        duration = row.get("recording_duration_seconds")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            block(recording_id, "reviewed_recording_duration_missing_or_invalid")
            duration = None
        if row.get("complete_record_coverage_attested") is not True:
            block(recording_id, "complete_record_coverage_not_attested")
        if row.get("quality_and_excluded_intervals_reviewed") is not True:
            block(recording_id, "quality_review_not_attested")
        declared_count = immutable["summary_declared_event_count"]
        events = row.get("reviewed_event_intervals_seconds")
        if not isinstance(events, list):
            block(recording_id, "reviewed_event_intervals_missing")
            events = []
        valid_events = []
        for event in events:
            if isinstance(event, dict):
                start, end = event.get("start_seconds"), event.get("end_seconds")
            elif isinstance(event, (list, tuple)) and len(event) == 2:
                start, end = event
            else:
                block(recording_id, "reviewed_event_interval_invalid")
                continue
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start or duration is None or end > duration):
                block(recording_id, "reviewed_event_interval_out_of_bounds")
                continue
            valid_events.append((float(start), float(end)))
        if any(right[0] < left[1] for left, right in zip(sorted(valid_events), sorted(valid_events)[1:])):
            block(recording_id, "reviewed_event_intervals_overlap")

        if type(declared_count) is int and declared_count > 0:
            if row.get("sidecar_parsed_and_reconciled") is not True:
                block(recording_id, "seizure_sidecar_not_reconciled_by_reviewer")
            if not sidecar_path or not row.get("sidecar_sha256"):
                block(recording_id, "seizure_sidecar_source_missing")
            notes = row.get("reconciliation_notes")
            if not isinstance(notes, str) or not notes.strip():
                block(recording_id, "seizure_reconciliation_notes_missing")
            if not valid_events:
                block(recording_id, "official_positive_recording_has_no_reconciled_event")
            reviewed_events += len(valid_events)
        elif declared_count == 0:
            if valid_events:
                block(recording_id, "summary_zero_has_reviewed_seizure_intervals")
            if row.get("verified_seizure_free") is not True:
                block(recording_id, "summary_zero_not_verified_seizure_free")
            elif (row.get("complete_record_coverage_attested") is True
                  and duration is not None and not valid_events):
                verified_background += 1
        else:
            block(recording_id, "summary_event_count_unavailable")

    ready = not blockers and len(by_id) == len(expected)
    return {
        "status": "annotation_review_complete" if ready else "annotation_review_incomplete",
        "annotation_evidence_complete": ready,
        "inventory_sha256": inventory.get("inventory_sha256"),
        "selection_sha256": selection.get("selection_sha256"),
        "reviewed_recording_count": sum(row.get("review_status") == "reviewed" for row in by_id.values()),
        "selected_recording_count": len(expected),
        "reconciled_event_count": reviewed_events,
        "verified_background_recording_count": verified_background,
        "training_labels_created": False,
        "blockers": blockers,
    }


def write_review_packet(packet: dict, output: Path | str) -> Path:
    output = Path(output).expanduser().absolute()
    if any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError("review_output_symlink_rejected")
    resolved = output.resolve()
    if resolved.is_relative_to(REPOSITORY) or any(resolved.is_relative_to(root) for root in EPHEMERAL):
        raise ValueError("review_output_must_be_persistent_and_outside_repository")
    if output.exists():
        raise FileExistsError("review_packet_will_not_overwrite_existing_file")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    pending = output.with_name(output.name + ".pending")
    data = (json.dumps(packet, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, output)
    os.chmod(output, 0o600)
    return output


def inspect_archived_recording(root, temporary_root, inventory, selection, approval_path, recording_id):
    """Authorized header/annotation consistency evidence, with no human decisions."""
    from .archive import extracted_recording, safe_file, safe_root, verify_original
    from .edf import inspect_edf
    from .adapters.chb_mit import inspect_summary
    validate_selection(selection, inventory)
    selected = next((r for r in selection["recordings"] if r["path"] == recording_id), None)
    if selected is None:
        raise ValueError("recording_not_in_proposed_selection")
    files = {r["path"]:r for r in selection["support_files"]}
    with extracted_recording(root,temporary_root,inventory,selection,approval_path,recording_id) as edf:
        metadata = inspect_edf(edf)
        summary_path = safe_file(safe_root(root),"sources/"+selected["summary_path"])
        verify_original(summary_path,files[selected["summary_path"]])
        errors = []
        summary = None
        try:
            parsed,_,_ = inspect_summary(summary_path,selected["case_id"],filename=edf.name)
            summary = parsed[0]
            if metadata.duration_seconds != summary.duration_seconds:
                errors.append("summary_edf_duration_disagreement")
            if tuple(c.label for c in metadata.channels) != summary.channels:
                errors.append("summary_edf_channels_disagree")
        except ValueError as exc:
            errors.append(str(exc))
        sidecar_events = None
        if selected.get("sidecar_path"):
            path = safe_file(safe_root(root),"sources/"+selected["sidecar_path"])
            verify_original(path,files[selected["sidecar_path"]])
            try:
                rates = {c.sampling_rate_hz for c in metadata.channels if c.label != "-"}
                if len(rates) != 1:
                    raise ValueError("sidecar_timebase_needs_review")
                with path.open("rb") as stream:
                    sidecar_events = parse_seizure_sidecar(stream.read(1024**2+1),sampling_rate_hz=rates.pop(),
                                                          duration_seconds=metadata.duration_seconds)
                if summary and (len(sidecar_events) != len(summary.events)
                        or any(abs(a["start_seconds"]-b.start_seconds)>.5 or abs(a["end_seconds"]-b.end_seconds)>.5
                               for a,b in zip(sidecar_events,summary.events))):
                    errors.append("summary_sidecar_events_disagree")
            except ValueError as exc:
                errors.append(str(exc))
    return {"schema_version":"chbmit-recording-inspection-v1","recording_id":recording_id,
            "inventory_sha256":inventory["inventory_sha256"],"selection_sha256":selection["selection_sha256"],
            "edf":metadata.model_dump(mode="json"),"summary":summary.model_dump(mode="json") if summary else None,
            "sidecar_events":sidecar_events,"published_age":selected.get("age"),"discrepancies":errors,
            "review_status":"unreviewed","background_verified":False,"quality_review":"pending",
            "training_labels_created":False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-packet", type=Path,
                        help="Audit a reviewer-completed packet without creating labels.")
    parser.add_argument("--inspect-recording", help="Inspect one authorized archive member; does not attest review.")
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--approval-record", type=Path)
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if args.inspect_recording:
        if args.audit_packet or not all((args.archive_root,args.temporary_root,args.approval_record)):
            parser.error("inspection requires archive root, temporary root and processing approval")
        report = inspect_archived_recording(args.archive_root,args.temporary_root,inventory,selection,
                                            args.approval_record,args.inspect_recording)
        if args.output:
            write_review_packet(report,args.output)
        print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
        return 0 if not report["discrepancies"] else 2
    if args.audit_packet is not None:
        if args.output is not None:
            parser.error("--output cannot be used with --audit-packet")
        packet = json.loads(args.audit_packet.read_text(encoding="utf-8"))
        report = audit_review_packet(inventory, selection, packet)
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0 if report["annotation_evidence_complete"] else 2
    if args.output is None:
        parser.error("--output is required unless --audit-packet is supplied")
    packet = build_review_packet(inventory, selection)
    target = write_review_packet(packet, args.output)
    print(json.dumps({"status": packet["status"], "recordings": len(packet["recordings"]),
                      "declared_events_unreconciled": packet["declared_event_count_unreconciled"],
                      "verified_background_recordings": 0, "output": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
