"""Explicitly gated, read-only bounded EDF smoke test for the reviewed pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .edf import inspect_edf
from .preprocessing import PreprocessingConfig
from .readiness import _check_file_manifest, _read_json, check_readiness
from .training import PILOT_FILES
from .waveform import read_edf


def decode_smoke_window(root: Path | str, relative_path: str,
                        config: PreprocessingConfig, expected_sha256: str) -> dict:
    """Read one configured window after verifying source identity, without writes."""
    if relative_path not in PILOT_FILES:
        raise ValueError("smoke_recording_not_in_frozen_pilot")
    path = Path(root) / relative_path
    metadata = inspect_edf(path)
    if metadata.sha256 != expected_sha256:
        raise ValueError("smoke_recording_sha256_mismatch")
    indices = tuple(channel.source_index for channel in config.channels)
    if any(index >= len(metadata.channels) for index in indices):
        raise ValueError("smoke_channel_mapping_out_of_bounds")
    for channel in config.channels:
        source = metadata.channels[channel.source_index]
        if (source.label != channel.source_label
                or source.sampling_rate_hz != channel.expected_source_rate_hz):
            raise ValueError("smoke_channel_mapping_metadata_mismatch")
    duration = min(float(config.window_seconds), metadata.duration_seconds)
    result = read_edf(root, relative_path, channel_indices=indices,
                      start_seconds=0.0, end_seconds=duration, max_samples=2_000_000)
    sample_counts = [len(channel.samples_uv) for channel in result.channels]
    if not sample_counts or len(set(sample_counts)) != 1 or not all(sample_counts):
        raise ValueError("smoke_decode_sample_alignment_failed")
    return {"status": "decoded", "recording": relative_path,
            "duration_seconds": duration, "channels_decoded": len(result.channels),
            "samples_per_channel": sample_counts[0], "source_sha256": metadata.sha256,
            "writes_artifacts": False}


def run_smoke(*, root: Path | str, relative_path: str,
              cohort_manifest_path: Path | str, preprocessing_config_path: Path | str,
              annotation_evidence_path: Path | str, approval_record_path: Path | str,
              file_manifest_path: Path | str, authorized: bool) -> dict:
    if not authorized:
        raise PermissionError("explicit_real_data_smoke_test_authorization_required")
    report = check_readiness(
        root=root, cohort_manifest_path=cohort_manifest_path,
        preprocessing_config_path=preprocessing_config_path,
        annotation_evidence_path=annotation_evidence_path,
        approval_record_path=approval_record_path, file_manifest_path=file_manifest_path,
    )
    if not report["ready_for_real_data_processing"]:
        raise PermissionError("readiness_gate_blocked")
    file_manifest, _ = _check_file_manifest(Path(file_manifest_path))
    entry = next((item for item in file_manifest["recordings"] if item["path"] == relative_path), None)
    if entry is None:
        raise ValueError("smoke_recording_not_in_frozen_pilot")
    config = PreprocessingConfig.model_validate(_read_json(Path(preprocessing_config_path)))
    result = decode_smoke_window(root, relative_path, config, entry["expected_sha256"])
    result["readiness"] = "passed"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--preprocessing-config", required=True)
    parser.add_argument("--annotation-evidence", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--file-manifest", required=True)
    parser.add_argument("--confirm-authorized-processing", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_smoke(
            root=args.root, relative_path=args.recording,
            cohort_manifest_path=args.cohort_manifest,
            preprocessing_config_path=args.preprocessing_config,
            annotation_evidence_path=args.annotation_evidence,
            approval_record_path=args.approval_record,
            file_manifest_path=args.file_manifest,
            authorized=args.confirm_authorized_processing,
        )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
