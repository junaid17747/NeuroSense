"""Audit the existing EEG V2 CSV and publish its Phase 1 subject split; no training."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.eeg_v2_data import (
    DATA_FILE, ROOT, SEED, build_manifest, dataset_fingerprint,
    feature_audit, read_dataset, structural_audit,
)


OUTPUT_DIR = ROOT / "reports/eeg_v2_phase1"


def json_text(payload):
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_new_or_identical(path, payload):
    """Idempotent reruns are allowed, but never overwrite an existing artifact."""
    text = json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite differing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def run_audit(data_file=DATA_FILE, output_dir=OUTPUT_DIR, seed=SEED):
    data_file, output_dir = Path(data_file).resolve(), Path(output_dir).resolve()
    if not data_file.is_relative_to(ROOT) or not output_dir.is_relative_to(ROOT):
        raise ValueError("Phase 1 input and output must stay inside the NeuroSense folder.")
    before = dataset_fingerprint(data_file)
    print(f"Reading {before['path']} ({before['size_bytes']:,} bytes)", flush=True)
    df = read_dataset(data_file)
    integrity = structural_audit(df)
    report = {
        "audit_version": 1, "dataset": before, "integrity": integrity,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "source_fingerprints": [dataset_fingerprint(ROOT / source) for source in (
            "src/model/eeg_v2_data.py", "src/model/audit_eeg_v2.py",
            "src/model/generate_synthetic_eeg_v2_1m.py", "src/features/eeg_features_v1.py",
        )],
    }
    manifest = None
    if not integrity["errors"]:
        report["features"] = feature_audit(df)
        try:
            manifest = build_manifest(df, before, seed)
        except ValueError as exc:
            integrity["errors"].append(str(exc))
    after = dataset_fingerprint(data_file)
    report["dataset_unchanged"] = before == after
    if before != after:
        integrity["errors"].append("Dataset changed during the audit.")
    report["status"] = "failed" if integrity["errors"] else "passed_with_warnings"
    if not integrity["errors"]:
        report["split_summary"] = {
            name: {key: value for key, value in split.items() if key != "subjects"}
            for name, split in manifest["splits"].items()
        }
        report["leakage_checks"] = {
            "pairwise_subject_overlap": 0, "row_overlap": 0, "unassigned_rows": 0,
            "cross_partition_identical_feature_vectors": 0,
            "all_classes_in_every_partition": True,
            "model_features_exclude_identifiers_and_label": True,
        }
    # Check both destinations before writing either one, so differing artifacts
    # cannot be partly replaced by a rerun with a different seed or dataset.
    artifacts = {output_dir / "dataset_audit.json": report}
    if not integrity["errors"]:
        artifacts[output_dir / "split_manifest.json"] = manifest
    for path, payload in artifacts.items():
        if path.exists() and path.read_text(encoding="utf-8") != json_text(payload):
            raise FileExistsError(f"Refusing to overwrite differing artifact: {path}")
    for path, payload in artifacts.items():
        write_new_or_identical(path, payload)
    print(f"Audit: {report['status']}")
    print(f"Rows: {len(df):,}; subjects: {integrity.get('subject_count', 0):,}; features: {integrity['feature_count']}")
    print(f"SHA-256: {before['sha256']}")
    for error in integrity["errors"]:
        print(f"ERROR: {error}")
    for name, split in report.get("split_summary", {}).items():
        print(f"{name}: {split['subject_count']} subjects; {split['rows']:,} rows; classes={split['class_counts']}")
    for warning in report.get("features", {}).get("warnings", []):
        print(f"WARNING {warning['code']}: rows={warning.get('rows', 'not measurable')}; {warning['detail']}")
    print(f"Dataset unchanged: {report['dataset_unchanged']}")
    print(f"Artifacts: {output_dir.relative_to(ROOT)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = run_audit(args.data, args.output_dir, args.seed)
    raise SystemExit(1 if report["integrity"]["errors"] else 0)


if __name__ == "__main__":
    main()
