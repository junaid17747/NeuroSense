"""Public-metadata cohort investigation and deterministic proposed patient splits.

These reports contain proposals, never reviewer attestations or training labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from .archive import read_json
from .archive_selection import validate_selection
from .inventory import BASE, canonical_patient
from .splitting import digest
from .training import PILOT_CHANNELS


def proposed_split(inventory: dict, selection: dict, *, seed: int = 42) -> dict:
    validate_selection(selection, inventory)
    if type(seed) is not int:
        raise ValueError("split_seed_must_be_integer")
    patients = sorted({canonical_patient(r["case_id"]) for r in selection["recordings"]},
                      key=lambda p: (digest([seed, p]), p))
    if len(patients) < 3:
        raise ValueError("three_distinct_patients_required")
    n_train = min(max(1, int(len(patients) * .6)), len(patients) - 2)
    n_val = min(max(1, int(len(patients) * .2)), len(patients) - n_train - 1)
    membership = {p: "train" if n < n_train else "validation" if n < n_train + n_val else "test"
                  for n, p in enumerate(patients)}
    rows = []
    for r in sorted(selection["recordings"], key=lambda r: r["path"]):
        patient = canonical_patient(r["case_id"])
        if r["canonical_patient_id"] != patient:
            raise ValueError("selection_canonical_identity_mismatch")
        channels = (r.get("summary") or {}).get("summary_channels", [])
        rows.append({"path": r["path"], "case_id": r["case_id"],
                     "canonical_patient_id": patient, "split": membership[patient],
                     "source_sha256": r["sha256"],
                     "age_group": (r.get("age") or {}).get("age_group", "unknown"),
                     "positive_candidate": bool(r["official_seizure_positive"]),
                     "background_candidate": bool(r["background_candidate"]),
                     "declared_events_unreviewed": (r.get("summary") or {}).get("declared_seizures"),
                     "missing_standard_channels": sorted(set(PILOT_CHANNELS) - set(channels)),
                     "channel_review": "requires_EDF_header_and_human_review"})
    distribution = {}
    for name in ("train", "validation", "test"):
        group = [r for r in rows if r["split"] == name]
        distribution[name] = {
            "patients": sorted({r["canonical_patient_id"] for r in group}),
            "recordings": len(group),
            "positive_candidates": sum(r["positive_candidate"] for r in group),
            "background_candidates": sum(r["background_candidate"] for r in group),
            "declared_events_unreviewed": sum(r["declared_events_unreviewed"] or 0 for r in group),
            "recordings_by_published_age": dict(sorted(Counter(r["age_group"] for r in group).items())),
            "patients_by_published_age": {age: len({r["canonical_patient_id"] for r in group
                                                   if r["age_group"] == age})
                                          for age in sorted({r["age_group"] for r in group})},
        }
    result = {"schema_version": "chbmit-multipatient-proposed-split-v1",
              "status": "proposed_not_reviewed", "algorithm": "sha256-json-seed-patient-v1",
              "seed": seed, "train_fraction": .6, "validation_fraction": .2,
              "inventory_sha256": inventory["inventory_sha256"],
              "selection_sha256": selection["selection_sha256"],
              "assignments": rows, "distribution": distribution,
              "policy": "Freeze before outcomes; exclusions retain original patient assignment. No age-specific validation claimed.",
              "training_labels_created": False}
    result["split_proposal_sha256"] = digest(result)
    return result


def validate_proposed_split(inventory: dict, selection: dict, proposal: dict) -> None:
    if proposed_split(inventory, selection, seed=proposal.get("seed")) != proposal:
        raise ValueError("multipatient_split_changed_or_noncanonical")


def annotation_investigation(inventory: dict, selection: dict) -> dict:
    validate_selection(selection, inventory)
    files = {r["path"]: r for r in inventory["files"]}
    recordings = {r["path"]: r for r in inventory["recordings"]}
    selected = {r["path"] for r in selection["recordings"]}
    issues = []
    for issue in inventory["annotation_issues"]:
        r = recordings[issue["path"]]
        sidecar = files.get(r["path"] + ".seizures")
        alternatives = [x for x in selection["recordings"] if x["case_id"] == r["case_id"]]
        issues.append({"path": r["path"], "selected": r["path"] in selected,
                       "gaps": issue["gaps"], "summary": r.get("summary"),
                       "listed_with_seizures": r["in_RECORDS_WITH_SEIZURES"],
                       "sidecar_listed": sidecar is not None,
                       "sidecar_bytes": sidecar["bytes"] if sidecar else None,
                       "sidecar_sha256": sidecar["sha256"] if sidecar else None,
                       "summary_url": BASE + r["case_id"] + "/" + r["case_id"] + "-summary.txt",
                       "summary_sha256": inventory["source_metadata_sha256"][r["case_id"] + "/" + r["case_id"] + "-summary.txt"],
                       "seizure_list_url": BASE + "RECORDS-WITH-SEIZURES",
                       "seizure_list_sha256": inventory["source_metadata_sha256"]["RECORDS-WITH-SEIZURES"],
                       "directory_url": BASE + r["case_id"] + "/",
                       "directory_sha256": inventory["source_metadata_sha256"][r["case_id"] + "/listing.html"],
                       "resolved_by_public_metadata": False,
                       "disposition": "Excluded from original proposal; do not repair labels by assumption.",
                       "existing_annotation_consistent_alternatives": [x["path"] for x in alternatives],
                       "human_action": "Reconcile official evidence after authorized staging; seek dataset maintainer clarification if needed."})
    background = [{"path": r["path"], "canonical_patient_id": r["canonical_patient_id"],
                   "summary_sha256": files[r["summary_path"]]["sha256"],
                   "summary_declares_zero": (r.get("summary") or {}).get("declared_seizures") == 0,
                   "listed_with_seizures": r["in_RECORDS_WITH_SEIZURES"],
                   "sidecar_listed": bool(r.get("sidecar_path")),
                   "verified_background": False, "reviewed_intervals": [],
                   "required_action": "Review full duration; explicitly record seizure-free intervals and quality exclusions."}
                  for r in selection["recordings"] if r["background_candidate"]]
    result = {"schema_version": "chbmit-annotation-discrepancy-investigation-v1",
              "inventory_sha256": inventory["inventory_sha256"], "selection_sha256": selection["selection_sha256"],
              "source_metadata_sha256": inventory["source_metadata_sha256"],
              "issues": issues, "background_candidates": background,
              "verified_background_recordings": 0, "labels_created": False,
              "sidecar_contents_reviewed": False,
              "finding": "Public summaries and list/sidecar presence do not resolve the conflicts or certify background."}
    result["report_sha256"] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("split", "investigate"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    inventory, selection = read_json(args.inventory), read_json(args.selection)
    result = (proposed_split(inventory, selection, seed=args.seed) if args.command == "split"
              else annotation_investigation(inventory, selection))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != data:
        raise ValueError("report_exists_with_different_content_use_new_path")
    args.output.write_text(data)
    print(json.dumps({"status": "proposal_only_no_labels", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
