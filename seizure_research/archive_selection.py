"""Deterministic prospective archive selection, separate from the frozen pilot."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .inventory import canonical_patient, safe_relative
from .splitting import digest

ARCHIVE_LIMIT = 2_147_483_648
METADATA_RESERVE = 16 * 1024**2
ZIP_OVERHEAD_LIMIT = 65536


def validate_inventory(inventory: dict) -> None:
    if (inventory.get("schema_version") != "chbmit-complete-inventory-v1"
            or inventory.get("inventory_sha256") != digest({k: v for k, v in inventory.items() if k != "inventory_sha256"})):
        raise ValueError("inventory_hash_or_schema_invalid")
    files = {item["path"]: item for item in inventory["files"]}
    if len(files) != len(inventory["files"]):
        raise ValueError("inventory_duplicate_file")
    paths = set()
    for item in inventory["recordings"]:
        path = safe_relative(item["path"])
        if (path in paths or path.split("/")[0] != item["case_id"]
                or item["case_id"] not in inventory["cases"]
                or canonical_patient(item["case_id"]) != item["canonical_patient_id"]
                or item["bytes"] != files[path]["bytes"] or item["sha256"] != files[path]["sha256"]):
            raise ValueError("inventory_recording_identity_invalid")
        paths.add(path)


def select_cohort(inventory: dict, *, budget: int = ARCHIVE_LIMIT) -> dict:
    """Coverage, then seizure/background balance, without assuming compression gain.

    No labels are made. Summary-zero files remain background *candidates* until
    genuine complete-coverage review. EDF byte bounds include a ZIP allowance;
    the writer uses ZIP_STORED if DEFLATE would exceed the admitted bound.
    """
    validate_inventory(inventory)
    if type(budget) is not int or not METADATA_RESERVE < budget <= ARCHIVE_LIMIT:
        raise ValueError("archive_budget_invalid")
    all_records = inventory["recordings"]
    patients = sorted({item["canonical_patient_id"] for item in all_records})
    eligible = [item for item in all_records if item["archive_eligible"]]
    by_patient = {p: [item for item in eligible if item["canonical_patient_id"] == p] for p in patients}
    selected = {}
    # Retain all available public summaries and core provenance, not just selected-case text.
    metadata_paths = {item["path"] for item in inventory["files"] if item["path"].endswith("-summary.txt")}
    metadata_paths.update(("RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO", "ANNOTATORS", "SHA256SUMS.txt"))
    files = {item["path"]: item for item in inventory["files"]}
    source_metadata_bytes = sum(files[path]["bytes"] for path in metadata_paths)
    used = METADATA_RESERVE + source_metadata_bytes

    def cost(item):
        return item["bytes"] + ZIP_OVERHEAD_LIMIT + (files[item["sidecar_path"]]["bytes"] if item["sidecar_path"] else 0)

    def order(item):
        # Raw-size bound first maximizes coverage; positive evidence breaks equal-size ties.
        return (cost(item), not item["official_seizure_positive"], item["path"])

    def add(item, reason):
        nonlocal used
        if item["path"] in selected or used + cost(item) > budget:
            return False
        selected[item["path"]] = {**item, "selection_reason": reason,
                                  "archive_byte_bound": item["bytes"] + ZIP_OVERHEAD_LIMIT,
                                  "actual_archive_bytes": None, "compression_ratio": None}
        used += cost(item)
        if item["sidecar_path"]:
            metadata_paths.add(item["sidecar_path"])
        return True

    minimums = [min(items, key=order) for items in by_patient.values() if items]
    minimum_all_patient_bound = used + sum(cost(item) for item in minimums)
    for item in sorted(minimums, key=order):
        add(item, "minimum_cost_canonical_patient_coverage")
    for flag, reason in (("official_seizure_positive", "add_official_seizure_coverage"),
                         ("verified_background", "add_reviewed_background_coverage"),
                         ("background_candidate", "add_summary_zero_candidate_requires_review")):
        additions = []
        for patient, candidates in by_patient.items():
            if any(item["canonical_patient_id"] == patient and item[flag] for item in selected.values()):
                continue
            matching = [item for item in candidates if item[flag] and item["path"] not in selected]
            if matching:
                additions.append(min(matching, key=order))
        for item in sorted(additions, key=order):
            add(item, reason)
    # Stop after coverage/balance rounds; don't spend the budget on arbitrary extra positives.
    report = []
    for patient in patients:
        official = [item for item in all_records if item["canonical_patient_id"] == patient]
        retained = [item for item in selected.values() if item["canonical_patient_id"] == patient]
        age_records = {item["case_id"]: item["age"] for item in official}
        report.append({"canonical_patient_id": patient, "case_ids": sorted(age_records),
            "official_recordings": len(official), "official_seizure_positive_recordings": sum(item["official_seizure_positive"] for item in official),
            "verified_background_recordings": sum(item["verified_background"] for item in official),
            "summary_zero_candidates": sum(item["background_candidate"] for item in official),
            "total_official_edf_bytes": sum(item["bytes"] for item in official),
            "ages_by_case": age_records, "archive_eligible": bool(by_patient[patient]),
            "age_specific_training_eligible": False,
            "selected_recordings": [item["path"] for item in retained],
            "selected_original_bytes": sum(item["bytes"] for item in retained),
            "selected_archive_byte_bound": sum(cost(item) for item in retained),
            "selected_actual_compressed_bytes": None,
            "selected_declared_seizure_events": sum((item["summary"] or {}).get("declared_seizures") or 0 for item in retained),
            "selected_verified_background_recordings": sum(item["verified_background"] for item in retained),
            "selected_background_candidates": sum(item["background_candidate"] for item in retained),
            "missing_patient_reason": None if retained else "no_annotation_consistent_candidate" if not by_patient[patient] else "budget_insufficient",
            "excluded_recordings": [{"path": item["path"], "reasons": item["annotation_gaps"] or [
                "not_needed_for_coverage_balance_or_budget_bound"]} for item in official if item["path"] not in selected]})
    selected_age_cases = Counter((item["age"] or {}).get("age_group", "unknown")
                                 for item in selected.values())
    selected_age_patients = Counter()
    for patient in patients:
        age_groups = {item["age"]["age_group"] if item["age"] else "unknown"
                      for item in selected.values() if item["canonical_patient_id"] == patient}
        selected_age_patients["/".join(sorted(age_groups)) if age_groups else "unknown"] += bool(age_groups)
    proposal = {"schema_version": "chbmit-archive-selection-v1", "cohort_id": "chbmit-multipatient-2gib-v1",
        "inventory_sha256": inventory["inventory_sha256"], "dataset_version": "1.0.0",
        "policy": "canonical-minimum-cost-coverage-then-positive-and-background-v1",
        "status": "proposed_not_authorized_or_acquired", "budget_bytes": budget,
        "metadata_reserve_bytes": METADATA_RESERVE, "total_retained_byte_bound": used,
        "remaining_bound_bytes": budget - used, "minimum_all_eligible_patient_bound": minimum_all_patient_bound,
        "official_case_count": inventory["case_count"], "official_patient_count": len(patients),
        "selected_patient_count": len({item["canonical_patient_id"] for item in selected.values()}), "selected_recording_count": len(selected),
        "selected_original_edf_bytes": sum(item["bytes"] for item in selected.values()),
        "actual_compressed_bytes": None, "compression_ratio": None,
        "selected_age_group_case_counts": dict(sorted(selected_age_cases.items())),
        "selected_age_group_canonical_patient_counts": dict(sorted(selected_age_patients.items())),
        "selected_official_seizure_positive_recordings": sum(item["official_seizure_positive"] for item in selected.values()),
        "selected_summary_zero_candidates": sum(item["background_candidate"] for item in selected.values()),
        "selected_verified_background_recordings": sum(item["verified_background"] for item in selected.values()),
        "selected_declared_seizure_events": sum((item["summary"] or {}).get("declared_seizures") or 0 for item in selected.values()),
        "human_reconciled_event_count": None, "training_ready": False, "patient_split": None,
        "training_blockers": ["genuine_permissions_and_processing_review_required", "complete_annotation_and_quality_review_required",
                              "age_and_channel_eligibility_review_required", "new_canonical_patient_split_required"],
        "recordings": sorted(selected.values(), key=lambda item: item["path"]),
        "support_files": [files[path] for path in sorted(metadata_paths)], "patients": report,
        "clinical_validation": False, "training_labels_created": False}
    proposal["selection_sha256"] = digest(proposal)
    return proposal


def validate_selection(selection: dict, inventory: dict) -> None:
    expected = select_cohort(inventory, budget=selection.get("budget_bytes"))
    if selection != expected:
        raise ValueError("selection_not_exact_deterministic_inventory_proposal")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-bytes", type=int, default=ARCHIVE_LIMIT)
    args = parser.parse_args(argv)
    result = select_cohort(json.loads(args.inventory.read_text()), budget=args.budget_bytes)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("selected_patient_count", "selected_recording_count", "selected_original_edf_bytes", "total_retained_byte_bound", "remaining_bound_bytes", "selection_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
