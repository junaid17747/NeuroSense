"""Archive-backed multipatient CHB-MIT workflow, separate from the frozen pilot.

Reuses the reviewed ingestion adapter, bounded feature extraction, XGBoost
trainer and event evaluator. No command downloads data or creates approval.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from .adapters.base import IngestionPlan
from .adapters.chb_mit import CHBMITAdapter
from .annotation_review import audit_review_packet, parse_seizure_sidecar
from .archive import (ArchiveBudget, _inputs, authorize, extracted_recording,
                      file_hash, read_json, safe_file, safe_root, verify_archive, verify_original)
from .cohort import validate_proposed_split
from .events import ScoredWindow, evaluate_events
from .features import FEATURE_VERSION, feature_names, iter_verified_recording_windows, write_feature_rows
from .ingestion import CohortManifest, normalized_plan
from .inventory import canonical_patient
from .preprocessing import PreprocessingConfig
from .schemas import TimeInterval
from .splitting import SplitManifest, build_explicit_split_manifest, checked_recordings, digest, validate_split_manifest
from .training import (DiskBudget, PILOT_CHANNELS, PilotRunConfig, _concat_feature_tables,
                       _jsonable, _write_budgeted_json, build_test_artifact_payloads,
                       evaluate_xgboost_test, load_feature_table, train_xgboost)
from .waveform import EDFStreamReader

COHORT_KIND = "chbmit-multipatient-archive-v1"
DEFAULT_RUN = PilotRunConfig(experiment_id="chbmit-multipatient-xgb-v1")


def _intervals(value) -> tuple[TimeInterval, ...]:
    if not isinstance(value, list):
        raise ValueError("explicit_reviewed_intervals_required")
    return tuple(TimeInterval.model_validate(v if isinstance(v, dict) else {
        "start_seconds": v[0], "end_seconds": v[1]}) for v in value)


def validate_review_inputs(inventory, selection, proposal, plan, packet, configs):
    """Validate human inputs without decoding EEG or inferring missing labels."""
    validate_proposed_split(inventory, selection, proposal)
    plan = normalized_plan(IngestionPlan.model_validate(plan))
    audit = audit_review_packet(inventory, selection, packet)
    if not audit["annotation_evidence_complete"]:
        raise PermissionError("annotation_review_packet_incomplete")
    rows = {r["recording_id"]: r for r in packet["recordings"]}
    if any(r.get("eligibility_status") not in ("eligible", "excluded") for r in rows.values()):
        raise PermissionError("explicit_recording_eligibility_decisions_required")
    eligible = {k for k, r in rows.items() if r["eligibility_status"] == "eligible"}
    if set(x.edf_path for x in plan.recordings) != eligible or set(configs) != eligible:
        raise ValueError("review_plan_and_preprocessing_must_cover_exact_eligible_cohort")
    selection_by_id = {r["path"]: r for r in selection["recordings"]}
    identities = {r.subject_id: r.canonical_patient_id for r in plan.identities.entries}
    if any(patient != canonical_patient(case) for case, patient in identities.items()):
        raise ValueError("canonical_patient_identity_mismatch")
    if not plan.dataset_review.verified or plan.permissions.status != "approved" or not plan.identities.review.verified:
        raise PermissionError("dataset_and_identity_review_required")
    prepared = {}
    for item in plan.recordings:
        selected, row = selection_by_id[item.edf_path], rows[item.edf_path]
        if (item.subject_id != selected["case_id"]
                or item.expected_signal_sha256 != selected["sha256"]
                or item.expected_summary_sha256 != row["summary_sha256"]
                or item.summary_path != row["summary_path"]
                or not item.metadata_review.verified or not item.annotation_review.verified
                or item.annotation_review.evidence_ref != row["evidence_ref"]
                or item.quality.status != "acceptable" or item.annotation_coverage != "complete"):
            raise ValueError("review_plan_source_or_review_binding_mismatch")
        published = selected.get("age") or {}
        # Missing published age is not imputed. A revised, independently reviewed
        # source/selection is needed to admit a case absent from SUBJECT-INFO.
        if published.get("age_years") is None:
            raise ValueError("unknown_published_age_requires_explicit_exclusion")
        if (item.age.completed_years != math.floor(published["age_years"])
                or item.age.group.value != published["age_group"]):
            raise ValueError("reviewed_age_conflicts_with_published_metadata")
        config = PreprocessingConfig.model_validate(configs[item.edf_path])
        if (tuple(c.target_label for c in config.channels) != PILOT_CHANNELS
                or item.montage.channels != PILOT_CHANNELS
                or item.montage.kind != "bipolar" or config.reference_mode != "none"
                or config.target_sampling_rate_hz != 256 or config.allow_resampling
                or config.age_group != item.age.group
                or config.window_seconds != 5 or config.step_seconds != 1
                or config.quality_thresholds.window_seconds != 5
                or config.quality_thresholds.step_seconds != 1):
            raise ValueError("multipatient_preprocessing_profile_or_montage_mismatch")
        if tuple((c.source_index, c.source_label, c.target_label) for c in config.channels) != tuple(
                (c.source_index, c.source_label, c.output_label) for c in item.channels if c.action == "keep"):
            raise ValueError("preprocessing_differs_from_reviewed_channel_selection")
        events = _intervals(row["reviewed_event_intervals_seconds"])
        negatives = _intervals(row.get("reviewed_seizure_free_intervals_seconds"))
        exclusions = _intervals(row.get("excluded_intervals_seconds"))
        if exclusions != item.quality.excluded_intervals:
            raise ValueError("reviewed_quality_exclusions_mismatch")
        # An explicit partition of the whole recording prevents unknown gaps
        # from becoming negative training labels or metric denominators.
        intervals = sorted((*events, *negatives, *exclusions), key=lambda v: v.start_seconds)
        cursor = 0.0
        for interval in intervals:
            if abs(interval.start_seconds - cursor) > 1e-6:
                raise ValueError("reviewed_interval_partition_has_gap_or_overlap")
            cursor = interval.end_seconds
        if abs(cursor - row["recording_duration_seconds"]) > 1e-6:
            raise ValueError("reviewed_interval_partition_incomplete")
        prepared[item.edf_path] = config
    return plan, prepared


def authorize_training(root, temporary_root, approval_path, inventory, selection,
                       proposal, plan, packet, configs, run_config):
    approval = authorize(root, temporary_root, approval_path, inventory, selection, processing=True)
    if (approval.training_status != "approved" or not all((approval.training_approver,
            approval.training_approved_on, approval.training_approval_ref, approval.training_runs_root))):
        raise PermissionError("separate_model_training_authorization_required")
    bindings = {
        "training_plan_sha256": digest(plan.model_dump(mode="json")),
        "training_split_sha256": proposal["split_proposal_sha256"],
        "training_preprocessing_sha256": digest({p: c.model_dump(mode="json") for p, c in configs.items()}),
        "training_review_packet_sha256": packet["packet_sha256"],
        "training_run_config_sha256": digest(run_config.model_dump(mode="json")),
    }
    if any(getattr(approval, k) != v for k, v in bindings.items()):
        raise PermissionError("training_authorization_input_binding_mismatch")
    runs_root = safe_root(approval.training_runs_root)
    archive_root, temporary = safe_root(root), safe_root(temporary_root)
    if any(runs_root.is_relative_to(p) or p.is_relative_to(runs_root) for p in (archive_root, temporary)):
        raise ValueError("training_artifacts_must_be_separate_from_archive_and_temporary_storage")
    from .readiness import EPHEMERAL_ROOTS
    if any(runs_root.is_relative_to(p) for p in EPHEMERAL_ROOTS):
        raise ValueError("training_runs_require_persistent_external_storage")
    return approval, runs_root


@contextlib.contextmanager
def reviewed_archive_source(root, temporary_root, inventory, selection, approval_path, item, plan):
    """Extract one verified EDF and copy only its reviewed support metadata."""
    with extracted_recording(root, temporary_root, inventory, selection, approval_path, item.edf_path) as edf:
        source_root = edf.parent.parent
        approval = authorize(root, temporary_root, approval_path, inventory, selection, processing=True)
        budget = ArchiveBudget(root, temporary_root, temporary_limit=approval.temporary_byte_limit)
        files = {r["path"]: r for r in selection["support_files"]}
        paths = [item.summary_path]
        sidecar = item.edf_path + ".seizures"
        if sidecar in files:
            paths.append(sidecar)
        for relative in paths:
            if relative not in files:
                raise ValueError("reviewed_annotation_source_not_in_selection")
            src = safe_file(safe_root(root), "sources/" + relative)
            verify_original(src, files[relative])
            budget.check(temporary_add=files[relative]["bytes"])
            dest = safe_file(source_root, relative)
            dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(src, dest)
            os.chmod(dest, 0o600)
            verify_original(dest, files[relative])
        entry = CHBMITAdapter().ingest(source_root, item, plan)
        if entry.recording.annotations:
            with safe_file(source_root, sidecar).open("rb") as stream:
                parsed = _intervals(parse_seizure_sidecar(
                    stream.read(1024**2 + 1), sampling_rate_hz=entry.recording.sampling_rate_hz,
                    duration_seconds=entry.recording.duration_seconds))
            if (len(parsed) != len(entry.recording.annotations)
                    or any(abs(a.start_seconds - b.start_seconds) > .5
                           or abs(a.end_seconds - b.end_seconds) > .5
                           for a, b in zip(parsed, entry.recording.annotations))):
                raise ValueError("sidecar_summary_event_disagreement")
        yield source_root, entry


def _validate_annotation_binding(entry, row):
    reviewed = _intervals(row["reviewed_event_intervals_seconds"])
    recording = entry.recording
    if (recording.duration_seconds != row["recording_duration_seconds"]
            or len(reviewed) != len(recording.annotations)
            or any(abs(a.start_seconds - b.start_seconds) > .5 or abs(a.end_seconds - b.end_seconds) > .5
                   for a, b in zip(reviewed, recording.annotations))):
        raise ValueError("reviewed_events_or_duration_disagree_with_verified_source")


def evaluation_details(records, scored, candidates, *, seed=42):
    """Recording/age coverage and patient-bootstrap intervals, never window CIs."""
    per_recording = {}
    per_age = {}
    for record in records:
        local_windows = tuple(w for w in scored if w.recording_id == record.recording_id)
        local_events = tuple(e for e in candidates if e.recording_id == record.recording_id)
        per_recording[record.recording_id] = _jsonable(evaluate_events((record,), local_windows, local_events))
    for age in sorted({r.age.group.value for r in records}):
        local = tuple(r for r in records if r.age.group.value == age)
        ids = {r.recording_id for r in local}
        per_age[age] = {"patients": len({r.patient_group_id for r in local}),
                        "recordings": len(local), "age_specific_model_validated": False,
                        "metrics": _jsonable(evaluate_events(local, tuple(w for w in scored if w.recording_id in ids),
                                                              tuple(e for e in candidates if e.recording_id in ids)))}
    aggregate = evaluate_events(records, scored, candidates)
    patients = list(aggregate.per_patient.values())
    ci = {"method": "canonical-patient bootstrap percentile 95%; exploratory",
          "seed": seed, "replicates": 2000, "patients": len(patients), "intervals": {}}
    samples = {key: [] for key in ("event_sensitivity", "false_alarms_per_hour", "median_detection_latency_seconds")}
    if len(patients) >= 3:
        rng = np.random.default_rng(seed)
        for _ in range(ci["replicates"]):
            group = [patients[i] for i in rng.integers(0, len(patients), len(patients))]
            ref, found = sum(p.reference_events for p in group), sum(p.detected_events for p in group)
            hours, false = sum(p.evaluable_hours for p in group), sum(p.false_events for p in group)
            latency = [x for p in group for x in p.trigger_latencies_seconds]
            if ref: samples["event_sensitivity"].append(found / ref)
            if hours: samples["false_alarms_per_hour"].append(false / hours)
            if latency: samples["median_detection_latency_seconds"].append(float(np.median(latency)))
    for name, values in samples.items():
        ci["intervals"][name] = {"bounds": np.quantile(values, [.025, .975]).tolist() if values else None,
                                 "valid_replicates": len(values)}
    return {"per_recording": per_recording, "per_age": per_age, "confidence_intervals": ci,
            "test_patients": len(patients), "test_recordings": len(records),
            "reviewed_recording_hours": sum(r.usable_seconds for r in records) / 3600,
            "quality_exclusions": {r.recording_id: [e.model_dump(mode="json") for e in r.quality.excluded_intervals]
                                   for r in records},
            "matching_policy": "One-to-one, event overlap with trigger inside the reference interval; no onset tolerance. "
                               "Reviewed reference seizures lost to algorithmic quality abstention remain misses. "
                               "False-alarm denominator is the union of evaluable non-seizure windows after warmup."}


def run_multipatient(root, temporary_root, inventory, selection, *, approval_path,
                     proposal, review_plan, review_packet, preprocessing, output_dir,
                     confirmed=False, run_config=DEFAULT_RUN):
    if not confirmed:
        raise PermissionError("explicit_multipatient_training_confirmation_required")
    inventory, selection = _inputs(inventory, selection)
    run_config = PilotRunConfig.model_validate(run_config)
    plan, configs = validate_review_inputs(inventory, selection, proposal, review_plan, review_packet, preprocessing)
    approval, runs_root = authorize_training(root, temporary_root, approval_path, inventory, selection,
                                            proposal, plan, review_packet, configs, run_config)
    integrity = verify_archive(root, inventory, selection)
    if integrity["status"] != "complete_verified":
        raise PermissionError("complete_verified_archive_required")
    output = safe_root(output_dir)
    if output.exists() or output.parent != runs_root:
        raise ValueError("new_run_directory_under_approved_runs_root_required")
    runs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    budget = DiskBudget(runs_root)
    budget.check(100 * 1024**2)
    rows = {r["recording_id"]: r for r in review_packet["recordings"]}
    entries = []
    for item in plan.recordings:
        with reviewed_archive_source(root, temporary_root, inventory, selection, approval_path, item, plan) as (_, entry):
            _validate_annotation_binding(entry, rows[item.edf_path])
            entries.append(entry)
    cohort = CohortManifest(adapter_id="chb_mit", adapter_version=CHBMITAdapter.adapter_version,
                            plan_sha256=digest(plan.model_dump(mode="json")), review_plan=plan, records=tuple(entries))
    records = checked_recordings(e.recording for e in entries)
    membership = {r["canonical_patient_id"]: r["split"] for r in proposal["assignments"]}
    split = build_explicit_split_manifest(records, {r.patient_group_id: membership[r.patient_group_id] for r in records},
                                          seed=proposal["seed"])
    assignments = {a.recording_id: a.split for a in split.assignments}
    for name in ("train", "validation", "test"):
        group = [r for r in records if assignments[r.recording_id] == name]
        if not any(r.annotations for r in group) or not any(not r.annotations for r in group):
            raise ValueError("every_partition_requires_reviewed_positive_and_background_recordings")
    output.mkdir(mode=0o700)
    shards = {}
    old_umask = os.umask(0o077)
    try:
        def write(name, payload):
            _write_budgeted_json(output / name, payload, budget)
        write("cohort_manifest.json", cohort.model_dump(mode="json"))
        write("split_manifest.json", split.model_dump(mode="json"))
        write("split_proposal.json", proposal)
        write("preprocessing.json", {p: c.model_dump(mode="json") for p, c in configs.items()})
        write("review_packet.json", review_packet)
        write("run_config.json", run_config.model_dump(mode="json"))
        # Extract test windows only after the train/validation model is frozen.
        def extract_partition(name):
            for item, entry in zip(plan.recordings, entries):
                if assignments[entry.recording.recording_id] != name:
                    continue
                authorize_training(root, temporary_root, approval_path, inventory, selection,
                                   proposal, plan, review_packet, configs, run_config)
                with reviewed_archive_source(root, temporary_root, inventory, selection, approval_path, item, plan) as (source, actual):
                    if actual != entry:
                        raise ValueError("reviewed_source_changed_before_feature_extraction")
                    with EDFStreamReader(source, item.edf_path) as reader:
                        windows = iter_verified_recording_windows(
                            reader, cohort, entry, configs[item.edf_path], split,
                            chunk_seconds=run_config.chunk_seconds, halo_seconds=run_config.halo_seconds,
                            max_samples=run_config.max_decoded_samples, space_check=budget.check)
                        shard = output / "features" / name / (Path(item.edf_path).stem + ".parquet")
                        shards[entry.recording.recording_id] = write_feature_rows(
                            windows, shard, feature_schema=feature_names(18), space_check=budget.check)
            tables = [load_feature_table(v["path"], split=name, manifest=cohort, split_manifest=split)
                      for k, v in shards.items() if assignments[k] == name]
            return _concat_feature_tables(tables, name)
        train, validation = extract_partition("train"), extract_partition("validation")
        run = train_xgboost(train, validation, recordings=records, split_manifest=split,
                            cohort_sha256=cohort.sha256, annotation_evidence_sha256=review_packet["packet_sha256"],
                            output_dir=output / "model", config=run_config, storage=budget, authorized=True)
        frozen = {"model_sha256": run["provenance"]["model_sha256"],
                  "scaler_sha256": run["provenance"]["scaler_sha256"],
                  "validation_selection_sha256": run["provenance"]["validation_selection_sha256"],
                  "threshold": run["threshold"], "split_sha256": split.sha256,
                  "frozen_before_test_extraction_at_utc": datetime.now(timezone.utc).isoformat()}
        write("test_freeze.json", frozen)
        del train, validation
        test = extract_partition("test")
        test_result = evaluate_xgboost_test(run, test, recordings=records, split_manifest=split)
        predictions, evaluation = build_test_artifact_payloads(test_result)
        test_records = tuple(r for r in records if assignments[r.recording_id] == "test")
        scored = tuple(ScoredWindow(**w) for w in predictions["windows"])
        details = evaluation_details(test_records, scored, test_result["candidates"], seed=run_config.seed)
        write("test_predictions.json", predictions)
        write("test_evaluation.json", evaluation)
        write("evaluation_details.json", details)
        feature_manifest = {"cohort_kind": COHORT_KIND, "feature_version": FEATURE_VERSION,
                            "feature_names": list(feature_names(18)), "cohort_manifest_sha256": cohort.sha256,
                            "split_manifest_sha256": split.sha256, "recordings": shards}
        write("feature_manifest.json", feature_manifest)
        run["provenance"].update({"cohort_kind": COHORT_KIND,
                                  "inventory_sha256": inventory["inventory_sha256"],
                                  "selection_sha256": selection["selection_sha256"],
                                  "split_proposal_sha256": proposal["split_proposal_sha256"],
                                  "feature_manifest_sha256": digest(feature_manifest),
                                  "preprocessing_config_sha256": digest({p: c.model_dump(mode="json")
                                                                          for p, c in configs.items()}),
                                  "approval_record_sha256": digest(read_json(Path(approval_path))),
                                  "evaluation_details_sha256": digest(details),
                                  "age_specific_validation": False})
        provenance_path = output / "model/provenance.json"
        data = json.dumps(run["provenance"], sort_keys=True, indent=2, allow_nan=False) + "\n"
        budget.check(len(data.encode()))
        provenance_path.write_text(data)
        # Completion is the final exclusive write; interrupted runs stay unavailable.
        files = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
        write("complete.json", {"schema_version": COHORT_KIND, "status": "complete",
                                "files_sha256": files, "bundle_sha256": digest(files),
                                "clinical_validation": False})
        return {"status": "complete", "run_dir": str(output), "clinical_validation": False,
                "patients": len({r.patient_group_id for r in records}), "recordings": len(records),
                "metrics": evaluation["metrics"]}
    finally:
        os.umask(old_umask)


def load_multipatient_artifacts(run_dir, *, root, temporary_root, inventory, selection,
                                approval_path, review_path):
    """Read a completed, source-bound archive experiment; never execute a model."""
    from .artifacts import read_json as artifact_json, validate_event_artifacts
    inventory, selection = _inputs(inventory, selection)
    run_dir = safe_root(run_dir)
    complete_path = safe_file(run_dir, "complete.json")
    before = complete_path.stat()
    complete = artifact_json(complete_path)
    files = complete.get("files_sha256")
    if (complete.get("schema_version") != COHORT_KIND or complete.get("status") != "complete"
            or complete.get("clinical_validation") is not False or not isinstance(files, dict)
            or complete.get("bundle_sha256") != digest(files)):
        raise ValueError("multipatient_completion_record_invalid")
    required = {"cohort_manifest.json", "split_manifest.json", "split_proposal.json", "preprocessing.json",
                "review_packet.json", "run_config.json", "test_freeze.json", "feature_manifest.json",
                "test_predictions.json", "test_evaluation.json", "evaluation_details.json",
                "model/xgboost.json", "model/scaler.npz", "model/provenance.json", "model/validation_selection.json"}
    if not required <= set(files):
        raise ValueError("multipatient_bundle_incomplete")
    for relative, checksum in files.items():
        if file_hash(safe_file(run_dir, relative)) != checksum:
            raise ValueError("multipatient_bundle_checksum_mismatch")
    def read(name):
        return artifact_json(safe_file(run_dir, name))
    cohort = CohortManifest.model_validate(read("cohort_manifest.json"))
    proposal, packet, preprocessing = read("split_proposal.json"), read("review_packet.json"), read("preprocessing.json")
    if packet != read_json(Path(review_path)):
        raise ValueError("current_annotation_review_differs_from_training_evidence")
    plan, configs = validate_review_inputs(inventory, selection, proposal, cohort.review_plan, packet, preprocessing)
    config = PilotRunConfig.model_validate(read("run_config.json"))
    approval, runs_root = authorize_training(root, temporary_root, approval_path, inventory, selection,
                                            proposal, plan, packet, configs, config)
    if verify_archive(root, inventory, selection)["status"] != "complete_verified":
        raise ValueError("multipatient_source_archive_not_complete")
    if run_dir.parent != runs_root or cohort.plan_sha256 != digest(plan.model_dump(mode="json")):
        raise ValueError("multipatient_run_storage_or_plan_mismatch")
    split = SplitManifest.model_validate(read("split_manifest.json"))
    records = checked_recordings(e.recording for e in cohort.records)
    validate_split_manifest(records, split)
    membership = {r["canonical_patient_id"]: r["split"] for r in proposal["assignments"]}
    expected = build_explicit_split_manifest(records, {r.patient_group_id: membership[r.patient_group_id] for r in records},seed=proposal["seed"])
    if split != expected:
        raise ValueError("multipatient_split_differs_from_frozen_proposal")
    by_path = {r["path"]: r for r in selection["recordings"]}
    review_rows = {r["recording_id"]: r for r in packet["recordings"]}
    plan_rows = {r.edf_path: r for r in plan.recordings}
    if set(e.signal.relative_path for e in cohort.records) != set(plan_rows):
        raise ValueError("multipatient_cohort_coverage_mismatch")
    for entry in cohort.records:
        r, item = entry.recording, plan_rows[entry.signal.relative_path]
        source = by_path[entry.signal.relative_path]
        if (entry.signal.sha256 != source["sha256"] or entry.signal.size_bytes != source["bytes"]
                or r.provenance.signal_sha256 != source["sha256"]
                or r.patient_group_id != canonical_patient(r.subject_id)
                or r.age != item.age or r.quality != item.quality or r.montage != item.montage
                or r.permissions != plan.permissions
                or entry.recording_review_sha256 != digest(item.model_dump(mode="json"))):
            raise ValueError("multipatient_recording_source_binding_mismatch")
        _validate_annotation_binding(entry, review_rows[entry.signal.relative_path])
    provenance, feature_manifest, frozen = read("model/provenance.json"), read("feature_manifest.json"), read("test_freeze.json")
    validation = read("model/validation_selection.json")
    threshold = provenance.get("selected_validation_threshold")
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("multipatient_threshold_invalid")
    bindings = {"cohort_kind": COHORT_KIND, "inventory_sha256": inventory["inventory_sha256"],
                "selection_sha256": selection["selection_sha256"], "cohort_manifest_sha256": cohort.sha256,
                "split_manifest_sha256": split.sha256, "split_proposal_sha256": proposal["split_proposal_sha256"],
                "model_sha256": files["model/xgboost.json"], "scaler_sha256": files["model/scaler.npz"],
                "validation_selection_sha256": files["model/validation_selection.json"],
                "feature_manifest_sha256": digest(feature_manifest), "preprocessing_config_sha256": digest(preprocessing),
                "annotation_evidence_sha256": packet["packet_sha256"],
                "approval_record_sha256": digest(read_json(Path(approval_path))), "clinical_validation": False}
    if (any(provenance.get(k) != v for k,v in bindings.items())
            or frozen["model_sha256"] != bindings["model_sha256"]
            or frozen["scaler_sha256"] != bindings["scaler_sha256"]
            or frozen["threshold"] != threshold or frozen["split_sha256"] != split.sha256
            or validation["selected_threshold"] != threshold or validation["best_iteration"] != provenance["best_iteration"]):
        raise ValueError("multipatient_model_provenance_mismatch")
    prediction, evaluation, details = read("test_predictions.json"), read("test_evaluation.json"), read("evaluation_details.json")
    validate_event_artifacts(prediction,evaluation,records,split,threshold,bindings["model_sha256"],cohort.sha256)
    from .events import windows_to_events
    scored = tuple(ScoredWindow(**w) for w in prediction["windows"])
    test_ids = {a.recording_id for a in split.assignments if a.split == "test"}
    expected_details = evaluation_details(tuple(r for r in records if r.recording_id in test_ids),
                                           scored,windows_to_events(scored,threshold_on=threshold),seed=config.seed)
    if details != expected_details or provenance["evaluation_details_sha256"] != digest(details):
        raise ValueError("multipatient_evaluation_details_mismatch")
    after = complete_path.stat()
    if (before.st_ino,before.st_mtime_ns,before.st_size) != (after.st_ino,after.st_mtime_ns,after.st_size):
        raise ValueError("multipatient_artifacts_changed_during_read")
    return {"predictions": prediction,"evaluation": evaluation,"model_provenance": provenance,
            "evaluation_details":details,"cohort":cohort,"preprocessing":configs}


def read_archive_waveform(root, temporary_root, inventory, selection, *, approval_path,
                          proposal, review_plan, review_packet, preprocessing,
                          relative_path, channel_labels, start_seconds, end_seconds):
    """Load at most 30 seconds/4 reviewed channels without requiring a model."""
    if (not math.isfinite(start_seconds) or not math.isfinite(end_seconds)
            or start_seconds < 0 or not 0 < end_seconds-start_seconds <= 30
            or not 1 <= len(channel_labels) <= 4 or len(set(channel_labels)) != len(channel_labels)):
        raise ValueError("archive_waveform_bounds_invalid")
    inventory, selection = _inputs(inventory, selection)
    authorize(root,temporary_root,approval_path,inventory,selection,processing=True)
    plan,configs = validate_review_inputs(inventory,selection,proposal,review_plan,review_packet,preprocessing)
    item = next((i for i in plan.recordings if i.edf_path == relative_path),None)
    if item is None:
        raise ValueError("archive_waveform_recording_not_eligible")
    mappings = {c.target_label:c.source_index for c in configs[relative_path].channels}
    if any(c not in mappings for c in channel_labels):
        raise ValueError("archive_waveform_unreviewed_channel")
    row = next(r for r in review_packet["recordings"] if r["recording_id"] == relative_path)
    with reviewed_archive_source(root,temporary_root,inventory,selection,approval_path,item,plan) as (source,entry):
        _validate_annotation_binding(entry,row)
        with EDFStreamReader(source,item.edf_path) as reader:
            data = reader.read_range(start_seconds,end_seconds,channel_indices=tuple(mappings[c] for c in channel_labels),max_samples=2_000_000)
    return {"time_seconds":data.channels[0].timestamps_seconds,
            "samples_uv":np.column_stack([c.samples_uv for c in data.channels]),
            "channel_labels":tuple(channel_labels), "annotations":entry.recording.annotations,
            "recording_id":entry.recording.recording_id,"duration_seconds":entry.recording.duration_seconds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--temporary-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--approval-record", required=True, type=Path)
    parser.add_argument("--split-proposal", required=True, type=Path)
    parser.add_argument("--review-plan", required=True, type=Path)
    parser.add_argument("--review-packet", required=True, type=Path)
    parser.add_argument("--preprocessing", required=True, type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-authorized-training", action="store_true")
    args = parser.parse_args(argv)
    try:
        missing = [name for name in ("inventory", "selection", "approval_record", "split_proposal",
                                     "review_plan", "review_packet", "preprocessing") if not getattr(args, name).is_file()]
        if missing:
            raise PermissionError("missing_required_inputs:" + ",".join(missing))
        inventory, selection = _inputs(args.inventory, args.selection)
        proposal, plan, packet, configs = (read_json(p) for p in
                                          (args.split_proposal, args.review_plan, args.review_packet, args.preprocessing))
        run_config = PilotRunConfig.model_validate(read_json(args.run_config)) if args.run_config else DEFAULT_RUN
        if args.command == "check":
            plan, prepared = validate_review_inputs(inventory, selection, proposal, plan, packet, configs)
            authorize_training(args.root, args.temporary_root, args.approval_record, inventory, selection,
                               proposal, plan, packet, prepared, run_config)
            result = verify_archive(args.root, inventory, selection)
            if result["status"] != "complete_verified":
                raise PermissionError("complete_verified_archive_required")
            print(json.dumps({"status": "inputs_verified_no_training", "archive": result}))
        else:
            if not args.output_dir:
                raise ValueError("output_directory_required")
            result = run_multipatient(args.root, args.temporary_root, inventory, selection,
                                      approval_path=args.approval_record, proposal=proposal,
                                      review_plan=plan, review_packet=packet, preprocessing=configs,
                                      output_dir=args.output_dir, confirmed=args.confirm_authorized_training,
                                      run_config=run_config)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, PermissionError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
