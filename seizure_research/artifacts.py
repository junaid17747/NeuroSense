"""Fail-closed loading of retrospective experimental results; no model inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .events import ScoredWindow, evaluate_events, windows_to_events
from .ingestion import CohortManifest
from .preprocessing import PreprocessingConfig
from .splitting import SplitManifest, digest, validate_split_manifest
from .training import PILOT_FILES, PILOT_SPLITS, _jsonable, _sha256_file, validate_pilot_preprocessing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "CHB-MIT Scalp EEG Database v1.0.0"
MAX_JSON_BYTES = 128 * 1024**2


def read_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("artifact_json_duplicate_key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("artifact_json_nonfinite_number")

    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("artifact_json_size_limit_exceeded")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def load_prediction_artifacts(run_dir: Path | str, *, readiness_report: Mapping,
                              approval_record: Mapping) -> dict:
    if readiness_report.get("ready_for_real_data_processing") is not True:
        raise PermissionError("real_prediction_readiness_missing")
    if (approval_record.get("processing_status") != "approved"
            or not all(approval_record.get(key) for key in (
                "processing_approver", "processing_approved_on", "processing_approval_ref"))):
        raise PermissionError("real_prediction_processing_approval_missing")
    requested = Path(run_dir).absolute()
    if any(path.is_symlink() for path in (requested, *requested.parents)):
        raise ValueError("artifact_root_symlink_not_allowed")
    root = requested.resolve(strict=True)
    if root.is_relative_to(PROJECT_ROOT):
        raise ValueError("research_artifacts_must_be_outside_repository")

    def artifact(relative):
        path = root / relative
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("artifact_symlink_not_allowed")
        path = path.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("artifact_path_outside_run_root")
        return path

    # Missing files deliberately propagate as unavailable, never placeholder results.
    model_hash = _sha256_file(artifact("model/xgboost.json"))
    provenance = read_json(artifact("model/provenance.json"))
    prediction = read_json(artifact("test_predictions.json"))
    evaluation = read_json(artifact("test_evaluation.json"))
    try:
        cohort = CohortManifest.model_validate(read_json(artifact("cohort_manifest.json")))
        split = SplitManifest.model_validate(read_json(artifact("split_manifest.json")))
        config = PreprocessingConfig.model_validate(read_json(artifact("preprocessing.json")))
        features = read_json(artifact("feature_manifest.json"))
        selection = read_json(artifact("model/validation_selection.json"))
        if not all(isinstance(value, dict) for value in (
                provenance, prediction, evaluation, features, selection)):
            raise ValueError("real_prediction_artifact_structure_invalid")
        checks = readiness_report["checks"]
        approved_root = Path(checks["approved_storage"]["root"]).resolve(strict=True)
        if not root.is_relative_to(approved_root):
            raise ValueError("artifact_run_outside_approved_storage")
        if (cohort.sha256 != checks["cohort_manifest"]["sha256"]
                or cohort.sha256 != approval_record["cohort_manifest_sha256"]
                or provenance["cohort_manifest_sha256"] != cohort.sha256
                or provenance["file_manifest_sha256"] != checks["file_manifest"]["sha256"]
                or provenance["approval_record_sha256"] != digest(dict(approval_record))):
            raise ValueError("artifact_cohort_or_approval_binding_mismatch")
        records = tuple(item.recording for item in cohort.records)
        validate_split_manifest(records, split)
        validate_pilot_preprocessing(config)
        if {item.signal.relative_path for item in cohort.records} != PILOT_FILES:
            raise ValueError("artifact_cohort_not_frozen_pilot")
        assignments = {item.recording_id: item for item in split.assignments}
        for entry in cohort.records:
            if assignments[entry.recording.recording_id].split != PILOT_SPLITS[entry.recording.subject_id]:
                raise ValueError("artifact_patient_partition_mismatch")
        threshold = provenance["selected_validation_threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("artifact_validation_threshold_invalid")
        if (provenance["schema_version"] != "seizure-xgb-model-v1"
                or provenance["clinical_validation"] is not False
                or provenance["model_sha256"] != model_hash
                or provenance["split_manifest_sha256"] != split.sha256
                or provenance["scaler_sha256"] != _sha256_file(artifact("model/scaler.npz"))
                or provenance["validation_selection_sha256"] != _sha256_file(artifact("model/validation_selection.json"))
                or selection["selected_threshold"] != threshold
                or selection["best_iteration"] != provenance["best_iteration"]
                or provenance["feature_manifest_sha256"] != digest(features)
                or provenance["preprocessing_config_sha256"] != digest(config.model_dump(mode="json"))
                or features["cohort_manifest_sha256"] != cohort.sha256
                or features["split_manifest_sha256"] != split.sha256
                or features["preprocessing_config_sha256"] != provenance["preprocessing_config_sha256"]
                or features["annotation_evidence_sha256"] != provenance["annotation_evidence_sha256"]):
            raise ValueError("real_prediction_provenance_missing_or_inconsistent")
        validate_event_artifacts(prediction, evaluation, records, split, threshold, model_hash, cohort.sha256)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("real_prediction_artifact_structure_invalid") from exc
    return {"predictions": prediction, "model_provenance": provenance, "evaluation": evaluation}


def validate_event_artifacts(prediction, evaluation, records, split, threshold, model_hash, cohort_sha256):
    """Shared pilot/archive result validation against reviewed test annotations."""
    assignments = {item.recording_id: item for item in split.assignments}
    shared = {"model_sha256": model_hash, "cohort_manifest_sha256": cohort_sha256,
              "split_manifest_sha256": split.sha256, "validation_threshold": threshold,
              "threshold_selected_on": "validation", "evaluation_partition": "test",
              "clinical_validation": False}
    for payload, version in ((prediction, "seizure-window-predictions-v1"),
                             (evaluation, "seizure-event-evaluation-v1")):
        if (payload["schema_version"] != version or payload["dataset"] != DATASET
                or any(payload["provenance"].get(key) != value for key, value in shared.items())
                or payload["provenance"].get("clinical_validation") is not False):
            raise ValueError("artifact_evaluation_provenance_mismatch")
    evaluation_hash = digest({key: value for key, value in evaluation.items()
                              if key != "evaluation_artifact_sha256"})
    if (evaluation["clinical_validation"] is not False
            or prediction["split"] != "test"
            or evaluation["evaluation_artifact_sha256"] != evaluation_hash
            or prediction["provenance"]["evaluation_artifact_sha256"] != evaluation_hash
            or prediction["provenance"]["prediction_artifact_sha256"] != digest({
                "windows": prediction["windows"], "candidate_events": prediction["candidate_events"]})):
        raise ValueError("real_prediction_artifact_hash_mismatch")
    test_records = tuple(item for item in records if assignments[item.recording_id].split == "test")
    by_id = {item.recording_id: item for item in test_records}
    if not isinstance(prediction["windows"], list) or not prediction["windows"]:
        raise ValueError("real_prediction_windows_missing")
    scored = []
    for row in prediction["windows"]:
        record = by_id.get(row["recording_id"])
        if (record is None or row["patient_group_id"] != record.patient_group_id
                or row["split"] != "test"
                or any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
                       for key in ("start_seconds", "end_seconds", "probability"))
                or not 0 <= row["start_seconds"] < row["end_seconds"] <= record.duration_seconds
                or not 0 <= row["probability"] <= 1):
            raise ValueError("real_prediction_window_invalid")
        scored.append(ScoredWindow(**row))
    candidates = windows_to_events(scored, threshold_on=threshold)
    if digest(_jsonable(candidates)) != digest(prediction["candidate_events"]):
        raise ValueError("artifact_candidates_disagree_with_window_scores")
    metrics = evaluate_events(test_records, scored, candidates)
    if digest(_jsonable(metrics)) != digest(evaluation["metrics"]):
        raise ValueError("artifact_metrics_disagree_with_predictions_and_annotations")
