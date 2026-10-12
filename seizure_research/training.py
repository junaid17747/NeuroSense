"""Approved-data seizure research pipeline; never downloads or claims clinical use.

CHB-MIT annotation policy: summary time ranges are the current adapter's primary
structured annotations. A seizure file is trainable only when a reviewer has
reconciled its summary with the exact `.edf.seizures` sidecar bytes and attested
the reviewed event intervals. The sidecar format is intentionally not guessed
or auto-parsed here. A zero-seizure summary alone does not certify background:
the reviewer must attest complete, full-record review and the exact recording
duration. Any missing evidence, source-hash change, count mismatch, or timing
difference over half a second blocks the cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import scipy
import sklearn
import xgboost as xgb
from pydantic import Field, StrictBool
from sklearn.preprocessing import StandardScaler

from .events import ScoredWindow, evaluate_events, select_validation_threshold, windows_to_events
from .features import (
    FEATURE_VERSION, extract_approved_recording_features, feature_names,
)
from .ingestion import CohortManifest, load_plan, validate_manifest
from .preprocessing import PreprocessingConfig
from .schemas import EEGRecording, ResearchSchema, SHA256, Text
from .splitting import (
    SplitManifest, build_explicit_split_manifest, digest, validate_split_manifest,
)
from .waveform import _safe_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PILOT_SPLITS = {
    "chb02": "train", "chb05": "train", "chb08": "train",
    "chb09": "validation", "chb06": "test", "chb10": "test",
}
PILOT_FILES = {
    "chb02/chb02_01.edf", "chb02/chb02_16.edf",
    "chb05/chb05_01.edf", "chb05/chb05_06.edf",
    "chb08/chb08_02.edf", "chb08/chb08_03.edf",
    "chb09/chb09_01.edf", "chb09/chb09_06.edf",
    "chb06/chb06_01.edf", "chb06/chb06_02.edf",
    "chb10/chb10_01.edf", "chb10/chb10_12.edf",
}
PILOT_SEIZURE_FILES = {
    "chb02/chb02_16.edf", "chb05/chb05_06.edf", "chb08/chb08_02.edf",
    "chb09/chb09_06.edf", "chb06/chb06_01.edf", "chb10/chb10_12.edf",
}
PILOT_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3",
    "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
    "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ",
)
MAX_PILOT_BYTES = 5 * 1024**3
MIN_FREE_BYTES = 20 * 1024**3


class AnnotationEventEvidence(ResearchSchema):
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)


class RecordingAnnotationEvidence(ResearchSchema):
    recording_id: Text
    policy: Literal["summary_sidecar_reconciled", "summary_zero_full_review"]
    summary_sha256: SHA256
    sidecar_path: Text | None = None
    sidecar_sha256: SHA256 | None = None
    coverage_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    coverage_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    events: tuple[AnnotationEventEvidence, ...] = ()
    reviewer: Text
    reviewed_on: date
    evidence_ref: Text
    reconciliation_ref: Text
    complete_coverage_attested: StrictBool


class AnnotationEvidenceManifest(ResearchSchema):
    schema_version: Literal["1"] = "1"
    source_policy_version: Literal["chbmit-summary-sidecar-reconciliation-v1"] = (
        "chbmit-summary-sidecar-reconciliation-v1"
    )
    recordings: tuple[RecordingAnnotationEvidence, ...] = Field(min_length=1)

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class PilotRunConfig(ResearchSchema):
    experiment_id: Text = "chbmit-child-xgb-pilot-v1"
    confirmation_required: StrictBool = True
    chunk_seconds: float = Field(default=300.0, gt=0, le=300, allow_inf_nan=False)
    halo_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)
    max_decoded_samples: int = Field(default=2_000_000, ge=1, le=10_000_000)
    positive_occupancy: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    seed: int = 42
    n_estimators: int = Field(default=300, ge=1, le=2000)
    max_depth: int = Field(default=3, ge=1, le=12)
    learning_rate: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)
    early_stopping_rounds: int = Field(default=30, ge=1, le=500)
    n_jobs: int = Field(default=4, ge=1, le=16)
    max_false_events_per_hour: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    max_alert_occupancy: float = Field(default=0.10, ge=0, le=1, allow_inf_nan=False)


class DiskBudget:
    """Enforce a per-pilot directory cap and leave reserved system space."""

    def __init__(self, root: Path | str, *, cap_bytes: int = MAX_PILOT_BYTES,
                 reserve_bytes: int = MIN_FREE_BYTES):
        path = Path(root)
        if path.is_symlink():
            raise ValueError("storage_root_symlink_not_allowed")
        self.root = path.resolve(strict=True)
        if self.root.is_relative_to(REPOSITORY_ROOT):
            raise ValueError("research_data_must_be_outside_repository")
        if type(cap_bytes) is not int or cap_bytes <= 0 or type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("storage_limits_invalid")
        self.cap_bytes = cap_bytes
        self.reserve_bytes = reserve_bytes

    def used_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("storage_symlink_not_allowed")
            if path.is_file():
                total += path.stat().st_size
        return total

    def check(self, additional_bytes: int = 0) -> None:
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("storage_estimate_invalid")
        used = self.used_bytes()
        if used + additional_bytes > self.cap_bytes:
            raise OSError("pilot_storage_cap_exceeded")
        free = shutil.disk_usage(self.root).free
        if free - additional_bytes < self.reserve_bytes:
            raise OSError("minimum_system_free_space_reserve_would_be_violated")


def validate_pilot_cohort(manifest: CohortManifest,
                          evidence: AnnotationEvidenceManifest,
                          root: Path | str) -> tuple[EEGRecording, ...]:
    """Check reviewed metadata, the exact six-case target, and annotation sources."""
    manifest = CohortManifest.model_validate(manifest)
    evidence = AnnotationEvidenceManifest.model_validate(evidence)
    validate_manifest(root, manifest)
    records = tuple(item.recording for item in manifest.records)
    subjects = {item.subject_id for item in records}
    if subjects != set(PILOT_SPLITS):
        raise ValueError("pilot_subject_set_must_match_frozen_six_patient_plan")
    if {item.signal.relative_path for item in manifest.records} != PILOT_FILES:
        raise ValueError("pilot_recording_files_must_match_frozen_twelve_edf_plan")
    if len(records) != 12 or len({item.patient_group_id for item in records}) != 6:
        raise ValueError("pilot_requires_two_recordings_for_each_of_six_distinct_patients")
    by_subject = {}
    for item in records:
        if item.age.group.value != "child":
            raise ValueError("pilot_scope_is_child_age_only")
        prior = by_subject.setdefault(item.subject_id, item.patient_group_id)
        if prior != item.patient_group_id:
            raise ValueError("case_identity_maps_to_multiple_canonical_patients")
        if item.permissions.status != "approved" or item.annotation_coverage != "complete":
            raise ValueError("approved_permissions_and_complete_annotations_required")
        if not item.annotation_protocol_ref or not item.provenance.annotation_provenance_ref:
            raise ValueError("annotation_provenance_reference_required")
        if not item.annotations and not item.annotation_protocol_ref:
            raise ValueError("seizure_free_recording_needs_reviewed_annotation_protocol")
    if len(set(by_subject.values())) != len(PILOT_SPLITS):
        raise ValueError("pilot_case_alias_or_duplicate_patient_detected")

    # Retain the known identity constraint even though neither alias belongs
    # to this six-case child cohort.
    identities = {item.subject_id: item.canonical_patient_id
                  for item in manifest.review_plan.identities.entries}
    if "chb01" in identities and "chb21" in identities and identities["chb01"] != identities["chb21"]:
        raise ValueError("known_chb01_chb21_alias_conflict")
    if subjects.intersection({"chb01", "chb21"}):
        raise ValueError("chb01_chb21_excluded_from_frozen_pilot_target")

    evidence_by_id = {item.recording_id: item for item in evidence.recordings}
    if len(evidence_by_id) != len(evidence.recordings) or set(evidence_by_id) != {
            item.recording_id for item in records}:
        raise ValueError("annotation_evidence_must_cover_exact_cohort")
    entry_by_id = {item.recording.recording_id: item for item in manifest.records}
    reviewed_inputs = {item.edf_path: item for item in manifest.review_plan.recordings}
    for recording in records:
        item = evidence_by_id[recording.recording_id]
        entry = entry_by_id[recording.recording_id]
        expected_seizures = entry.signal.relative_path in PILOT_SEIZURE_FILES
        if bool(recording.annotations) != expected_seizures:
            raise ValueError("pilot_annotation_event_presence_differs_from_frozen_recording_plan")
        reviewed_input = reviewed_inputs.get(entry.signal.relative_path)
        validate_recording_annotation_evidence(
            recording, entry, item, reviewed_input, root)
    return records


def validate_recording_annotation_evidence(recording: EEGRecording, entry,
                                          evidence: RecordingAnnotationEvidence,
                                          reviewed_input, root: Path | str) -> None:
    """Apply the versioned summary/sidecar and full-background review policy."""
    if (evidence.summary_sha256 != entry.annotations.sha256
            or not evidence.complete_coverage_attested
            or evidence.coverage_start_seconds != 0
            or evidence.coverage_end_seconds != recording.duration_seconds
            or reviewed_input is None
            or not reviewed_input.annotation_review.verified
            or evidence.evidence_ref != reviewed_input.annotation_review.evidence_ref
            or evidence.reconciliation_ref != recording.provenance.annotation_provenance_ref):
        raise ValueError("annotation_source_hash_or_complete_coverage_mismatch")
    reviewed_events = tuple(sorted(evidence.events, key=lambda event: event.start_seconds))
    reference_events = tuple(sorted(recording.annotations, key=lambda event: event.start_seconds))
    if len(reviewed_events) != len(reference_events):
        raise ValueError("annotation_event_count_mismatch")
    for reviewed, reference in zip(reviewed_events, reference_events):
        if (abs(reviewed.start_seconds - reference.start_seconds) > 0.5
                or abs(reviewed.end_seconds - reference.end_seconds) > 0.5):
            raise ValueError("annotation_timing_mismatch")
    if reference_events:
        expected_sidecar = entry.signal.relative_path + ".seizures"
        if (evidence.policy != "summary_sidecar_reconciled"
                or evidence.sidecar_path != expected_sidecar or not evidence.sidecar_sha256):
            raise ValueError("seizure_recording_requires_reconciled_seizures_sidecar")
        sidecar = _safe_path(root, evidence.sidecar_path)
        if _sha256_file(sidecar) != evidence.sidecar_sha256:
            raise ValueError("seizure_sidecar_hash_mismatch")
    elif (evidence.policy != "summary_zero_full_review" or evidence.sidecar_path is not None
          or evidence.sidecar_sha256 is not None):
        raise ValueError("zero_seizure_interval_requires_full_record_review_policy")


def validate_pilot_preprocessing(config: PreprocessingConfig) -> None:
    config = PreprocessingConfig.model_validate(config)
    if (config.age_group.value != "child" or len(config.channels) != 18
            or tuple(item.target_label for item in config.channels) != PILOT_CHANNELS
            or config.target_sampling_rate_hz != 256
            or config.allow_resampling or config.line_noise_hz is not None
            or config.window_seconds != 5 or config.step_seconds != 1
            or config.quality_thresholds.window_seconds != 5
            or config.quality_thresholds.step_seconds != 1):
        raise ValueError("preprocessing_config_differs_from_frozen_pilot_profile")


def _sha256_file(path: Path) -> str:
    digestor = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digestor.update(chunk)
    return digestor.hexdigest()


@dataclass(frozen=True)
class FeatureTable:
    split: Literal["train", "validation", "test"]
    feature_names: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray
    patient_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    starts: np.ndarray
    ends: np.ndarray
    event_ids: tuple[tuple[str, ...], ...]
    sha256: str


def load_feature_table(path: Path | str, *, split: Literal["train", "validation", "test"],
                       manifest: CohortManifest, split_manifest: SplitManifest,
                       schema: Sequence[str] | None = None) -> FeatureTable:
    """Load and bind one partition shard to its reviewed patient split."""
    import pyarrow.parquet as pq

    file = Path(path)
    table = pq.read_table(file)
    data = table.to_pydict()
    names = tuple(schema or [name for name in table.column_names if name.startswith("channel_")
                            or name.startswith("across_channels__")])
    if len(names) != 378 or any(name not in data for name in names):
        raise ValueError("feature_schema_mismatch")
    recordings = {item.recording.recording_id: item.recording for item in manifest.records}
    assignment = {item.recording_id: item for item in split_manifest.assignments}
    if not data.get("window_id"):
        raise ValueError("feature_shard_empty")
    for row, recording_id in enumerate(data["recording_id"]):
        record = recordings.get(recording_id)
        item = assignment.get(recording_id)
        if record is None or item is None:
            raise ValueError("feature_row_unknown_recording")
        if (item.split != split or data["patient_group_id"][row] != record.patient_group_id
                or item.patient_group_id != record.patient_group_id):
            raise ValueError("feature_row_patient_split_leakage")
    labels = np.asarray([1 if item == "positive" else 0 if item == "negative" else -1
                         for item in data["label"]], dtype=np.int8)
    values = np.column_stack([np.asarray(data[name], dtype=np.float32) for name in names])
    if not np.isfinite(values).all():
        raise ValueError("feature_table_nonfinite")
    values.setflags(write=False)
    labels.setflags(write=False)
    return FeatureTable(
        split=split, feature_names=names, values=values, labels=labels,
        patient_ids=tuple(data["patient_group_id"]), recording_ids=tuple(data["recording_id"]),
        window_ids=tuple(data["window_id"]), starts=np.asarray(data["start_seconds"], dtype=float),
        ends=np.asarray(data["end_seconds"], dtype=float),
        event_ids=tuple(tuple(row or ()) for row in data["event_ids"]),
        sha256=_sha256_file(file))


def _fit_rows(table: FeatureTable) -> np.ndarray:
    return table.labels >= 0


def patient_sample_weights(patient_ids: Sequence[str], labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Equalize each patient's total base weight; calculate class ratio on train only."""
    patients = np.asarray(patient_ids, dtype=object)
    labels = np.asarray(labels, dtype=np.int8)
    if len(patients) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("training_partition_requires_both_classes")
    counts = Counter(patients.tolist())
    base = np.asarray([1.0 / counts[patient] for patient in patients], dtype=np.float64)
    base *= len(base) / base.sum()
    negative_weight = float(base[labels == 0].sum())
    positive_weight = float(base[labels == 1].sum())
    return base.astype(np.float32), negative_weight / positive_weight


def _probabilities(model: xgb.XGBClassifier, scaler: StandardScaler, table: FeatureTable,
                   best_iteration: int) -> np.ndarray:
    return model.predict_proba(scaler.transform(table.values),
                               iteration_range=(0, best_iteration + 1))[:, 1]


def train_xgboost(
    train: FeatureTable,
    validation: FeatureTable,
    *,
    recordings: Sequence[EEGRecording],
    split_manifest: SplitManifest,
    cohort_sha256: str,
    annotation_evidence_sha256: str,
    output_dir: Path | str,
    config: PilotRunConfig = PilotRunConfig(),
    storage: DiskBudget | None = None,
    authorized: bool = False,
) -> dict:
    """Fit an independent XGBoost baseline; test artifacts are not accepted."""
    if not authorized:
        raise PermissionError("explicit_real_data_training_authorization_required")
    if train.split != "train" or validation.split != "validation":
        raise ValueError("trainer_accepts_only_train_and_validation_partitions")
    if train.feature_names != validation.feature_names or len(train.feature_names) != 378:
        raise ValueError("train_validation_feature_schema_mismatch")
    split_manifest = SplitManifest.model_validate(split_manifest)
    validate_split_manifest(recordings, split_manifest)
    allowed = {name: {item.patient_group_id for item in split_manifest.assignments if item.split == name}
               for name in ("train", "validation", "test")}
    if (not set(train.patient_ids) <= allowed["train"]
            or not set(validation.patient_ids) <= allowed["validation"]
            or set(train.patient_ids) & set(validation.patient_ids)):
        raise ValueError("patient_leakage_between_training_partitions")
    train_mask, val_mask = _fit_rows(train), _fit_rows(validation)
    X_train, y_train = train.values[train_mask], train.labels[train_mask]
    X_val, y_val = validation.values[val_mask], validation.labels[val_mask]
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
        raise ValueError("train_and_validation_need_both_unambiguous_classes")
    train_patients = np.asarray(train.patient_ids, dtype=object)[train_mask]
    sample_weight, positive_weight_ratio = patient_sample_weights(train_patients, y_train)
    # Fit all learned feature transform state on training rows only.
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train)
    normalized_train = scaler.transform(X_train)
    normalized_val = scaler.transform(X_val)
    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cpu",
        n_estimators=config.n_estimators, max_depth=config.max_depth,
        learning_rate=config.learning_rate, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5, reg_alpha=0, max_bin=256,
        random_state=config.seed, n_jobs=config.n_jobs, eval_metric="aucpr",
        early_stopping_rounds=config.early_stopping_rounds,
        scale_pos_weight=positive_weight_ratio,
    )
    model.fit(normalized_train, y_train, sample_weight=sample_weight,
              eval_set=[(normalized_val, y_val)], verbose=False)
    best_iteration = int(model.best_iteration)
    val_probabilities = _probabilities(model, scaler, validation, best_iteration)
    validation_recordings = tuple(item for item in recordings
                                  if item.patient_group_id in allowed["validation"])
    scored = tuple(ScoredWindow(
        recording_id=validation.recording_ids[index],
        patient_group_id=validation.patient_ids[index],
        start_seconds=float(validation.starts[index]), end_seconds=float(validation.ends[index]),
        probability=float(val_probabilities[index]), split="validation")
        for index in range(len(val_probabilities)))
    operating_threshold, curve = select_validation_threshold(
        validation_recordings, scored,
        max_false_events_per_hour=config.max_false_events_per_hour,
        max_alert_occupancy=config.max_alert_occupancy)
    output = Path(output_dir)
    if output.is_symlink() or output.exists():
        raise ValueError("model_output_must_be_new_directory")
    output_parent = output.parent.resolve(strict=True)
    if output_parent.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("model_artifacts_must_be_outside_repository")
    if storage is not None:
        storage.check(100 * 1024**2)
    output.mkdir(mode=0o700)
    try:
        model.save_model(output / "xgboost.json")
        np.savez_compressed(output / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_,
                            var=scaler.var_, n_features_in_=np.asarray(scaler.n_features_in_))
        validation_payload = {
            "selected_threshold": operating_threshold,
            "threshold_curve": {str(key): value.__dict__ for key, value in curve.items()},
            "best_iteration": best_iteration,
        }
        (output / "validation_selection.json").write_text(
            json.dumps(validation_payload, sort_keys=True, default=lambda value: value.__dict__,
                       allow_nan=False) + "\n", encoding="utf-8")
        provenance = {
            "schema_version": "seizure-xgb-model-v1", "model_type": "xgboost_cpu_binary_classifier",
            "experiment_id": config.experiment_id, "feature_version": FEATURE_VERSION,
            "feature_names": train.feature_names, "feature_count": len(train.feature_names),
            "cohort_manifest_sha256": cohort_sha256,
            "annotation_evidence_sha256": annotation_evidence_sha256,
            "split_manifest_sha256": split_manifest.sha256,
            "train_feature_sha256": train.sha256, "validation_feature_sha256": validation.sha256,
            "training_patient_ids": sorted(allowed["train"]),
            "validation_patient_ids": sorted(allowed["validation"]),
            "test_patient_ids": sorted(allowed["test"]), "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)), "training_label_counts": dict(Counter(y_train.tolist())),
            "validation_label_counts": dict(Counter(y_val.tolist())),
            "patient_base_weight_policy": "equal_total_per_patient",
            "scale_pos_weight": positive_weight_ratio, "best_iteration": best_iteration,
            "selected_validation_threshold": operating_threshold,
            "config": config.model_dump(mode="json"),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "xgboost": xgb.__version__},
            "clinical_validation": False,
        }
        provenance["model_sha256"] = _sha256_file(output / "xgboost.json")
        provenance["scaler_sha256"] = _sha256_file(output / "scaler.npz")
        code_files = (Path(__file__), Path(__file__).with_name("features.py"),
                      Path(__file__).with_name("events.py"),
                      Path(__file__).with_name("preprocessing.py"),
                      Path(__file__).with_name("waveform.py"))
        provenance["code_sha256"] = digest({path.name: _sha256_file(path) for path in code_files})
        (output / "provenance.json").write_text(
            json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(output)
        raise
    return {"model_dir": str(output), "threshold": operating_threshold,
            "best_iteration": best_iteration, "provenance": provenance,
            "validation_curve": curve, "model": model, "scaler": scaler}


def evaluate_xgboost_test(
    run: dict,
    test: FeatureTable,
    *,
    recordings: Sequence[EEGRecording],
    split_manifest: SplitManifest,
) -> dict:
    """Evaluate one frozen validation-selected model on test patients."""
    if test.split != "test":
        raise ValueError("frozen_test_evaluator_requires_test_partition")
    if test.feature_names != run["provenance"]["feature_names"]:
        raise ValueError("test_feature_schema_mismatch")
    assignments = {item.recording_id: item for item in split_manifest.assignments}
    for recording_id, patient_id in zip(test.recording_ids, test.patient_ids):
        assignment = assignments.get(recording_id)
        if assignment is None or assignment.split != "test" or assignment.patient_group_id != patient_id:
            raise ValueError("test_feature_patient_split_mismatch")
    probabilities = _probabilities(run["model"], run["scaler"], test,
                                   run["best_iteration"] if "best_iteration" in run else
                                   run["provenance"]["best_iteration"])
    scored = tuple(ScoredWindow(
        recording_id=test.recording_ids[index], patient_group_id=test.patient_ids[index],
        start_seconds=float(test.starts[index]), end_seconds=float(test.ends[index]),
        probability=float(probabilities[index]), split="test") for index in range(len(probabilities)))
    candidates = windows_to_events(scored, threshold_on=run["threshold"])
    test_patients = {item.patient_group_id for item in split_manifest.assignments if item.split == "test"}
    test_recordings = tuple(item for item in recordings if item.patient_group_id in test_patients)
    metrics = evaluate_events(test_recordings, scored, candidates)
    return {"threshold": run["threshold"], "metrics": metrics, "candidates": candidates,
            "clinical_validation": False}


def run_approved_pilot(root: Path | str, manifest: CohortManifest,
                       config: PreprocessingConfig, annotation_evidence: AnnotationEvidenceManifest,
                       output_dir: Path | str, *, authorized: bool = False,
                       run_config: PilotRunConfig = PilotRunConfig()) -> dict:
    """End-to-end extraction, train/validation selection, then frozen test score."""
    if not authorized:
        raise PermissionError("explicit_real_data_pilot_authorization_required")
    records = validate_pilot_cohort(manifest, annotation_evidence, root)
    validate_pilot_preprocessing(config)
    by_subject = {record.subject_id: record.patient_group_id for record in records}
    patient_splits = {by_subject[subject]: split for subject, split in PILOT_SPLITS.items()}
    split_manifest = build_explicit_split_manifest(records, patient_splits, seed=run_config.seed)
    storage_root = Path(root).resolve(strict=True)
    budget = DiskBudget(storage_root)
    budget.check()
    output = Path(output_dir)
    if output.is_symlink() or output.exists():
        raise ValueError("pilot_output_must_be_new_directory")
    if not output.resolve().is_relative_to(storage_root):
        raise ValueError("pilot_output_must_be_under_storage_root")
    budget.check(100 * 1024**2)
    output.mkdir(parents=True, mode=0o700)
    evidence_by_id = {item.recording_id: item for item in annotation_evidence.recordings}
    split_by_recording = {item.recording_id: item.split for item in split_manifest.assignments}
    shards = {}
    try:
        for recording in records:
            split = split_by_recording[recording.recording_id]
            shard = output / "features" / split / f"{recording.recording_id.replace(':', '_')}.parquet"
            shard.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            evidence = evidence_by_id[recording.recording_id]
            # Recheck annotation bytes immediately before extracting this source.
            entry = next(item for item in manifest.records if item.recording.recording_id == recording.recording_id)
            if evidence.summary_sha256 != entry.annotations.sha256:
                raise ValueError("annotation_source_hash_mismatch")
            if evidence.sidecar_path:
                sidecar = _safe_path(root, evidence.sidecar_path)
                if _sha256_file(sidecar) != evidence.sidecar_sha256:
                    raise ValueError("annotation_sidecar_changed_before_extraction")
            shards[recording.recording_id] = extract_approved_recording_features(
                root, manifest, recording.recording_id, config, split_manifest, shard,
                chunk_seconds=run_config.chunk_seconds, halo_seconds=run_config.halo_seconds,
                max_samples=run_config.max_decoded_samples,
                space_check=lambda estimate: budget.check(estimate))
            budget.check()
        feature_manifest = {
            "schema_version": "seizure-feature-dataset-v1",
            "experiment_id": run_config.experiment_id,
            "cohort_manifest_sha256": manifest.sha256,
            "annotation_evidence_sha256": annotation_evidence.sha256,
            "split_manifest_sha256": split_manifest.sha256,
            "preprocessing_config_sha256": digest(config.model_dump(mode="json")),
            "run_config_sha256": digest(run_config.model_dump(mode="json")),
            "feature_version": FEATURE_VERSION,
            "feature_names": feature_names(len(config.channels)),
            "storage_policy": {"root_cap_bytes": MAX_PILOT_BYTES,
                               "minimum_free_bytes": MIN_FREE_BYTES},
            "recordings": [{
                "recording_id": item.recording.recording_id,
                "subject_id": item.recording.subject_id,
                "patient_group_id": item.recording.patient_group_id,
                "split": split_by_recording[item.recording.recording_id],
                "source_edf_sha256": item.signal.sha256,
                "summary_annotation_sha256": item.annotations.sha256,
                "annotation_sidecar_sha256": evidence_by_id[item.recording.recording_id].sidecar_sha256,
                "annotation_evidence_ref": evidence_by_id[item.recording.recording_id].evidence_ref,
                "feature_shard": shards[item.recording.recording_id],
            } for item in manifest.records],
            "clinical_validation": False,
        }
        budget.check()
        (output / "feature_manifest.json").write_text(
            json.dumps(feature_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        split_manifest_path = output / "split_manifest.json"
        split_manifest_path.write_text(json.dumps(split_manifest.model_dump(mode="json"),
                                                 sort_keys=True, indent=2) + "\n", encoding="utf-8")
        annotation_hash = annotation_evidence.sha256
        def combine(split: str, recording_ids: Sequence[str]) -> FeatureTable:
            paths = [Path(shards[item]["path"]) for item in recording_ids]
            tables = [load_feature_table(path, split=split, manifest=manifest,
                                         split_manifest=split_manifest,
                                         schema=feature_names(len(config.channels))) for path in paths]
            return _concat_feature_tables(tables, split)
        ids_by_split = {name: [recording_id for recording_id, split in split_by_recording.items()
                               if split == name] for name in ("train", "validation", "test")}
        train = combine("train", ids_by_split["train"])
        validation = combine("validation", ids_by_split["validation"])
        budget.check()
        run = train_xgboost(
            train, validation, recordings=records, split_manifest=split_manifest,
            cohort_sha256=manifest.sha256, annotation_evidence_sha256=annotation_hash,
            output_dir=output / "model", config=run_config, storage=budget, authorized=True)
        # Test shards are loaded only after the artifact and validation threshold are frozen.
        test = combine("test", ids_by_split["test"])
        test_result = evaluate_xgboost_test(run, test, recordings=records,
                                            split_manifest=split_manifest)
        test_json = _jsonable(test_result)
        (output / "test_evaluation.json").write_text(
            json.dumps(test_json, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        budget.check()
        return {"split_manifest": split_manifest, "feature_shards": shards,
                "model": run, "test_evaluation": test_result, "clinical_validation": False}
    except Exception:
        # Keep already-created research artifacts for audit; callers must use a
        # new run directory after any partial failure rather than overwriting.
        raise


def _concat_feature_tables(tables: Sequence[FeatureTable], split: str) -> FeatureTable:
    if not tables or any(item.split != split for item in tables):
        raise ValueError("feature_table_partition_mismatch")
    names = tables[0].feature_names
    if any(item.feature_names != names for item in tables):
        raise ValueError("feature_table_schema_mismatch")
    values = np.concatenate([item.values for item in tables], axis=0)
    labels = np.concatenate([item.labels for item in tables])
    return FeatureTable(
        split=split, feature_names=names, values=values, labels=labels,
        patient_ids=tuple(value for item in tables for value in item.patient_ids),
        recording_ids=tuple(value for item in tables for value in item.recording_ids),
        window_ids=tuple(value for item in tables for value in item.window_ids),
        starts=np.concatenate([item.starts for item in tables]), ends=np.concatenate([item.ends for item in tables]),
        event_ids=tuple(value for item in tables for value in item.event_ids),
        sha256=digest([item.sha256 for item in tables]))


def _jsonable(value):
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json(path: str):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("run-pilot", help="Extract and train on an explicitly reviewed local cohort")
    pilot.add_argument("--root", required=True, help="Authorized local source root; no download is attempted")
    pilot.add_argument("--cohort-manifest", required=True)
    pilot.add_argument("--preprocessing-config", required=True)
    pilot.add_argument("--annotation-evidence", required=True)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--approve-real-data-processing", action="store_true",
                       help="Explicitly authorize this local real-recording run")
    args = parser.parse_args(argv)
    try:
        manifest = CohortManifest.model_validate(_load_json(args.cohort_manifest))
        config = PreprocessingConfig.model_validate(_load_json(args.preprocessing_config))
        evidence = AnnotationEvidenceManifest.model_validate(_load_json(args.annotation_evidence))
        run_approved_pilot(args.root, manifest, config, evidence, args.output_dir,
                           authorized=args.approve_real_data_processing)
        return 0
    except (ValueError, OSError, PermissionError, KeyError):
        print(json.dumps({"error": "pilot_input_gate_or_execution_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
