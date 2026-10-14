"""Cached, fail-closed status snapshots for the authenticated EEG research page."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import streamlit as st

from seizure_research.annotation_review import audit_review_packet
from seizure_research.archive import authorize as authorize_archive, verify_archive
from seizure_research.archive_selection import validate_inventory, validate_selection
from seizure_research.artifacts import read_json
from seizure_research.splitting import digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = PROJECT_ROOT / "reports/checkpoint_d7/official_inventory.json"
SELECTION = PROJECT_ROOT / "reports/checkpoint_d7/selection_proposal.json"
POLL_SECONDS = 5
CACHE_TTL_SECONDS = 15


def artifact_fingerprint(paths: list[Path]) -> str:
    """Cheap invalidation key; reads metadata only, never decodes an EDF."""
    rows = []
    for path in sorted(set(Path(item) for item in paths), key=str):
        try:
            stat = path.stat()
            rows.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        except OSError:
            rows.append((str(path), None, None, None))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _archive_signature(root: Path) -> list[Path]:
    paths = [root / name for name in ("index.json", "inventory.json", "selection.json")]
    if root.is_dir():
        # Stat the small selected archive file set so a changed/corrupt archive
        # invalidates the cached full checksum verification.
        paths.extend(item for item in root.rglob("*.zip") if item.is_file())
        sources = root / "sources"
        if sources.is_dir():
            paths.extend(item for item in sources.rglob("*") if item.is_file())
    return paths


@st.cache_data(max_entries=4, show_spinner=False)
def _verified_archive(fingerprint, root_text, inventory_hash, selection_hash, _inventory, _selection):
    """Full decompression only when source metadata changes, not every poll."""
    return verify_archive(Path(root_text), _inventory, _selection)


@st.cache_data(max_entries=4, show_spinner=False)
def _multipatient_bundle(fingerprint, run_text, archive_text, temporary_text, approval_text, review_text,
                         inventory_hash, selection_hash, _inventory, _selection):
    from seizure_research.multipatient import load_multipatient_artifacts
    return load_multipatient_artifacts(
        run_text,root=archive_text,temporary_root=temporary_text,approval_path=approval_text,
        review_path=review_text,inventory=_inventory,selection=_selection)


def status_fingerprint(*, archive_root: Path, approval_path: Path, review_path: Path,
                       pilot_root: Path | None, run_root: Path | None,
                       test_artifact: Path | None) -> str:
    paths = [INVENTORY, SELECTION, approval_path, review_path]
    paths.extend(_archive_signature(archive_root))
    if pilot_root:
        paths.extend(pilot_root / name for name in (
            "cohort-manifest.json", "preprocessing.json", "annotation-evidence.json",
            "faculty-approval.json", "SHA256SUMS.txt"))
        manifest = Path(os.environ.get(
            "NEUROSENSE_CHBMIT_FILE_MANIFEST",
            str(PROJECT_ROOT / "docs/chbmit_pilot_file_manifest.json"),
        )).expanduser()
        paths.append(manifest)
        try:
            file_manifest = read_json(manifest)
            paths.extend(pilot_root / row["path"] for row in file_manifest.get("recordings", []))
        except (OSError, ValueError, TypeError):
            pass
    if run_root:
        paths.extend(run_root / name for name in (
            "model/xgboost.json", "model/provenance.json", "model/scaler.npz",
            "model/validation_selection.json", "test_predictions.json", "test_evaluation.json",
            "cohort_manifest.json", "split_manifest.json", "preprocessing.json",
            "feature_manifest.json"))
    if test_artifact:
        paths.append(test_artifact)
    return artifact_fingerprint(paths)


def _status_template() -> dict:
    return {
        "inventory": "unavailable", "archive": "not staged",
        "archive_integrity": "not verified", "archive_bytes": 0,
        "recordings_available": 0, "recordings_verified": 0,
        "annotation_review": "not reviewed", "data_use_staging": "not approved",
        "processing_authorization": "not approved", "preprocessing": "not ready",
        "training_readiness": "not ready",
        "model": "unavailable", "training_status": "not trained or validated",
        "model_eligibility": "not established", "evaluation": "unavailable",
        "dashboard_artifacts": "unavailable", "provenance": {},
        "metrics": None, "test_metrics": None, "errors": [],
        "readiness_blockers": [],
    }


def validate_test_evaluation_artifact(path: Path | str) -> dict:
    """Load an explicitly opted-in, visibly test-only metrics fixture."""
    requested = Path(path).expanduser().absolute()
    if any(item.is_symlink() for item in (requested, *requested.parents)):
        raise ValueError("test_evaluation_artifact_symlink_rejected")
    requested = requested.resolve(strict=True)
    if requested.is_relative_to(PROJECT_ROOT) or requested.stat().st_size > 1024**2:
        raise ValueError("test_evaluation_artifact_must_be_small_and_external")
    before = requested.stat()
    payload = read_json(requested)
    after = requested.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("test_evaluation_artifact_changed_during_read")
    if (payload.get("schema_version") != "neurosense-dashboard-test-evaluation-v1"
            or payload.get("artifact_kind") != "SYNTHETIC_TEST_ONLY"
            or payload.get("clinical_validation") is not False
            or payload.get("research_performance_claim") is not False):
        raise ValueError("test_evaluation_artifact_marker_invalid")
    supplied_hash = payload.get("artifact_sha256")
    unsigned = dict(payload)
    unsigned["artifact_sha256"] = None
    if supplied_hash != digest(unsigned):
        raise ValueError("test_evaluation_artifact_hash_invalid")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("test_evaluation_metrics_invalid")
    int_fields = ("detected_events", "reference_events", "missed_events", "false_events", "matched_events")
    if any(type(metrics.get(key)) is not int or metrics[key] < 0 for key in int_fields):
        raise ValueError("test_evaluation_counts_invalid")
    if metrics["detected_events"] + metrics["missed_events"] != metrics["reference_events"]:
        raise ValueError("test_evaluation_event_counts_inconsistent")
    if metrics["matched_events"] != metrics["detected_events"]:
        raise ValueError("test_evaluation_match_count_inconsistent")
    hours = metrics.get("evaluable_hours")
    sensitivity = metrics.get("event_sensitivity")
    false_rate = metrics.get("false_alarms_per_hour")
    latency = metrics.get("median_detection_latency_seconds")
    if type(hours) not in (float, int) or not math.isfinite(hours) or hours < 0:
        raise ValueError("test_evaluation_hours_invalid")
    if sensitivity is not None and (type(sensitivity) not in (float, int) or not math.isfinite(sensitivity) or not 0 <= sensitivity <= 1):
        raise ValueError("test_evaluation_sensitivity_invalid")
    if ((metrics["reference_events"] == 0 and sensitivity is not None)
            or (metrics["reference_events"] > 0 and
                (sensitivity is None or not math.isclose(
                    sensitivity, metrics["detected_events"] / metrics["reference_events"], rel_tol=1e-9)))):
        raise ValueError("test_evaluation_sensitivity_inconsistent")
    if false_rate is not None and (type(false_rate) not in (float, int) or not math.isfinite(false_rate) or false_rate < 0):
        raise ValueError("test_evaluation_false_rate_invalid")
    if ((hours == 0 and false_rate is not None)
            or (hours > 0 and (false_rate is None or not math.isclose(
                false_rate, metrics["false_events"] / hours, rel_tol=1e-9)))):
        raise ValueError("test_evaluation_false_rate_inconsistent")
    if latency is not None and (type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0):
        raise ValueError("test_evaluation_latency_invalid")
    return payload


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_dashboard_snapshot(fingerprint: str, archive_root_text: str,
                               temporary_root_text: str, approval_path_text: str, review_path_text: str,
                               pilot_root_text: str, run_root_text: str,
                               test_artifact_text: str) -> dict:
    status = _status_template()
    archive_root = Path(archive_root_text)
    temporary_root = Path(temporary_root_text)
    approval_path, review_path = Path(approval_path_text), Path(review_path_text)
    pilot_root = Path(pilot_root_text) if pilot_root_text else None
    run_root = Path(run_root_text) if run_root_text else None
    test_artifact = Path(test_artifact_text) if test_artifact_text else None
    try:
        inventory, selection = read_json(INVENTORY), read_json(SELECTION)
        validate_inventory(inventory)
        validate_selection(selection, inventory)
        status["inventory"] = "verified official metadata inventory"
        status["recordings_available"] = int(selection["selected_recording_count"])
        status["provenance"].update({
            "dataset": "CHB-MIT Scalp EEG Database v1.0.0",
            "cohort": selection["cohort_id"],
            "manifest_sha256": selection["selection_sha256"],
            "canonical_patients": selection["selected_patient_count"],
            "recordings": selection["selected_recording_count"],
        })
    except (OSError, ValueError, TypeError, KeyError):
        status["errors"].append("Official inventory or selection metadata is missing or invalid.")
        return status

    if review_path.is_file():
        try:
            review = read_json(review_path)
            audit = audit_review_packet(inventory, selection, review)
            status["annotation_review"] = (
                f"reviewed: {audit['reviewed_recording_count']}/{audit['selected_recording_count']} recordings; "
                f"{audit['verified_background_recording_count']} verified background"
                if audit["annotation_evidence_complete"] else
                f"incomplete: {audit['reviewed_recording_count']}/{audit['selected_recording_count']} recordings; "
                f"{audit['verified_background_recording_count']} verified background"
            )
        except (OSError, ValueError, TypeError, KeyError):
            status["annotation_review"] = "invalid review evidence"

    archive_exists = archive_root.is_dir() and any(archive_root.iterdir())
    if approval_path.is_file():
        try:
            approval = authorize_archive(archive_root, temporary_root,
                                         approval_path, inventory, selection)
            status["data_use_staging"] = "approved"
            if archive_exists:
                archive_result = _verified_archive(
                    artifact_fingerprint(_archive_signature(archive_root)),str(archive_root),
                    inventory["inventory_sha256"],selection["selection_sha256"],inventory,selection)
                status["archive"] = "staged" if archive_result["status"] == "complete_verified" else "incomplete archive"
                status["archive_integrity"] = archive_result["status"]
                status["archive_bytes"] = int(archive_result["actual_compressed_eeg_bytes"])
                status["recordings_verified"] = int(archive_result["verified_recordings"])
            try:
                authorize_archive(archive_root, temporary_root, approval_path,
                                  inventory, selection, processing=True)
                status["processing_authorization"] = "approved"
            except (OSError, ValueError, PermissionError, KeyError, TypeError):
                status["processing_authorization"] = "not approved"
        except (OSError, ValueError, PermissionError, KeyError, TypeError):
            status["archive"] = ("files present; approval or verification unavailable"
                                  if archive_exists else "not staged")
            status["archive_integrity"] = "not verified"
            status["data_use_staging"] = "not verified"
    elif archive_exists:
        status["archive"] = "files present; approval or verification unavailable"
        status["archive_integrity"] = "not verified"
        status["data_use_staging"] = "not verified"

    multipatient_run = bool(run_root and (run_root / "complete.json").is_file())
    if multipatient_run:
        try:
            bundle = _multipatient_bundle(fingerprint,str(run_root),str(archive_root),str(temporary_root),
                                          str(approval_path),str(review_path),inventory["inventory_sha256"],
                                          selection["selection_sha256"],inventory,selection)
            provenance = bundle["model_provenance"]
            status.update(model="validated multipatient seizure model",training_status="trained; complete bundle validated",
                          evaluation="validated held-out patient evaluation",dashboard_artifacts="ready",
                          training_readiness="complete",preprocessing="reviewed recording-specific profiles",
                          model_eligibility="pooled experimental cohort; no age-specific model validation",
                          cohort_kind="multipatient",metrics=bundle["evaluation"]["metrics"],
                          metrics_by_age=bundle["evaluation_details"]["per_age"],
                          evaluation_timestamp=bundle["evaluation"].get("evaluated_at_utc"))
            status["provenance"].update({
                "cohort": "multipatient archive experiment", "model_name":provenance["experiment_id"],
                "model_version":provenance["schema_version"],"training_timestamp":provenance["trained_at_utc"],
                "preprocessing_version":provenance["preprocessing_config_sha256"],
                "dataset_manifest_sha256":provenance["cohort_manifest_sha256"],
                "canonical_patients":len({r.recording.patient_group_id for r in bundle["cohort"].records}),
                "recordings":len(bundle["cohort"].records),
                "split_summary": {name:len(provenance[name]) for name in
                                  ("training_patient_ids","validation_patient_ids","test_patient_ids")}})
        except (OSError,ValueError,PermissionError,KeyError,TypeError):
            status["errors"].append("Multipatient results are incomplete, changed, or not bound to current review decisions.")

    if not multipatient_run and pilot_root and pilot_root.is_dir():
        try:
            from seizure_research.readiness import check_readiness
            readiness = check_readiness(
                root=pilot_root, cohort_manifest_path=pilot_root / "cohort-manifest.json",
                preprocessing_config_path=pilot_root / "preprocessing.json",
                annotation_evidence_path=pilot_root / "annotation-evidence.json",
                approval_record_path=pilot_root / "faculty-approval.json",
                file_manifest_path=Path(os.environ.get(
                    "NEUROSENSE_CHBMIT_FILE_MANIFEST",
                    str(PROJECT_ROOT / "docs/chbmit_pilot_file_manifest.json"),
                )).expanduser())
            if readiness["checks"].get("preprocessing_config", {}).get("status") == "valid":
                status["preprocessing"] = "approved profile validated"
            if readiness["ready_for_real_data_processing"]:
                status["processing_authorization"] = "approved and readiness passed"
                if run_root:
                    from seizure_research.artifacts import load_prediction_artifacts
                    approval_data = read_json(pilot_root / "faculty-approval.json")
                    bundle = load_prediction_artifacts(
                        run_root, readiness_report=readiness, approval_record=approval_data)
                    status["model"] = "validated experimental seizure model"
                    status["training_status"] = "trained; complete artifact bundle validated"
                    status["evaluation"] = "validated held-out test evaluation"
                    status["dashboard_artifacts"] = "ready"
                    status["training_readiness"] = "complete model and held-out evaluation artifacts validated"
                    status["metrics"] = bundle["evaluation"]["metrics"]
                    status["evaluation_timestamp"] = bundle["evaluation"].get("evaluated_at_utc")
                    provenance = bundle["model_provenance"]
                    model_age = (provenance.get("config") or {}).get("age_group")
                    status["model_eligibility"] = (
                        f"frozen {model_age} pilot scope only; not clinically validated"
                        if model_age else "age eligibility unavailable")
                    status["provenance"].update({
                        "model_name": provenance.get("experiment_id", "XGBoost seizure baseline"),
                        "model_version": provenance.get("schema_version"),
                        "training_timestamp": provenance.get("trained_at_utc"),
                        "preprocessing_version": provenance.get("preprocessing_config_sha256"),
                        "split_summary": {
                            "training_patients": len(provenance.get("training_patient_ids", [])),
                            "validation_patients": len(provenance.get("validation_patient_ids", [])),
                            "test_patients": len(provenance.get("test_patient_ids", [])),
                        },
                        "dataset_manifest_sha256": provenance.get("cohort_manifest_sha256"),
                    })
            else:
                status["readiness_blockers"] = list(readiness.get("blockers", []))
                status["processing_authorization"] = "blocked by readiness checks"
                status["training_readiness"] = "blocked by readiness checks"
        except (OSError, ValueError, PermissionError, KeyError, TypeError) as exc:
            reason = str(exc)
            safe_code = reason if re.fullmatch(r"[a-z0-9_]+", reason) else type(exc).__name__
            status["errors"].append(
                "Configured model/readiness artifacts unavailable (" + safe_code + ").")

    if test_artifact:
        try:
            payload = validate_test_evaluation_artifact(test_artifact)
            status["test_metrics"] = payload["metrics"]
            status["test_artifact_label"] = "SYNTHETIC TEST ONLY — NOT RESEARCH PERFORMANCE"
        except (OSError, ValueError, TypeError, KeyError):
            status["errors"].append("Explicit synthetic test evaluation artifact is invalid.")
    return status


def load_dashboard_snapshot(*, archive_root: Path, approval_path: Path,
                            temporary_root: Path | None = None,
                            review_path: Path, pilot_root: Path | None = None,
                            run_root: Path | None = None,
                            test_artifact: Path | None = None) -> dict:
    paths = [INVENTORY, SELECTION, approval_path, review_path]
    paths.extend(_archive_signature(archive_root))
    if pilot_root:
        paths.extend(pilot_root.glob("*"))
    if run_root:
        paths.extend(run_root / name for name in (
            "model/xgboost.json", "model/provenance.json", "model/scaler.npz",
            "model/validation_selection.json", "test_predictions.json", "test_evaluation.json",
            "cohort_manifest.json", "split_manifest.json", "preprocessing.json", "feature_manifest.json"))
        # Bounded to the configured run; catches partial completion and changed
        # feature/model/review artifacts before serving a cached result.
        if run_root.is_dir():
            paths.extend(p for p in run_root.rglob("*") if p.is_file())
    if test_artifact:
        paths.append(test_artifact)
    fingerprint = artifact_fingerprint(paths)
    return _cached_dashboard_snapshot(
        fingerprint, str(archive_root), str(temporary_root or (archive_root.parent / "temporary")),
        str(approval_path), str(review_path),
        str(pilot_root) if pilot_root else "", str(run_root) if run_root else "",
        str(test_artifact) if test_artifact else "")


def clear_dashboard_status_cache() -> None:
    _cached_dashboard_snapshot.clear()
    _verified_archive.clear()
    _multipatient_bundle.clear()

