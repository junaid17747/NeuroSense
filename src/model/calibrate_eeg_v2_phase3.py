"""Freeze sigmoid calibration, then evaluate the synthetic EEG test set once."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import xgboost

from src.model.eeg_v2_data import (
    CLASSES, DATA_FILE, FEATURE_NAMES, ROOT, SPLIT_COUNTS,
    dataset_fingerprint, load_manifest_dataset, model_features,
)
from src.model.train_eeg_v2_phase2 import (
    MANIFEST_FILE, MODEL_ROOT, OUTPUT_DIR as PHASE2_DIR,
    assert_feature_matrix, load_phase2_model, reserve_run_directory, write_json_new,
)
from src.model.eeg_v2_evaluation import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, LOG_EPSILON, RELIABILITY_BINS,
    checked_probabilities, cluster_bootstrap, per_subject_table, point_metrics,
    reliability_bins, subject_statistics,
)


OUTPUT_DIR = MODEL_ROOT / "phase3_sigmoid_v1_seed42"
TEST_LEDGER = MODEL_ROOT / "phase3_test_evaluation.lock.json"
DATA_SHA256 = "6c36f1bf2c1a6a6d4ee133d9a1aac0e0ec2324bc2a79bdec65347bbf3ad4f99d"
MANIFEST_SHA256 = "2a427ef2895b9d19c4b297186ce1e8f404337fd6691e5f48c24a5c29b3cf4c29"
BASE_MODEL_SHA256 = "315b603ae4ca56d3d76305337328a548af738aa4f78f65dfe609b65ba1cf6172"
LIMITATION = (
    "Synthetic EEG feature-space benchmark results only. No clinical validity, "
    "real-world attention detection performance, or raw-EEG pipeline validity is established. "
    "Synthetic RMS/variance, kurtosis and absolute power-scale inconsistencies are preserved. "
    "The test results are final descriptive estimates, not inputs to further tuning."
)


def sha256(path):
    return dataset_fingerprint(path)["sha256"]


def booster_digest(model):
    return hashlib.sha256(model.get_booster().save_raw(raw_format="ubj")).hexdigest()


@dataclass(frozen=True)
class Partition:
    name: str
    X: pd.DataFrame
    y: np.ndarray


def assert_partition(partition, manifest, required):
    if required not in {"calibration", "test"} or partition.name != required:
        raise ValueError(f"Only the {required} partition is permitted here.")
    assert_feature_matrix(partition.X)
    if not isinstance(partition.X.index, pd.MultiIndex) or partition.X.index.names != ["subject_id", "sample_id"]:
        raise ValueError("Window provenance is required.")
    expected = manifest["splits"][required]
    subjects = set(partition.X.index.get_level_values("subject_id"))
    forbidden = set().union(*(set(split["subjects"]) for name, split in manifest["splits"].items() if name != required))
    if subjects & forbidden or subjects != set(expected["subjects"]):
        raise ValueError("Partition subject leakage or membership mismatch.")
    if len(subjects) != SPLIT_COUNTS[required] or len(partition.X) != expected["rows"]:
        raise ValueError("Partition subject/window count mismatch.")
    counts = partition.X.index.get_level_values("subject_id").value_counts()
    sample_ids = partition.X.index.get_level_values("sample_id").to_numpy()
    if not partition.X.index.is_unique or not counts.eq(manifest["samples_per_subject"]).all():
        raise ValueError("Duplicate or incomplete windows in partition.")
    if not np.isin(sample_ids, np.arange(1, manifest["samples_per_subject"] + 1)).all():
        raise ValueError("Invalid sample IDs in partition.")
    if partition.y.shape != (len(partition.X),) or partition.y.dtype.kind not in "iu":
        raise ValueError("Encoded labels must align with partition rows.")
    if set(np.unique(partition.y)) != {0, 1, 2}:
        raise ValueError("Partition must contain all three encoded classes.")
    if manifest["feature_names"] != list(FEATURE_NAMES) or manifest["classes"] != list(CLASSES):
        raise ValueError("Manifest feature/class schema mismatch.")


def select_partition(df, indices, manifest, name):
    if name not in {"calibration", "test"}:
        raise ValueError("Phase 3 only selects calibration or test partitions.")
    expected = np.flatnonzero(df["subject_id"].isin(manifest["splits"][name]["subjects"]).to_numpy())
    if not np.array_equal(indices[name], expected):
        raise ValueError("Indices differ from the verified subject manifest.")
    frame = df.iloc[indices[name]]
    X = model_features(frame)
    X.index = pd.MultiIndex.from_frame(frame[["subject_id", "sample_id"]])
    mapping = {name: i for i, name in enumerate(CLASSES)}
    labels = frame["label"].map(mapping)
    if labels.isna().any():
        raise ValueError("Unknown label in selected partition.")
    partition = Partition(name, X, labels.to_numpy(dtype=np.int64))
    assert_partition(partition, manifest, name)
    return partition


def fit_sigmoid(base_model, calibration, manifest):
    assert_partition(calibration, manifest, "calibration")
    before = booster_digest(base_model)
    calibrator = CalibratedClassifierCV(
        FrozenEstimator(base_model), method="sigmoid", ensemble=False, cv=5, n_jobs=1,
    )
    started = time.perf_counter()
    calibrator.fit(calibration.X, calibration.y)
    duration = time.perf_counter() - started
    if booster_digest(base_model) != before:
        raise RuntimeError("Frozen XGBoost state changed during calibration.")
    if len(calibrator.calibrated_classifiers_) != 1:
        raise RuntimeError("Expected one calibrated frozen estimator.")
    embedded = calibrator.calibrated_classifiers_[0].estimator
    if not isinstance(embedded, FrozenEstimator) or booster_digest(embedded.estimator) != before:
        raise RuntimeError("Calibrator did not retain the unchanged frozen base model.")
    # Freeze the calibrator itself, as well as the XGBoost estimator inside it.
    return FrozenEstimator(calibrator), duration, before


@dataclass(frozen=True)
class LoadedCalibration:
    model: FrozenEstimator
    freeze_record: dict

    def predict_proba(self, features):
        assert_feature_matrix(features)
        return checked_probabilities(self.model.predict_proba(features), len(features))


def load_calibrated_model(directory=OUTPUT_DIR):
    directory = Path(directory)
    record = json.loads((directory / "freeze_record.json").read_text())
    for name, fingerprint in record["frozen_artifacts"].items():
        if Path(name).name != name or sha256(directory / name) != fingerprint["sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    schema = json.loads((directory / "feature_schema.json").read_text())
    mapping = json.loads((directory / "class_mapping.json").read_text())
    if schema["feature_names"] != list(FEATURE_NAMES) or mapping != {str(i): c for i, c in enumerate(CLASSES)}:
        raise ValueError("Frozen feature/class schema mismatch.")
    # This is a local, checksum-verified artifact; joblib is not for untrusted files.
    model = joblib.load(directory / "calibrated_model.joblib")
    if not isinstance(model, FrozenEstimator) or not isinstance(model.estimator, CalibratedClassifierCV):
        raise ValueError("Calibrated model is not frozen.")
    calibrator = model.estimator
    if calibrator.method != "sigmoid" or calibrator.ensemble is not False:
        raise ValueError("Unexpected calibration configuration.")
    if not isinstance(calibrator.estimator, FrozenEstimator) or len(calibrator.calibrated_classifiers_) != 1:
        raise ValueError("Base estimator is not frozen.")
    embedded = calibrator.calibrated_classifiers_[0].estimator
    if not isinstance(embedded, FrozenEstimator) or booster_digest(embedded.estimator) != record["base_booster_state_sha256"]:
        raise ValueError("Embedded base model state mismatch.")
    if tuple(model.feature_names_in_) != FEATURE_NAMES or not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Calibrated model feature/class order mismatch.")
    return LoadedCalibration(model, record)


def freeze_artifacts(directory, frozen, base_digest, manifest_bytes, provenance, configuration):
    with (directory / "calibrated_model.joblib").open("xb") as stream:
        joblib.dump(frozen, stream, compress=3)
    with (directory / "uncalibrated_xgboost.ubj").open("xb") as stream:
        stream.write((PHASE2_DIR / "xgboost.ubj").read_bytes())
    with (directory / "split_manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    write_json_new(directory / "class_mapping.json", {str(i): c for i, c in enumerate(CLASSES)})
    write_json_new(directory / "feature_schema.json", {
        "feature_names": list(FEATURE_NAMES), "feature_count": 20,
        "excluded_model_columns": ["subject_id", "sample_id", "label"],
        "input": "numeric finite DataFrame; exact column order required",
    })
    write_json_new(directory / "calibration_config.json", configuration)
    write_json_new(directory / "provenance.json", provenance)
    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_file_sha256": BASE_MODEL_SHA256,
        "base_booster_state_sha256": base_digest,
        "test_inference_started": False,
        "frozen_artifacts": {p.name: dataset_fingerprint(p) for p in sorted(directory.iterdir()) if p.is_file()},
    }
    write_json_new(directory / "freeze_record.json", record)
    return load_calibrated_model(directory)


def evaluate_test_once(calibrated, base, test, manifest, directory, ledger=TEST_LEDGER):
    assert_partition(test, manifest, "test")
    if not (directory / "freeze_record.json").is_file():
        raise ValueError("Calibrator must be saved and frozen before test evaluation.")
    write_json_new(ledger, {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(directory.relative_to(ROOT)),
        "freeze_record_sha256": sha256(directory / "freeze_record.json"),
        "rule": "Single test inference per model. Never rerun; reuse saved predictions for statistics.",
    })
    # Exactly one call to each predictive model on the full held-out partition.
    calibrated_probabilities = calibrated.predict_proba(test.X)
    uncalibrated_probabilities = checked_probabilities(base.predict_proba(test.X), len(test.X))
    payload = {
        "subjects": test.X.index.get_level_values("subject_id").to_numpy(dtype=str),
        "sample_ids": test.X.index.get_level_values("sample_id").to_numpy(dtype=np.int64),
        "labels": test.y, "calibrated": calibrated_probabilities,
        "uncalibrated": uncalibrated_probabilities,
    }
    with (directory / "test_predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **payload)
    write_json_new(directory / "test_inference_completed.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibrated_test_predict_calls": 1, "uncalibrated_test_predict_calls": 1,
        "test_rows": len(test.X), "test_subjects": len(np.unique(payload["subjects"])),
        "predictions_sha256": sha256(directory / "test_predictions.npz"),
        "freeze_record_sha256": sha256(directory / "freeze_record.json"),
    })
    return payload


def plot_reliability(directory, tables):
    with tempfile.TemporaryDirectory(dir=directory) as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=(10, 8), layout="constrained")
            FigureCanvasAgg(figure)
            for axis, label in zip(figure.subplots(2, 2).flat, (*CLASSES, "top_label")):
                axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
                for name, table in tables.items():
                    occupied = [b for b in table["series"][label]["bins"] if b["count"]]
                    axis.plot([b["mean_probability"] for b in occupied],
                              [b["observed_frequency"] for b in occupied], "o-", label=name)
                axis.set(title=label.replace("_", " "), xlabel="Mean predicted probability",
                         ylabel="Observed frequency", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8)
            figure.suptitle("Synthetic EEG feature-space benchmark\nHeld-out test reliability; 10 fixed equal-width bins")
            with (directory / "reliability_diagrams.png").open("xb") as stream:
                figure.savefig(stream, format="png", dpi=160)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def summarize_saved_predictions(directory, majority_class):
    """All analyses share one immutable prediction cache; no model calls here."""
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as saved:
        labels, subjects = saved["labels"], saved["subjects"]
        probabilities = {name: saved[name] for name in ("calibrated", "uncalibrated")}
    probabilities["majority_class"] = np.tile(np.eye(3)[majority_class], (len(labels), 1))
    stats = {name: subject_statistics(labels, p, subjects) for name, p in probabilities.items()}
    metrics = {name: point_metrics(values) for name, values in stats.items()}
    with (directory / "per_subject_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        per_subject_table(stats).to_csv(stream, index=False)
    intervals, weights = cluster_bootstrap(stats)
    write_json_new(directory / "bootstrap_confidence_intervals.json", intervals)
    with (directory / "bootstrap_subject_counts.npz").open("xb") as stream:
        np.savez_compressed(stream, subjects=next(iter(stats.values()))["subjects"], weights=weights)
    tables = {name: reliability_bins(labels, p) for name, p in probabilities.items()}
    write_json_new(directory / "reliability_bins.json", tables)
    plot_reliability(directory, tables)
    return metrics


def verify_inputs():
    if sha256(DATA_FILE) != DATA_SHA256 or sha256(MANIFEST_FILE) != MANIFEST_SHA256:
        raise ValueError("Phase 1 dataset or manifest checksum mismatch.")
    if sha256(PHASE2_DIR / "xgboost.ubj") != BASE_MODEL_SHA256:
        raise ValueError("Phase 2 model checksum mismatch.")
    manifest_bytes = MANIFEST_FILE.read_bytes()
    if (PHASE2_DIR / "split_manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("Phase 1 and Phase 2 manifests differ.")
    benchmark = load_phase2_model(PHASE2_DIR)
    manifest = json.loads(manifest_bytes)
    if benchmark.report["dataset"] != manifest["dataset"] or benchmark.report["manifest"]["sha256"] != MANIFEST_SHA256:
        raise ValueError("Phase 2 provenance does not match Phase 1.")
    for source in benchmark.report["source_fingerprints"]:
        if dataset_fingerprint(ROOT / source["path"]) != source:
            raise ValueError(f"Frozen source changed since Phase 2: {source['path']}")
    df, indices = load_manifest_dataset(DATA_FILE, MANIFEST_FILE)
    if MANIFEST_FILE.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during verification.")
    return benchmark, manifest, manifest_bytes, df, indices


def run_phase3(output_dir=OUTPUT_DIR):
    if TEST_LEDGER.exists():
        raise FileExistsError("Phase 3 test evaluation has already started. Reuse saved predictions; do not rerun inference.")
    directory = reserve_run_directory(output_dir)
    started = time.perf_counter()
    print(LIMITATION, flush=True)
    print("Verifying dataset, manifest, frozen feature schema and Phase 2 model checksums...", flush=True)
    base, manifest, manifest_bytes, df, indices = verify_inputs()
    audit_path = ROOT / "reports/eeg_v2_phase1/dataset_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["dataset"] != manifest["dataset"] or audit["integrity"]["errors"]:
        raise ValueError("Phase 1 audit provenance mismatch or integrity errors.")
    ratios = df["variance"].to_numpy() / df[list(FEATURE_NAMES[:5])].sum(axis=1).to_numpy()
    inconsistencies = {
        "phase1_measured_counts": audit["features"]["counts"],
        "rms_variance": "RMS squared must equal variance plus mean squared for one signal window; independent synthetic RMS noise violates this in 500045 rows.",
        "kurtosis": "6928 values are below -2; 6791 are below the unbiased Fisher lower bound -2.0039177277179236 for 1024-sample windows. Generator clips at -2.5.",
        "power_scale": {
            "description": "Generator sets variance to pre-noise summed band power times lognormal(4.5, 0.25), a median multiplier near 90, then perturbs spectral powers without rescaling variance. Absolute spectral/time-domain powers therefore lack a shared raw-signal-derived scale. Summed 1-40 Hz bands need not exactly equal raw variance, but this arbitrary synthetic multiplier is not a validated physical scaling.",
            "variance_over_stored_band_sum": {"min": float(ratios.min()), "median": float(np.median(ratios)),
                                             "mean": float(ratios.mean()), "max": float(ratios.max())},
        },
        "raw_windows": "No corresponding raw EEG windows are available to validate physical equivalence.",
        "dataset_modified": False,
    }
    training_counts = manifest["splits"]["train"]["class_counts"]
    majority_class = int(np.argmax([training_counts[c] for c in CLASSES]))
    configuration = {
        "method": "sigmoid", "estimator": "CalibratedClassifierCV(FrozenEstimator(Phase2 XGBClassifier))",
        "ensemble": False, "cv": 5, "n_jobs": 1,
        "cv_note": "Frozen base predictions only; all folds are within calibration rows, every base fit is a no-op, and one sigmoid per class is fitted using all calibration rows.",
        "fit_partition": "calibration", "fit_subjects": 100, "fit_rows": 100000,
        "excluded_fit_partitions": ["train", "validation", "test"],
        "base_retrained": False, "hyperparameter_search": False,
        "post_fit_freeze": "FrozenEstimator wraps the fitted calibrator before serialization and test inference",
        "class_order": list(CLASSES), "feature_order": list(FEATURE_NAMES),
        "majority_baseline": {"class": CLASSES[majority_class], "encoded_class": majority_class,
                              "selected_from": "Phase 1 training class counts only", "training_class_counts": training_counts,
                              "probabilities": np.eye(3)[majority_class].tolist()},
        "log_loss": {"dtype": "float64", "epsilon": LOG_EPSILON, "clipping": "[epsilon, 1-epsilon], then row normalization", "natural_log": True},
        "brier_score": "mean over windows of sum over 3 classes (p_k - 1[y=k])^2; unscaled range [0,2]",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence_level": 0.95, "unit": "subject", "paired": True},
        "reliability": {"bins": RELIABILITY_BINS, "strategy": "uniform", "plots": [*CLASSES, "top_label"]},
    }
    provenance = {
        "benchmark": "synthetic EEG feature-space", "limitation": LIMITATION,
        "dataset": manifest["dataset"], "manifest": dataset_fingerprint(MANIFEST_FILE),
        "phase2_model": dataset_fingerprint(PHASE2_DIR / "xgboost.ubj"),
        "phase2_training_report": dataset_fingerprint(PHASE2_DIR / "training_report.json"),
        "phase1_audit": dataset_fingerprint(audit_path), "known_inconsistencies": inconsistencies,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "source_fingerprints": [dataset_fingerprint(p) for p in (Path(__file__), ROOT / "src/model/eeg_v2_evaluation.py")],
    }
    calibration = select_partition(df, indices, manifest, "calibration")
    print("Fitting sigmoid calibration on 100,000 rows from 100 calibration subjects only...", flush=True)
    frozen, duration, base_digest = fit_sigmoid(base.model, calibration, manifest)
    calibration_probabilities = checked_probabilities(frozen.predict_proba(calibration.X), len(calibration.X))
    configuration["fitted_sigmoid_parameters"] = [
        {"class": label, "a": float(sigmoid.a_), "b": float(sigmoid.b_)}
        for label, sigmoid in zip(CLASSES, frozen.estimator.calibrated_classifiers_[0].calibrators)
    ]
    loaded = freeze_artifacts(directory, frozen, base_digest, manifest_bytes, provenance, configuration)
    restored_probabilities = loaded.predict_proba(calibration.X)
    if not np.array_equal(calibration_probabilities, restored_probabilities):
        raise RuntimeError("Calibrated model reload predictions differ on calibration rows.")
    write_json_new(directory / "reload_verification.json", {
        "partition": "calibration", "rows": len(calibration.X),
        "probabilities_exactly_equal": True, "maximum_absolute_difference": 0.0,
    })
    print(f"Calibrator frozen and reload verified; fit took {duration:.3f} seconds. Starting the single test evaluation...", flush=True)
    test = select_partition(df, indices, manifest, "test")
    del df, indices, calibration, calibration_probabilities, restored_probabilities
    evaluate_test_once(loaded, base, test, manifest, directory)
    del test
    print("Test predictions saved. Computing metrics, reliability and 2,000 subject bootstrap replicates from cache...", flush=True)
    metrics = summarize_saved_predictions(directory, majority_class)
    if booster_digest(base.model) != base_digest or sha256(PHASE2_DIR / "xgboost.ubj") != BASE_MODEL_SHA256:
        raise RuntimeError("Phase 2 model integrity changed.")
    if sha256(DATA_FILE) != DATA_SHA256 or sha256(MANIFEST_FILE) != MANIFEST_SHA256:
        raise RuntimeError("Dataset/manifest changed during Phase 3.")
    load_calibrated_model(directory)  # Checks bytes and state; does not predict again.
    report = {
        "phase": 3, "benchmark": "synthetic EEG feature-space", "limitation": LIMITATION,
        "test_metrics": metrics, "class_order": list(CLASSES),
        "calibration_duration_seconds": duration, "total_duration_seconds": time.perf_counter() - started,
        "test_inference_calls": {"calibrated": 1, "uncalibrated": 1},
        "test_rows": 100000, "test_subjects": 100, "further_tuning_performed": False,
        "base_model_state_unchanged": True, "dataset_unchanged": True,
        "frozen_model_sha256": sha256(directory / "calibrated_model.joblib"),
        "uncalibrated_model_sha256": BASE_MODEL_SHA256,
        "prediction_cache_sha256": sha256(directory / "test_predictions.npz"),
        "calibration_config": "calibration_config.json", "provenance": "provenance.json",
        "bootstrap_confidence_intervals": "bootstrap_confidence_intervals.json",
        "per_subject_metrics": "per_subject_metrics.csv", "reliability_diagrams": "reliability_diagrams.png",
    }
    write_json_new(directory / "evaluation_report.json", report)
    write_json_new(directory / "artifact_checksums.json", {
        p.name: dataset_fingerprint(p) for p in sorted(directory.iterdir()) if p.is_file()
    })
    print(json.dumps({"calibration_duration_seconds": duration, "test_metrics": metrics}, indent=2), flush=True)
    print(f"Artifacts: {directory.relative_to(ROOT)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_phase3(args.output_dir)


if __name__ == "__main__":
    main()
