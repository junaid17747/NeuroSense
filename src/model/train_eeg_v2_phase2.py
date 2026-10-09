"""Phase 2: uncalibrated synthetic EEG feature-space XGBoost benchmark.

Adapt the existing EEG V2 XGBoost settings without changing its uncommitted
legacy trainer. Only train/validation data can reach the fitting interface.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss,
)
from sklearn.preprocessing import LabelEncoder
import xgboost
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from src.model.eeg_v2_data import (
    CLASSES, DATA_FILE, FEATURE_NAMES, ROOT, SEED, SPLIT_COUNTS,
    dataset_fingerprint, load_manifest_dataset, model_features,
)


MANIFEST_FILE = ROOT / "reports/eeg_v2_phase1/split_manifest.json"
MODEL_ROOT = ROOT / "models/eeg_v2"
OUTPUT_DIR = MODEL_ROOT / "phase2_seed42"
LIMITATION = (
    "Synthetic EEG feature-space benchmark results only. This uncalibrated model "
    "does not establish real EEG accuracy, raw-signal pipeline validity, or clinical validity. "
    "Known synthetic feature inconsistencies are preserved. Calibration and test "
    "subjects are excluded from fitting, early stopping, and model selection."
)


@dataclass(frozen=True)
class TrainingConfig:
    max_rounds: int = 1000
    patience: int = 50
    seed: int = SEED
    n_jobs: int = 4
    max_depth: int = 7
    learning_rate: float = 0.08
    subsample: float = 0.85
    colsample_bytree: float = 0.85

    def validate(self):
        if type(self.max_rounds) is not int or not 1 <= self.max_rounds <= 1000:
            raise ValueError("Maximum boosting rounds must be between 1 and 1000.")
        if self.patience != 50 or type(self.patience) is not int:
            raise ValueError("Phase 2 early stopping patience must be 50 rounds.")
        if type(self.n_jobs) is not int or self.n_jobs < 1:
            raise ValueError("Worker count must be a positive integer.")
        if type(self.seed) is not int or self.seed != SEED:
            raise ValueError("Phase 2 uses the fixed seed 42.")

    def model_parameters(self):
        self.validate()
        return {
            "n_estimators": self.max_rounds, "max_depth": self.max_depth,
            "learning_rate": self.learning_rate, "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "objective": "multi:softprob", "eval_metric": "mlogloss",
            "tree_method": "hist", "device": "cpu", "n_jobs": self.n_jobs,
            "random_state": self.seed,
        }


@dataclass(frozen=True)
class TrainingData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_validation: pd.DataFrame
    y_validation: np.ndarray
    classes: tuple
    manifest: dict


def assert_feature_matrix(matrix):
    if not isinstance(matrix, pd.DataFrame) or tuple(matrix.columns) != FEATURE_NAMES:
        raise ValueError("Model input must contain exactly the frozen 20 features in order.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise ValueError("Model features must be numeric.")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Model features must be finite.")


def prepare_training_data(df, indices, manifest):
    """Select no calibration/test features or labels after manifest verification."""
    if set(indices) != set(SPLIT_COUNTS):
        raise ValueError("All four verified partitions are required.")
    if manifest["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Manifest feature order differs from the frozen schema.")
    seen = set()
    for name, count in SPLIT_COUNTS.items():
        split = manifest["splits"][name]
        subjects = set(split["subjects"])
        if len(subjects) != count or len(split["subjects"]) != count or seen & subjects:
            raise ValueError("Invalid or overlapping manifest subjects.")
        seen.update(subjects)
        expected = np.flatnonzero(df["subject_id"].isin(subjects).to_numpy())
        if not np.array_equal(indices[name], expected) or len(expected) != split["rows"]:
            raise ValueError(f"The {name} indices differ from the verified manifest.")
    if seen != set(df["subject_id"]):
        raise ValueError("Manifest does not cover every dataset subject.")

    frames = {name: df.iloc[indices[name]] for name in ("train", "validation")}
    encoder = LabelEncoder().fit(frames["train"]["label"])
    if tuple(encoder.classes_) != CLASSES or manifest["classes"] != list(CLASSES):
        raise ValueError("Training label classes do not match the manifest.")

    def features(name):
        frame = frames[name]
        matrix = model_features(frame)
        matrix.index = pd.MultiIndex.from_frame(frame[["subject_id", "sample_id"]])
        return matrix

    data = TrainingData(
        features("train"), encoder.transform(frames["train"]["label"]),
        features("validation"), encoder.transform(frames["validation"]["label"]),
        tuple(encoder.classes_), manifest,
    )
    assert_training_boundary(data)
    return data


def assert_training_boundary(data):
    """Recheck provenance immediately before fit, including injected held-out rows."""
    if data.classes != CLASSES:
        raise ValueError("Wrong training class order.")
    used = set()
    held_out = set(data.manifest["splits"]["calibration"]["subjects"]) | set(
        data.manifest["splits"]["test"]["subjects"]
    )
    for name, matrix, labels in (
        ("train", data.X_train, data.y_train),
        ("validation", data.X_validation, data.y_validation),
    ):
        assert_feature_matrix(matrix)
        if not isinstance(matrix.index, pd.MultiIndex) or matrix.index.names != ["subject_id", "sample_id"]:
            raise ValueError("Training window provenance is required.")
        subjects = set(matrix.index.get_level_values("subject_id"))
        expected = data.manifest["splits"][name]
        if subjects & held_out or subjects & used:
            raise ValueError("Calibration/test or train/validation subject leakage detected.")
        if subjects != set(expected["subjects"]) or len(matrix) != expected["rows"]:
            raise ValueError(f"The {name} data does not match manifest membership/counts.")
        if not matrix.index.is_unique:
            raise ValueError("Duplicate subject/sample windows in training input.")
        if len(labels) != len(matrix) or set(np.unique(labels)) != set(range(len(CLASSES))):
            raise ValueError("Encoded labels are incomplete or misaligned in length.")
        used.update(subjects)


def fit_training_data(data, config=TrainingConfig(), verbose=25):
    assert_training_boundary(data)
    parameters = config.model_parameters()
    # Explicit dataset/metric names avoid relying on the eval_set ordering default.
    stopping = EarlyStopping(
        rounds=config.patience, data_name="validation_1", metric_name="mlogloss",
        maximize=False, min_delta=0.0, save_best=True,
    )
    model = XGBClassifier(**parameters, callbacks=[stopping])
    started = time.perf_counter()
    model.fit(
        data.X_train, data.y_train,
        eval_set=[(data.X_train, data.y_train), (data.X_validation, data.y_validation)],
        verbose=verbose,
    )
    return model, time.perf_counter() - started


def validation_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    encoded_classes = np.arange(len(CLASSES))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=encoded_classes)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=encoded_classes).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=encoded_classes, target_names=list(CLASSES),
            output_dict=True, zero_division=0,
        ),
    }


def write_json_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def reserve_run_directory(output_dir):
    output_dir = Path(output_dir).resolve()
    base = MODEL_ROOT.resolve()
    if not base.is_relative_to(ROOT) or not output_dir.is_relative_to(base) or output_dir == base:
        raise ValueError("New run directory must be strictly inside NeuroSense/models/eeg_v2/.")
    # This is the run lock and overwrite guard. Never reuse even an empty run folder.
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_model_new(model, path):
    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        temporary_model = Path(temporary) / "model.ubj"
        model.save_model(temporary_model)
        with path.open("xb") as stream:
            stream.write(temporary_model.read_bytes())


def save_learning_curves(output_dir, history, best_iteration):
    train_loss = history["validation_0"]["mlogloss"]
    validation_loss = history["validation_1"]["mlogloss"]
    if len(train_loss) != len(validation_loss):
        raise ValueError("Training and validation history lengths differ.")
    write_json_new(output_dir / "learning_curves.json", {
        "iteration_indexing": "zero-based", "train_mlogloss": train_loss,
        "validation_mlogloss": validation_loss, "best_iteration": best_iteration,
    })
    with (output_dir / "learning_curves.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration_zero_based", "round_one_based", "train_mlogloss", "validation_mlogloss"])
        writer.writerows((i, i + 1, t, v) for i, (t, v) in enumerate(zip(train_loss, validation_loss)))
    # Keep matplotlib's cache inside the project, and remove only our temporary cache.
    with tempfile.TemporaryDirectory(dir=output_dir) as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=(8, 4.5), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            rounds = np.arange(1, len(train_loss) + 1)
            axes.plot(rounds, train_loss, label="Training subjects")
            axes.plot(rounds, validation_loss, label="Validation subjects")
            axes.axvline(best_iteration + 1, linestyle="--", color="gray", label=f"Best round {best_iteration + 1}")
            axes.set(xlabel="Boosting round (one-based)", ylabel="Multiclass log-loss",
                     title="Synthetic EEG feature-space benchmark: Phase 2")
            axes.legend()
            axes.grid(alpha=0.2)
            with (output_dir / "learning_curves.png").open("xb") as stream:
                figure.savefig(stream, format="png", dpi=160)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


@dataclass(frozen=True)
class LoadedBenchmark:
    model: XGBClassifier
    classes: tuple
    report: dict

    def predict_proba(self, features):
        assert_feature_matrix(features)
        return self.model.predict_proba(features)


def load_phase2_model(output_dir):
    output_dir = Path(output_dir)
    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
    if report["feature_names"] != list(FEATURE_NAMES) or report["classes"] != list(CLASSES):
        raise ValueError("Saved model feature/class schema does not match the frozen schema.")
    for name in ("xgboost.ubj", "class_mapping.json", "split_manifest.json"):
        digest = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        if digest != report["artifacts"][name]["sha256"]:
            raise ValueError(f"Saved artifact checksum mismatch: {name}")
    mapping = json.loads((output_dir / "class_mapping.json").read_text(encoding="utf-8"))
    if mapping != {str(i): name for i, name in enumerate(CLASSES)}:
        raise ValueError("Saved class mapping is invalid.")
    model = XGBClassifier(n_jobs=report["configuration"]["n_jobs"])
    model.load_model(output_dir / "xgboost.ubj")
    if tuple(model.get_booster().feature_names) != FEATURE_NAMES:
        raise ValueError("Model's embedded feature order is invalid.")
    if model.best_iteration != report["best_iteration_zero_based"]:
        raise ValueError("Saved best iteration differs from training metadata.")
    if model.get_booster().num_boosted_rounds() != model.best_iteration + 1:
        raise ValueError("Saved model must contain only rounds through the best iteration.")
    if not np.array_equal(model.classes_, np.arange(len(CLASSES))):
        raise ValueError("Model's encoded classes are invalid.")
    return LoadedBenchmark(model, tuple(report["classes"]), report)


def train(data_file=DATA_FILE, manifest_file=MANIFEST_FILE, output_dir=OUTPUT_DIR,
          config=TrainingConfig()):
    config.validate()
    data_file, manifest_file = Path(data_file).resolve(), Path(manifest_file).resolve()
    if not data_file.is_relative_to(ROOT) or not manifest_file.is_relative_to(ROOT):
        raise ValueError("Phase 2 inputs must remain inside NeuroSense.")
    output_dir = reserve_run_directory(output_dir)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    print(LIMITATION, flush=True)
    print("Verifying Phase 1 dataset checksum and all four subject partitions...", flush=True)
    df, indices = load_manifest_dataset(data_file, manifest_file)
    if manifest_file.read_bytes() != manifest_bytes:
        raise ValueError("Split manifest changed while loading data.")
    data = prepare_training_data(df, indices, manifest)
    del df, indices
    print(f"Fitting: {len(data.X_train):,} train rows; early stopping: {len(data.X_validation):,} validation rows.", flush=True)
    print(f"Maximum rounds: {config.max_rounds}; patience: {config.patience}; workers: {config.n_jobs}.", flush=True)
    model, duration = fit_training_data(data, config)
    history = model.evals_result()
    best_iteration = int(model.best_iteration)
    probabilities = model.predict_proba(data.X_validation)
    metrics = validation_metrics(data.y_validation, probabilities)
    if dataset_fingerprint(data_file) != manifest["dataset"] or manifest_file.read_bytes() != manifest_bytes:
        raise ValueError("Dataset or manifest changed during training; refusing to publish model.")
    save_model_new(model, output_dir / "xgboost.ubj")
    write_json_new(output_dir / "class_mapping.json", {str(i): name for i, name in enumerate(data.classes)})
    with (output_dir / "split_manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    write_json_new(output_dir / "booster_config.json", json.loads(model.get_booster().save_config()))
    save_learning_curves(output_dir, history, best_iteration)
    rounds_run = len(history["validation_1"]["mlogloss"])
    report = {
        "report_version": 1, "phase": 2, "benchmark": "synthetic EEG feature-space",
        "limitation": LIMITATION, "model_type": "uncalibrated XGBClassifier",
        "started_at_utc": started_at, "dataset": manifest["dataset"],
        "manifest": {"path": str(manifest_file.relative_to(ROOT)), "sha256": manifest_sha256},
        "configuration": asdict(config), "model_parameters": config.model_parameters(),
        "seeds": {"subject_split": manifest["seed"], "xgboost": config.seed},
        "feature_names": list(FEATURE_NAMES), "classes": list(data.classes),
        "preprocessing": "None; all 20 frozen numeric features used without changes.",
        "label_encoder_fit_partition": "train",
        "split_summary": {name: {k: v for k, v in split.items() if k != "subjects"}
                          for name, split in manifest["splits"].items()},
        "data_usage": {"fit": ["train"], "curve_monitoring": ["train", "validation"],
                       "early_stopping": ["validation"], "metrics": ["validation"],
                       "excluded_from_fitting_and_selection": ["calibration", "test"],
                       "calibration_performed": False, "test_evaluated": False,
                       "hyperparameter_search_performed": False},
        "early_stopping": {"callback": "xgboost.callback.EarlyStopping", "rounds": config.patience,
                           "data_name": "validation_1", "metric_name": "mlogloss",
                           "maximize": False, "min_delta": 0.0, "save_best": True},
        "training_duration_seconds": duration,
        "best_iteration_zero_based": best_iteration, "best_round_one_based": best_iteration + 1,
        "best_validation_mlogloss": float(model.best_score), "rounds_executed": rounds_run,
        "saved_boosting_rounds": model.get_booster().num_boosted_rounds(),
        "stopping_reason": "early_stopping" if rounds_run < config.max_rounds else "maximum_rounds",
        "validation_metrics": metrics,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "platform": platform.platform()},
        "source_fingerprints": [dataset_fingerprint(path) for path in (
            Path(__file__), ROOT / "src/model/eeg_v2_data.py", ROOT / "src/features/eeg_features_v1.py")],
        "artifacts": {path.name: dataset_fingerprint(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "total_duration_before_report_seconds": time.perf_counter() - started,
    }
    write_json_new(output_dir / "training_report.json", report)
    restored = load_phase2_model(output_dir)
    restored_probabilities = restored.predict_proba(data.X_validation)
    max_difference = float(np.max(np.abs(restored_probabilities - probabilities)))
    if not np.array_equal(probabilities, restored_probabilities):
        raise RuntimeError("Reloaded model validation probabilities differ.")
    write_json_new(output_dir / "reload_verification.json", {
        "partition": "validation", "rows": len(data.X_validation),
        "probabilities_exactly_equal": True, "maximum_absolute_difference": max_difference,
        "saved_boosting_rounds": restored.model.get_booster().num_boosted_rounds(),
    })
    print(f"Training duration: {duration:.3f} seconds; rounds executed: {rounds_run}.", flush=True)
    print(f"Best iteration: {best_iteration} (zero-based); saved rounds: {best_iteration + 1}.", flush=True)
    print("Validation metrics: " + json.dumps(metrics, sort_keys=True), flush=True)
    print(f"Model reload verified on all validation rows; artifacts: {output_dir.relative_to(ROOT)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    train(args.data, args.manifest, args.output_dir, TrainingConfig(n_jobs=args.jobs))


if __name__ == "__main__":
    main()
