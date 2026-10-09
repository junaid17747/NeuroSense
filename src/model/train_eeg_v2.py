"""Train and evaluate the EEG V2 synthetic baseline on disjoint subjects."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features.eeg_features_v1 import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/neurosense_eeg_v2_synthetic_1m.csv"
CLASSES = {"Focused", "Neutral", "Distracted"}
SEED = 42
LIMITATION = (
    "Synthetic feature-space baseline only. These results do not establish "
    "accuracy on real EEG or validate the raw-signal inference pipeline."
)


def validate_dataset(df):
    expected = ["subject_id", "sample_id", *FEATURE_NAMES, "label"]
    if list(df.columns) != expected:
        raise ValueError(f"Expected columns in this order: {expected}")
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset must be nonempty and contain no missing values.")
    if not np.isfinite(df[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("EEG features must all be finite.")
    if set(df["label"]) != CLASSES:
        raise ValueError(f"Dataset must contain exactly these classes: {sorted(CLASSES)}")
    if df["subject_id"].str.strip().eq("").any():
        raise ValueError("Subject IDs must not be blank.")
    sample_ids = df["sample_id"].to_numpy(dtype=float)
    if not (
        np.isfinite(sample_ids).all()
        and (sample_ids >= 1).all()
        and (sample_ids == np.floor(sample_ids)).all()
    ):
        raise ValueError("Sample IDs must be positive integers.")
    if df.duplicated(["subject_id", "sample_id"]).any():
        raise ValueError("Duplicate subject/sample windows detected.")
    if df["subject_id"].nunique() < 10:
        raise ValueError("At least 10 subjects are required for the three-way split.")


def split_subjects(df):
    """Reserve 20% of subjects, then split them into calibration and test sets."""
    groups = df["subject_id"]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train, holdout = next(outer.split(df, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
    calibration, test = next(
        inner.split(df.iloc[holdout], groups=groups.iloc[holdout])
    )
    splits = {"train": train, "calibration": holdout[calibration], "test": holdout[test]}
    seen = set()
    for name, indices in splits.items():
        subjects = set(groups.iloc[indices])
        if seen & subjects:
            raise ValueError("Subject leakage detected.")
        seen.update(subjects)
        if set(df["label"].iloc[indices]) != CLASSES:
            raise ValueError(f"The {name} split must contain all three classes.")
    return splits


def evaluate(y_true, probabilities, class_names):
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "multiclass_brier_score": float(
            np.mean(np.sum((probabilities - np.eye(len(labels))[y_true]) ** 2, axis=1))
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, predictions, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }


def train(data_file=DATA_FILE, output_dir=ROOT / "models", n_estimators=350, n_jobs=4):
    if n_estimators < 1 or n_jobs < 1:
        raise ValueError("Trees and worker count must be positive integers.")
    data_file, output_dir = Path(data_file), Path(output_dir)
    print(f"Loading {data_file}", flush=True)
    df = pd.read_csv(data_file, dtype={"subject_id": str})
    validate_dataset(df)
    splits = split_subjects(df)
    encoder = LabelEncoder().fit(df["label"].iloc[splits["train"]])
    y = encoder.transform(df["label"])
    X = df[FEATURE_NAMES]

    split_report = {}
    for name, indices in splits.items():
        subjects = sorted(df["subject_id"].iloc[indices].unique().tolist())
        split_report[name] = {
            "rows": len(indices), "subject_count": len(subjects), "subjects": subjects,
            "class_counts": df["label"].iloc[indices].value_counts().to_dict(),
        }
        print(f"{name}: {len(indices):,} rows, {len(subjects)} subjects", flush=True)

    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=7, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist", n_jobs=n_jobs, random_state=SEED,
    )
    start = time.monotonic()
    print("Training EEG V2 XGBoost...", flush=True)
    model.fit(X.iloc[splits["train"]], y[splits["train"]])
    print("Calibrating on separate subjects...", flush=True)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X.iloc[splits["calibration"]], y[splits["calibration"]])

    X_test, y_test = X.iloc[splits["test"]], y[splits["test"]]
    class_names = encoder.classes_.tolist()
    priors = np.bincount(y[splits["train"]], minlength=len(class_names))
    priors = priors / priors.sum()
    report = {
        "model_version": "eeg-v2-synthetic-calibrated-v1",
        "limitation": LIMITATION,
        "dataset": str(data_file.resolve()),
        "rows": len(df), "feature_names": FEATURE_NAMES, "classes": class_names,
        "random_seed": SEED, "splits": split_report,
        "model_parameters": model.get_params(),
        "calibration_method": "sigmoid",
        "evaluation": {
            "class_prior_baseline": evaluate(
                y_test, np.tile(priors, (len(y_test), 1)), class_names,
            ),
            "uncalibrated": evaluate(y_test, model.predict_proba(X_test), class_names),
            "calibrated": evaluate(y_test, calibrated.predict_proba(X_test), class_names),
        },
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "xgboost_eeg_v2.pkl")
    joblib.dump(encoder, output_dir / "xgboost_eeg_v2_label_encoder.pkl")
    joblib.dump(calibrated, output_dir / "calibrated_xgboost_eeg_v2.pkl")
    joblib.dump(encoder, output_dir / "calibrated_xgboost_eeg_v2_label_encoder.pkl")
    report_file = output_dir / "eeg_v2_evaluation.json"
    report_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for name, metrics in report["evaluation"].items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, log_loss={metrics['log_loss']:.4f}",
            flush=True,
        )
    print(f"Report: {report_file}\n{LIMITATION}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--trees", type=int, default=350)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    train(args.data, args.output_dir, args.trees, args.jobs)


if __name__ == "__main__":
    main()
