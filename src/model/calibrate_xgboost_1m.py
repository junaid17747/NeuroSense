from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

DATA_FILE = Path("data/processed/neurosense_synthetic_1m.csv")
BASE_MODEL_FILE = Path("models/xgboost_synthetic_1m.pkl")
ENCODER_FILE = Path("models/xgboost_synthetic_1m_label_encoder.pkl")

OUTPUT_MODEL = Path("models/calibrated_xgboost_synthetic_1m.pkl")
OUTPUT_ENCODER = Path(
    "models/calibrated_xgboost_synthetic_1m_label_encoder.pkl"
)

FEATURES = [
    "theta",
    "alpha",
    "beta",
    "heart_rate",
    "motion_level",
]

print("Loading dataset and trained model...")

df = pd.read_csv(DATA_FILE)
model = joblib.load(BASE_MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

# --------------------------------------------------
# IMPORTANT:
# Base model trained on S0001-S1000 except the
# GroupShuffleSplit test subjects from random_state=42.
# Reproduce that exact split first.
# --------------------------------------------------

from sklearn.model_selection import GroupShuffleSplit

X = df[FEATURES]
y = encoder.transform(df["label"])
groups = df["subject_id"]

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    outer_split.split(X, y, groups=groups)
)

# Only use the original held-out 200 subjects.
X_holdout = X.iloc[test_idx]
y_holdout = y[test_idx]
holdout_groups = groups.iloc[test_idx]

# Split those 200 subjects:
# 100 calibration subjects + 100 final evaluation subjects.
inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=43,
)

cal_local_idx, eval_local_idx = next(
    inner_split.split(
        X_holdout,
        y_holdout,
        groups=holdout_groups,
    )
)

X_cal = X_holdout.iloc[cal_local_idx]
y_cal = y_holdout[cal_local_idx]

X_eval = X_holdout.iloc[eval_local_idx]
y_eval = y_holdout[eval_local_idx]

cal_subjects = set(
    holdout_groups.iloc[cal_local_idx]
)
eval_subjects = set(
    holdout_groups.iloc[eval_local_idx]
)

print("Calibration rows:", f"{len(X_cal):,}")
print("Evaluation rows:", f"{len(X_eval):,}")
print("Calibration subjects:", len(cal_subjects))
print("Evaluation subjects:", len(eval_subjects))
print(
    "Calibration/evaluation overlap:",
    len(cal_subjects & eval_subjects),
)

if cal_subjects & eval_subjects:
    raise RuntimeError("Subject leakage detected!")

# --------------------------------------------------
# Probability calibration
# --------------------------------------------------

print("\nCalibrating probabilities...")

calibrated_model = CalibratedClassifierCV(
    FrozenEstimator(model),
    method="sigmoid",
)

calibrated_model.fit(X_cal, y_cal)

# --------------------------------------------------
# Final untouched evaluation subset
# --------------------------------------------------

pred = calibrated_model.predict(X_eval)

balanced_accuracy = balanced_accuracy_score(
    y_eval,
    pred,
)

macro_f1 = f1_score(
    y_eval,
    pred,
    average="macro",
)

print("\n================================")
print("CALIBRATED SYNTHETIC 1M RESULTS")
print("================================")

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)
print(
    f"Macro F1:          {macro_f1:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_eval,
        pred,
        target_names=encoder.classes_,
        digits=4,
    )
)

# Probability sanity check
probabilities = calibrated_model.predict_proba(
    X_eval.iloc[:5]
)

print("Example calibrated probabilities:")
for row in probabilities:
    print(
        {
            label: round(float(prob), 4)
            for label, prob in zip(
                encoder.classes_,
                row,
            )
        },
        "sum=",
        round(float(np.sum(row)), 4),
    )

# --------------------------------------------------
# Save
# --------------------------------------------------

joblib.dump(
    calibrated_model,
    OUTPUT_MODEL,
)

joblib.dump(
    encoder,
    OUTPUT_ENCODER,
)

print("\nSaved:")
print(OUTPUT_MODEL)
print(OUTPUT_ENCODER)

print(
    "\nSynthetic baseline only — "
    "not validated on real headset data."
)
