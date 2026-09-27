import joblib
import pandas as pd

from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "neurosense_training_data.csv"

MODEL_PATH = ROOT / "models" / "calibrated_xgboost_model.pkl"
ENCODER_PATH = ROOT / "models" / "calibrated_label_encoder.pkl"


FEATURES = [
    "theta",
    "alpha",
    "beta",
    "heart_rate",
    "motion_level"
]


df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
groups = df["subject_id"]

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])


# --------------------------------------------------
# SUBJECT-INDEPENDENT TRAIN / CALIBRATION / TEST
# --------------------------------------------------

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

dev_idx, test_idx = next(
    split1.split(X, y, groups)
)

X_dev = X.iloc[dev_idx]
y_dev = y[dev_idx]
groups_dev = groups.iloc[dev_idx]

X_test = X.iloc[test_idx]
y_test = y[test_idx]


split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, calibration_idx = next(
    split2.split(
        X_dev,
        y_dev,
        groups_dev
    )
)

X_train = X_dev.iloc[train_idx]
y_train = y_dev[train_idx]

X_cal = X_dev.iloc[calibration_idx]
y_cal = y_dev[calibration_idx]


# --------------------------------------------------
# TRAIN BASE XGBOOST
# --------------------------------------------------

base_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

base_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# CALIBRATION
# --------------------------------------------------

try:
    from sklearn.frozen import FrozenEstimator

    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model),
        method="sigmoid"
    )

except ImportError:
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv="prefit"
    )


calibrated_model.fit(
    X_cal,
    y_cal
)


# --------------------------------------------------
# TEST
# --------------------------------------------------

predictions = calibrated_model.predict(
    X_test
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print("=" * 60)
print("NeuroSense Calibrated XGBoost")
print("=" * 60)

print("Training samples:", len(X_train))
print("Calibration samples:", len(X_cal))
print("Testing samples:", len(X_test))

print(
    "Balanced Accuracy:",
    round(balanced_accuracy, 3)
)

print(
    "Macro F1:",
    round(macro_f1, 3)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

joblib.dump(
    calibrated_model,
    MODEL_PATH
)

joblib.dump(
    encoder,
    ENCODER_PATH
)

print("\nCalibrated model saved:")
print(MODEL_PATH)

print("\nEncoder saved:")
print(ENCODER_PATH)
