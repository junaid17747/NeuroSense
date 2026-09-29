from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


DATA_FILE = Path("data/processed/neurosense_synthetic_1m.csv")
MODEL_FILE = Path("models/xgboost_synthetic_1m.pkl")
ENCODER_FILE = Path("models/xgboost_synthetic_1m_label_encoder.pkl")

FEATURES = [
    "theta",
    "alpha",
    "beta",
    "heart_rate",
    "motion_level",
]


print("Loading 1M synthetic dataset...")
df = pd.read_csv(DATA_FILE)

print("Rows:", f"{len(df):,}")
print("Subjects:", df["subject_id"].nunique())

X = df[FEATURES]
groups = df["subject_id"]

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

print("Classes:", list(encoder.classes_))


# -------------------------------------------------
# Subject-independent split
# -------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_subjects = set(groups.iloc[train_idx])
test_subjects = set(groups.iloc[test_idx])

print("\nTraining rows:", f"{len(X_train):,}")
print("Testing rows:", f"{len(X_test):,}")

print("Training subjects:", len(train_subjects))
print("Testing subjects:", len(test_subjects))

overlap = train_subjects.intersection(test_subjects)

print("Subject overlap:", len(overlap))

if overlap:
    raise RuntimeError("Subject leakage detected!")


# -------------------------------------------------
# XGBoost
# -------------------------------------------------

model = XGBClassifier(
    n_estimators=350,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

print("\nTraining XGBoost...")
start = time.time()

model.fit(
    X_train,
    y_train,
)

elapsed = time.time() - start

print(f"Training finished in {elapsed:.1f} seconds.")


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions,
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)

print("\n================================")
print("SYNTHETIC 1M TEST RESULTS")
print("================================")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        digits=4,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# -------------------------------------------------
# Save synthetic baseline
# -------------------------------------------------

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    model,
    MODEL_FILE,
)

joblib.dump(
    encoder,
    ENCODER_FILE,
)

print("\nSaved synthetic model:")
print(MODEL_FILE)

print("Saved label encoder:")
print(ENCODER_FILE)

print(
    "\nNOTE: This model is trained entirely on synthetic "
    "rule-generated data and is NOT a validated real-world "
    "attention model."
)
