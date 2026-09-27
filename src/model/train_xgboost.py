import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "neurosense_training_data.csv"

MODEL_PATH = ROOT / "models" / "xgboost_attention_model.pkl"
ENCODER_PATH = ROOT / "models" / "xgboost_label_encoder.pkl"


FEATURES = [
    "theta",
    "alpha",
    "beta",
    "heart_rate",
    "motion_level"
]


def train_xgboost():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["label"]
    groups = df["subject_id"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    group_kfold = GroupKFold(n_splits=5)

    macro_f1_scores = []
    balanced_scores = []
    accuracies = []

    all_true = []
    all_pred = []

    print("=" * 60)
    print("NeuroSense XGBoost — GroupKFold Evaluation")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(X, y_encoded, groups),
        start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        balanced_acc = balanced_accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        accuracies.append(accuracy)
        balanced_scores.append(balanced_acc)
        macro_f1_scores.append(macro_f1)

        all_true.extend(y_test)
        all_pred.extend(predictions)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.3f} | "
            f"Balanced Accuracy={balanced_acc:.3f} | "
            f"Macro F1={macro_f1:.3f}"
        )

    print("\n" + "=" * 60)
    print("GROUPED CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print("Mean Accuracy:",
          round(np.mean(accuracies), 3))

    print("Mean Balanced Accuracy:",
          round(np.mean(balanced_scores), 3))

    print("Mean Macro F1:",
          round(np.mean(macro_f1_scores), 3))

    print("\nClassification Report:")

    print(
        classification_report(
            all_true,
            all_pred,
            target_names=encoder.classes_
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )

    # Train final model using all synthetic data
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    final_model.fit(X, y_encoded)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    print("\nFinal XGBoost model saved:")
    print(MODEL_PATH)

    print("\nLabel encoder saved:")
    print(ENCODER_PATH)


if __name__ == "__main__":
    train_xgboost()
