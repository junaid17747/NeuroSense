import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "neurosense_training_data.csv"
MODEL_PATH = ROOT / "models" / "xgboost_attention_model.pkl"
ENCODER_PATH = ROOT / "models" / "xgboost_label_encoder.pkl"


def train():
    df = pd.read_csv(DATA_PATH)

    features = [
        "theta",
        "alpha",
        "beta",
        "heart_rate",
        "motion_level"
    ]

    X = df[features]
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=250,
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

    print("=" * 55)
    print("NeuroSense XGBoost Training")
    print("=" * 55)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClasses:")
    print(list(encoder.classes_))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("\nXGBoost model saved to:")
    print(MODEL_PATH)

    print("\nLabel encoder saved to:")
    print(ENCODER_PATH)


if __name__ == "__main__":
    train()
