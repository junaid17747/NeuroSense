import joblib
import pandas as pd
import shap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "xgboost_attention_model.pkl"
ENCODER_PATH = ROOT / "models" / "xgboost_label_encoder.pkl"

FEATURES = [
    "theta",
    "alpha",
    "beta",
    "heart_rate",
    "motion_level"
]

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

explainer = shap.TreeExplainer(model)


def explain_prediction(
    theta,
    alpha,
    beta,
    heart_rate,
    motion_level
):
    row = pd.DataFrame([{
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "heart_rate": heart_rate,
        "motion_level": motion_level
    }])

    prediction_encoded = model.predict(row)[0]
    prediction = encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        class_values = shap_values[prediction_encoded][0]
    else:
        values = shap_values

        if len(values.shape) == 3:
            class_values = values[0, :, prediction_encoded]
        elif len(values.shape) == 2:
            class_values = values[0]
        else:
            class_values = values

    explanation = []

    for feature, value in zip(FEATURES, class_values):
        explanation.append({
            "feature": feature,
            "impact": float(value)
        })

    explanation.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return prediction, explanation


if __name__ == "__main__":
    prediction, explanation = explain_prediction(
        theta=20,
        alpha=80,
        beta=70,
        heart_rate=75,
        motion_level=0.25
    )

    print("Prediction:", prediction)
    print()

    print("Top feature impacts:")

    for item in explanation:
        print(
            item["feature"],
            round(item["impact"], 4)
        )
