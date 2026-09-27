import joblib
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "calibrated_xgboost_model.pkl"
ENCODER_PATH = ROOT / "models" / "calibrated_label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


FEATURES = [
    "theta",
    "alpha",
    "beta",
    "heart_rate",
    "motion_level"
]


def predict_calibrated(
    theta,
    alpha,
    beta,
    heart_rate,
    motion_level
):
    data = pd.DataFrame([{
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "heart_rate": heart_rate,
        "motion_level": motion_level
    }])

    probabilities = model.predict_proba(data)[0]

    class_names = encoder.inverse_transform(
        range(len(probabilities))
    )

    probs = {
        name: float(prob)
        for name, prob in zip(class_names, probabilities)
    }

    focused = probs.get("Focused", 0.0)
    neutral = probs.get("Neutral", 0.0)
    distracted = probs.get("Distracted", 0.0)

    attention_score = 100 * (
        focused + 0.5 * neutral
    )

    predicted_state = max(
        probs,
        key=probs.get
    )

    confidence = probs[predicted_state] * 100

    return {
        "state": predicted_state,
        "confidence": round(confidence, 1),
        "score": round(attention_score, 1),
        "focused_probability": focused,
        "neutral_probability": neutral,
        "distracted_probability": distracted
    }


if __name__ == "__main__":
    result = predict_calibrated(
        theta=20,
        alpha=80,
        beta=70,
        heart_rate=75,
        motion_level=0.25
    )

    print(result)
