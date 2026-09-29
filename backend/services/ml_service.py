import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from model.calibrated_predict import predict_calibrated


class MLService:
    """
    Production-facing NeuroSense ML inference service.

    The API communicates with this service instead of loading
    ML implementation details directly.
    """

    def predict(
        self,
        theta: float,
        alpha: float,
        beta: float,
        heart_rate: float,
        motion_level: float,
    ):
        result = predict_calibrated(
            theta,
            alpha,
            beta,
            heart_rate,
            motion_level,
        )

        return {
            "state": result["state"],
            "confidence": result["confidence"],
            "attention_score": result["score"],
            "focused_probability": result["focused_probability"],
            "neutral_probability": result["neutral_probability"],
            "distracted_probability": result["distracted_probability"],
        }


ml_service = MLService()
