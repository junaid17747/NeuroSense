import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from preprocessing.eeg_filter import bandpass_filter
from features.eeg_features import extract_eeg_features
from features.signal_quality import (
    evaluate_eeg_quality,
    evaluate_aux_quality,
    overall_signal_status,
)
from backend.services.ml_service import ml_service


class SensorInferenceService:

    EXPECTED_EEG_SAMPLES = 1024

    def predict_from_sensor(
        self,
        eeg,
        heart_rate: float,
        motion_level: float,
    ):
        eeg = np.asarray(eeg, dtype=float)

        if len(eeg) != self.EXPECTED_EEG_SAMPLES:
            raise ValueError(
                f"Expected {self.EXPECTED_EEG_SAMPLES} EEG samples "
                f"(4 seconds at 256 Hz), received {len(eeg)}."
            )

        eeg_quality = evaluate_eeg_quality(eeg)

        aux_quality = evaluate_aux_quality(
            heart_rate=heart_rate,
            motion_level=motion_level
        )

        quality = overall_signal_status(
            eeg_quality,
            aux_quality
        )

        if not quality["allow_prediction"]:
            return {
                "prediction_allowed": False,
                "signal_quality": quality,
                "prediction": None
            }

        filtered_eeg = bandpass_filter(eeg)

        features = extract_eeg_features(filtered_eeg)

        result = ml_service.predict(
            theta=features["theta"],
            alpha=features["alpha"],
            beta=features["beta"],
            heart_rate=heart_rate,
            motion_level=motion_level,
        )

        return {
            "prediction_allowed": True,
            "signal_quality": quality,
            "features": {
                "theta": float(features["theta"]),
                "alpha": float(features["alpha"]),
                "beta": float(features["beta"]),
            },
            "prediction": result
        }


sensor_inference_service = SensorInferenceService()
