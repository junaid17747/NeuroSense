import numpy as np


def evaluate_eeg_quality(eeg_signal):
    if eeg_signal is None or len(eeg_signal) == 0:
        return {
            "quality": 0.0,
            "status": "missing",
            "usable": False
        }

    eeg_signal = np.asarray(eeg_signal, dtype=float)

    if not np.all(np.isfinite(eeg_signal)):
        return {
            "quality": 0.0,
            "status": "invalid",
            "usable": False
        }

    std = np.std(eeg_signal)
    peak = np.max(np.abs(eeg_signal))

    if std < 0.5:
        return {
            "quality": 0.2,
            "status": "flat_signal",
            "usable": False
        }

    if peak > 500:
        return {
            "quality": 0.3,
            "status": "possible_artifact",
            "usable": False
        }

    quality = 1.0

    return {
        "quality": quality,
        "status": "good",
        "usable": True
    }


def evaluate_aux_quality(heart_rate, motion_level):
    heart_ok = heart_rate is not None and 40 <= heart_rate <= 180
    motion_ok = motion_level is not None and 0.0 <= motion_level <= 1.0

    return {
        "heart_rate_available": heart_ok,
        "motion_available": motion_ok
    }


def overall_signal_status(eeg_quality, aux_quality):
    if not eeg_quality["usable"]:
        return {
            "status": "insufficient_signal",
            "allow_prediction": False
        }

    if not aux_quality["heart_rate_available"]:
        return {
            "status": "partial_signal",
            "allow_prediction": True
        }

    return {
        "status": "good",
        "allow_prediction": True
    }


if __name__ == "__main__":
    sample_eeg = np.random.normal(0, 20, 512)

    eeg_quality = evaluate_eeg_quality(sample_eeg)

    aux_quality = evaluate_aux_quality(
        heart_rate=75,
        motion_level=0.25
    )

    overall = overall_signal_status(
        eeg_quality,
        aux_quality
    )

    print("EEG Quality:", eeg_quality)
    print("Aux Quality:", aux_quality)
    print("Overall:", overall)
