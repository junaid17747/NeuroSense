import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


FEATURE_NAMES = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "low_gamma_power",
    "relative_delta",
    "relative_theta",
    "relative_alpha",
    "relative_beta",
    "relative_low_gamma",
    "theta_beta_ratio",
    "alpha_beta_ratio",
    "spectral_entropy",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "rms",
    "variance",
    "kurtosis",
    "line_length",
]


def _band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)

    if not np.any(mask):
        return 0.0

    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_eeg_features_v1(eeg, sampling_rate=256):
    eeg = np.asarray(eeg, dtype=float)

    if eeg.ndim != 1:
        raise ValueError("EEG must be a 1-D signal.")

    if len(eeg) != sampling_rate * 4:
        raise ValueError(
            f"Expected {sampling_rate * 4} EEG samples "
            f"for a 4-second window, got {len(eeg)}."
        )

    if not np.all(np.isfinite(eeg)):
        raise ValueError("EEG contains non-finite values.")

    freqs, psd = welch(
        eeg,
        fs=sampling_rate,
        nperseg=min(len(eeg), sampling_rate * 2),
    )

    delta = _band_power(freqs, psd, 1, 4)
    theta = _band_power(freqs, psd, 4, 8)
    alpha = _band_power(freqs, psd, 8, 13)
    beta = _band_power(freqs, psd, 13, 30)
    low_gamma = _band_power(freqs, psd, 30, 40)

    total_power = delta + theta + alpha + beta + low_gamma
    eps = 1e-12

    normalized_psd = psd / max(float(np.sum(psd)), eps)
    normalized_psd = normalized_psd[normalized_psd > 0]

    spectral_entropy = float(
        -np.sum(normalized_psd * np.log2(normalized_psd))
    )

    activity = float(np.var(eeg))

    first_derivative = np.diff(eeg)
    second_derivative = np.diff(first_derivative)

    var_d1 = float(np.var(first_derivative))
    var_d2 = float(np.var(second_derivative))

    mobility = float(
        np.sqrt(var_d1 / max(activity, eps))
    )

    second_mobility = float(
        np.sqrt(var_d2 / max(var_d1, eps))
    )

    complexity = float(
        second_mobility / max(mobility, eps)
    )

    features = {
        "delta_power": delta,
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "low_gamma_power": low_gamma,

        "relative_delta": delta / max(total_power, eps),
        "relative_theta": theta / max(total_power, eps),
        "relative_alpha": alpha / max(total_power, eps),
        "relative_beta": beta / max(total_power, eps),
        "relative_low_gamma": low_gamma / max(total_power, eps),

        "theta_beta_ratio": theta / max(beta, eps),
        "alpha_beta_ratio": alpha / max(beta, eps),

        "spectral_entropy": spectral_entropy,

        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,

        "rms": float(np.sqrt(np.mean(eeg ** 2))),
        "variance": activity,
        "kurtosis": float(kurtosis(eeg, fisher=True, bias=False)),
        "line_length": float(np.sum(np.abs(np.diff(eeg)))),
    }

    for name, value in features.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Feature {name} produced a non-finite value."
            )

    return features


if __name__ == "__main__":
    fs = 256
    t = np.arange(fs * 4) / fs

    eeg = (
        20 * np.sin(2 * np.pi * 10 * t)
        + 10 * np.sin(2 * np.pi * 20 * t)
        + np.random.normal(0, 5, len(t))
    )

    features = extract_eeg_features_v1(eeg, fs)

    print("EEG FEATURE SCHEMA V1")
    print("Feature count:", len(features))

    for name in FEATURE_NAMES:
        print(f"{name}: {features[name]:.6f}")
