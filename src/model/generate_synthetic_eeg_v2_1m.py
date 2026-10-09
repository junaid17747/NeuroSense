from pathlib import Path

import numpy as np
import pandas as pd

from src.features.eeg_features_v1 import FEATURE_NAMES


SEED = 42
SUBJECTS = 1000
SAMPLES_PER_SUBJECT = 1000
TOTAL = SUBJECTS * SAMPLES_PER_SUBJECT

rng = np.random.default_rng(SEED)

print("=" * 55)
print("NEUROSENSE EEG V2 SYNTHETIC DATA GENERATOR")
print("=" * 55)
print(f"Subjects: {SUBJECTS:,}")
print(f"Windows per subject: {SAMPLES_PER_SUBJECT:,}")
print(f"Total windows: {TOTAL:,}")
print(f"EEG features: {len(FEATURE_NAMES)}")


# ---------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------

subject_id = np.repeat(
    [f"S{i:04d}" for i in range(1, SUBJECTS + 1)],
    SAMPLES_PER_SUBJECT,
)

sample_id = np.tile(
    np.arange(1, SAMPLES_PER_SUBJECT + 1),
    SUBJECTS,
)


# ---------------------------------------------------------
# Subject-to-subject EEG variability
# ---------------------------------------------------------

subject_scale = rng.lognormal(
    mean=0.0,
    sigma=0.18,
    size=SUBJECTS,
)

subject_scale = np.repeat(
    subject_scale,
    SAMPLES_PER_SUBJECT,
)


# ---------------------------------------------------------
# Latent attention state
#
# IMPORTANT:
# Label is NOT deterministically calculated from one feature.
# We create overlapping distributions.
# ---------------------------------------------------------

class_names = np.array(
    ["Focused", "Neutral", "Distracted"]
)

class_probabilities = np.array(
    [0.34, 0.33, 0.33]
)

labels = rng.choice(
    class_names,
    size=TOTAL,
    p=class_probabilities,
)

focused = labels == "Focused"
neutral = labels == "Neutral"
distracted = labels == "Distracted"


# ---------------------------------------------------------
# Base band powers
# ---------------------------------------------------------

delta = rng.lognormal(0.10, 0.40, TOTAL)
theta = rng.lognormal(0.20, 0.38, TOTAL)
alpha = rng.lognormal(0.30, 0.36, TOTAL)
beta = rng.lognormal(0.25, 0.38, TOTAL)
gamma = rng.lognormal(-0.20, 0.42, TOTAL)


# ---------------------------------------------------------
# Soft class tendencies
# Intentionally overlapping — not perfect separation.
# ---------------------------------------------------------

beta[focused] *= rng.normal(
    1.28, 0.18, focused.sum()
)
theta[focused] *= rng.normal(
    0.86, 0.13, focused.sum()
)
alpha[focused] *= rng.normal(
    0.94, 0.15, focused.sum()
)

alpha[neutral] *= rng.normal(
    1.08, 0.16, neutral.sum()
)

theta[distracted] *= rng.normal(
    1.25, 0.18, distracted.sum()
)
beta[distracted] *= rng.normal(
    0.86, 0.14, distracted.sum()
)


# ---------------------------------------------------------
# Apply participant variability
# ---------------------------------------------------------

delta *= subject_scale
theta *= subject_scale
alpha *= subject_scale
beta *= subject_scale
gamma *= subject_scale

eps = 1e-9

total_power = (
    delta +
    theta +
    alpha +
    beta +
    gamma
)


# ---------------------------------------------------------
# Relative band powers
# ---------------------------------------------------------

relative_delta = delta / np.maximum(total_power, eps)
relative_theta = theta / np.maximum(total_power, eps)
relative_alpha = alpha / np.maximum(total_power, eps)
relative_beta = beta / np.maximum(total_power, eps)
relative_gamma = gamma / np.maximum(total_power, eps)


# ---------------------------------------------------------
# Ratios
# ---------------------------------------------------------

theta_beta_ratio = theta / np.maximum(beta, eps)
alpha_beta_ratio = alpha / np.maximum(beta, eps)


# ---------------------------------------------------------
# Time-domain / complexity features
#
# Correlated with signal power but with independent noise.
# ---------------------------------------------------------

variance = (
    total_power *
    rng.lognormal(4.5, 0.25, TOTAL)
)

rms = np.sqrt(
    np.maximum(variance, eps)
) * rng.normal(1.0, 0.04, TOTAL)

hjorth_activity = variance.copy()

hjorth_mobility = np.clip(
    rng.normal(0.55, 0.10, TOTAL)
    + relative_beta * 0.20
    + relative_gamma * 0.12,
    0.05,
    None,
)

hjorth_complexity = np.clip(
    rng.normal(2.3, 0.38, TOTAL)
    + relative_theta * 0.30,
    0.1,
    None,
)

spectral_entropy = np.clip(
    rng.normal(3.0, 0.30, TOTAL)
    + relative_delta * 0.30
    + relative_theta * 0.25,
    0.1,
    None,
)

kurtosis_value = np.clip(
    rng.normal(-0.4, 0.65, TOTAL),
    -2.5,
    5.0,
)

line_length = (
    rms *
    rng.normal(420, 45, TOTAL)
    * (1.0 + relative_beta * 0.25)
)


# ---------------------------------------------------------
# Realistic feature noise / session variability
# ---------------------------------------------------------

noise_scale = rng.normal(
    1.0,
    0.04,
    (TOTAL, 5),
)

bands = np.column_stack([
    delta,
    theta,
    alpha,
    beta,
    gamma,
])

bands *= noise_scale

delta, theta, alpha, beta, gamma = bands.T


# Recalculate derived spectral features after noise
total_power = delta + theta + alpha + beta + gamma

relative_delta = delta / np.maximum(total_power, eps)
relative_theta = theta / np.maximum(total_power, eps)
relative_alpha = alpha / np.maximum(total_power, eps)
relative_beta = beta / np.maximum(total_power, eps)
relative_gamma = gamma / np.maximum(total_power, eps)

theta_beta_ratio = theta / np.maximum(beta, eps)
alpha_beta_ratio = alpha / np.maximum(beta, eps)


# ---------------------------------------------------------
# Build exact frozen EEG feature schema
# ---------------------------------------------------------

df = pd.DataFrame({
    "subject_id": subject_id,
    "sample_id": sample_id,

    "delta_power": delta,
    "theta_power": theta,
    "alpha_power": alpha,
    "beta_power": beta,
    "low_gamma_power": gamma,

    "relative_delta": relative_delta,
    "relative_theta": relative_theta,
    "relative_alpha": relative_alpha,
    "relative_beta": relative_beta,
    "relative_low_gamma": relative_gamma,

    "theta_beta_ratio": theta_beta_ratio,
    "alpha_beta_ratio": alpha_beta_ratio,

    "spectral_entropy": spectral_entropy,

    "hjorth_activity": hjorth_activity,
    "hjorth_mobility": hjorth_mobility,
    "hjorth_complexity": hjorth_complexity,

    "rms": rms,
    "variance": variance,
    "kurtosis": kurtosis_value,
    "line_length": line_length,

    "label": labels,
})


# ---------------------------------------------------------
# Schema validation
# ---------------------------------------------------------

actual_features = [
    column
    for column in df.columns
    if column not in {
        "subject_id",
        "sample_id",
        "label",
    }
]

if actual_features != FEATURE_NAMES:
    raise RuntimeError(
        "\nFeature schema mismatch!\n"
        f"Expected:\n{FEATURE_NAMES}\n\n"
        f"Actual:\n{actual_features}"
    )

if df[FEATURE_NAMES].isna().any().any():
    raise RuntimeError("NaN detected in EEG features.")

if not np.isfinite(
    df[FEATURE_NAMES].to_numpy()
).all():
    raise RuntimeError(
        "Non-finite EEG feature detected."
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = Path(
    "data/processed/neurosense_eeg_v2_synthetic_1m.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    output,
    index=False,
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("GENERATION COMPLETE")
print("=" * 55)

print("Rows:", f"{len(df):,}")
print(
    "Subjects:",
    df["subject_id"].nunique()
)

print(
    "Feature count:",
    len(actual_features)
)

print("\nClass distribution:")
print(
    df["label"].value_counts()
)

print("\nClass percentages:")
print(
    (
        df["label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nFrozen feature order:")

for index, feature in enumerate(
    FEATURE_NAMES,
    start=1,
):
    print(
        f"{index:02d}. {feature}"
    )

print("\nSaved:")
print(output)

print(
    "\n✅ EEG V2 1M DATASET READY"
)
