from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
SUBJECTS = 1000
SAMPLES_PER_SUBJECT = 1000
TOTAL = SUBJECTS * SAMPLES_PER_SUBJECT

rng = np.random.default_rng(SEED)

print(f"Generating {TOTAL:,} synthetic NeuroSense samples...")

subject_id = np.repeat(
    [f"S{i:04d}" for i in range(1, SUBJECTS + 1)],
    SAMPLES_PER_SUBJECT
)

sample_id = np.tile(
    np.arange(1, SAMPLES_PER_SUBJECT + 1),
    SUBJECTS
)

# Synthetic subject-to-subject variability
subject_effect = rng.normal(1.0, 0.12, SUBJECTS)
subject_effect = np.repeat(subject_effect, SAMPLES_PER_SUBJECT)

theta = rng.gamma(2.5, 0.35, TOTAL) * subject_effect
alpha = rng.gamma(3.0, 0.30, TOTAL) * subject_effect
beta = rng.gamma(2.8, 0.32, TOTAL) * subject_effect

heart_rate = np.clip(
    rng.normal(78, 13, TOTAL),
    45,
    160
)

motion_level = np.clip(
    rng.beta(2.0, 4.0, TOTAL),
    0,
    1
)

attention_index = beta / np.maximum(alpha + theta, 1e-6)

labels = np.full(TOTAL, "Distracted", dtype=object)

neutral_mask = (
    (attention_index >= 0.35) &
    (motion_level < 0.70)
)

focused_mask = (
    (attention_index >= 0.70) &
    (motion_level < 0.40)
)

labels[neutral_mask] = "Neutral"
labels[focused_mask] = "Focused"

df = pd.DataFrame({
    "subject_id": subject_id,
    "sample_id": sample_id,
    "theta": theta,
    "alpha": alpha,
    "beta": beta,
    "heart_rate": heart_rate,
    "motion_level": motion_level,
    "label": labels
})

output = Path("data/processed/neurosense_synthetic_1m.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("\nDONE")
print("Rows:", f"{len(df):,}")
print("Subjects:", df["subject_id"].nunique())

print("\nClass distribution:")
print(df["label"].value_counts())
print("\nClass percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\nSaved:", output)
