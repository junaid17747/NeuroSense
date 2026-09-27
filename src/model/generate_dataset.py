import random
import pandas as pd


def generate_dataset(subjects=15, samples_per_subject=120):
    rows = []

    for subject_num in range(1, subjects + 1):
        subject_id = f"S{subject_num:02d}"

        for sample_num in range(samples_per_subject):
            theta = round(random.uniform(5, 80), 2)
            alpha = round(random.uniform(10, 220), 2)
            beta = round(random.uniform(5, 120), 2)

            heart_rate = random.randint(60, 110)
            motion_level = round(random.uniform(0.0, 1.0), 2)

            attention_index = beta / max(alpha + theta, 1)

            if attention_index >= 0.70 and motion_level < 0.40:
                label = "Focused"

            elif attention_index >= 0.35 and motion_level < 0.70:
                label = "Neutral"

            else:
                label = "Distracted"

            rows.append({
                "subject_id": subject_id,
                "sample_id": sample_num,
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "heart_rate": heart_rate,
                "motion_level": motion_level,
                "label": label
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()

    output = "data/processed/neurosense_training_data.csv"

    df.to_csv(output, index=False)

    print("=" * 55)
    print("NeuroSense Grouped Synthetic Dataset")
    print("=" * 55)

    print("Rows:", len(df))
    print("Subjects:", df["subject_id"].nunique())

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nSaved to:")
    print(output)
