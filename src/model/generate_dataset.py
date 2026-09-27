import random
import pandas as pd


def generate_dataset(samples=1500):
    rows = []

    for _ in range(samples):
        theta = round(random.uniform(5, 80), 2)
        alpha = round(random.uniform(10, 220), 2)
        beta = round(random.uniform(5, 120), 2)
        heart_rate = random.randint(60, 110)
        motion_level = round(random.uniform(0.0, 1.0), 2)

        attention_index = beta / max(alpha + theta, 1)

        if attention_index >= 0.7 and motion_level < 0.4:
            label = "Focused"
        elif attention_index >= 0.35 and motion_level < 0.7:
            label = "Moderate"
        else:
            label = "Distracted"

        rows.append({
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

    print("Dataset created successfully")
    print("Rows:", len(df))
    print()
    print(df["label"].value_counts())
    print()
    print("Saved to:", output)
