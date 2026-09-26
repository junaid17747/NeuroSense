from acquisition.eeg_reader import EEGReader
from acquisition.heart_rate_reader import HeartRateReader

from preprocessing.eeg_filter import bandpass_filter

from features.eeg_features import (
    extract_eeg_features,
    calculate_attention_index
)

from features.sensor_features import (
    attention_score,
    attention_state
)


def main():

    print("=" * 60)
    print("NeuroSense")
    print("Multi-Sensor Brain Attention Monitoring System")
    print("=" * 60)

    # -----------------------------
    # SENSOR ACQUISITION
    # -----------------------------

    eeg_sensor = EEGReader()
    heart_sensor = HeartRateReader()

    _, raw_eeg = eeg_sensor.generate_eeg()

    heart_rate = heart_sensor.read_heart_rate()

    # -----------------------------
    # EEG PREPROCESSING
    # -----------------------------

    filtered_eeg = bandpass_filter(raw_eeg)

    # -----------------------------
    # FEATURE EXTRACTION
    # -----------------------------

    features = extract_eeg_features(filtered_eeg)

    attention_index = calculate_attention_index(features)

    score = attention_score(attention_index)

    state = attention_state(score)

    # -----------------------------
    # OUTPUT
    # -----------------------------

    print("\nSYSTEM STATUS: RUNNING")

    print("\nSensor Data")
    print("-" * 30)

    print("Heart Rate:", heart_rate, "BPM")

    print("\nEEG Frequency Bands")
    print("-" * 30)

    print("Theta Power:", round(features["theta"], 2))
    print("Alpha Power:", round(features["alpha"], 2))
    print("Beta Power:", round(features["beta"], 2))

    print("\nAttention Analysis")
    print("-" * 30)

    print("Attention Index:", round(attention_index, 3))
    print("NeuroSense Score:", score, "/ 100")
    print("Attention State:", state)

    print("\nNeuroSense analysis completed successfully.")


if __name__ == "__main__":
    main()