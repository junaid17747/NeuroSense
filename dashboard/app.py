import sys
from time import sleep
from time import sleep
from pathlib import Path

import streamlit as st
import pandas as pd


# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from acquisition.eeg_reader import EEGReader
from storage.data_logger import log_reading
from acquisition.heart_rate_reader import HeartRateReader
from acquisition.motion_reader import MotionReader
from acquisition.unified_sensor_manager import UnifiedSensorManager
from preprocessing.eeg_filter import bandpass_filter
from features.eeg_features import (
    extract_eeg_features,
    calculate_attention_index
)
from features.sensor_fusion import calculate_fusion_score, fusion_state
from model.predict import predict_attention
from model.calibrated_predict import predict_calibrated
from model.shap_explainer import explain_prediction
from features.sensor_features import (
    attention_score,
    attention_state
)


st.set_page_config(
    page_title="NeuroSense",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroSense")
st.subheader("Multi-Sensor Brain Attention Monitoring System")

st.caption(
    "Prototype attention monitoring using EEG, heart-rate and motion data."
)


# -----------------------------
# Generate Sensor Data
# -----------------------------


sensor_manager = UnifiedSensorManager()
sensor_data = sensor_manager.read_all()

time_axis = sensor_data["time"]
raw_eeg = sensor_data["eeg"]
heart_rate = sensor_data["heart_rate"]
motion_level = sensor_data["motion_level"]
sensor_mode = sensor_data["mode"]

st.caption(f"Sensor Mode: {sensor_mode.upper()}")

filtered_eeg = bandpass_filter(raw_eeg)

features = extract_eeg_features(filtered_eeg)

attention_index = calculate_attention_index(features)

score = attention_score(attention_index)

fusion_score = calculate_fusion_score(
    score,
    heart_rate,
    motion_level
)
state = fusion_state(fusion_score)

ml_prediction, ml_confidence = predict_attention(
    features["theta"],
    features["alpha"],
    features["beta"],
    heart_rate,
    motion_level
)

calibrated_result = predict_calibrated(
    features["theta"],
    features["alpha"],
    features["beta"],
    heart_rate,
    motion_level
)

shap_prediction, shap_explanation = explain_prediction(
    features["theta"],
    features["alpha"],
    features["beta"],
    heart_rate,
    motion_level
)

calibrated_state = calibrated_result["state"]
calibrated_confidence = calibrated_result["confidence"]
raw_calibrated_score = calibrated_result["score"]

p_focused = calibrated_result["focused_probability"]
p_neutral = calibrated_result["neutral_probability"]
p_distracted = calibrated_result["distracted_probability"]

# Exponential moving average smoothing
if "smoothed_attention_score" not in st.session_state:
    st.session_state.smoothed_attention_score = raw_calibrated_score

alpha_smooth = 0.35

st.session_state.smoothed_attention_score = (
    alpha_smooth * raw_calibrated_score
    + (1 - alpha_smooth)
    * st.session_state.smoothed_attention_score
)

calibrated_score = round(
    st.session_state.smoothed_attention_score,
    1
)

# -----------------------------
# Live History + Data Logging
# -----------------------------

log_reading(
    heart_rate,
    motion_level,
    score,
    fusion_score,
    state
)

if "attention_history" not in st.session_state:
    st.session_state.attention_history = []

st.session_state.attention_history.append({
    "Attention Score": fusion_score,
    "Heart Rate": heart_rate,
    "Motion Level": motion_level
})

# Keep only latest 30 readings
st.session_state.attention_history = st.session_state.attention_history[-30:]
state = attention_state(score)

# -----------------------------
# Main Metrics
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="❤️ Heart Rate",
        value=f"{heart_rate} BPM"
    )

with col2:
    st.metric(
        label="🏃 Motion Level",
        value=f"{motion_level:.2f}"
    )

with col3:
    st.metric(
        label="🧠 Attention Score",
        value=f"{fusion_score}/100"
    )

with col4:
    st.metric(
        label="🎯 Attention State",
        value=state
    )

st.divider()

# -----------------------------
# EEG Graph
# -----------------------------

st.subheader("Live EEG Signal")

eeg_dataframe = pd.DataFrame({
    "Time": time_axis,
    "Raw EEG": raw_eeg,
    "Filtered EEG": filtered_eeg
})

st.line_chart(
    eeg_dataframe.set_index("Time")
)

# -----------------------------
# Frequency Bands
# -----------------------------

st.subheader("EEG Frequency Bands")

band_data = pd.DataFrame({
    "Band": ["Theta", "Alpha", "Beta"],
    "Power": [
        features["theta"],
        features["alpha"],
        features["beta"]
    ]
})

st.bar_chart(
    band_data.set_index("Band")
)

# -----------------------------
# Detailed Analysis
# -----------------------------

st.subheader("Attention Analysis")

st.write(
    "Experimental Attention Index:",
    round(attention_index, 3)
)

st.progress(
    min(int(fusion_score), 100)
)

if state == "Focused":
    st.success("Focused")

elif state == "Moderate":
    st.warning("Moderate Attention")

else:
    st.error("Distracted")



# -----------------------------
# Machine Learning Prediction
# -----------------------------

st.subheader("Machine Learning Analysis")

ml_col1, ml_col2 = st.columns(2)

with ml_col1:
    st.metric(
        "🤖 ML Prediction",
        ml_prediction
    )

with ml_col2:
    st.metric(
        "📊 ML Confidence",
        f"{ml_confidence}%"
    )


# -----------------------------
# Calibrated XGBoost Analysis
# -----------------------------

st.subheader("Calibrated XGBoost Attention Analysis")

cal1, cal2, cal3 = st.columns(3)

with cal1:
    st.metric(
        "🧠 Calibrated Attention Score",
        f"{calibrated_score}/100"
    )

with cal2:
    st.metric(
        "🎯 Calibrated State",
        calibrated_state
    )

with cal3:
    st.metric(
        "📊 Confidence",
        f"{calibrated_confidence}%"
    )

probability_df = pd.DataFrame({
    "State": [
        "Focused",
        "Neutral",
        "Distracted"
    ],
    "Probability": [
        p_focused,
        p_neutral,
        p_distracted
    ]
})

st.bar_chart(
    probability_df.set_index("State")
)

st.caption(
    "Attention score = 100 × "
    "[P(Focused) + 0.5 × P(Neutral)]. "
    "Displayed score is smoothed across readings."
)


# -----------------------------
# SHAP Explainability
# -----------------------------

st.subheader("Why This Prediction?")

st.caption(
    f"SHAP explanation for current XGBoost prediction: {shap_prediction}"
)

shap_df = pd.DataFrame(shap_explanation[:5])

if not shap_df.empty:
    shap_df["absolute_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values(
        "absolute_impact",
        ascending=False
    )

    st.bar_chart(
        shap_df.set_index("feature")[["impact"]]
    )

    st.write("Top contributing features:")

    for _, row in shap_df.iterrows():
        direction = (
            "increased"
            if row["impact"] > 0
            else "decreased"
        )

        st.write(
            f"• {row['feature']}: "
            f"{direction} support for the predicted class "
            f"(SHAP impact {row['impact']:.4f})"
        )

# -----------------------------
# Attention History
# -----------------------------

st.subheader("Live Attention History")

history_df = pd.DataFrame(st.session_state.attention_history)

if not history_df.empty:
    st.line_chart(history_df[["Attention Score"]])

st.caption("Displaying the latest 30 attention readings.")

st.info(
    "This is an experimental academic prototype and is not intended "
    "for medical diagnosis."
)
# -----------------------------
# Auto Refresh
# -----------------------------
sleep(2)
st.rerun()
