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
from preprocessing.eeg_filter import bandpass_filter
from features.eeg_features import (
    extract_eeg_features,
    calculate_attention_index
)
from features.sensor_fusion import calculate_fusion_score, fusion_state
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
    "Prototype attention monitoring using simulated EEG and heart-rate data."
)

# -----------------------------
# Generate Sensor Data
# -----------------------------

eeg_sensor = EEGReader()
heart_sensor = HeartRateReader()
motion_sensor = MotionReader()

time, raw_eeg = eeg_sensor.generate_eeg()
heart_rate = heart_sensor.read_heart_rate()
motion_level = motion_sensor.read_motion()

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
    "Time": time,
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
