"""NeuroSense research dashboard with authenticated, user-scoped views."""

from __future__ import annotations

import sys
from pathlib import Path
from time import sleep
from uuid import uuid4

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_PATH):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from backend.storage.database import initialize_database  # noqa: E402
from dashboard import auth_client, data_client  # noqa: E402


initialize_database()
st.set_page_config(page_title="NeuroSense", page_icon="🧠", layout="wide")

CSS_PATH = Path(__file__).parent / "static" / "neurosense.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _brand() -> None:
    st.markdown(
        '<div class="ns-brand"><div class="ns-kicker">Academic research prototype</div>'
        '<h1>🧠 NeuroSense</h1><p>Multi-sensor attention monitoring</p></div>',
        unsafe_allow_html=True,
    )


def _set_auth(result: dict) -> None:
    _reset_user_workspace()
    st.session_state.auth_token = result["access_token"]
    st.session_state.auth_user = result["user"]


def _reset_user_workspace() -> None:
    """Drop all in-memory sensor and chart state tied to the signed-in user."""
    for key in (
        "sensor_manager",
        "eeg_session_id",
        "attention_history",
        "smoothed_attention_score",
        "workspace_page",
    ):
        st.session_state.pop(key, None)


def _clear_auth() -> None:
    token = st.session_state.get("auth_token")
    auth_client.logout(token)
    st.session_state.pop("auth_token", None)
    st.session_state.pop("auth_user", None)
    _reset_user_workspace()


def render_authentication() -> None:
    _brand()
    st.markdown('<div class="ns-auth">', unsafe_allow_html=True)
    st.markdown('<div class="ns-kicker">Secure workspace</div>', unsafe_allow_html=True)
    st.subheader("Sign in to NeuroSense")
    login_tab, signup_tab = st.tabs(["Login", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", autocomplete="email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        if submitted:
            try:
                _set_auth(auth_client.login(email, password))
                st.success("Welcome back.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with signup_tab:
        with st.form("signup_form"):
            display_name = st.text_input("Display name")
            email = st.text_input("Email address", key="signup_email", autocomplete="email")
            password = st.text_input("Password", type="password", key="signup_password", autocomplete="new-password")
            confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
            st.caption("Use at least 10 characters with a lowercase letter, uppercase letter, and number.")
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    _set_auth(auth_client.register(email, password, display_name))
                    st.success("Account created.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Synthetic EEG feature-space benchmark interface. This prototype is not a medical device.")


def _sensor_snapshot() -> dict:
    from acquisition.unified_sensor_manager import UnifiedSensorManager
    from features.eeg_features import calculate_attention_index, extract_eeg_features
    from features.sensor_features import attention_score, attention_state
    from features.sensor_fusion import calculate_fusion_score, fusion_state
    from features.signal_quality import evaluate_aux_quality, evaluate_eeg_quality, overall_signal_status
    from model.calibrated_predict import predict_calibrated
    from model.predict import predict_attention
    from model.shap_explainer import explain_prediction
    from preprocessing.eeg_filter import bandpass_filter

    if "sensor_manager" not in st.session_state:
        st.session_state.sensor_manager = UnifiedSensorManager()
    sensor_data = st.session_state.sensor_manager.read_all()
    if "eeg_session_id" not in st.session_state:
        st.session_state.eeg_session_id = "dashboard-" + uuid4().hex
    sensor_data["session_id"] = st.session_state.eeg_session_id
    time_axis, raw_eeg = sensor_data["time"], sensor_data["eeg"]
    heart_rate, motion_level = sensor_data["heart_rate"], sensor_data["motion_level"]
    eeg_quality = evaluate_eeg_quality(raw_eeg)
    aux_quality = evaluate_aux_quality(heart_rate, motion_level)
    signal_status = overall_signal_status(eeg_quality, aux_quality)
    filtered_eeg = bandpass_filter(raw_eeg)
    features = extract_eeg_features(filtered_eeg)
    attention_index = calculate_attention_index(features)
    score = attention_score(attention_index)
    fusion_score = calculate_fusion_score(score, heart_rate, motion_level)
    state = fusion_state(fusion_score)
    ml_prediction, ml_confidence = predict_attention(
        features["theta"], features["alpha"], features["beta"], heart_rate, motion_level
    )
    calibrated_result = predict_calibrated(
        features["theta"], features["alpha"], features["beta"], heart_rate, motion_level
    )
    shap_prediction, shap_explanation = explain_prediction(
        features["theta"], features["alpha"], features["beta"], heart_rate, motion_level
    )
    raw_calibrated_score = calibrated_result["score"]
    if "smoothed_attention_score" not in st.session_state:
        st.session_state.smoothed_attention_score = raw_calibrated_score
    st.session_state.smoothed_attention_score = (
        0.35 * raw_calibrated_score + 0.65 * st.session_state.smoothed_attention_score
    )
    result = {
        "sensor_data": sensor_data, "time": time_axis, "raw_eeg": raw_eeg,
        "filtered_eeg": filtered_eeg, "heart_rate": heart_rate, "motion_level": motion_level,
        "sensor_mode": sensor_data["mode"], "eeg_quality": eeg_quality,
        "signal_status": signal_status, "features": features, "attention_index": attention_index,
        "score": score, "fusion_score": fusion_score, "state": state,
        "ml_prediction": ml_prediction, "ml_confidence": ml_confidence,
        "calibrated_result": calibrated_result,
        "calibrated_score": round(st.session_state.smoothed_attention_score, 1),
        "shap_prediction": shap_prediction, "shap_explanation": shap_explanation,
    }
    user = st.session_state.auth_user
    try:
        data_client.save_live_reading(user, sensor_data, {
            "prediction_allowed": signal_status["allow_prediction"],
            "signal_quality": signal_status,
            "prediction": {
                **calibrated_result,
                "attention_score": calibrated_result.get("score"),
                "model_version": "calibrated_xgboost_v1",
            },
        })
    except Exception as exc:
        st.warning(f"Live reading could not be saved: {exc}")
    from storage.data_logger import log_reading
    log_reading(heart_rate, motion_level, score, fusion_score, state)
    history = st.session_state.setdefault("attention_history", [])
    history.append({"Attention Score": fusion_score, "Heart Rate": heart_rate, "Motion Level": motion_level})
    st.session_state.attention_history = history[-30:]
    return result


def _render_metrics(snapshot: dict) -> None:
    columns = st.columns(4)
    values = [
        ("❤️ Heart Rate", f"{snapshot['heart_rate']} BPM"),
        ("🏃 Motion Level", f"{snapshot['motion_level']:.2f}"),
        ("🧠 Attention Score", f"{snapshot['fusion_score']}/100"),
        ("🎯 Attention State", snapshot["state"]),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)


def _navigate(page: str) -> None:
    st.session_state.workspace_page = page


def render_overview(snapshot: dict) -> None:
    st.title("Overview")
    st.caption(f"Sensor mode: {snapshot['sensor_mode'].upper()} · Live authenticated workspace")
    _render_metrics(snapshot)
    st.button(
        "📷 Start Webcam Monitoring",
        key="start_webcam_monitoring",
        type="primary",
        use_container_width=True,
        on_click=_navigate,
        args=("Webcam Monitoring",),
    )
    st.divider()
    q1, q2, q3 = st.columns(3)
    q1.metric("EEG Quality", f"{snapshot['eeg_quality']['quality'] * 100:.0f}%")
    q2.metric("EEG Status", snapshot["eeg_quality"]["status"])
    q3.metric("Overall Signal", snapshot["signal_status"]["status"])
    if not snapshot["signal_status"]["allow_prediction"]:
        st.warning("Signal quality is insufficient; treat predictions as experimental.")
    st.subheader("Attention trend")
    history_df = pd.DataFrame(st.session_state.get("attention_history", []))
    if not history_df.empty:
        st.line_chart(history_df.set_index(history_df.index)[["Attention Score", "Heart Rate", "Motion Level"]])
    st.info("Synthetic EEG feature-space benchmark results for academic prototyping; no clinical validity is implied.")


def render_webcam() -> None:
    from dashboard.webcam import webcam_preview

    st.title("Webcam Monitoring")
    st.button(
        "Back to Overview",
        key="back_to_overview",
        on_click=_navigate,
        args=("Overview",),
    )
    st.caption("Enable Camera, align your face, then choose Start Monitoring when calibration is complete.")
    webcam_preview(key="webcam_preview")
    st.caption("Video stays in your browser. No video is recorded or uploaded.")


def render_live(snapshot: dict) -> None:
    st.title("Live EEG")
    _render_metrics(snapshot)
    st.caption(f"Sensor mode: {snapshot['sensor_mode'].upper()}")
    st.subheader("Live EEG Signal")
    eeg_df = pd.DataFrame({"Time": snapshot["time"], "Raw EEG": snapshot["raw_eeg"], "Filtered EEG": snapshot["filtered_eeg"]})
    st.line_chart(eeg_df.set_index("Time"))
    st.subheader("EEG Frequency Bands")
    band_df = pd.DataFrame({"Band": ["Theta", "Alpha", "Beta"], "Power": [snapshot["features"]["theta"], snapshot["features"]["alpha"], snapshot["features"]["beta"]]})
    st.bar_chart(band_df.set_index("Band"))
    st.subheader("Attention Analysis")
    st.write("Experimental Attention Index:", round(snapshot["attention_index"], 3))
    st.progress(min(int(snapshot["fusion_score"]), 100))
    if snapshot["state"] == "Focused": st.success("Focused")
    elif snapshot["state"] == "Moderate": st.warning("Moderate Attention")
    else: st.error("Distracted")
    st.subheader("Machine Learning Analysis")
    ml_col1, ml_col2 = st.columns(2)
    ml_col1.metric("🤖 ML Prediction", snapshot["ml_prediction"])
    ml_col2.metric("📊 ML Confidence", f"{snapshot['ml_confidence']}%")
    calibrated = snapshot["calibrated_result"]
    st.subheader("Calibrated XGBoost Attention Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("🧠 Calibrated Attention Score", f"{snapshot['calibrated_score']}/100")
    c2.metric("🎯 Calibrated State", calibrated["state"])
    c3.metric("📊 Confidence", f"{calibrated['confidence']}%")
    probability_df = pd.DataFrame({"State": ["Focused", "Neutral", "Distracted"], "Probability": [calibrated["focused_probability"], calibrated["neutral_probability"], calibrated["distracted_probability"]]})
    st.bar_chart(probability_df.set_index("State"))
    st.caption("Attention score = 100 × [P(Focused) + 0.5 × P(Neutral)]. Displayed score is smoothed across readings.")
    st.subheader("Why This Prediction?")
    st.caption(f"SHAP explanation for current XGBoost prediction: {snapshot['shap_prediction']}")
    shap_df = pd.DataFrame(snapshot["shap_explanation"][:5])
    if not shap_df.empty:
        shap_df["absolute_impact"] = shap_df["impact"].abs()
        st.bar_chart(shap_df.sort_values("absolute_impact", ascending=False).set_index("feature")[["impact"]])
        st.write("Top contributing features:")
        for _, row in shap_df.sort_values("absolute_impact", ascending=False).iterrows():
            direction = "increased" if row["impact"] > 0 else "decreased"
            st.write(f"• {row['feature']}: {direction} support for the predicted class (SHAP impact {row['impact']:.4f})")
    st.subheader("Live Attention History")
    history_df = pd.DataFrame(st.session_state.get("attention_history", []))
    if not history_df.empty: st.line_chart(history_df[["Attention Score"]])
    st.caption("Displaying the latest 30 attention readings.")


def render_sessions(user: dict) -> None:
    st.title("Sessions")
    sessions = data_client.list_sessions(user)
    if not sessions:
        st.info("No saved EEG sessions yet. Start Live EEG to create one.")
        return
    frame = pd.DataFrame(sessions)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect session", [row["session_id"] for row in sessions])
    records = data_client.list_records(user, selected)
    if records:
        st.subheader("Prediction history")
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def render_analytics(user: dict) -> None:
    st.title("Analytics")
    records = data_client.list_records(user, limit=5000)
    if not records:
        st.info("Analytics will appear after authenticated EEG readings are recorded.")
        return
    frame = pd.DataFrame(records)
    m1, m2, m3 = st.columns(3)
    m1.metric("Readings", len(frame))
    m2.metric("Sessions", frame["session_id"].nunique())
    m3.metric("Mean attention", f"{frame['attention_score'].dropna().mean():.1f}")
    st.line_chart(frame.sort_values("timestamp").set_index("timestamp")[["attention_score", "heart_rate", "motion_level"]])
    st.bar_chart(frame["state"].value_counts())


def render_profile(user: dict) -> None:
    st.title("Profile")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write(f"**Display name:** {user['display_name']}")
    st.write(f"**Email:** {user['email']}")
    st.write(f"**Account created:** {user.get('created_at', 'Unknown')}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Your EEG sessions and prediction history are scoped to this account.")


def main() -> None:
    if not st.session_state.get("auth_token") or not st.session_state.get("auth_user"):
        render_authentication()
        st.stop()
    user = auth_client.current_user(st.session_state.auth_token)
    if user is None:
        _clear_auth()
        st.rerun()
    st.session_state.auth_user = user
    with st.sidebar:
        _brand()
        st.write(f"Signed in as **{user['display_name']}**")
        if st.button("Log out", use_container_width=True):
            _clear_auth()
            st.rerun()
        page = st.radio(
            "Workspace",
            ["Overview", "Live EEG", "Webcam Monitoring", "Sessions", "Analytics", "Profile"],
            key="workspace_page",
            label_visibility="collapsed",
        )
    if page in {"Overview", "Live EEG"}:
        snapshot = _sensor_snapshot()
        if page == "Overview": render_overview(snapshot)
        else: render_live(snapshot)
    elif page == "Webcam Monitoring": render_webcam()
    elif page == "Sessions": render_sessions(user)
    elif page == "Analytics": render_analytics(user)
    else: render_profile(user)
    if page == "Live EEG":
        sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
