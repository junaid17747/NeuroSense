"""Authenticated research-only seizure lab components and artifact gates."""

from __future__ import annotations

import os
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seizure_research.artifacts import read_json as _read_json
from seizure_research.schemas import AgeGroup, TimeInterval
from seizure_research.training import PILOT_CHANNELS, PILOT_FILES, PILOT_SPLITS
from dashboard.research_status import (POLL_SECONDS, clear_dashboard_status_cache,
                                       load_dashboard_snapshot)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = {
    "train": ("chb02", "chb05", "chb08"),
    "validation": ("chb09",),
    "test": ("chb06", "chb10"),
}
KNOWN_ALIAS = frozenset(("chb01", "chb21"))
SUPPORTED_PILOT_AGES = frozenset((AgeGroup.CHILD.value,))
MAX_SYNTHETIC_SECONDS = 60
MAX_REAL_VIEW_SECONDS = 30
LIVE_WINDOW_SECONDS = 10
MAX_LIVE_WINDOW_SECONDS = 30
LIVE_SAMPLE_RATE_HZ = 128


def validate_partitions(partitions: Mapping[str, Sequence[str]] = PARTITIONS) -> None:
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("patient_partition_names_invalid")
    values = [patient for group in partitions.values() for patient in group]
    if len(values) != len(set(values)):
        raise ValueError("patient_leakage_between_partitions")
    if set(values) & KNOWN_ALIAS:
        raise ValueError("known_chb01_chb21_alias_must_not_enter_pilot")
    if set(values) != set(PILOT_SPLITS):
        raise ValueError("patient_partition_cohort_mismatch")


def age_group_status(age_group: str) -> dict:
    try:
        normalized = AgeGroup(age_group).value
    except ValueError:
        return {"supported": False, "reason": "unknown_age_group"}
    if normalized not in SUPPORTED_PILOT_AGES:
        return {"supported": False, "reason": "unsupported_age_group_for_frozen_pilot"}
    return {"supported": True,
            "reason": "child_age_group_is_the_frozen_pilot_target; individual review remains required"}


def lab_empty_states(*, data_present: bool = False, model_present: bool = False,
                     approvals_present: bool = False, prediction_present: bool = False,
                     age_group: str = AgeGroup.CHILD.value) -> dict[str, str]:
    status = age_group_status(age_group)
    return {
        "data": "available" if data_present else "No approved real recordings are staged.",
        "model": "available" if model_present else "No real-data seizure model artifact exists.",
        "approvals": "available" if approvals_present else "Approval and readiness evidence has not been supplied to this lab session.",
        "predictions": "available" if prediction_present else "No real prediction artifact exists.",
        "age": "supported pilot age group" if status["supported"] else status["reason"],
    }


def synthetic_waveform(*, duration_seconds: int = 30, sampling_rate_hz: int = 128,
                       channels: Sequence[str] = ("Synthetic Fp1-F7", "Synthetic F7-T7")) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic bounded demonstration signal, never patient data."""
    if type(duration_seconds) is not int or not 1 <= duration_seconds <= MAX_SYNTHETIC_SECONDS:
        raise ValueError("synthetic_duration_out_of_bounds")
    if type(sampling_rate_hz) is not int or not 32 <= sampling_rate_hz <= 512:
        raise ValueError("synthetic_sampling_rate_out_of_bounds")
    if not channels or len(channels) > 4:
        raise ValueError("synthetic_channel_count_out_of_bounds")
    time = np.arange(duration_seconds * sampling_rate_hz, dtype=np.float64) / sampling_rate_hz
    traces = [35 * np.sin(2 * np.pi * (7 + index * 3) * time)
              + 8 * np.sin(2 * np.pi * 1.3 * time + index)
              for index in range(len(channels))]
    return time, np.column_stack(traces)


def synthetic_scrolling_waveform(elapsed_seconds: float, *,
                                 sampling_rate_hz: int = LIVE_SAMPLE_RATE_HZ,
                                 window_seconds: int = LIVE_WINDOW_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Bounded, deterministic moving window for the synthetic signal source."""
    if (not np.isfinite(elapsed_seconds) or elapsed_seconds < 0
            or type(sampling_rate_hz) is not int or not 32 <= sampling_rate_hz <= 512
            or type(window_seconds) is not int or not 1 <= window_seconds <= MAX_LIVE_WINDOW_SECONDS):
        raise ValueError("synthetic_stream_bounds_invalid")
    end = max(float(elapsed_seconds), 0.05)
    start = max(0.0, end - window_seconds)
    count = max(2, int((end - start) * sampling_rate_hz))
    axis = np.linspace(start, end, count, endpoint=False, dtype=np.float64)
    traces = [35 * np.sin(2 * np.pi * (7 + index * 3) * axis)
              + 8 * np.sin(2 * np.pi * 1.3 * axis + index)
              for index in range(2)]
    return axis, np.column_stack(traces)


def _elapsed_acquisition_seconds(mode: str, accumulated_seconds: float,
                                 started_at: float, now: float) -> float:
    """Compute the displayed monotonic clock without advancing paused/stopped playback."""
    if mode == "playing":
        return accumulated_seconds + max(0.0, now - started_at)
    if mode == "paused":
        return accumulated_seconds
    return 0.0


def playback_position(mode: str, elapsed: float, started: float, now: float,
                      speed: float, duration: float) -> float:
    """Bounded prerecorded clock; reaching the end never silently loops."""
    position = elapsed + max(0.0, now - started) * speed if mode == "playing" else elapsed
    return min(duration, max(0.0, position))


def _render_playback_controls(st, *, source: str, time_seconds: Sequence[float],
                              samples_uv: np.ndarray, channel_labels: Sequence[str],
                              state_key: str, annotations: Iterable = (),
                              predictions: Iterable = ()) -> None:
    """Seekable, speed-controlled playback of one bounded reviewed EDF segment."""
    import altair as alt
    axis, signal = np.asarray(time_seconds, dtype=float), np.asarray(samples_uv, dtype=float)
    if len(axis) < 2 or signal.shape != (len(axis), len(channel_labels)):
        raise ValueError("playback_segment_invalid")
    duration = float(axis[-1] - axis[0] + (axis[1] - axis[0]))
    mode_key, elapsed_key, started_key, speed_key = (f"{state_key}_{name}" for name in
                                                    ("mode", "elapsed", "started", "speed"))
    st.session_state.setdefault(mode_key, "stopped")
    st.session_state.setdefault(elapsed_key, 0.0)
    st.session_state.setdefault(speed_key, 1.0)
    now = time.monotonic()
    mode = st.session_state[mode_key]
    elapsed = playback_position(mode, st.session_state[elapsed_key],
                                st.session_state.get(started_key, now), now,
                                st.session_state[speed_key], duration)
    st.session_state[elapsed_key], st.session_state[started_key] = elapsed, now
    if elapsed >= duration and mode == "playing":
        st.session_state[mode_key] = mode = "ended"
    start_col, pause_col, resume_col, stop_col = st.columns(4)
    if start_col.button("Start", key=f"{state_key}_start", disabled=mode == "playing"):
        st.session_state[elapsed_key] = elapsed = 0.0
        st.session_state[mode_key] = mode = "playing"
    if pause_col.button("Pause", key=f"{state_key}_pause", disabled=mode != "playing"):
        st.session_state[mode_key] = mode = "paused"
    if resume_col.button("Resume", key=f"{state_key}_resume", disabled=mode != "paused"):
        st.session_state[mode_key] = mode = "playing"
    if stop_col.button("Stop / Reset", key=f"{state_key}_stop", disabled=mode == "stopped"):
        st.session_state[elapsed_key] = elapsed = 0.0
        st.session_state[mode_key] = mode = "stopped"
    speed = st.selectbox("Playback speed", (0.25, 0.5, 1.0, 2.0, 4.0), index=2,
                         format_func=lambda value: f"{value:g}×", key=f"{state_key}_speed_widget")
    st.session_state[speed_key] = speed
    def seek():
        st.session_state[elapsed_key] = float(st.session_state[f"{state_key}_seek"])
        st.session_state[started_key] = time.monotonic()
        if st.session_state[mode_key] in ("stopped", "ended"):
            st.session_state[mode_key] = "paused"
    st.slider("Seek within loaded segment (seconds)", min_value=0.0,
              max_value=float(duration), value=0.0, step=0.1,
              key=f"{state_key}_seek", on_change=seek)
    cursor = axis[0] + max(elapsed, min(duration, 0.05))
    begin = max(axis[0], cursor - LIVE_WINDOW_SECONDS)
    mask = (axis >= begin) & (axis <= cursor)
    if mask.sum() < 2:
        mask[:2] = True
    frame = pd.DataFrame(signal[mask], columns=channel_labels)
    frame.insert(0, "Recording time (s)", axis[mask])
    base = alt.Chart(frame).transform_fold(list(channel_labels), as_=["Channel", "EEG (µV)"])
    line = base.mark_line().encode(
        x=alt.X("Recording time (s):Q", scale=alt.Scale(zero=False)),
        y=alt.Y("EEG (µV):Q"), color=alt.Color("Channel:N"),
        tooltip=["Recording time (s):Q", "Channel:N", "EEG (µV):Q"])
    layers = []
    for intervals, color, label in ((annotations, "#e9b949", "Reviewed seizure"),
                                    (predictions, "#9c7bff", "Experimental candidate")):
        overlays = clip_overlays(intervals, float(axis[mask][0]), float(axis[mask][-1]))
        if overlays:
            overlay = pd.DataFrame([{"start":a, "end":b, "Type":label} for a,b in overlays])
            layers.append(alt.Chart(overlay).mark_rect(color=color, opacity=.22).encode(
                x="start:Q", x2="end:Q", tooltip=["Type:N", "start:Q", "end:Q"]))
    st.altair_chart(alt.layer(*layers, line).properties(height=320).interactive(), width="stretch")
    st.caption(f"Prerecorded CHB-MIT EDF playback · {source} · {mode} · "
               f"elapsed {elapsed:.1f}/{duration:.1f} s · recording time {axis[0]+elapsed:.1f} s · "
               f"{speed:g}× · µV · Gold: reviewed seizures; violet: experimental candidates")


def _render_live_synthetic_player(st) -> None:
    @st.fragment(run_every=1)
    def player():
        state_key = "synthetic_eeg_player"
        mode_key, elapsed_key, started_key = (f"{state_key}_{name}" for name in
                                              ("mode", "elapsed", "started"))
        st.session_state.setdefault(mode_key, "stopped")
        st.session_state.setdefault(elapsed_key, 0.0)
        mode = st.session_state[mode_key]
        now = time.monotonic()
        start_col, pause_col, resume_col, stop_col, elapsed_col = st.columns(5)
        if start_col.button("Start", key=f"{state_key}_start", disabled=mode == "playing"):
            st.session_state[elapsed_key] = 0.0
            st.session_state[started_key] = time.monotonic()
            st.session_state[mode_key] = mode = "playing"
            now = st.session_state[started_key]
        if pause_col.button("Pause", key=f"{state_key}_pause", disabled=mode != "playing"):
            st.session_state[elapsed_key] += max(
                0.0, now - st.session_state.get(started_key, now))
            st.session_state[mode_key] = mode = "paused"
        if resume_col.button("Resume", key=f"{state_key}_resume", disabled=mode != "paused"):
            st.session_state[started_key] = time.monotonic()
            st.session_state[mode_key] = mode = "playing"
        if stop_col.button("Stop / Reset", key=f"{state_key}_stop", disabled=mode == "stopped"):
            st.session_state[mode_key] = mode = "stopped"
            st.session_state[elapsed_key] = 0.0

        channel_col, window_col = st.columns(2)
        labels = ("Synthetic Fp1-F7", "Synthetic F7-T7")
        selected_channels = channel_col.multiselect(
            "Channels", labels, default=list(labels), key="synthetic_eeg_player_channels")
        window_seconds = window_col.slider(
            "Display window (seconds)", min_value=5, max_value=MAX_LIVE_WINDOW_SECONDS,
            value=LIVE_WINDOW_SECONDS, step=1, key="synthetic_eeg_display_window")
        elapsed = _elapsed_acquisition_seconds(
            mode, float(st.session_state.get(elapsed_key, 0.0)),
            float(st.session_state.get(started_key, now)), now)
        axis, values = synthetic_scrolling_waveform(
            elapsed, window_seconds=window_seconds)
        frame = pd.DataFrame(values, columns=labels)
        frame.insert(0, "Acquisition time (s)", axis)
        state_label = {"playing": "Acquiring", "paused": "Paused", "stopped": "Stopped / reset"}[mode]
        elapsed_col.caption("Elapsed time")
        elapsed_col.write(f"**{elapsed:.1f} s**")
        st.caption(f"Live Synthetic EEG · Status: {state_label} · 128 Hz · Simulated signal, not patient data")
        if selected_channels:
            st.line_chart(frame[["Acquisition time (s)", *selected_channels]],
                          x="Acquisition time (s)", y=selected_channels,
                          height=320, width="stretch")
        else:
            st.info("Select at least one channel to show the waveform.")
    player()


def _metric_text(metrics: Mapping | None) -> tuple[str, str, str, str]:
    if not metrics:
        return ("Not evaluated", "Not evaluated", "Not evaluated", "Not evaluated")
    detected = metrics.get("detected_events")
    reference = metrics.get("reference_events")
    sensitivity = metrics.get("event_sensitivity")
    if reference == 0 or sensitivity is None:
        sensitivity_text = "Unavailable (no evaluable reference events)"
    else:
        sensitivity_text = f"{detected}/{reference} · {100 * sensitivity:.1f}%"
    missed = metrics.get("missed_events")
    missed_text = "Unavailable" if missed is None else str(missed)
    rate, hours = metrics.get("false_alarms_per_hour"), metrics.get("evaluable_hours")
    false_text = ("Unavailable (no evaluable non-seizure hours)" if rate is None or not hours
                  else f"{metrics.get('false_events', '—')} / {hours:.2f} h · {rate:.3f}/h")
    latency, matched = metrics.get("median_detection_latency_seconds"), metrics.get("detected_events")
    latency_text = ("Unavailable (no matched events)" if latency is None
                    else f"{latency:.2f} s median · {matched} matched")
    return sensitivity_text, missed_text, false_text, latency_text


def _render_live_research_status(st, *, selected_age: str) -> None:
    @st.fragment(run_every=POLL_SECONDS)
    def status_fragment():
        archive_root = Path(os.environ.get(
            "NEUROSENSE_CHBMIT_ARCHIVE_ROOT",
            str(Path.home() / "NeuroSense-EEG-Data" / "archive")))
        temporary_root = Path(os.environ.get(
            "NEUROSENSE_CHBMIT_TEMPORARY_ROOT",
            str(Path.home() / "NeuroSense-EEG-Data" / "temporary")))
        approval_path = Path(os.environ.get(
            "NEUROSENSE_CHBMIT_ARCHIVE_APPROVAL",
            str(Path.home() / "NeuroSense-EEG-Data" / "manifests" / "archive-approval.json")))
        review_path = Path(os.environ.get(
            "NEUROSENSE_CHBMIT_REVIEW_PACKET",
            str(Path.home() / "NeuroSense-EEG-Data" / "reports" / "annotation-review-packet.json")))
        pilot_root = Path(os.environ["NEUROSENSE_CHBMIT_ROOT"]) if os.environ.get("NEUROSENSE_CHBMIT_ROOT") else None
        run_root = Path(os.environ["NEUROSENSE_SEIZURE_RUN_DIR"]) if os.environ.get("NEUROSENSE_SEIZURE_RUN_DIR") else None
        test_artifact = (Path(os.environ["NEUROSENSE_SEIZURE_TEST_ARTIFACT"])
                         if os.environ.get("NEUROSENSE_SEIZURE_TEST_ARTIFACT") else None)
        try:
            snapshot = load_dashboard_snapshot(
                archive_root=archive_root, approval_path=approval_path,
                temporary_root=temporary_root, review_path=review_path, pilot_root=pilot_root,
                run_root=run_root, test_artifact=test_artifact)
        except Exception:
            snapshot = {"errors": ["Status refresh failed; research results remain unavailable."],
                        "metrics": None, "test_metrics": None, "provenance": {}}
        multipatient = snapshot.get("cohort_kind") == "multipatient"
        if multipatient:
            real_metrics = snapshot.get("metrics_by_age", {}).get(selected_age, {}).get("metrics")
        else:
            real_metrics = snapshot.get("metrics") if age_group_status(selected_age)["supported"] else None
        for column, label, value in zip(st.columns(4),
                ("Event sensitivity", "Missed seizures", "False alarms/hour", "Detection latency"),
                _metric_text(real_metrics)):
            column.metric(label, value)
        if real_metrics is not None:
            st.caption(("Pooled-model held-out results for the selected age group; no age-specific validation. " if multipatient else "Held-out research evaluation · ")
                       + "Latency is the median delay for matched seizures.")
        elif multipatient:
            st.caption("Not evaluated: no reviewed held-out results exist for the selected age group.")
        elif not age_group_status(selected_age)["supported"]:
            st.caption("Not evaluated: the selected age group is outside the child pilot scope.")
        else:
            st.caption("Not evaluated: no verified real-model evaluation is available. Synthetic EEG demonstrates playback only.")

        dataset_status = ("recordings verified" if snapshot.get("archive_integrity") == "complete_verified"
                          else "recordings pending verification" if snapshot.get("archive") != "not staged"
                          else "recordings not staged")
        st.write(f"**Research status:** CHB-MIT {dataset_status} · "
                 f"Model: {snapshot.get('model', 'unavailable')} · Research only, not a diagnostic tool.")
        with st.expander("Research Details", expanded=False):
            st.button("Refresh now", key="seizure_status_refresh",
                      on_click=clear_dashboard_status_cache)
            st.caption("Updated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
            if _render_dataset_overview(st) != selected_age:
                st.rerun()
            st.subheader("Readiness checks")
            status_rows = (
                ("Dataset inventory", snapshot.get("inventory", "unavailable")),
                ("Archive staging", snapshot.get("archive", "not staged")),
                ("Archive checksum verification", snapshot.get("archive_integrity", "not verified")),
                ("Archive recordings verified", f"{snapshot.get('recordings_verified', 0)} / {snapshot.get('recordings_available', 0)}"),
                ("Annotation review", snapshot.get("annotation_review", "not reviewed")),
                ("Data-use / staging approval", snapshot.get("data_use_staging", "not approved")),
                ("Processing authorization", snapshot.get("processing_authorization", "not approved")),
                ("Preprocessing readiness", snapshot.get("preprocessing", "not ready")),
                ("Training / evaluation readiness", snapshot.get("training_readiness", "not ready")),
                ("Seizure model", snapshot.get("model", "unavailable")),
                ("Model training status", snapshot.get("training_status", "not trained or validated")),
                ("Model eligibility", snapshot.get("model_eligibility", "not established")),
                ("Held-out evaluation", snapshot.get("evaluation", "unavailable")),
                ("Dashboard artifact readiness", snapshot.get("dashboard_artifacts", "unavailable")),
            )
            for label, value in status_rows:
                st.write(f"**{label}:** {value}")
            if snapshot.get("errors"):
                for error in snapshot["errors"]:
                    st.warning(error)
            if snapshot.get("readiness_blockers"):
                st.write("**Readiness blocker codes:**", snapshot["readiness_blockers"])

            if snapshot.get("test_metrics") is not None:
                st.warning(snapshot.get("test_artifact_label", "SYNTHETIC TEST ONLY — NOT RESEARCH PERFORMANCE"))
                st.json({"synthetic_test_metrics": snapshot["test_metrics"]})
            provenance = snapshot.get("provenance", {})
            st.subheader("Research provenance")
            readable = {key: provenance.get(key, "Unavailable") for key in (
                "dataset", "cohort", "manifest_sha256", "canonical_patients", "recordings",
                "model_name", "model_version", "training_timestamp", "dataset_manifest_sha256",
                "split_summary", "preprocessing_version")}
            readable["evaluation_timestamp"] = (snapshot.get("evaluation_timestamp")
                                                or (real_metrics or {}).get("evaluated_at_utc")
                                                or "Unavailable")
            for label, value in readable.items():
                st.write(f"**{label.replace('_', ' ').title()}:** {value}")
            st.json({"provenance": provenance, "dataset_status": snapshot.get("inventory"),
                     "archive_status": snapshot.get("archive"),
                     "evaluation_status": snapshot.get("evaluation"),
                     "test_fixture_only": snapshot.get("test_metrics") is not None})
    status_fragment()


def _interval_pair(value) -> tuple[float, float]:
    if isinstance(value, TimeInterval):
        start, end = value.start_seconds, value.end_seconds
    elif isinstance(value, Mapping):
        start, end = value["start_seconds"], value["end_seconds"]
    else:
        start, end = value
    start, end = float(start), float(end)
    if not np.isfinite((start, end)).all() or start < 0 or end <= start:
        raise ValueError("overlay_interval_invalid")
    return start, end


def plot_waveform(time_seconds: Sequence[float], signal_uv: np.ndarray,
                  channel_labels: Sequence[str], *, annotations: Iterable = (),
                  predictions: Iterable = (), title: str = "EEG waveform"):
    """Plot signal and independently colored reviewed/predicted intervals."""
    time = np.asarray(time_seconds, dtype=float)
    signals = np.asarray(signal_uv, dtype=float)
    labels = tuple(channel_labels)
    if (time.ndim != 1 or signals.ndim != 2 or signals.shape[0] != len(time)
            or signals.shape[1] != len(labels) or not len(time)
            or not np.isfinite(time).all() or not np.isfinite(signals).all()
            or np.any(np.diff(time) <= 0)):
        raise ValueError("waveform_shape_or_timebase_invalid")
    if len(time) > 512 * MAX_SYNTHETIC_SECONDS:
        raise ValueError("waveform_plot_sample_limit_exceeded")
    fig, axis = plt.subplots(figsize=(12, max(3.0, 1.5 * len(labels))), layout="constrained")
    spacing = max(150.0, float(np.ptp(signals)) * 1.2)
    for index, label in enumerate(labels):
        axis.plot(time, signals[:, index] + index * spacing, linewidth=0.65, label=label)
    duration_end = float(time[-1] + (np.median(np.diff(time)) if len(time) > 1 else 0))
    for start, end in map(_interval_pair, annotations):
        if end > duration_end:
            plt.close(fig)
            raise ValueError("annotation_interval_outside_waveform")
        axis.axvspan(start, end, color="#f4b942", alpha=0.22,
                     label="Reviewed seizure annotation" if not any(
                         patch.get_label() == "Reviewed seizure annotation" for patch in axis.patches) else "_nolegend_")
    for start, end in map(_interval_pair, predictions):
        if end > duration_end:
            plt.close(fig)
            raise ValueError("prediction_interval_outside_waveform")
        axis.axvspan(start, end, color="#b38cff", alpha=0.24,
                     label="Experimental model candidate" if not any(
                         patch.get_label() == "Experimental model candidate" for patch in axis.patches) else "_nolegend_")
    axis.set(title=title, xlabel="Recording time (seconds)", ylabel="Offset amplitude (µV)")
    axis.legend(loc="upper right", fontsize="small")
    axis.grid(alpha=0.18)
    return fig


def load_real_prediction_artifacts(run_dir: Path | str, *, readiness_report: Mapping,
                                   approval_record: Mapping) -> dict:
    """Load artifacts from the independent seizure model without attention inference."""
    from seizure_research.artifacts import load_prediction_artifacts
    return load_prediction_artifacts(run_dir, readiness_report=readiness_report,
                                     approval_record=approval_record)


def clip_overlays(intervals: Iterable, start_seconds: float, end_seconds: float) -> tuple[dict, ...]:
    """Intersect full-record annotations/candidates with the displayed range."""
    result = []
    for left, right in map(_interval_pair, intervals):
        left, right = max(left, start_seconds), min(right, end_seconds)
        if left < right:
            result.append({"start_seconds": left, "end_seconds": right})
    return tuple(result)


def read_approved_real_waveform(root: Path | str, relative_path: str, *,
                                channel_indices: Sequence[int], start_seconds: float,
                                end_seconds: float, readiness_report: Mapping,
                                approval_record: Mapping) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Decode one bounded real EDF range only after readiness and authorization."""
    if readiness_report.get("ready_for_real_data_processing") is not True:
        raise PermissionError("real_waveform_readiness_missing")
    if (approval_record.get("processing_status") != "approved"
            or not approval_record.get("processing_approver")
            or not approval_record.get("processing_approved_on")
            or not approval_record.get("processing_approval_ref")):
        raise PermissionError("real_waveform_processing_approval_missing")
    if relative_path not in PILOT_FILES:
        raise ValueError("recording_outside_frozen_cohort")
    requested_root = Path(root)
    if requested_root.is_symlink():
        raise ValueError("real_waveform_root_symlink_not_allowed")
    resolved_root = requested_root.resolve(strict=True)
    if resolved_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("research_data_must_be_outside_repository")
    if (not np.isfinite((start_seconds, end_seconds)).all() or start_seconds < 0
            or end_seconds <= start_seconds or end_seconds - start_seconds > MAX_REAL_VIEW_SECONDS):
        raise ValueError("real_waveform_window_out_of_bounds")
    from seizure_research.waveform import read_edf
    result = read_edf(resolved_root, relative_path, channel_indices=channel_indices,
                      start_seconds=start_seconds, end_seconds=end_seconds,
                      max_samples=2_000_000)
    rates = {item.metadata.sampling_rate_hz for item in result.channels}
    if len(rates) != 1:
        raise ValueError("real_waveform_mixed_sampling_rates_unsupported")
    return (result.channels[0].timestamps_seconds,
            np.column_stack([item.samples_uv for item in result.channels]),
            tuple(item.metadata.label for item in result.channels))


def _approved_lab_assets(root: Path | str) -> tuple[dict, dict, object, object]:
    """Run the read-only D4 gate and load only reviewed metadata after a click."""
    from seizure_research.preprocessing import PreprocessingConfig
    from seizure_research.readiness import check_readiness
    from seizure_research.training import AnnotationEvidenceManifest
    from seizure_research.ingestion import CohortManifest

    base = Path(root).resolve(strict=True)
    cohort_path = base / "cohort-manifest.json"
    preprocessing_path = base / "preprocessing.json"
    evidence_path = base / "annotation-evidence.json"
    approval_path = base / "faculty-approval.json"
    report = check_readiness(
        root=base, cohort_manifest_path=cohort_path,
        preprocessing_config_path=preprocessing_path,
        annotation_evidence_path=evidence_path, approval_record_path=approval_path,
        file_manifest_path=Path(os.environ.get(
            "NEUROSENSE_CHBMIT_FILE_MANIFEST",
            str(PROJECT_ROOT / "docs/chbmit_pilot_file_manifest.json"),
        )).expanduser())
    if not report["ready_for_real_data_processing"]:
        return report, {}, None, None
    approval = _read_json(approval_path)
    return (report, approval, CohortManifest.model_validate(_read_json(cohort_path)),
            PreprocessingConfig.model_validate(_read_json(preprocessing_path)))


def _render_dataset_overview(st) -> str:
    """Secondary dataset information and the existing pilot-age selector."""
    # Public inventory and deterministic proposal are descriptive metadata only.
    # A proposal is not an acquired archive, reviewed annotation set, or model.
    proposal_path = PROJECT_ROOT / "reports" / "checkpoint_d7" / "selection_proposal.json"
    inventory_path = PROJECT_ROOT / "reports" / "checkpoint_d7" / "official_inventory.json"
    proposal = inventory = None
    try:
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    st.subheader("Dataset overview")
    if proposal and inventory:
        st.write(f"**CHB-MIT version:** {inventory.get('dataset_version', 'Unavailable')} · **Official case directories:** {inventory.get('case_count', 'Unavailable')} · **Canonical patients:** {inventory.get('canonical_patient_count', 'Unavailable')} · **Official EDF inventory:** {inventory.get('recording_count', 'Unavailable')} recordings")
        st.write(f"**Proposed cohort:** {proposal.get('selected_patient_count', 0)} canonical patients · {proposal.get('selected_recording_count', 0)} EDFs · {proposal.get('selected_original_edf_bytes', 0):,} original bytes")
        st.write(f"**Archive budget:** {proposal.get('budget_bytes', 0):,} bytes · conservative retained bound {proposal.get('total_retained_byte_bound', 0):,} bytes · proposal only, not staged")
        st.write(f"**Declared events in selected summaries:** {proposal.get('selected_declared_seizure_events', 0)} (unreconciled) · official positive recordings: {proposal.get('selected_official_seizure_positive_recordings', 0)} · verified seizure-free recordings: 0")
        st.caption(f"Inventory SHA-256: {inventory.get('inventory_sha256', 'unavailable')} · selection SHA-256: {proposal.get('selection_sha256', 'unavailable')}")
    else:
        st.info("Official inventory and cohort proposal are not available in this checkout.")
    st.subheader("Storage status")
    st.write(f"**System disk free:** {shutil.disk_usage('/').free:,} bytes · minimum reserve: 21,474,836,480 bytes")
    st.write("**Compression:** No authorized EDF sample was available for benchmarking; actual lossless compression ratio is unknown.")
    st.subheader("Age eligibility")
    if proposal:
        st.write(f"Published case-age groups in the selected proposal: {proposal.get('selected_age_group_case_counts', {})}. chb24 age is unknown; neonatal, infant and older-adult cases are not represented. Child, adolescent and adult ages are represented, but no age-specific model is validated or trained.")
    else:
        st.write("Age eligibility has not been established for a staged cohort.")
    st.subheader("Frozen patient-independent cohort")
    cohort = pd.DataFrame([{"Partition": split.title(), "Patients": ", ".join(PARTITIONS[split]),
                            "Recordings": {"train": 6, "validation": 2, "test": 4}[split]}
                           for split in ("train", "validation", "test")])
    st.dataframe(cohort, hide_index=True, use_container_width=True)
    st.caption("Six patients · 12 EDFs · chb01/chb21 are recognized as the same patient and excluded.")

    selected_age = st.selectbox("Age group", [item.value for item in AgeGroup], index=2,
                                key="seizure_age_group")
    age = age_group_status(selected_age)
    if age["supported"]:
        st.info("Child is the frozen pilot target. Individual age metadata, permissions, annotation coverage, and signal quality still require review; model availability is checked below.")
    else:
        st.info(f"This age group is outside the frozen pilot: {age['reason']}.")

    return selected_age


def _archive_viewer_settings():
    base = Path.home() / "NeuroSense-EEG-Data"
    return {
        "root": Path(os.environ.get("NEUROSENSE_CHBMIT_ARCHIVE_ROOT", str(base / "archive"))),
        "temporary_root": Path(os.environ.get("NEUROSENSE_CHBMIT_TEMPORARY_ROOT", str(base / "temporary"))),
        "approval_path": Path(os.environ.get("NEUROSENSE_CHBMIT_ARCHIVE_APPROVAL", str(base / "manifests/archive-approval.json"))),
        "review_path": Path(os.environ.get("NEUROSENSE_CHBMIT_REVIEW_PACKET", str(base / "reports/annotation-review-packet.json"))),
        "plan_path": Path(os.environ.get("NEUROSENSE_CHBMIT_REVIEW_PLAN", str(base / "manifests/review-plan.json"))),
        "preprocessing_path": Path(os.environ.get("NEUROSENSE_CHBMIT_PREPROCESSING", str(base / "manifests/preprocessing.json"))),
        "proposal_path": Path(os.environ.get("NEUROSENSE_CHBMIT_SPLIT_PROPOSAL", str(PROJECT_ROOT / "reports/final/proposed_patient_split.json"))),
    }


def _render_archive_edf_viewer(st, selected_age):
    from seizure_research.archive import authorize
    from seizure_research.multipatient import validate_review_inputs, read_archive_waveform
    settings = _archive_viewer_settings()
    try:
        inventory = _read_json(PROJECT_ROOT / "reports/checkpoint_d7/official_inventory.json")
        selection = _read_json(PROJECT_ROOT / "reports/checkpoint_d7/selection_proposal.json")
        authorize(settings["root"],settings["temporary_root"],settings["approval_path"],inventory,selection,processing=True)
        proposal, packet, plan_data, configs = (_read_json(settings[key]) for key in
                                              ("proposal_path","review_path","plan_path","preprocessing_path"))
        plan, _ = validate_review_inputs(inventory,selection,proposal,plan_data,packet,configs)
    except (OSError,ValueError,PermissionError,KeyError,TypeError):
        st.session_state.pop("archive_edf_player_segment", None)
        st.info("Real CHB-MIT playback is unavailable until recordings and their reviews are ready.")
        return
    eligible = [r for r in plan.recordings if r.age.group.value == selected_age]
    if not eligible:
        st.session_state.pop("archive_edf_player_segment", None)
        st.info("No reviewed recordings are available for the selected age group.")
        return
    patient = st.selectbox("Patient", sorted({plan.identities.resolve(r.subject_id) for r in eligible}),key="archive_edf_patient")
    relative = st.selectbox("Recording", [r.edf_path for r in eligible if plan.identities.resolve(r.subject_id) == patient],
                            key="archive_edf_recording")
    selected_channels = st.multiselect("EDF channels (up to 4)",PILOT_CHANNELS,default=list(PILOT_CHANNELS[:2]),
                                       max_selections=4,key="archive_edf_channels")
    review = next(r for r in packet["recordings"] if r["recording_id"] == relative)
    duration = float(review["recording_duration_seconds"])
    start = st.number_input("Recording time start (seconds)",min_value=0.0,max_value=max(0.0,duration-5),
                            value=0.0,step=1.0,key="archive_edf_start")
    length = st.slider("Playback segment length (seconds)",5,MAX_REAL_VIEW_SECONDS,20,5,key="archive_edf_length")
    request = (relative,tuple(selected_channels),start,length,packet["packet_sha256"],selected_age)
    if st.session_state.get("archive_edf_player_request") != request:
        st.session_state.pop("archive_edf_player_segment",None)
    if st.button("Verify and load EDF segment",key="archive_edf_load"):
        st.session_state.pop("archive_edf_player_segment",None)
        try:
            segment = read_archive_waveform(settings["root"],settings["temporary_root"],inventory,selection,
                                            approval_path=settings["approval_path"],proposal=proposal,
                                            review_plan=plan_data,review_packet=packet,preprocessing=configs,
                                            relative_path=relative,channel_labels=tuple(selected_channels),
                                            start_seconds=float(start),end_seconds=min(duration,start+length))
            segment.pop("duration_seconds")
            recording_id = segment.pop("recording_id")
            segment.update(source=relative,state_key="archive_edf_player",predictions=())
            run_path = os.environ.get("NEUROSENSE_SEIZURE_RUN_DIR")
            if run_path:
                try:
                    from seizure_research.multipatient import load_multipatient_artifacts
                    bundle = load_multipatient_artifacts(run_path,root=settings["root"],temporary_root=settings["temporary_root"],
                                                         inventory=inventory,selection=selection,approval_path=settings["approval_path"],
                                                         review_path=settings["review_path"])
                    segment["predictions"] = tuple(e for e in bundle["predictions"]["candidate_events"]
                                                    if e["recording_id"] == recording_id)
                except (OSError,ValueError,PermissionError,KeyError,TypeError):
                    pass
            st.session_state["archive_edf_player_segment"] = segment
            st.session_state["archive_edf_player_request"] = request
            st.session_state["archive_edf_player_mode"] = "stopped"
            st.session_state["archive_edf_player_elapsed"] = 0.0
            st.session_state.pop("archive_edf_player_seek",None)
        except (OSError,ValueError,PermissionError,KeyError,TypeError):
            st.error("The recording could not be verified or decoded. Review its source and annotation evidence.")
    if st.session_state.get("archive_edf_player_segment"):
        @st.fragment(run_every=1)
        def archive_player():
            # Re-read current authorization before exposing retained session data.
            try:
                authorize(settings["root"],settings["temporary_root"],settings["approval_path"],inventory,selection,processing=True)
                if _read_json(settings["review_path"]).get("packet_sha256") != packet["packet_sha256"]:
                    raise ValueError("review_evidence_changed")
            except (OSError,ValueError,PermissionError):
                st.session_state.pop("archive_edf_player_segment",None)
                st.info("Recording access is no longer authorized.")
                return
            segment = st.session_state.get("archive_edf_player_segment")
            if segment:
                _render_playback_controls(st,**segment)
        archive_player()


def render_seizure_lab(st) -> None:
    """Render the authenticated Streamlit section; no data is auto-loaded."""
    validate_partitions()
    st.title("EEG Seizure Research")
    selected_age = st.session_state.get("seizure_age_group", AgeGroup.CHILD.value)
    age = age_group_status(selected_age)
    signal_source = st.radio("Signal source", (
        "Live Synthetic EEG", "Verified CHB-MIT EDF playback",
        "Compatible EEG headset stream"), horizontal=True, key="seizure_signal_source")
    if signal_source == "Live Synthetic EEG":
        _render_live_synthetic_player(st)
    elif signal_source == "Compatible EEG headset stream":
        st.info("No compatible headset stream is connected. Hardware live inference is not implemented.")
    else:
        st.caption("Prerecorded CHB-MIT EEG · Select and load an approved recording to begin playback.")
    root_setting = os.environ.get("NEUROSENSE_CHBMIT_ROOT")
    archive_configured = _archive_viewer_settings()["plan_path"].is_file()
    use_archive = not root_setting
    if signal_source == "Verified CHB-MIT EDF playback" and root_setting and archive_configured:
        use_archive = st.selectbox("Recording cohort",("Multipatient archive","Frozen six-patient pilot"),
                                   key="seizure_recording_cohort") == "Multipatient archive"
    if signal_source != "Verified CHB-MIT EDF playback":
        pass
    elif use_archive:
        _render_archive_edf_viewer(st,selected_age)
    elif not age["supported"]:
        st.info("Real-data waveforms, seizure models and results are unavailable for this age group.")
    elif not root_setting:
        st.info("Approved real-data storage is not configured. Reviewed annotations and real waveform access are unavailable.")
    else:
        patient = st.selectbox("Pilot patient",sorted(PILOT_SPLITS),key="seizure_edf_patient")
        relative = st.selectbox("Pilot EDF recording", sorted(p for p in PILOT_FILES if p.startswith(patient+"/")), key="seizure_edf_recording")
        selected_channels = st.multiselect(
            "EDF channels (up to 4)", PILOT_CHANNELS, default=list(PILOT_CHANNELS[:2]),
            key="seizure_edf_channels", max_selections=4)
        window_start = st.number_input("Recording time start (seconds)", min_value=0,
                                       max_value=86400, value=0, step=10,
                                       key="seizure_edf_window_start")
        window_length = st.slider("Playback segment length (seconds)", 5, MAX_REAL_VIEW_SECONDS,
                                  20, 5, key="seizure_edf_window_length")
        request = (root_setting, relative, tuple(selected_channels), window_start, window_length)
        if st.session_state.get("approved_edf_player_request") != request:
            st.session_state.pop("approved_edf_player_segment", None)
        if st.button("Verify access and load bounded EDF segment", key="seizure_lab_load_real"):
            st.session_state.pop("approved_edf_player_segment", None)
            try:
                readiness, approval, manifest, config = _approved_lab_assets(root_setting)
                if not readiness["ready_for_real_data_processing"]:
                    st.error("Real-data access is locked. The readiness audit found outstanding requirements.")
                    st.caption("Private approval-record contents are not shown.")
                else:
                    entry = next(item for item in manifest.records
                                 if item.signal.relative_path == relative)
                    end_seconds = float(window_start + window_length)
                    annotations = clip_overlays(entry.recording.annotations, window_start, end_seconds)
                    channel_map = {item.target_label: item.source_index for item in config.channels}
                    channel_indices = tuple(channel_map[label] for label in selected_channels
                                             if label in channel_map)
                    if not channel_indices:
                        raise ValueError("select_at_least_one_supported_channel")
                    real_time, real_signal, real_labels = read_approved_real_waveform(
                        root_setting, relative, channel_indices=channel_indices,
                        start_seconds=window_start, end_seconds=end_seconds,
                        readiness_report=readiness, approval_record=approval)
                    real_predictions = ()
                    run_setting = os.environ.get("NEUROSENSE_SEIZURE_RUN_DIR")
                    if run_setting:
                        try:
                            bundle = load_real_prediction_artifacts(
                                run_setting, readiness_report=readiness,
                                approval_record=approval)
                            real_predictions = tuple(
                                {"start_seconds": item["start_seconds"],
                                 "end_seconds": item["end_seconds"]}
                                for item in bundle["predictions"]["candidate_events"]
                                if item["recording_id"] == entry.recording.recording_id)
                        except (OSError, ValueError, PermissionError, KeyError, TypeError):
                            real_predictions = ()
                    st.session_state["approved_edf_player_request"] = request
                    st.session_state["approved_edf_player_mode"] = "stopped"
                    st.session_state["approved_edf_player_elapsed"] = 0.0
                    st.session_state.pop("approved_edf_player_seek",None)
                    st.session_state["approved_edf_player_segment"] = dict(
                        source=relative, time_seconds=real_time, samples_uv=real_signal,
                        channel_labels=real_labels, annotations=annotations,
                        predictions=clip_overlays(real_predictions, window_start, end_seconds),
                        state_key="approved_edf_player")
            except (OSError, ValueError, PermissionError, KeyError, TypeError, StopIteration) as exc:
                st.error(f"Real-data viewer remains unavailable: {exc}")
        if st.session_state.get("approved_edf_player_segment"):
            @st.fragment(run_every=1)
            def edf_player():
                st.caption("Gold: reviewed seizures · Violet: experimental model candidates")
                _render_playback_controls(st, **st.session_state["approved_edf_player_segment"])
            edf_player()

    _render_live_research_status(st, selected_age=selected_age)
