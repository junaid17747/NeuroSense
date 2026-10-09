"""Browser-only camera positioning; mounted only in the authenticated workspace."""

from pathlib import Path

import streamlit as st


STATIC = Path(__file__).parent / "static"

webcam_preview = st.components.v2.component(
    "neurosense_webcam_preview",
    html="""
    <section class="webcam-panel" aria-label="Face alignment camera-positioning check">
        <div class="alignment-heading">
            <span class="alignment-kicker">CAMERA SETUP</span>
            <h3>Find your frame</h3>
            <p>Center your face, look straight ahead, and keep your eyes comfortably open.</p>
        </div>
        <div class="webcam-controls">
            <button id="enable-camera" type="button">Enable Camera</button>
            <button id="disable-camera" type="button" disabled>Disable Camera</button>
        </div>
        <p id="camera-status" role="status" aria-live="polite">Camera is off.</p>
        <div class="webcam-stage">
            <p id="camera-placeholder">Enable your camera to begin the positioning check.</p>
            <div id="video-viewport" hidden>
                <video id="camera-video" aria-label="Live webcam preview" autoplay muted playsinline hidden></video>
                <div id="alignment-guide" data-tone="neutral" aria-hidden="true"></div>
                <span class="preview-label">LIVE · ON DEVICE</span>
            </div>
        </div>
        <div class="alignment-feedback">
            <p id="alignment-message" role="status" aria-live="polite">Camera positioning has not started.</p>
            <div class="calibration-label">
                <label for="calibration-progress">Alignment calibration</label>
                <span id="calibration-value">0%</span>
            </div>
            <progress id="calibration-progress" max="1" value="0">0%</progress>
            <p class="alignment-hint">Keep one face centered for 3 continuous seconds.</p>
            <div class="webcam-controls">
                <button id="start-monitoring" type="button" disabled>Start Monitoring</button>
                <button id="pause-monitoring" type="button" disabled>Pause</button>
                <button id="resume-monitoring" type="button" disabled>Resume</button>
                <button id="stop-monitoring" type="button" disabled>Stop Monitoring</button>
                <button id="retry-alignment" type="button" hidden>Retry Alignment</button>
            </div>
        </div>
        <p class="alignment-disclaimer">Alignment is a camera-positioning check. All session measurements
        are experimental behavioral indicators for adult wellness research. They do not measure brainwaves
        or stress, diagnose fatigue or neurological disease, or detect seizures. No identity verification or face matching.</p>
        <div class="tiredness-inputs">
            <label>Before-session tiredness (optional, self-reported)
                <select id="tiredness-before" aria-label="Before-session tiredness">
                    <option value="">Not provided</option>
                    <option value="0">0 — Not tired</option><option>1</option><option>2</option><option>3</option><option>4</option>
                    <option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option value="10">10 — Very tired</option>
                </select>
            </label>
            <label>After-session tiredness (optional, self-reported)
                <select id="tiredness-after" aria-label="After-session tiredness" disabled>
                    <option value="">Not provided</option>
                    <option value="0">0 — Not tired</option><option>1</option><option>2</option><option>3</option><option>4</option>
                    <option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option value="10">10 — Very tired</option>
                </select>
            </label>
        </div>
        <p id="rating-summary" class="alignment-hint">Self-reports are separate from camera measurements.</p>
        <p id="session-status" role="status" aria-live="polite">No session started.</p>
        <div class="behavior-metrics" aria-label="Experimental behavioral measurements">
            <div class="behavior-card"><span>Blinks observed</span><strong id="blink-count">—</strong></div>
            <div class="behavior-card"><span>Blinks / valid minute</span><strong id="blink-rate">—</strong><small>Shown after 10 valid seconds</small></div>
            <div class="behavior-card"><span>Current eye closure</span><strong id="eye-closure">—</strong><small id="last-closure">Last closure: —</small></div>
            <div class="behavior-card"><span>Time eyes appear closed</span><strong id="closed-percent">—</strong><small>Percentage of valid observed time</small></div>
            <div class="behavior-card"><span>Head pose (approximate)</span><strong id="head-pose">—</strong><small>Yaw / pitch / roll, degrees</small></div>
            <div class="behavior-card"><span>Head movement</span><strong id="head-movement">—</strong><small>Smoothed angular speed</small></div>
            <div class="behavior-card"><span>Face tracking confidence</span><strong id="tracking-confidence">Unavailable</strong><small>Heuristic; not a model probability</small></div>
            <div class="behavior-card"><span>Measurement coverage</span><strong id="measurement-quality">—</strong><small id="valid-time">Valid observed time: 0.0 s</small></div>
        </div>
        <div class="behavior-chart">
            <h4>Session timeline <small>Latest 5 minutes · local time</small></h4>
            <svg id="behavior-timeline" viewBox="0 0 600 220" role="img" aria-label="Timestamped experimental eye closure and head movement timeline">
                <text x="8" y="16" class="chart-eye-label">Eyes closed, % of valid time (0–100%)</text>
                <path d="M 8 92 H 592 M 8 188 H 592" class="chart-axis"/>
                <path id="closed-timeline" class="chart-eye" fill="none"/>
                <text x="8" y="116" class="chart-motion-label">Head movement (0–60°/s; higher values clipped)</text>
                <path id="motion-timeline" class="chart-motion" fill="none"/>
                <text id="timeline-start" x="8" y="214">—</text>
                <text id="timeline-end" x="592" y="214" text-anchor="end">—</text>
            </svg>
            <p id="timeline-summary" class="alignment-hint">Start a session to see a timeline. Gaps represent unavailable measurements.</p>
        </div>
        <p class="alignment-hint">Glasses glare, occlusion, lighting, camera speed, and eye shape can reduce accuracy.
        Realign and Resume when quality recovers. No medical alerts are generated.</p>
        <p class="alignment-hint">Nothing is saved or uploaded. Measurements and ratings are cleared when you leave this page or start a new session.</p>
    </section>
    """,
    css=(STATIC / "webcam.css").read_text(encoding="utf-8"),
    # Streamlit serves one inline ES module; compose local modules without
    # relative browser imports or a separate frontend build toolchain.
    js="\n".join((STATIC / name).read_text(encoding="utf-8") for name in (
        "webcam_alignment.js", "webcam_behavior.js", "webcam_session_ui.js", "webcam_detector.js", "webcam.js",
    )),
)
