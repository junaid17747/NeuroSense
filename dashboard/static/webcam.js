// Pure alignment logic and the detector loader are composed into this module
// by dashboard/webcam.py. No component state or images are sent to Python.
export default function mountWebcam({ parentElement }) {
  const select = id => parentElement.querySelector(`#${id}`);
  const enable = select("enable-camera"), disable = select("disable-camera");
  const video = select("camera-video"), status = select("camera-status");
  const placeholder = select("camera-placeholder"), viewport = select("video-viewport");
  const guide = select("alignment-guide"), message = select("alignment-message");
  const readiness = select("monitoring-readiness");
  const diagnosticPanel = select("alignment-diagnostics");
  const progress = select("calibration-progress"), value = select("calibration-value");
  const start = select("start-monitoring"), retry = select("retry-alignment");
  const rppgEnabled = select("rppg-enabled");
  const tracker = new AlignmentTracker();
  const calibration = new EyeCalibration();
  const session = new BehavioralSession();
  const rppg = new RPPGProcessor();
  const referenceRows = [];
  const rppgCanvas = document.createElement("canvas");
  rppgCanvas.width = 160;
  rppgCanvas.height = 120;
  const rppgContext = rppgCanvas.getContext("2d", { willReadFrequently: true });
  let eyeBaseline = null;
  let lastMetricsRender = -Infinity, lastSessionState = null, lastReady = null;
  const canvas = document.createElement("canvas");
  let stream = null, detector = null, pending = false, disposed = false;
  let detectorState = "off";
  let requestId = 0, detectionId = 0, frameId = null, loadTimeout = null;
  let lastSample = -Infinity, lastVideoTime = -1, lastFreshFrame = null;
  let lastRppgSample = -Infinity;
  const diagnosticEnabled = new URLSearchParams(window.location?.search || "").get("webcamDebug") === "1" ||
    ["localhost", "127.0.0.1", "::1"].includes(window.location?.hostname);
  diagnosticPanel.hidden = !diagnosticEnabled;

  function renderRppg(snapshot = rppg.snapshot()) {
    const enabled = rppgEnabled.checked;
    const sampling = enabled && !!stream && session.state === "running";
    select("rppg-bpm").textContent = sampling && snapshot.bpm !== null ? `${snapshot.bpm} BPM` : "Unavailable";
    select("rppg-quality").textContent = sampling ? snapshot.quality : "Unavailable";
    select("rppg-progress").textContent = enabled ? `${(snapshot.progress * RPPG.minWindowSeconds).toFixed(1)} / ${RPPG.minWindowSeconds} s` : `0 / ${RPPG.minWindowSeconds} s`;
    select("rppg-duration").textContent = enabled ? `${snapshot.durationSeconds.toFixed(1)} s` : "0.0 s";
    select("rppg-coverage").textContent = enabled ? `${snapshot.coverage}%` : "0%";
    select("rppg-source").textContent = enabled ? "WEBCAM_RPPG_EXPERIMENTAL" : "UNAVAILABLE";
    const reason = !enabled ? "Disabled. No rPPG frames are being sampled."
      : !stream ? "Enable the camera to begin."
      : session.state === "quality_paused" || session.state === "paused" ? `${session.reason} rPPG sampling is paused and its buffer is cleared.`
      : session.state === "stopped" ? "Monitoring stopped. Start a new session to collect a fresh pulse window."
      : session.state !== "running" ? "Pulse sampling begins only while Monitoring is active."
      : snapshot.reason || "Pulse signal available for experimental review.";
    select("rppg-reason").textContent = reason;
    return snapshot;
  }

  function resetRppg(reason) {
    lastRppgSample = -Infinity;
    rppgCanvas.width = 0;
    rppgCanvas.height = 0;
    rppg.reset(reason);
    renderRppg();
  }

  function sampleRppg(faces, bounds, check, quality, now) {
    if (!rppgEnabled.checked || session.state !== "running") return;
    if (now - lastRppgSample < RPPG.sampleIntervalMs) return;
    lastRppgSample = now;
    let reason = "";
    if (faces.length !== 1) reason = faces.length > 1 ? "Only one face can be tracked for pulse sampling." : "Face tracking lost.";
    else if (!check.aligned) reason = check.diagnostics?.lighting === false ? check.message : "Please center your face and remain still.";
    else if (!quality.valid) reason = quality.reason;
    else if (!rppgContext) reason = "Local frame sampling is unavailable in this browser.";
    if (reason) {
      renderRppg(rppg.invalidate(reason));
      return;
    }
    try {
      const polygons = faceSkinPolygons(faces[0]);
      if (!polygons) { renderRppg(rppg.invalidate("Facial skin regions could not be tracked.")); return; }
      if (rppgCanvas.width !== 96 || rppgCanvas.height !== 72) {
        rppgCanvas.width = 96;
        rppgCanvas.height = 72;
      }
      rppgContext.drawImage(video, 0, 0, rppgCanvas.width, rppgCanvas.height);
      const pixels = rppgContext.getImageData(0, 0, rppgCanvas.width, rppgCanvas.height).data;
      const extracted = extractSkinRgb(pixels, rppgCanvas.width, rppgCanvas.height, polygons);
      if (!extracted.valid) { renderRppg(rppg.invalidate(extracted.reason)); return; }
      const state = rppg.add({ time: now, videoTime: video.currentTime, regions: extracted.regions, bounds, valid: true });
      renderRppg(state);
    } catch {
      renderRppg(rppg.invalidate("Local pulse signal processing failed."));
    }
  }

  function renderDiagnostics(check, state, reason) {
    if (!diagnosticEnabled) return;
    const show = (id, value) => { select(id).textContent = value; };
    const yesNo = value => value === null || value === undefined ? "Waiting" : value ? "Pass" : "Fail";
    show("diag-camera", stream && video.readyState >= 2 ? "Yes" : "No");
    show("diag-detector", detectorState === "ready" ? "Yes" : detectorState === "failed" ? "No · Failed" : "No");
    show("diag-one-face", yesNo(check?.diagnostics?.exactlyOneFace));
    show("diag-centered", yesNo(check?.diagnostics?.centered));
    show("diag-face-size", yesNo(check?.diagnostics?.faceSize));
    show("diag-lighting", yesNo(check?.diagnostics?.lighting));
    show("diag-head-pose", yesNo(check?.diagnostics?.headPose));
    show("diag-stability", `${((state?.progress || 0) * ALIGNMENT.stableMs / 1000).toFixed(1)} / 3.0 s`);
    show("diag-blocking", reason || "Ready");
  }

  function showAlignment(text, tone = "neutral", state = tracker.snapshot(performance.now())) {
    if (message.textContent !== text) message.textContent = text;
    guide.dataset.tone = tone;
    progress.value = state.progress;
    value.textContent = `${Math.floor(state.progress * 100)}%`;
    progress.setAttribute("aria-valuetext", `${Math.floor(state.progress * 100)} percent; 3 seconds required`);
    const now = performance.now();
    const sessionCanStart = ["idle", "stopped"].includes(session.state);
    const ready = state.ready && eyeBaseline !== null;
    const waiting = !stream ? "Enable the camera first."
      : detectorState === "loading" ? "Face alignment is loading."
      : detectorState === "failed" ? "Face alignment could not start; retry alignment."
      : detectorState !== "ready" ? "Face alignment is not ready."
      : document.hidden ? "Return to this page."
      : lastFreshFrame === null || now - lastFreshFrame > BEHAVIOR.maxGapMs ? "Waiting for a fresh camera frame."
      : !state.ready ? `Keep your face aligned for ${Math.max(0, Math.ceil((ALIGNMENT.stableMs - state.progress * ALIGNMENT.stableMs) / 1000))} more seconds.`
      : eyeBaseline === null ? "Keep both eyes comfortably open to finish eye calibration."
      : null;
    const reason = ["paused", "quality_paused"].includes(session.state)
      ? waiting ? `Resume is unavailable. ${waiting}` : "Ready. Resume Monitoring is available."
      : session.state === "running" ? "Monitoring is active. Pause and Stop are available."
      : !sessionCanStart ? `Start is unavailable while monitoring is ${session.state.replace("_", " ")}.`
      : !stream ? "Start is unavailable. Enable the camera first."
      : detectorState === "loading" ? "Start is unavailable. Face alignment is loading."
      : detectorState === "failed" ? "Start is unavailable. Face alignment could not start; retry alignment."
      : detectorState !== "ready" ? "Start is unavailable. Face alignment is not ready."
      : document.hidden ? "Start is unavailable while this page is hidden. Return to the page."
      : lastFreshFrame === null || now - lastFreshFrame > BEHAVIOR.maxGapMs ? "Start is unavailable. Waiting for a fresh camera frame."
      : !state.ready ? `Start is unavailable. Keep your face aligned for ${Math.max(0, Math.ceil((ALIGNMENT.stableMs - state.progress * ALIGNMENT.stableMs) / 1000))} more seconds.`
      : eyeBaseline === null ? "Alignment is stable. Keep both eyes comfortably open to finish eye calibration."
      : "Ready. Start Monitoring is available.";
    start.disabled = reason !== "Ready. Start Monitoring is available.";
    readiness.textContent = reason;
    start.setAttribute("aria-describedby", "monitoring-readiness");
    start.textContent = "Start Monitoring";
    if (now - lastMetricsRender >= 200 || session.state !== lastSessionState || ready !== lastReady) {
      renderBehaviorSession(select, session, now, ready);
      lastMetricsRender = now; lastSessionState = session.state; lastReady = ready;
    }
  }

  function resetAlignment(text, tone = "neutral") {
    tracker.reset();
    calibration.reset();
    eyeBaseline = null;
    showAlignment(text, tone);
    renderDiagnostics(null, tracker.snapshot(performance.now()), text);
  }

  function pauseForQuality(text) {
    session.pause(performance.now(), `Measurements paused: ${text} Realign, then click Resume.`, true);
    resetRppg(`Monitoring paused: ${text} The pulse window was cleared.`);
    resetAlignment(text, "warning");
  }

  function closeDetector(instance) {
    try { instance?.close(); } catch { /* Still release camera tracks on teardown failure. */ }
  }

  function stopDetection() {
    session.pause(performance.now(), "Measurements paused. Restore camera tracking, realign, then Resume.", true);
    detectionId++;
    if (frameId !== null) cancelAnimationFrame(frameId);
    if (loadTimeout !== null) clearTimeout(loadTimeout);
    frameId = null;
    loadTimeout = null;
    closeDetector(detector);
    detector = null;
    if (detectorState !== "failed") detectorState = "off";
    lastSample = -Infinity;
    lastVideoTime = -1;
    lastFreshFrame = null;
    resetRppg("Camera is off; temporary pulse samples were cleared.");
    canvas.width = 0;
    canvas.height = 0;
    retry.hidden = true;
    resetAlignment("Camera positioning has not started.");
  }

  function stopCamera(text = "Camera is off.") {
    requestId++;
    pending = false;
    session.stop(performance.now());
    stopDetection();
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    video.hidden = true;
    viewport.hidden = true;
    placeholder.hidden = false;
    enable.disabled = false;
    disable.disabled = true;
    status.textContent = text;
    showAlignment("Camera positioning has not started.");
    renderRppg();
    renderDiagnostics(null, tracker.snapshot(performance.now()), text);
  }

  function resizeGuide() {
    const shape = alignmentGuide(video.videoWidth, video.videoHeight);
    if (!shape) return;
    const aspect = video.videoWidth / video.videoHeight;
    viewport.style.aspectRatio = String(aspect);
    viewport.style.maxWidth = `${Math.min(760, 480 * aspect)}px`;
    guide.style.width = `${shape.rx * 200}%`;
    guide.style.height = `${shape.ry * 200}%`;
  }

  function sampleLighting(faces) {
    // The face ROI avoids averaging a dark face with a bright background.
    const bounds = faces.length === 1 ? faceBounds(faces[0]) : null;
    const x = Math.max(0, Math.min(0.99, bounds?.left ?? 0.25));
    const y = Math.max(0, Math.min(0.99, bounds?.top ?? 0.15));
    const w = Math.min(1 - x, bounds?.width ?? 0.5);
    const h = Math.min(1 - y, bounds?.height ?? 0.7);
    canvas.width = 48;
    canvas.height = 48;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas unavailable");
    context.drawImage(video, x * video.videoWidth, y * video.videoHeight,
      w * video.videoWidth, h * video.videoHeight, 0, 0, 48, 48);
    return lightingStats(context.getImageData(0, 0, 48, 48).data);
  }

  function detectionError(text) {
    detectorState = "failed";
    stopDetection();
    detectorState = "failed";
    resetAlignment(text, "warning");
    retry.hidden = false;
  }

  function tick(now) {
    frameId = null;
    if (disposed || !stream || !detector) return;
    try {
      if (document.hidden) {
        pauseForQuality("Return to this tab to align your face.");
      } else if (now - lastSample >= BEHAVIOR.sampleIntervalMs && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastSample = now;
        lastVideoTime = video.currentTime;
        resizeGuide();
        const result = detector.detectForVideo(video, now);
        const faces = result.faceLandmarks;
        const matrix = result.facialTransformationMatrixes?.[0];
        const check = evaluateAlignment({ faces, matrix,
          lighting: sampleLighting(faces), width: video.videoWidth, height: video.videoHeight });
        const eyes = faces.length === 1 ? eyeAspectRatios(faces[0], video.videoWidth, video.videoHeight) : null;
        const timely = performance.now() - now <= BEHAVIOR.maxGapMs &&
          (lastFreshFrame === null || now - lastFreshFrame <= BEHAVIOR.maxGapMs);
        const quality = behavioralQuality(check, eyes, session.measurements?.baseline, timely,
          faces.length === 1 ? faceBounds(faces[0]) : null);
        lastFreshFrame = now;
        if (session.state === "running") {
          session.sample({ eyes, pose: estimateHeadPose(matrix), valid: quality.valid, reason: `Measurements paused: ${quality.reason} Realign, then click Resume.` }, now);
          if (session.state !== "running") {
            resetRppg(`Monitoring paused: ${quality.reason} The pulse window was cleared.`);
            resetAlignment(quality.valid ? session.reason : quality.reason, "warning");
          }
          else {
            sampleRppg(faces, faces.length === 1 ? faceBounds(faces[0]) : null, check, quality, now);
            showAlignment("Monitoring active · Experimental behavioral indicators", "success");
            renderDiagnostics(check, tracker.snapshot(now), quality.valid ? "Monitoring active" : quality.reason);
          }
        } else {
          const aligned = check.aligned && quality.valid;
          const state = tracker.update(aligned, now);
          if (!aligned || state.progress === 0) calibration.reset();
          if (aligned) calibration.add(eyes, now);
          eyeBaseline = state.ready ? calibration.baseline() : null;
          const ready = state.ready && eyeBaseline !== null;
          const text = !quality.valid ? quality.reason : !check.aligned ? check.message
            : ready ? (["paused", "quality_paused"].includes(session.state) ? "Ready to Resume" : "Ready for Monitoring")
            : state.ready ? "Keep eyes comfortably open; reduce glasses glare to finish eye calibration."
            : "Aligned — hold still with eyes comfortably open for 3 seconds.";
          showAlignment(text, aligned ? "success" : check.code === "no_face" ? "neutral" : "warning", state);
          renderDiagnostics(check, state, !quality.valid ? quality.reason : !check.aligned ? check.message
            : state.ready && !eyeBaseline ? "Keep both eyes comfortably open to finish eye calibration."
            : state.ready ? "Ready" : "Hold this position until the 3-second timer completes.");
        }
      } else if (lastFreshFrame === null || now - lastFreshFrame > BEHAVIOR.maxGapMs) {
        pauseForQuality("Waiting for fresh camera frames. Measurement is unavailable.");
      }
      frameId = requestAnimationFrame(tick);
    } catch {
      detectionError("Face alignment encountered a problem. Retry Alignment or restart your camera.");
    }
  }

  async function beginDetection() {
    if (!stream || disposed) return;
    stopDetection();
    detectorState = "loading";
    const current = detectionId;
    resetAlignment("Loading face alignment model…");
    loadTimeout = setTimeout(() => {
      if (current === detectionId && !disposed) detectionError("Face model loading timed out. Check your connection and Retry Alignment.");
    }, 30000);
    try {
      const loaded = await createFaceDetector();
      if (disposed || !stream || current !== detectionId) { closeDetector(loaded); return; }
      clearTimeout(loadTimeout);
      loadTimeout = null;
      detector = loaded;
      detectorState = "ready";
      showAlignment("Face alignment is ready. Center your face to begin calibration.");
      frameId = requestAnimationFrame(tick);
    } catch {
      if (disposed || current !== detectionId) return;
      detectionError("Face model could not load. Check your connection or blocked model assets, then Retry Alignment.");
    }
  }

  async function enableCamera() {
    if (disposed || pending || stream) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      status.textContent = "Camera access requires HTTPS or localhost and a browser that supports webcams.";
      return;
    }
    pending = true;
    enable.disabled = true;
    disable.disabled = false;
    status.textContent = "Waiting for browser camera permission…";
    const currentRequest = ++requestId;
    let acquired = null;
    try {
      acquired = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (disposed || currentRequest !== requestId) {
        acquired.getTracks().forEach(track => track.stop());
        return;
      }
      stream = acquired;
      video.srcObject = stream;
      video.hidden = false;
      viewport.hidden = false;
      placeholder.hidden = true;
      stream.getVideoTracks().forEach(track => track.addEventListener("ended", () => {
        if (!disposed && stream === acquired) stopCamera("Camera disconnected. Enable Camera to try again.");
      }, { once: true }));
      await video.play();
      if (!disposed && currentRequest === requestId) {
        pending = false;
        resizeGuide();
        status.textContent = "Camera is on · Live preview";
        void beginDetection();
      }
    } catch (error) {
      if (acquired) acquired.getTracks().forEach(track => track.stop());
      if (disposed || currentRequest !== requestId) return;
      const messages = {
        NotAllowedError: "Camera permission was denied. Allow camera access in your browser, then click Enable Camera to retry.",
        NotFoundError: "No camera was found. Connect a webcam, then click Enable Camera to retry.",
        NotReadableError: "The camera is unavailable or in use. Close other camera apps, then click Enable Camera to retry.",
      };
      stopCamera(messages[error.name] || "The camera could not start. Check your camera and click Enable Camera to retry.");
    }
  }

  const disableCamera = () => stopCamera();
  const startMonitoring = () => {
    const now = performance.now();
    if (!stream || !detector || document.hidden || !eyeBaseline || lastFreshFrame === null || now - lastFreshFrame > BEHAVIOR.maxGapMs || !tracker.start(now)) {
      showAlignment("Monitoring cannot start yet. Follow the readiness instructions below.", "warning");
      return;
    }
    if (!session.start(now, Date.now(), eyeBaseline, select("tiredness-before").value)) return;
    resetRppg("Collecting pulse signal — please remain still.");
    select("tiredness-after").value = "";
    showAlignment("Monitoring active · Experimental behavioral indicators", "success");
  };
  const pauseMonitoring = () => {
    session.pause(performance.now());
    resetRppg("Monitoring paused; rPPG sampling stopped and the pulse window was cleared.");
    resetAlignment("Paused. Realign with eyes comfortably open before resuming.");
  };
  const resumeMonitoring = () => {
    const now = performance.now();
    if (!eyeBaseline || !stream || !detector || document.hidden || lastFreshFrame === null || now - lastFreshFrame > BEHAVIOR.maxGapMs || !tracker.start(now)) {
      showAlignment("Resume is unavailable. Realign your face and wait for the readiness message.", "warning");
      return;
    }
    if (session.resume(now, eyeBaseline)) {
      resetRppg("Collecting pulse signal — please remain still.");
      showAlignment("Monitoring active · Experimental behavioral indicators", "success");
    }
  };
  const stopMonitoring = () => {
    session.stop(performance.now());
    resetRppg("Monitoring stopped; the pulse window was cleared.");
    select("tiredness-before").value = "";
    resetAlignment("Session stopped. You may enter an optional after-session tiredness rating.");
  };
  const afterRatingChanged = () => {
    session.setAfter(select("tiredness-after").value);
    renderBehaviorSession(select, session, performance.now(), !!eyeBaseline && tracker.ready);
  };
  const rppgChanged = () => {
    resetRppg(rppgEnabled.checked ? "Pulse sampling begins only while Monitoring is active." : "Disabled. No rPPG frames are being sampled.");
  };
  const recordReference = () => {
    const source = select("rppg-reference-source").value.trim();
    const referenceBpm = Number(select("rppg-reference-bpm").value);
    if (!source || !Number.isFinite(referenceBpm) || referenceBpm < 25 || referenceBpm > 250) {
      select("rppg-reference-summary").textContent = "Enter a reference source and a BPM value from 25 to 250.";
      return;
    }
    const estimate = rppg.snapshot();
    const row = { timestamp: new Date().toISOString(), referenceSource: source, referenceBpm,
      rppgBpm: estimate.bpm, absoluteError: estimate.bpm === null ? null : Math.abs(estimate.bpm - referenceBpm),
      quality: estimate.quality, observationSeconds: estimate.durationSeconds, coveragePercent: estimate.coverage,
      lighting: select("rppg-reference-light").value, motion: select("rppg-reference-motion").value };
    referenceRows.push(row);
    const metrics = compareReferenceMeasurements(referenceRows);
    const valid = metrics.count, failed = referenceRows.length - valid;
    select("rppg-reference-summary").textContent = `${referenceRows.length} paired observations · ${valid} valid · ${failed} unavailable · valid coverage ${(metrics.validCoverage * 100).toFixed(0)}%` +
      (valid ? ` · MAE ${metrics.mae.toFixed(1)} BPM · RMSE ${metrics.rmse.toFixed(1)} BPM · bias ${metrics.bias.toFixed(1)} BPM` : "");
    select("rppg-export-reference").disabled = false;
  };
  const exportReference = () => {
    if (!referenceRows.length) return;
    const columns = ["timestamp", "referenceSource", "referenceBpm", "rppgBpm", "absoluteError", "quality", "observationSeconds", "coveragePercent", "lighting", "motion"];
    const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [columns.join(","), ...referenceRows.map(row => columns.map(key => quote(row[key])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url; link.download = "neurosense-rppg-reference-validation.csv";
    document.body.appendChild(link);
    link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const visibilityChanged = () => {
    if (document.hidden && stream) pauseForQuality("Return to this tab to align your face.");
  };
  enable.addEventListener("click", enableCamera);
  disable.addEventListener("click", disableCamera);
  start.addEventListener("click", startMonitoring);
  select("pause-monitoring").addEventListener("click", pauseMonitoring);
  select("resume-monitoring").addEventListener("click", resumeMonitoring);
  select("stop-monitoring").addEventListener("click", stopMonitoring);
  select("tiredness-after").addEventListener("change", afterRatingChanged);
  rppgEnabled.addEventListener("change", rppgChanged);
  select("rppg-record-reference").addEventListener("click", recordReference);
  select("rppg-export-reference").addEventListener("click", exportReference);
  retry.addEventListener("click", beginDetection);
  video.addEventListener("resize", resizeGuide);
  document.addEventListener("visibilitychange", visibilityChanged);
  window.addEventListener("pagehide", disableCamera);
  stopCamera();

  return () => {
    disposed = true;
    stopCamera();
    session.clear();
    select("tiredness-before").value = "";
    select("tiredness-after").value = "";
    renderBehaviorSession(select, session, performance.now(), false);
    enable.removeEventListener("click", enableCamera);
    disable.removeEventListener("click", disableCamera);
    start.removeEventListener("click", startMonitoring);
    select("pause-monitoring").removeEventListener("click", pauseMonitoring);
    select("resume-monitoring").removeEventListener("click", resumeMonitoring);
    select("stop-monitoring").removeEventListener("click", stopMonitoring);
    select("tiredness-after").removeEventListener("change", afterRatingChanged);
    rppgEnabled.removeEventListener("change", rppgChanged);
    select("rppg-record-reference").removeEventListener("click", recordReference);
    select("rppg-export-reference").removeEventListener("click", exportReference);
    referenceRows.length = 0;
    retry.removeEventListener("click", beginDetection);
    video.removeEventListener("resize", resizeGuide);
    document.removeEventListener("visibilitychange", visibilityChanged);
    window.removeEventListener("pagehide", disableCamera);
  };
}
