// Pure alignment logic and the detector loader are composed into this module
// by dashboard/webcam.py. No component state or images are sent to Python.
export default function mountWebcam({ parentElement }) {
  const select = id => parentElement.querySelector(`#${id}`);
  const enable = select("enable-camera"), disable = select("disable-camera");
  const video = select("camera-video"), status = select("camera-status");
  const placeholder = select("camera-placeholder"), viewport = select("video-viewport");
  const guide = select("alignment-guide"), message = select("alignment-message");
  const progress = select("calibration-progress"), value = select("calibration-value");
  const start = select("start-monitoring"), retry = select("retry-alignment");
  const tracker = new AlignmentTracker();
  const calibration = new EyeCalibration();
  const session = new BehavioralSession();
  let eyeBaseline = null;
  let lastMetricsRender = -Infinity, lastSessionState = null, lastReady = null;
  const canvas = document.createElement("canvas");
  let stream = null, detector = null, pending = false, disposed = false;
  let requestId = 0, detectionId = 0, frameId = null, loadTimeout = null;
  let lastSample = -Infinity, lastVideoTime = -1, lastFreshFrame = null;

  function showAlignment(text, tone = "neutral", state = tracker.snapshot(performance.now())) {
    if (message.textContent !== text) message.textContent = text;
    guide.dataset.tone = tone;
    progress.value = state.progress;
    value.textContent = `${Math.floor(state.progress * 100)}%`;
    progress.setAttribute("aria-valuetext", `${Math.floor(state.progress * 100)} percent; 3 seconds required`);
    const ready = state.ready && eyeBaseline !== null;
    start.disabled = !ready || !["idle", "stopped"].includes(session.state);
    start.textContent = "Start Monitoring";
    const now = performance.now();
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
  }

  function pauseForQuality(text) {
    session.pause(performance.now(), `Measurements paused: ${text} Realign, then click Resume.`, true);
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
    lastSample = -Infinity;
    lastVideoTime = -1;
    lastFreshFrame = null;
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
    stopDetection();
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
          if (session.state !== "running") resetAlignment(quality.valid ? session.reason : quality.reason, "warning");
          else showAlignment("Monitoring active · Experimental behavioral indicators", "success");
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
      resetAlignment("Align your face for 3 seconds before starting.");
      return;
    }
    if (!session.start(now, Date.now(), eyeBaseline, select("tiredness-before").value)) return;
    select("tiredness-after").value = "";
    showAlignment("Monitoring active · Experimental behavioral indicators", "success");
  };
  const pauseMonitoring = () => {
    session.pause(performance.now());
    resetAlignment("Paused. Realign with eyes comfortably open before resuming.");
  };
  const resumeMonitoring = () => {
    const now = performance.now();
    if (!eyeBaseline || !stream || !detector || document.hidden || lastFreshFrame === null || now - lastFreshFrame > BEHAVIOR.maxGapMs || !tracker.start(now)) return;
    if (session.resume(now, eyeBaseline)) showAlignment("Monitoring active · Experimental behavioral indicators", "success");
  };
  const stopMonitoring = () => {
    session.stop(performance.now());
    select("tiredness-before").value = "";
    resetAlignment("Session stopped. You may enter an optional after-session tiredness rating.");
  };
  const afterRatingChanged = () => {
    session.setAfter(select("tiredness-after").value);
    renderBehaviorSession(select, session, performance.now(), !!eyeBaseline && tracker.ready);
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
    retry.removeEventListener("click", beginDetection);
    video.removeEventListener("resize", resizeGuide);
    document.removeEventListener("visibilitychange", visibilityChanged);
    window.removeEventListener("pagehide", disableCamera);
  };
}
