// Run with: node --test tests/test_webcam_camera.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { frontalMatrix, face } from "./webcam_test_helpers.mjs";

const source = (await Promise.all(["webcam_alignment.js", "webcam_behavior.js", "webcam_session_ui.js", "webcam.js"].map(name =>
  readFile(new URL(`../dashboard/static/${name}`, import.meta.url), "utf8")))).join("\n");
const detectorStub = "let factory; export function setFactory(value) { factory = value; } function createFaceDetector() { return factory(); }\n";
const { default: mountWebcam, setFactory } = await import(`data:text/javascript;base64,${Buffer.from(detectorStub + source).toString("base64")}`);

function fakeStream() {
  const track = new EventTarget();
  track.stopped = false;
  track.stop = () => { track.stopped = true; };
  return { getTracks: () => [track], getVideoTracks: () => [track], track };
}

function setup(t, getUserMedia, loadDetector) {
  const nodes = new Map();
  for (const id of ["enable-camera", "disable-camera", "camera-video", "camera-status", "camera-placeholder",
    "video-viewport", "alignment-guide", "alignment-message", "calibration-progress", "calibration-value", "start-monitoring", "retry-alignment",
    "pause-monitoring", "resume-monitoring", "stop-monitoring", "tiredness-before", "tiredness-after", "session-status",
    "blink-count", "blink-rate", "eye-closure", "last-closure", "closed-percent", "head-pose", "head-movement", "tracking-confidence",
    "measurement-quality", "valid-time", "rating-summary", "closed-timeline", "motion-timeline", "timeline-start", "timeline-end", "timeline-summary"]) {
    const node = new EventTarget();
    node.play = async () => {};
    node.style = {};
    node.dataset = {};
    node.setAttribute = () => {};
    nodes.set(`#${id}`, node);
  }
  const video = nodes.get("#camera-video");
  Object.assign(video, { videoWidth: 640, videoHeight: 480, readyState: 2, currentTime: 0 });
  const detector = { closed: false, detectForVideo: () => ({ faceLandmarks: [face()], facialTransformationMatrixes: [frontalMatrix] }) };
  detector.close = () => { detector.closed = true; };
  setFactory(loadDetector || (async () => detector));
  const canvas = { width: 0, height: 0, getContext: () => ({ drawImage: () => {}, getImageData: () => ({ data: new Uint8ClampedArray([128, 128, 128, 255]) }) }) };
  const document = new EventTarget();
  document.createElement = () => canvas;
  document.hidden = false;
  let now = 0, nextId = 1;
  const frames = new Map(), timers = new Map();
  const window = new EventTarget();
  window.isSecureContext = true;
  const replacements = { navigator: { mediaDevices: { getUserMedia } }, window, document,
    performance: { now: () => now },
    requestAnimationFrame: fn => { const id = nextId++; frames.set(id, fn); return id; },
    cancelAnimationFrame: id => frames.delete(id),
    setTimeout: fn => { const id = nextId++; timers.set(id, fn); return id; },
    clearTimeout: id => timers.delete(id),
  };
  const originals = new Map(Object.keys(replacements).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, val] of Object.entries(replacements)) Object.defineProperty(globalThis, key, { configurable: true, value: val });
  t.after(() => {
    cleanup();
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const cleanup = mountWebcam({ parentElement: { querySelector: (id) => nodes.get(id) } });
  return {
    nodes, cleanup, window, document, detector, canvas, frames, timers,
    frame: (time, fresh = true) => {
      now = time;
      if (fresh) video.currentTime += .125;
      const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn(time));
    },
    click: (id) => nodes.get(`#${id}`).dispatchEvent(new Event("click")),
    status: () => nodes.get("#camera-status").textContent,
  };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

test("mounting never requests a camera, even with permission already granted", async (t) => {
  let calls = 0;
  const stream = fakeStream();
  const ui = setup(t, async (constraints) => {
    calls++;
    assert.deepEqual(constraints, { video: true, audio: false });
    return stream;
  });
  await settle();
  assert.equal(calls, 0);
  assert.equal(ui.status(), "Camera is off.");
  ui.click("enable-camera");
  await settle();
  assert.equal(calls, 1);
  assert.match(ui.status(), /Camera is on/);
  assert.equal(ui.nodes.get("#camera-video").srcObject, stream);
  ui.click("disable-camera");
  assert.equal(stream.track.stopped, true);
  assert.equal(ui.nodes.get("#camera-video").srcObject, null);
});

test("denied permission leaves camera off and supports an explicit retry", async (t) => {
  let calls = 0;
  const ui = setup(t, async () => {
    calls++;
    if (calls === 1) throw { name: "NotAllowedError" };
    return fakeStream();
  });
  ui.click("enable-camera");
  await settle();
  assert.match(ui.status(), /permission was denied/);
  assert.equal(ui.nodes.get("#enable-camera").disabled, false);
  assert.equal(ui.nodes.get("#camera-video").srcObject, null);
  ui.click("enable-camera");
  await settle();
  assert.match(ui.status(), /Camera is on/);
});

for (const action of ["cleanup", "disable", "pagehide"]) {
  test(`${action} stops active tracks`, async (t) => {
    const stream = fakeStream();
    const ui = setup(t, async () => stream);
    ui.click("enable-camera");
    await settle();
    if (action === "cleanup") ui.cleanup();
    if (action === "disable") ui.click("disable-camera");
    if (action === "pagehide") ui.window.dispatchEvent(new Event("pagehide"));
    assert.equal(stream.track.stopped, true);
    assert.equal(ui.nodes.get("#camera-video").srcObject, null);
  });

  test(`${action} while permission is pending stops a late stream`, async (t) => {
    let grant;
    const stream = fakeStream();
    const ui = setup(t, () => new Promise((resolve) => { grant = resolve; }));
    ui.click("enable-camera");
    if (action === "cleanup") ui.cleanup();
    if (action === "disable") ui.click("disable-camera");
    if (action === "pagehide") ui.window.dispatchEvent(new Event("pagehide"));
    grant(stream);
    await settle();
    assert.equal(stream.track.stopped, true);
    assert.equal(ui.nodes.get("#camera-video").srcObject, null);
  });
}

test("unsupported browsers show a useful message without requesting a camera", (t) => {
  const ui = setup(t, undefined);
  ui.click("enable-camera");
  assert.match(ui.status(), /HTTPS or localhost/);
});

for (const [name, message] of [["NotFoundError", /No camera was found/], ["NotReadableError", /unavailable or in use/]]) {
  test(`${name} keeps the camera off`, async (t) => {
    const ui = setup(t, async () => { throw { name }; });
    ui.click("enable-camera");
    await settle();
    assert.match(ui.status(), message);
    assert.equal(ui.nodes.get("#camera-video").srcObject, null);
  });
}

test("calibration enables a separate Start action and alignment loss revokes it", async (t) => {
  const ui = setup(t, async () => fakeStream());
  assert.equal(ui.nodes.get("#start-monitoring").disabled, true);
  ui.click("enable-camera");
  await settle();
  for (let time = 0; time < 3000; time += 50) {
    ui.frame(time);
    assert.equal(ui.nodes.get("#start-monitoring").disabled, true);
  }
  ui.frame(3000);
  assert.equal(ui.nodes.get("#alignment-message").textContent, "Ready for Monitoring");
  assert.equal(ui.nodes.get("#start-monitoring").disabled, false);
  ui.click("start-monitoring");
  assert.match(ui.nodes.get("#alignment-message").textContent, /Monitoring active/);
  ui.detector.detectForVideo = () => ({ faceLandmarks: [], facialTransformationMatrixes: [] });
  ui.frame(3050);
  assert.equal(ui.nodes.get("#start-monitoring").disabled, true);
  assert.equal(ui.nodes.get("#calibration-progress").value, 0);
});

for (const action of ["hidden tab", "frozen frame"]) test(`${action} resets calibration`, async (t) => {
  const ui = setup(t, async () => fakeStream());
  ui.click("enable-camera"); await settle();
  for (let time = 0; time <= 3000; time += 50) ui.frame(time);
  if (action === "hidden tab") {
    ui.document.hidden = true;
    ui.document.dispatchEvent(new Event("visibilitychange"));
  } else ui.frame(3700, false);
  assert.equal(ui.nodes.get("#start-monitoring").disabled, true);
  assert.equal(ui.nodes.get("#calibration-progress").value, 0);
});

test("cleanup cancels detection, clears pixels and closes the model", async (t) => {
  const ui = setup(t, async () => fakeStream());
  ui.click("enable-camera"); await settle(); ui.frame(0);
  ui.cleanup();
  assert.equal(ui.detector.closed, true);
  assert.equal(ui.frames.size, 0);
  assert.equal(ui.timers.size, 0);
  assert.equal(ui.canvas.width, 0);
  assert.equal(ui.nodes.get("#calibration-progress").value, 0);
});

test("a late model load is closed after navigation", async (t) => {
  let finish;
  const ui = setup(t, async () => fakeStream(), () => new Promise(resolve => { finish = resolve; }));
  ui.click("enable-camera"); await settle();
  ui.cleanup();
  finish(ui.detector); await settle();
  assert.equal(ui.detector.closed, true);
  assert.equal(ui.frames.size, 0);
});

test("missing model assets preserve preview and offer retry without another camera request", async (t) => {
  let requests = 0;
  const ui = setup(t, async () => { requests++; return fakeStream(); }, async () => { throw new Error("404"); });
  ui.click("enable-camera"); await settle();
  assert.match(ui.nodes.get("#alignment-message").textContent, /Face model could not load/);
  assert.equal(ui.nodes.get("#retry-alignment").hidden, false);
  assert.equal(ui.nodes.get("#start-monitoring").disabled, true);
  assert.notEqual(ui.nodes.get("#camera-video").srcObject, null);
  setFactory(async () => ui.detector);
  ui.click("retry-alignment"); await settle(); ui.frame(0);
  assert.equal(requests, 1);
  assert.equal(ui.nodes.get("#retry-alignment").hidden, true);
});

test("loading timeout and late resolution cannot start calibration", async (t) => {
  let finish;
  const ui = setup(t, async () => fakeStream(), () => new Promise(resolve => { finish = resolve; }));
  ui.click("enable-camera"); await settle();
  [...ui.timers.values()].forEach(fn => fn());
  assert.match(ui.nodes.get("#alignment-message").textContent, /timed out/);
  finish(ui.detector); await settle();
  assert.equal(ui.detector.closed, true);
  assert.equal(ui.frames.size, 0);
});

test("inference errors and close errors still allow camera shutdown", async (t) => {
  const stream = fakeStream();
  const ui = setup(t, async () => stream);
  ui.click("enable-camera"); await settle();
  ui.detector.detectForVideo = () => { throw new Error("WASM failure"); };
  ui.detector.close = () => { throw new Error("teardown failure"); };
  ui.frame(0);
  assert.match(ui.nodes.get("#alignment-message").textContent, /encountered a problem/);
  assert.equal(ui.nodes.get("#start-monitoring").disabled, true);
  ui.click("disable-camera");
  assert.equal(stream.track.stopped, true);
  assert.equal(ui.frames.size, 0);
});

test("Pause, Resume, Stop and a new session reuse one camera stream", async (t) => {
  let requests = 0;
  const ui = setup(t, async () => { requests++; return fakeStream(); });
  ui.click("enable-camera"); await settle();
  for (let time = 0; time <= 3000; time += 50) ui.frame(time);
  ui.nodes.get("#tiredness-before").value = "3";
  ui.click("start-monitoring");
  for (let time = 3050; time <= 4000; time += 50) ui.frame(time);
  ui.click("pause-monitoring");
  assert.equal(ui.nodes.get("#pause-monitoring").disabled,true);
  assert.equal(ui.nodes.get("#resume-monitoring").disabled,true);
  for (let time = 4050; time <= 7100; time += 50) ui.frame(time);
  assert.equal(ui.nodes.get("#resume-monitoring").disabled,false);
  ui.click("resume-monitoring"); ui.frame(7150);
  assert.match(ui.nodes.get("#session-status").textContent,/Monitoring active/);
  ui.click("stop-monitoring");
  assert.equal(ui.nodes.get("#tiredness-after").disabled,false);
  ui.nodes.get("#tiredness-after").value = "7";
  ui.nodes.get("#tiredness-after").dispatchEvent(new Event("change"));
  assert.match(ui.nodes.get("#rating-summary").textContent,/Before: 3; After: 7/);
  for (let time = 7200; time <= 10250; time += 50) ui.frame(time);
  ui.click("start-monitoring");
  assert.match(ui.nodes.get("#rating-summary").textContent,/Before: not provided; After: not provided/);
  assert.equal(ui.nodes.get("#blink-count").textContent,"0");
  assert.equal(requests,1);
});

test("component teardown clears session measurements and self-reports", async (t) => {
  const ui = setup(t,async () => fakeStream());
  ui.click("enable-camera"); await settle();
  for (let time = 0; time <= 3000; time += 50) ui.frame(time);
  ui.nodes.get("#tiredness-before").value = "9";
  ui.click("start-monitoring"); ui.frame(3050);
  ui.cleanup();
  assert.equal(ui.nodes.get("#blink-count").textContent,"—");
  assert.equal(ui.nodes.get("#tiredness-before").value,"");
  assert.equal(ui.nodes.get("#tiredness-after").value,"");
  assert.equal(ui.nodes.get("#session-status").textContent,"No session started.");
  assert.equal(ui.frames.size,0); assert.equal(ui.detector.closed,true);
});

test("tracking loss pauses an active session; recovery never automatically resumes", async (t) => {
  const ui = setup(t,async () => fakeStream());
  ui.click("enable-camera"); await settle();
  for (let time = 0; time <= 3000; time += 50) ui.frame(time);
  ui.click("start-monitoring"); ui.frame(3050);
  const validDetection = ui.detector.detectForVideo;
  ui.detector.detectForVideo = () => ({faceLandmarks:[],facialTransformationMatrixes:[]});
  ui.frame(3100);
  assert.match(ui.nodes.get("#session-status").textContent,/Measurements paused/);
  assert.equal(ui.nodes.get("#head-pose").textContent,"—");
  ui.detector.detectForVideo = validDetection;
  for (let time = 3150; time <= 6200; time += 50) ui.frame(time);
  assert.equal(ui.nodes.get("#resume-monitoring").disabled,false);
  assert.equal(ui.nodes.get("#pause-monitoring").disabled,true);
  assert.match(ui.nodes.get("#session-status").textContent,/Measurements paused/);
});
