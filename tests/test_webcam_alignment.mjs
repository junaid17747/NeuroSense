import assert from "node:assert/strict";
import test from "node:test";
import { loadAlignment, frontalMatrix, goodLight, face, rotation } from "./webcam_test_helpers.mjs";

const { ALIGNMENT, AlignmentTracker, alignmentGuide, evaluateAlignment, lightingStats, headRotationDegrees } = await loadAlignment();
const evaluate = (changes = {}) => evaluateAlignment({ faces: [face()], matrix: frontalMatrix, lighting: goodLight, width: 640, height: 480, ...changes });

test("one frontal centered face aligns, no face and multiple faces do not", () => {
  assert.equal(evaluate().aligned, true);
  assert.equal(evaluate({ faces: [] }).code, "no_face");
  assert.equal(evaluate({ faces: [face(), face()] }).code, "multiple_faces");
});

for (const [cx, cy, code, message] of [
  [.6, .5, "left", /Move right/], [.4, .5, "right", /Move left/],
  [.5, .4, "up", /Move down/], [.5, .6, "down", /Move up/],
]) test(`mirrored preview: ${code} gives corrective feedback`, () => {
  const result = evaluate({ faces: [face({ cx, cy })] });
  assert.equal(result.code, code);
  assert.match(result.message, message);
});

test("distance thresholds reject both too close and too far", () => {
  for (const fill of [ALIGNMENT.minFill + .001, ALIGNMENT.maxFill - .001]) assert.equal(evaluate({ faces: [face({ fill })] }).aligned, true);
  assert.equal(evaluate({ faces: [face({ fill: ALIGNMENT.minFill - .001 })] }).code, "too_far");
  assert.equal(evaluate({ faces: [face({ fill: ALIGNMENT.maxFill + .001 })] }).code, "too_close");
});

test("centering boundaries are measured relative to the actual guide", () => {
  const guide = alignmentGuide(640, 480);
  for (const sign of [-1, 1]) {
    assert.equal(evaluate({ faces: [face({ cx: .5 + sign * guide.rx * (ALIGNMENT.centerTolerance - .001) })] }).aligned, true);
    assert.equal(evaluate({ faces: [face({ cx: .5 + sign * guide.rx * (ALIGNMENT.centerTolerance + .001) })] }).aligned, false);
  }
});

for (const [width, height] of [[640, 480], [1920, 1080], [360, 640], [480, 480]]) {
  test(`oval and evaluation agree at ${width}x${height}`, () => {
    const guide = alignmentGuide(width, height);
    assert(Math.abs((guide.rx * width) / (guide.ry * height) - .72) < 1e-9);
    assert(guide.rx <= .40 + 1e-9 && guide.ry <= .38 + 1e-9);
    assert.equal(evaluate({ faces: [face({ width, height })], width, height }).aligned, true);
  });
}

for (const axis of ["x", "y", "z"]) test(`head ${axis} rotation is bounded in both directions`, () => {
  for (const sign of [-1, 1]) {
    assert.equal(evaluate({ matrix: rotation(sign * 19.9, axis) }).aligned, true);
    assert.equal(evaluate({ matrix: rotation(sign * 20.1, axis) }).code, "rotation");
  }
});

test("pose angle tolerates uniform scaling and fails closed on missing/invalid pose", () => {
  const matrix = rotation(10);
  matrix.data = matrix.data.map((v, i) => i < 12 ? v * 2 : v);
  assert(Math.abs(headRotationDegrees(matrix) - 10) < 1e-6);
  for (const matrix of [null, {}, { ...frontalMatrix, data: Array(16).fill(NaN) }]) assert.equal(evaluate({ matrix }).aligned, false);
  assert.equal(evaluate({ faces: [[{ x: NaN, y: .5 }]] }).aligned, false);
});

test("luminance and clipped areas detect poor lighting without saving pixels", () => {
  assert.equal(lightingStats(new Uint8ClampedArray([0, 0, 0, 255])).mean, 0);
  assert.equal(lightingStats(new Uint8ClampedArray([255, 255, 255, 255])).brightFraction, 1);
  for (const mean of [44.9, 220.1]) assert.equal(evaluate({ lighting: { ...goodLight, mean } }).aligned, false);
  for (const mean of [45, 220]) assert.equal(evaluate({ lighting: { ...goodLight, mean } }).aligned, true);
  assert.equal(evaluate({ lighting: { ...goodLight, darkFraction: .451 } }).code, "too_dark");
  assert.equal(evaluate({ lighting: { ...goodLight, brightFraction: .451 } }).code, "too_bright");
  assert.equal(evaluate({ lighting: null }).aligned, false);
});

test("three continuous seconds are required, starting is always explicit", () => {
  const tracker = new AlignmentTracker();
  assert.equal(tracker.start(0), false);
  for (let t = 0; t < 3000; t += 100) assert.equal(tracker.update(true, t).ready, false);
  assert.equal(tracker.update(true, 2999).ready, false);
  assert.equal(tracker.update(true, 3000).ready, true);
  assert.equal(tracker.monitoring, false);
  assert.equal(tracker.start(3001), true);
  assert.equal(tracker.monitoring, true);
});

test("alignment loss resets calibration and cancels active monitoring", () => {
  const tracker = new AlignmentTracker();
  for (let t = 0; t <= 3000; t += 100) tracker.update(true, t);
  tracker.start(3001);
  assert.deepEqual(tracker.update(false, 3100), { ready: false, monitoring: false, progress: 0 });
  assert.equal(tracker.update(true, 3200).progress, 0);
  assert.equal(tracker.update(true, 3300).ready, false);
});

test("stalls, duplicate timestamps, clock reversal, and stale Start clicks fail closed", () => {
  for (const gap of [601, -1, 0]) {
    const tracker = new AlignmentTracker();
    for (let t = 0; t <= 3000; t += 100) tracker.update(true, t);
    assert.equal(tracker.update(true, 3000 + gap).ready, false);
  }
  const tracker = new AlignmentTracker();
  for (let t = 0; t <= 3000; t += 100) tracker.update(true, t);
  assert.equal(tracker.start(3601), false);
  assert.equal(tracker.update(true, NaN).ready, false);
});
