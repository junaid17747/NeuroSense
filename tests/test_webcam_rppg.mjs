import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../dashboard/static/webcam_rppg.js", import.meta.url), "utf8");
const { RPPG, RPPGProcessor, estimatePos, faceSkinPolygons, extractSkinRgb, compareReferenceMeasurements } =
  await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const makePulse = (bpm, { noise = 0, seconds = 25, hz = 10, regions = ["forehead", "leftCheek", "rightCheek"] } = {}) => {
  const result = [];
  for (let index = 0; index < seconds * hz; index++) {
    const time = index / hz, pulse = Math.sin(2 * Math.PI * bpm / 60 * time);
    const regionSamples = {};
    for (const [regionIndex, region] of regions.entries()) {
      const jitter = noise ? noise * Math.sin(2 * Math.PI * 3.17 * time + regionIndex) : 0;
      const phase = regionIndex * .015;
      const wave = Math.sin(2 * Math.PI * bpm / 60 * time + phase) + jitter;
      regionSamples[region] = { r: 105 + .15 * wave, g: 125 + .45 * wave, b: 95 + .05 * wave };
    }
    result.push(regionSamples);
  }
  return result;
};

test("POS estimates known pulse frequencies from synthetic RGB variation", () => {
  for (const bpm of [54, 72, 96, 132, 168]) {
    const result = estimatePos(makePulse(bpm));
    assert.equal(result.valid, true, `${bpm}: ${result.reason || ""}`);
    assert(Math.abs(result.bpm - bpm) <= 3, `${bpm} estimated as ${result.bpm}`);
  }
});

test("POS tolerates modest synthetic noise and rejects nonperiodic or ambiguous signals", () => {
  const noisy = estimatePos(makePulse(84, { noise: .22 }));
  assert.equal(noisy.valid, true);
  assert(Math.abs(noisy.bpm - 84) <= 3);
  const flat = makePulse(72).map(sample => Object.fromEntries(Object.keys(sample).map(region => [region, { r: 105, g: 125, b: 95 }])));
  assert.equal(estimatePos(flat).valid, false);
  const ambiguous = Array.from({ length: 250 }, (_, index) => {
    const time = index / 10;
    const wave = .4 * Math.sin(2 * Math.PI * 72 / 60 * time) + .4 * Math.sin(2 * Math.PI * 108 / 60 * time);
    const regions = { forehead: 0, leftCheek: 0, rightCheek: 0 };
    for (const region of Object.keys(regions)) regions[region] = { r: 105 + .15 * wave, g: 125 + .45 * wave, b: 95 + .05 * wave };
    return regions;
  });
  assert.equal(estimatePos(ambiguous).valid, false);
  const short = estimatePos(makePulse(72, { seconds: 10 }));
  assert.equal(short.valid, false);
  assert.match(short.reason, /Collecting/);
});

test("POS rejects region disagreement and unsupported rates without clamping", () => {
  const regions = ["forehead", "leftCheek", "rightCheek"];
  const disagreement = makePulse(72).map((sample, index) => {
    const alternative = Math.sin(2 * Math.PI * 110 / 60 * index / 10);
    sample.rightCheek = { r: 105 + .15 * alternative, g: 125 + .45 * alternative, b: 95 + .05 * alternative };
    return sample;
  });
  assert.equal(estimatePos(disagreement).valid, false);
  assert.equal(estimatePos(makePulse(210)).valid, false);
  assert.deepEqual(regions.length, 3);
});

test("timestamp processor handles irregular sampling, computes metadata, and rejects stale or dropped frames", () => {
  const processor = new RPPGProcessor();
  const rgb = makePulse(72, { seconds: 28 });
  let snapshot;
  for (let index = 0; index < rgb.length; index++) {
    // Small deterministic cadence jitter exercises timestamp-aware resampling.
    const time = index * 100 + (index % 3) * 4;
    snapshot = processor.add({ time, videoTime: index / 10, regions: rgb[index], bounds: { cx: .5, cy: .5 } });
  }
  assert.equal(snapshot.source, "WEBCAM_RPPG_EXPERIMENTAL");
  assert.equal(snapshot.algorithmVersion, RPPG.algorithmVersion);
  assert(snapshot.bpm !== null, snapshot.reason);
  assert(Math.abs(snapshot.bpm - 72) <= 3);
  assert.equal(snapshot.quality, "Good");
  assert.equal(processor.add({ time: 26000, videoTime: 25.9, regions: rgb.at(-1), bounds: { cx: .5, cy: .5 } }).bpm, null);
  assert.match(processor.snapshot().reason, /stale/);

  processor.reset();
  processor.add({ time: 0, videoTime: 0, regions: rgb[0], bounds: { cx: .5, cy: .5 } });
  const gap = processor.add({ time: 450, videoTime: .45, regions: rgb[1], bounds: { cx: .5, cy: .5 } });
  assert.match(gap.reason, /interrupted/);
});

test("processor invalidates camera, tracking, motion and exposure failures and bounds memory", () => {
  const processor = new RPPGProcessor(), rgb = makePulse(72, { seconds: 40 });
  let snapshot;
  for (let i = 0; i < rgb.length; i++) snapshot = processor.add({ time: i * 100, videoTime: i / 10, regions: rgb[i], bounds: { cx: .5, cy: .5 } });
  assert(processor.samples.length <= RPPG.maxSamples);
  assert(snapshot.durationSeconds <= RPPG.windowSeconds);
  assert.equal(processor.add({ time: 4000, videoTime: 40, regions: null, bounds: null, valid: false, reason: "Face tracking lost." }).bpm, null);
  assert.equal(processor.samples.length, 0);
  const base = rgb[0];
  processor.add({ time: 0, videoTime: 0, regions: base, bounds: { cx: .5, cy: .5 } });
  assert.match(processor.add({ time: 100, videoTime: .1, regions: rgb[1], bounds: { cx: .7, cy: .5 } }).reason, /movement/);
  processor.reset();
  processor.add({ time: 0, videoTime: 0, regions: base, bounds: { cx: .5, cy: .5 } });
  const bright = Object.fromEntries(Object.keys(base).map(region => [region, { r: 180, g: 200, b: 170 }]));
  assert.match(processor.add({ time: 100, videoTime: .1, regions: bright, bounds: { cx: .5, cy: .5 } }).reason, /exposure/);
});

test("low-rate sampling cannot pass coverage and malformed timestamps fail closed", () => {
  const processor = new RPPGProcessor(), rgb = makePulse(72, { seconds: 40 });
  let snapshot;
  for (let i = 0; i <= 125; i++) snapshot = processor.add({ time: i * 200, videoTime: i / 5, regions: rgb[i], bounds: { cx: .5, cy: .5 } });
  assert.equal(snapshot.bpm, null);
  assert.match(snapshot.reason, /coverage/);
  assert.equal(processor.add({ time: NaN, videoTime: 12.6, regions: rgb[0], bounds: { cx: .5, cy: .5 } }).bpm, null);
  assert.match(processor.snapshot().reason, /timestamps/);
});

test("face-relative skin polygons and pixel extraction reject missing landmarks or small regions", () => {
  const landmarks = Array.from({ length: 478 }, (_, i) => ({ x: .25 + (i % 10) * .05, y: .2 + Math.floor(i / 10) * .01 }));
  const polygons = faceSkinPolygons(landmarks);
  assert.deepEqual(Object.keys(polygons), ["forehead", "leftCheek", "rightCheek"]);
  assert.equal(faceSkinPolygons([]), null);
  const width = 100, height = 100, pixels = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < pixels.length; i += 4) { pixels[i] = 120; pixels[i + 1] = 130; pixels[i + 2] = 110; pixels[i + 3] = 255; }
  const polygon = [{ x: .1, y: .1 }, { x: .9, y: .1 }, { x: .9, y: .9 }, { x: .1, y: .9 }];
  const extracted = extractSkinRgb(pixels, width, height, { forehead: polygon }, 20);
  assert.equal(extracted.valid, true);
  assert.equal(extracted.regions.forehead.r, 120);
  assert.equal(extractSkinRgb(pixels, width, height, { forehead: polygon }, 9000).valid, false);
  for (let i = 0; i < pixels.length; i += 4) pixels[i] = pixels[i + 1] = pixels[i + 2] = 0;
  assert.match(extractSkinRgb(pixels, width, height, { forehead: polygon }).reason, /lighting/i);
  for (const [w, h] of [[160, 90], [320, 240], [90, 160]]) {
    const image = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < image.length; i += 4) { image[i] = 120; image[i + 1] = 130; image[i + 2] = 110; image[i + 3] = 255; }
    assert.equal(extractSkinRgb(image, w, h, { forehead: polygon }).valid, true, `${w}x${h}`);
  }
});

test("reference workflow calculates error summaries without claiming validated accuracy", () => {
  const metrics = compareReferenceMeasurements([
    { referenceBpm: 70, rppgBpm: 72 }, { referenceBpm: 80, rppgBpm: 76 }, { referenceBpm: 90, rppgBpm: null },
  ]);
  assert.equal(metrics.count, 2);
  assert.equal(metrics.mae, 3);
  assert.equal(metrics.bias, -1);
  assert.equal(metrics.validCoverage, 2 / 3);
  assert.equal(compareReferenceMeasurements([]).mae, null);
});
