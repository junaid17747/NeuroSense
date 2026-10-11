// Experimental, browser-local rPPG processing. No frames, landmarks, or RGB
// samples are logged or sent across the Streamlit component bridge.
export const RPPG = Object.freeze({
  algorithmVersion: "pos-browser-1.0",
  sampleIntervalMs: 100,
  targetHz: 10,
  windowSeconds: 25,
  minWindowSeconds: 25,
  maxSamples: 320,
  maxGapMs: 300,
  minCoverage: 0.90,
  minRegionPixels: 18,
  minLuminance: 35,
  maxLuminance: 225,
  maxClippedFraction: 0.20,
  minBpm: 45,
  maxBpm: 180,
  minPeakRatio: 2.2,
  maxRegionDisagreementBpm: 8,
  maxMotionFraction: 0.025,
});

// MediaPipe Face Landmarker mesh indices. These polygons are bounded by the
// forehead/cheek contours and sit clear of eyes, eyebrows, nostrils, and lips.
export const SKIN_POLYGONS = Object.freeze({
  forehead: [10, 109, 67, 103, 54, 21, 162, 127, 9, 356, 389, 251, 284, 332, 297, 338],
  leftCheek: [116, 117, 118, 119, 100, 126, 142, 203, 206, 205, 50, 101],
  rightCheek: [345, 346, 347, 348, 329, 355, 371, 423, 426, 425, 280, 330],
});

const finite = Number.isFinite;
const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const std = values => {
  const center = mean(values);
  return Math.sqrt(mean(values.map(value => (value - center) ** 2)));
};

export function faceSkinPolygons(landmarks) {
  if (!Array.isArray(landmarks)) return null;
  const polygons = {};
  for (const [name, indices] of Object.entries(SKIN_POLYGONS)) {
    const points = indices.map(index => landmarks[index]);
    if (points.some(point => !point || !finite(point.x) || !finite(point.y) || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1)) return null;
    polygons[name] = points;
  }
  return polygons;
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i], b = polygon[j];
    if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

// `pixels` is a downsampled RGBA copy of the source video frame. Polygon
// coordinates remain normalized to the source frame, so mirroring CSS has no
// effect and differing camera resolutions/aspect ratios use the same mapping.
export function extractSkinRgb(pixels, width, height, polygons, minPixels = RPPG.minRegionPixels) {
  if (!pixels || pixels.length !== width * height * 4 || !polygons) return { valid: false, reason: "Skin-region sample unavailable." };
  const result = {};
  for (const [name, polygon] of Object.entries(polygons)) {
    let r = 0, g = 0, b = 0, count = 0, dark = 0, clipped = 0;
    const x0 = Math.max(0, Math.floor(Math.min(...polygon.map(point => point.x)) * width));
    const x1 = Math.min(width, Math.ceil(Math.max(...polygon.map(point => point.x)) * width));
    const y0 = Math.max(0, Math.floor(Math.min(...polygon.map(point => point.y)) * height));
    const y1 = Math.min(height, Math.ceil(Math.max(...polygon.map(point => point.y)) * height));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const nx = (x + .5) / width, ny = (y + .5) / height;
      if (!pointInPolygon(nx, ny, polygon)) continue;
      const offset = (y * width + x) * 4;
      const red = pixels[offset], green = pixels[offset + 1], blue = pixels[offset + 2];
      const luminance = .2126 * red + .7152 * green + .0722 * blue;
      r += red; g += green; b += blue; count++;
      if (luminance < 25) dark++;
      if (luminance > 245) clipped++;
    }
    if (count < minPixels) return { valid: false, reason: "Skin-region coverage is too low." };
    const rgb = { r: r / count, g: g / count, b: b / count, pixels: count };
    const luminance = .2126 * rgb.r + .7152 * rgb.g + .0722 * rgb.b;
    if (luminance < RPPG.minLuminance || luminance > RPPG.maxLuminance || dark / count > RPPG.maxClippedFraction || clipped / count > RPPG.maxClippedFraction) {
      return { valid: false, reason: luminance < RPPG.minLuminance ? "Insufficient lighting" : "Lighting is saturated or uneven." };
    }
    result[name] = rgb;
  }
  return { valid: true, regions: result };
}

function interpolate(samples, start, count, step) {
  const output = [];
  let cursor = 0;
  for (let i = 0; i < count; i++) {
    const time = start + i * step;
    while (cursor < samples.length - 2 && samples[cursor + 1].time < time) cursor++;
    const a = samples[cursor], b = samples[cursor + 1];
    if (!a || !b || time < a.time || time > b.time || b.time - a.time > RPPG.maxGapMs) return null;
    const fraction = (time - a.time) / (b.time - a.time);
    const regions = {};
    for (const region of ["forehead", "leftCheek", "rightCheek"]) {
      regions[region] = {};
      for (const channel of ["r", "g", "b"]) {
        regions[region][channel] = a.regions[region][channel] + fraction * (b.regions[region][channel] - a.regions[region][channel]);
      }
    }
    output.push(regions);
  }
  return output;
}

function removeMeanAndTrend(values) {
  const n = values.length;
  const centerX = (n - 1) / 2;
  let centerY = mean(values), covariance = 0, variance = 0;
  for (let i = 0; i < n; i++) {
    covariance += (i - centerX) * (values[i] - centerY);
    variance += (i - centerX) ** 2;
  }
  const slope = covariance / variance;
  return values.map((value, i) => value - (centerY + slope * (i - centerX)));
}

function fft(real, imag) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = -2 * Math.PI / length, wr0 = Math.cos(angle), wi0 = Math.sin(angle);
    for (let start = 0; start < n; start += length) {
      let wr = 1, wi = 0;
      for (let offset = 0; offset < length / 2; offset++) {
        const even = start + offset, odd = even + length / 2;
        const tr = wr * real[odd] - wi * imag[odd], ti = wr * imag[odd] + wi * real[odd];
        real[odd] = real[even] - tr; imag[odd] = imag[even] - ti;
        real[even] += tr; imag[even] += ti;
        [wr, wi] = [wr * wr0 - wi * wi0, wr * wi0 + wi * wr0];
      }
    }
  }
}

function analyzePos(samples, region, sampleHz) {
  const red = samples.map(sample => sample[region].r), green = samples.map(sample => sample[region].g), blue = samples.map(sample => sample[region].b);
  const rMean = mean(red), gMean = mean(green), bMean = mean(blue);
  if (Math.min(rMean, gMean, bMean) <= 1) return null;
  const x = green.map((value, i) => value / gMean - blue[i] / bMean);
  const y = red.map((value, i) => -2 * value / rMean + green[i] / gMean + blue[i] / bMean);
  const scale = std(y) > 1e-8 ? std(x) / std(y) : NaN;
  if (!finite(scale) || std(x) < 1e-6) return null;
  let signal = removeMeanAndTrend(x.map((value, i) => value + scale * y[i]));
  const n = signal.length;
  // A Hann taper limits spectral leakage; FFT is zero padded to next power of 2.
  signal = signal.map((value, i) => value * (.5 - .5 * Math.cos(2 * Math.PI * i / (n - 1))));
  let fftSize = 1;
  while (fftSize < n) fftSize <<= 1;
  const real = new Array(fftSize).fill(0), imag = new Array(fftSize).fill(0);
  for (let i = 0; i < n; i++) real[i] = signal[i];
  fft(real, imag);
  const minHz = RPPG.minBpm / 60, maxHz = RPPG.maxBpm / 60;
  const powers = [];
  const allBand = [];
  for (let k = 1; k < fftSize / 2; k++) {
    const hz = k * sampleHz / fftSize;
    if (hz >= .5 && hz <= Math.min(4.5, sampleHz / 2)) allBand.push({ k, hz, power: real[k] ** 2 + imag[k] ** 2 });
    if (hz < minHz || hz > maxHz) continue;
    powers.push({ k, hz, power: real[k] ** 2 + imag[k] ** 2 });
  }
  if (powers.length < 3) return null;
  powers.sort((a, b) => b.power - a.power);
  const unrestrictedPeak = allBand.sort((a, b) => b.power - a.power)[0];
  if (unrestrictedPeak && (unrestrictedPeak.hz < minHz || unrestrictedPeak.hz > maxHz) &&
      unrestrictedPeak.power > powers[0].power * 1.5) return { outOfRange: true };
  const strongest = powers[0], second = powers.find(item => Math.abs(item.hz - strongest.hz) > .12);
  if (!strongest || !second || strongest.power <= 0) return null;
  const ratio = strongest.power / Math.max(second.power, 1e-16);
  // Three-bin parabolic interpolation improves peak location without clamping.
  const left = real[strongest.k - 1] ** 2 + imag[strongest.k - 1] ** 2;
  const middle = strongest.power;
  const right = real[strongest.k + 1] ** 2 + imag[strongest.k + 1] ** 2;
  const denom = left - 2 * middle + right;
  const offset = denom === 0 ? 0 : .5 * (left - right) / denom;
  const bpm = 60 * (strongest.k + Math.max(-.5, Math.min(.5, offset))) * sampleHz / fftSize;
  return { bpm, ratio, amplitude: Math.sqrt(middle) / n };
}

export function estimatePos(samples, sampleHz = RPPG.targetHz) {
  if (!Array.isArray(samples) || samples.length < RPPG.minWindowSeconds * sampleHz) return { valid: false, reason: "Collecting pulse signal — please remain still." };
  const regions = ["forehead", "leftCheek", "rightCheek"];
  const estimates = regions.map(region => analyzePos(samples, region, sampleHz));
  if (estimates.some(value => value?.outOfRange)) return { valid: false, reason: "Pulse estimate outside the supported research range." };
  if (estimates.some(value => !value)) return { valid: false, reason: "Signal quality too low" };
  if (estimates.some(value => value.ratio < RPPG.minPeakRatio)) return { valid: false, reason: "Pulse frequency is ambiguous." };
  const rates = estimates.map(value => value.bpm);
  if (Math.max(...rates) - Math.min(...rates) > RPPG.maxRegionDisagreementBpm) return { valid: false, reason: "Facial regions disagree; improve lighting and remain still." };
  const bpm = median(rates);
  if (bpm < RPPG.minBpm || bpm > RPPG.maxBpm) return { valid: false, reason: "Pulse estimate outside the supported research range." };
  return { valid: true, bpm: Math.round(bpm), quality: "Good", peakRatio: Math.min(...estimates.map(value => value.ratio)), regionsBpm: rates };
}

export class RPPGProcessor {
  constructor() { this.reset(); }
  reset(reason = "Collecting pulse signal — please remain still.") {
    this.samples = [];
    this.lastFrameTime = null;
    this.lastVideoTime = null;
    this.lastBounds = null;
    this.blockingReason = reason;
    this.result = null;
    this.exposure = null;
    this.rateHistory = [];
    this.lastAcceptedRate = null;
    this.transitionHistory = [];
  }
  invalidate(reason) { this.reset(reason); return this.snapshot(); }
  add({ time, videoTime, regions, bounds, valid = true, reason = "" }) {
    if (!valid) return this.invalidate(reason || "Signal quality too low");
    if (!finite(time) || !finite(videoTime) || !regions || !bounds) return this.invalidate("Camera timestamps or face tracking are unavailable.");
    if (this.lastVideoTime !== null && videoTime <= this.lastVideoTime) return this.invalidate("Repeated or stale camera frame detected.");
    if (this.lastFrameTime !== null && (time <= this.lastFrameTime || time - this.lastFrameTime > RPPG.maxGapMs)) return this.invalidate("Camera sampling was interrupted; collecting a new window.");
    const luminance = .2126 * regions.forehead.r + .7152 * regions.forehead.g + .0722 * regions.forehead.b;
    if (this.exposure !== null && Math.abs(luminance - this.exposure) / Math.max(1, this.exposure) > .20) return this.invalidate("Camera exposure changed abruptly; collecting a new window.");
    if (this.lastBounds) {
      const movement = Math.hypot(bounds.cx - this.lastBounds.cx, bounds.cy - this.lastBounds.cy);
      if (movement > RPPG.maxMotionFraction) return this.invalidate("Excessive movement; please remain still.");
    }
    const sample = { time, videoTime, regions: Object.fromEntries(Object.entries(regions).map(([key, rgb]) => [key, { r: rgb.r, g: rgb.g, b: rgb.b }])) };
    this.samples.push(sample);
    // Keep one extra second before the analysis window so timestamp resampling
    // has a real bracket at its left edge rather than extrapolating.
    this.samples = this.samples.filter(item => time - item.time <= (RPPG.windowSeconds + 1) * 1000).slice(-RPPG.maxSamples);
    this.lastFrameTime = time;
    this.lastVideoTime = videoTime;
    this.lastBounds = bounds;
    this.exposure = luminance;
    this.blockingReason = "Collecting pulse signal — please remain still.";
    this.result = null;
    return this.snapshot();
  }
  snapshot(now = this.lastFrameTime) {
    const first = this.samples[0]?.time;
    const intervals = this.samples.slice(1).map((item, index) => item.time - this.samples[index].time).filter(value => value > 0);
    const samplePeriod = intervals.length ? median(intervals) : 1000 / RPPG.targetHz;
    const duration = first === undefined || !finite(now) ? 0 : Math.max(0, Math.min(RPPG.windowSeconds, (now - first + samplePeriod) / 1000));
    const nominal = Math.max(1, Math.floor(duration * RPPG.targetHz) + 1);
    const coverage = Math.min(1, this.samples.length / nominal);
    const progress = Math.min(1, duration / RPPG.minWindowSeconds);
    if (duration >= RPPG.minWindowSeconds && coverage < RPPG.minCoverage) {
      this.result = null;
      this.blockingReason = "Sampling coverage is too low; too many frames were dropped.";
      this.rateHistory = [];
      this.lastAcceptedRate = null;
      this.transitionHistory = [];
    } else if (duration >= RPPG.minWindowSeconds && coverage >= RPPG.minCoverage && this.samples.length >= 2) {
      const start = this.samples.at(-1).time - RPPG.minWindowSeconds * 1000;
      const count = RPPG.minWindowSeconds * RPPG.targetHz;
      const uniform = interpolate(this.samples, start, count, 1000 / RPPG.targetHz);
        if (!uniform) {
          this.result = null;
          this.rateHistory = [];
          this.lastAcceptedRate = null;
          this.transitionHistory = [];
          this.blockingReason = "Dropped frames or sampling gaps are too large.";
        }
      else {
        const estimate = estimatePos(uniform);
        if (!estimate.valid) {
          this.result = null;
          this.rateHistory = [];
            this.lastAcceptedRate = null;
            this.transitionHistory = [];
          this.blockingReason = estimate.reason;
        } else {
          if (this.lastAcceptedRate !== null && Math.abs(estimate.bpm - this.lastAcceptedRate) > 15) {
            const previousCandidate = this.transitionHistory.length ? median(this.transitionHistory) : null;
            this.transitionHistory = previousCandidate !== null && Math.abs(estimate.bpm - previousCandidate) <= 6
              ? [...this.transitionHistory, estimate.bpm].slice(-5) : [estimate.bpm];
            if (this.transitionHistory.length < 5) {
              this.result = null;
              this.blockingReason = "Pulse estimate is unstable; waiting for five consistent rolling windows.";
            } else {
              this.rateHistory = [...this.transitionHistory];
              this.lastAcceptedRate = median(this.transitionHistory);
              this.transitionHistory = [];
              this.result = { ...estimate, bpm: Math.round(this.lastAcceptedRate) };
              this.blockingReason = "";
            }
          } else {
            this.transitionHistory = [];
            this.rateHistory = [...this.rateHistory, estimate.bpm].slice(-5);
            this.lastAcceptedRate = median(this.rateHistory);
            this.result = { ...estimate, bpm: Math.round(this.lastAcceptedRate) };
            this.blockingReason = "";
          }
        }
      }
    }
    const quality = this.result ? "Good" : this.samples.length ? "Limited"
      : this.blockingReason.startsWith("Collecting pulse signal") ? "Limited" : "Poor";
    return { source: "WEBCAM_RPPG_EXPERIMENTAL", timestamp: this.lastFrameTime,
      bpm: this.result?.bpm ?? null, quality, durationSeconds: duration,
      coverage: Math.round(coverage * 100), algorithmVersion: RPPG.algorithmVersion,
      reason: this.blockingReason || (progress < 1 ? "Collecting pulse signal — please remain still." : "Pulse estimate unavailable."),
      progress, waveform: this.result?.waveform ?? null };
  }
}

export function compareReferenceMeasurements(rows) {
  const valid = rows.filter(row => finite(row.referenceBpm) && finite(row.rppgBpm));
  if (!valid.length) return { count: 0, mae: null, rmse: null, bias: null, validCoverage: 0, failureRate: 1 };
  const errors = valid.map(row => row.rppgBpm - row.referenceBpm);
  const mae = mean(errors.map(Math.abs));
  return { count: valid.length, mae, rmse: Math.sqrt(mean(errors.map(value => value ** 2))), bias: mean(errors),
    validCoverage: valid.length / rows.length, failureRate: 1 - valid.length / rows.length };
}
