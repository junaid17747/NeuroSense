// Experimental behavioral measurements. No identity, health inference, or I/O.
export const BEHAVIOR = Object.freeze({
  sampleIntervalMs: 40, maxGapMs: 150, calibrationSamples: 30,
  closeRatio: 0.65, openRatio: 0.82, minBlinkMs: 60, maxBlinkMs: 700,
  minClosedSamples: 2, minRateMs: 10000, timelineLimit: 300,
});
export const EYE_POINTS = Object.freeze({
  left: [33, 160, 158, 133, 153, 144],
  right: [362, 385, 387, 263, 373, 380],
});

export function eyeAspectRatios(landmarks, width, height) {
  if (!(width > 0 && height > 0)) return null;
  const eye = indices => {
    const points = indices.map(i => landmarks?.[i]);
    if (points.some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) return null;
    const distance = (a, b) => Math.hypot((a.x - b.x) * width, (a.y - b.y) * height);
    const span = distance(points[0], points[3]);
    if (span < 12) return null;
    const ratio = (distance(points[1], points[5]) + distance(points[2], points[4])) / (2 * span);
    return ratio >= 0 && ratio <= .65 ? { ratio, span } : null;
  };
  const left = eye(EYE_POINTS.left), right = eye(EYE_POINTS.right);
  if (!left || !right || left.span / right.span < .55 || left.span / right.span > 1.8) return null;
  return { left: left.ratio, right: right.ratio };
}

const percentile = (values, fraction) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) * fraction)];
export class EyeCalibration {
  constructor() { this.reset(); }
  reset() { this.samples = []; }
  add(eyes, now) {
    if (!eyes || !Number.isFinite(now) || ![eyes.left, eyes.right].every(v => Number.isFinite(v) && v >= 0 && v <= .65)) { this.reset(); return; }
    this.samples.push({ left: eyes.left, right: eyes.right, time: now });
    this.samples = this.samples.filter(s => now - s.time <= 3500).slice(-100);
  }
  baseline() {
    if (this.samples.length < BEHAVIOR.calibrationSamples || this.samples.at(-1).time - this.samples[0].time < 2500) return null;
    const baseline = {};
    for (const side of ["left", "right"]) {
      const values = this.samples.map(s => s[side]);
      const open = percentile(values, .8);
      if (!Number.isFinite(open) || open < .18 || open > .5 || percentile(values, .5) < open * .75) return null;
      baseline[side] = open;
    }
    return baseline;
  }
}

export function estimateHeadPose(matrix) {
  if (headRotationDegrees(matrix) === null) return null;
  const m = matrix.data, scale = Math.hypot(m[0], m[1], m[2]), degrees = 180 / Math.PI;
  // MediaPipe face geometry uses a column-major rigid transform. Angles are
  // approximate camera-relative Euler rotations, not anatomical measurements.
  return { yaw: Math.asin(Math.max(-1, Math.min(1, -m[2] / scale))) * degrees,
    pitch: Math.atan2(m[6], m[10]) * degrees, roll: Math.atan2(m[1], m[0]) * degrees };
}

export function behavioralQuality(check, eyes, baseline, timely, bounds) {
  if (!timely) return { valid: false, reason: "Camera sampling is too slow or interrupted. Try a faster device or close other apps." };
  // Small translations are useful movement observations once a session starts.
  if (!["aligned", "left", "right", "up", "down"].includes(check.code)) return { valid: false, reason: check.message };
  if (!bounds || bounds.left < 0 || bounds.top < 0 || bounds.right > 1 || bounds.bottom > 1 || !eyes) {
    return { valid: false, reason: "Eyes are not clear. Reduce glasses glare and keep both eyes unobstructed." };
  }
  const left = eyes.left / (baseline?.left || .3), right = eyes.right / (baseline?.right || .3);
  if (![left, right].every(Number.isFinite) || Math.abs(left - right) > .55) {
    return { valid: false, reason: "Eye tracking is inconsistent. Check glare or partial occlusion." };
  }
  return { valid: true, reason: "Tracking usable · Experimental measurements" };
}

export class BlinkMeasurements {
  constructor(baseline) {
    this.baseline = { ...baseline };
    this.count = 0; this.validMs = 0; this.closedMs = 0; this.lastClosureMs = null;
    this.interrupt();
  }
  interrupt() {
    this.lastTime = null; this.closedSince = null; this.eyeState = "unknown";
    this.armed = false; this.closedSamples = 0;
  }
  update(eyes, now) {
    if (!eyes || !Number.isFinite(now) || ![eyes.left, eyes.right].every(v => Number.isFinite(v) && v >= 0 && v <= .65)) { this.interrupt(); return false; }
    const ratios = [eyes.left / this.baseline.left, eyes.right / this.baseline.right];
    if (!ratios.every(Number.isFinite) || Math.abs(ratios[0] - ratios[1]) > .55) { this.interrupt(); return false; }
    const dt = this.lastTime === null ? 0 : now - this.lastTime;
    if (dt < 0 || (this.lastTime !== null && dt === 0) || dt > BEHAVIOR.maxGapMs) { this.interrupt(); return false; }
    const wasClosed = this.eyeState === "closed";
    let next = this.eyeState;
    if (ratios.every(r => r <= BEHAVIOR.closeRatio)) next = "closed";
    else if (ratios.every(r => r >= BEHAVIOR.openRatio)) next = "open";
    if (next === "unknown") { this.lastTime = null; return false; }
    if (this.eyeState !== "unknown") {
      this.validMs += dt;
      this.closedMs += dt * ((wasClosed ? 1 : 0) + (next === "closed" ? 1 : 0)) / 2;
    }
    if (next === "closed") {
      if (!wasClosed) { this.closedSince = this.lastTime === null ? now : now - dt / 2; this.closedSamples = 0; }
      this.closedSamples++;
    } else {
      if (wasClosed) {
        const duration = now - dt / 2 - this.closedSince;
        this.lastClosureMs = duration;
        if (this.armed && this.closedSamples >= BEHAVIOR.minClosedSamples && duration >= BEHAVIOR.minBlinkMs && duration <= BEHAVIOR.maxBlinkMs) this.count++;
      }
      this.closedSince = null;
      this.closedSamples = 0;
      this.armed = true;
    }
    this.eyeState = next;
    this.lastTime = now;
    return true;
  }
  snapshot(now) {
    return { blinkCount: this.count, validMs: this.validMs,
      blinkRate: this.validMs >= BEHAVIOR.minRateMs ? this.count * 60000 / this.validMs : null,
      closedPercent: this.validMs > 0 ? 100 * this.closedMs / this.validMs : null,
      closureMs: this.closedSince === null || this.lastTime === null ? null : Math.max(0, Math.min(now, this.lastTime) - this.closedSince),
      lastClosureMs: this.lastClosureMs };
  }
}

export function tirednessRating(value) {
  if (value === "" || value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 && number <= 10 ? number : null;
}

export class BehavioralSession {
  constructor() { this.clear(); }
  clear() {
    this.state = "idle"; this.measurements = null; this.timeline = [];
    this.before = null; this.after = null; this.startedAt = null; this.wallStart = null;
    this.runningMs = 0; this.runSince = null; this.lastPoint = null;
    this.pose = null; this.previousPose = null; this.motion = null; this.lastValid = null;
    this.reason = "No session started.";
  }
  start(now, wallTime, baseline, rating = null) {
    if (!["idle", "stopped"].includes(this.state) || !Number.isFinite(now) || !Number.isFinite(wallTime) ||
        !baseline || ![baseline.left, baseline.right].every(v => Number.isFinite(v) && v >= .18 && v <= .5)) return false;
    this.clear(); this.measurements = new BlinkMeasurements(baseline);
    this.state = "running"; this.startedAt = now; this.wallStart = wallTime; this.runSince = now;
    this.before = tirednessRating(rating); this.reason = "Monitoring active · Experimental behavioral indicators";
    return true;
  }
  pause(now, reason = "Paused by you.", quality = false) {
    if (this.state !== "running") return;
    now = Number.isFinite(now) ? Math.max(now, this.runSince, this.lastValid ?? this.runSince) : (this.lastValid ?? this.runSince);
    this.runningMs += Math.max(0, now - this.runSince);
    this.runSince = null;
    this.state = quality ? "quality_paused" : "paused";
    this.reason = reason;
    this.measurements.interrupt(); this.lastValid = null; this.previousPose = null; this.pose = null; this.motion = null;
    this.record(now, true);
  }
  resume(now, baseline) {
    if (!["paused", "quality_paused"].includes(this.state) || !Number.isFinite(now) || now < this.startedAt ||
        !baseline || ![baseline.left, baseline.right].every(v => Number.isFinite(v) && v >= .18 && v <= .5)) return false;
    this.measurements.baseline = { ...baseline }; this.measurements.interrupt();
    this.state = "running"; this.runSince = now; this.lastPoint = null;
    this.reason = "Monitoring active · Experimental behavioral indicators";
    return true;
  }
  stop(now) {
    if (["idle", "stopped"].includes(this.state)) return;
    if (this.state === "running") this.pause(now);
    this.state = "stopped"; this.reason = "Session stopped. Results remain only in this page until you leave or start again.";
  }
  setAfter(value) { if (this.state === "stopped") this.after = tirednessRating(value); }
  sample({ eyes, pose, valid, reason }, now) {
    if (this.state !== "running") return;
    if (!valid || !pose || !Object.values(pose).every(Number.isFinite) || !this.measurements.update(eyes, now)) {
      this.pause(now, reason || "Measurement interrupted. Realign and Resume when ready.", true); return;
    }
    if (this.previousPose && this.lastValid !== null) {
      const dt = now - this.lastValid;
      if (dt > 0) {
        const speed = Math.hypot(pose.yaw - this.previousPose.yaw, pose.pitch - this.previousPose.pitch, pose.roll - this.previousPose.roll) * 1000 / dt;
        this.motion = this.motion === null ? speed : .2 * speed + .8 * this.motion;
      }
    }
    this.previousPose = { ...pose }; this.pose = { ...pose }; this.lastValid = now;
    this.record(now);
  }
  record(now, gap = false) {
    if (!this.measurements || (this.lastPoint !== null && now - this.lastPoint < 1000 && !gap)) return;
    this.lastPoint = now;
    const stats = this.measurements.snapshot(now);
    this.timeline.push({ time: this.wallStart + now - this.startedAt, gap,
      closedPercent: gap ? null : stats.closedPercent, motion: gap ? null : this.motion,
      blinks: stats.blinkCount, yaw: gap ? null : this.pose?.yaw,
      pitch: gap ? null : this.pose?.pitch, roll: gap ? null : this.pose?.roll });
    const currentTime = this.wallStart + now - this.startedAt;
    this.timeline = this.timeline.filter(point => currentTime - point.time <= 300000).slice(-BEHAVIOR.timelineLimit);
  }
  snapshot(now) {
    const stats = this.measurements?.snapshot(now);
    const total = this.runningMs + (this.runSince === null ? 0 : Math.max(0, now - this.runSince));
    const coverage = stats && total > 0 ? Math.min(100, stats.validMs / total * 100) : null;
    return { ...stats, state: this.state, coverage, pose: this.pose, motion: this.motion,
      confidence: this.state !== "running" || this.lastValid === null ? "Unavailable" : coverage >= 90 ? "High (heuristic)" : "Limited (heuristic)" };
  }
}
