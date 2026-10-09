// Pure camera-positioning heuristics. These are not medical measurements.
export const ALIGNMENT = Object.freeze({
  stableMs: 3000,
  maxGapMs: 600,
  sampleIntervalMs: 125,
  centerTolerance: 0.18, // Fraction of the oval's radius.
  minFill: 0.68,
  maxFill: 1.04,
  maxRotationDegrees: 20,
  minLuminance: 45,
  maxLuminance: 220,
  maxClippedFraction: 0.45,
});

// Fit the same physically proportioned oval to landscape or portrait video.
// Coordinates are relative to the uncropped source frame, not its container.
export function alignmentGuide(width, height) {
  if (!(width > 0 && height > 0)) return null;
  const ry = Math.min(0.38, 0.40 * width / (0.72 * height));
  return { cx: 0.5, cy: 0.5, rx: ry * 0.72 * height / width, ry };
}

export function faceBounds(landmarks) {
  if (!landmarks?.length || landmarks.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) return null;
  const xs = landmarks.map(p => p.x), ys = landmarks.map(p => p.y);
  const left = Math.min(...xs), right = Math.max(...xs);
  const top = Math.min(...ys), bottom = Math.max(...ys);
  if (right <= left || bottom <= top) return null;
  return { left, right, top, bottom, width: right - left, height: bottom - top,
    cx: (left + right) / 2, cy: (top + bottom) / 2 };
}

export function headRotationDegrees(matrix) {
  const m = matrix?.data;
  if (matrix?.rows !== 4 || matrix?.columns !== 4 || m?.length !== 16 || !Array.from(m).every(Number.isFinite)) return null;
  // The trace gives the angle away from the canonical frontal pose. It is
  // unchanged by row/column storage order. Remove uniform face-scale first.
  const scale = Math.hypot(m[0], m[1], m[2]);
  if (scale < 1e-6) return null;
  const cosine = ((m[0] + m[5] + m[10]) / scale - 1) / 2;
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}

export function lightingStats(rgba) {
  if (!rgba?.length || rgba.length % 4) return null;
  let total = 0, dark = 0, bright = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const value = 0.2126 * rgba[i] + 0.7152 * rgba[i + 1] + 0.0722 * rgba[i + 2];
    total += value;
    if (value < 25) dark++;
    if (value > 245) bright++;
  }
  const count = rgba.length / 4;
  return { mean: total / count, darkFraction: dark / count, brightFraction: bright / count };
}

const feedback = (code, message, tone = "warning") => ({ code, message, tone, aligned: false });

export function evaluateAlignment({ faces, matrix, lighting, width, height }, thresholds = ALIGNMENT) {
  if (!Array.isArray(faces)) return feedback("invalid", "Face check unavailable. Try again.");
  if (faces.length > 1) return feedback("multiple_faces", "Only one face should be in view.");
  if (!lighting || ![lighting.mean, lighting.darkFraction, lighting.brightFraction].every(Number.isFinite)) {
    return feedback("invalid", "Unable to check lighting. Try again.");
  }
  if (lighting.mean < thresholds.minLuminance || lighting.darkFraction > thresholds.maxClippedFraction) {
    return feedback("too_dark", "Add light in front of your face.");
  }
  if (lighting.mean > thresholds.maxLuminance || lighting.brightFraction > thresholds.maxClippedFraction) {
    return feedback("too_bright", "Reduce glare or bright light on your face.");
  }
  if (!faces.length) return feedback("no_face", "Position your face inside the oval.", "neutral");
  const bounds = faceBounds(faces[0]);
  const guide = alignmentGuide(width, height);
  const rotation = headRotationDegrees(matrix);
  if (!bounds || !guide || rotation === null) return feedback("invalid", "Keep your full face visible.");
  if (rotation > thresholds.maxRotationDegrees) return feedback("rotation", "Look straight at the camera; keep your head upright.");
  const fillX = bounds.width / (2 * guide.rx), fillY = bounds.height / (2 * guide.ry);
  if (fillX > thresholds.maxFill || fillY > thresholds.maxFill) return feedback("too_close", "Move a little farther away.");
  if (fillX < thresholds.minFill || fillY < thresholds.minFill) return feedback("too_far", "Move a little closer.");
  // Preview is mirrored; directions refer to movement in that preview.
  const dx = (1 - bounds.cx - guide.cx) / guide.rx;
  const dy = (bounds.cy - guide.cy) / guide.ry;
  if (dx < -thresholds.centerTolerance) return feedback("left", "Move right toward the center.");
  if (dx > thresholds.centerTolerance) return feedback("right", "Move left toward the center.");
  if (dy < -thresholds.centerTolerance) return feedback("up", "Move down toward the center.");
  if (dy > thresholds.centerTolerance) return feedback("down", "Move up toward the center.");
  return { code: "aligned", message: "Hold still for 3 seconds.", tone: "success", aligned: true };
}

export class AlignmentTracker {
  constructor(thresholds = ALIGNMENT) { this.thresholds = thresholds; this.reset(); }
  reset() { this.since = null; this.last = null; this.ready = false; this.monitoring = false; }
  update(aligned, now) {
    const stale = this.last !== null && (now <= this.last || now - this.last > this.thresholds.maxGapMs);
    if (!Number.isFinite(now) || !aligned || stale) this.reset();
    if (!Number.isFinite(now) || !aligned) return this.snapshot(now);
    if (this.since === null) this.since = now;
    this.last = now;
    this.ready = now - this.since >= this.thresholds.stableMs;
    return this.snapshot(now);
  }
  snapshot(now) {
    return { ready: this.ready, monitoring: this.monitoring,
      progress: this.since === null ? 0 : Math.min(1, Math.max(0, (now - this.since) / this.thresholds.stableMs)) };
  }
  start(now) {
    if (!Number.isFinite(now) || !this.ready || this.last === null || now < this.last || now - this.last > this.thresholds.maxGapMs) {
      this.reset();
      return false;
    }
    this.monitoring = true;
    return true;
  }
}
