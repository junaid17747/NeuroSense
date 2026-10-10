// Pure camera-positioning heuristics. These are not medical measurements.
export const ALIGNMENT = Object.freeze({
  stableMs: 3000,
  maxGapMs: 600,
  sampleIntervalMs: 125,
  centerTolerance: 0.22, // Fraction of the guide radius; forgiving of ordinary webcam framing.
  minFill: 0.45, // Face keypoints may occupy less of the guide at normal laptop distance.
  maxFill: 0.94, // Keep a visible margin so accepted landmarks remain inside the guide.
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
  // MediaPipe normalized coordinates are relative to the actual source frame.
  // Use stable facial-oval anchors instead of min/max over all 478 points:
  // irises, lips, and occasional outlier landmarks otherwise distort size.
  const anchors = [landmarks?.[10], landmarks?.[152], landmarks?.[234], landmarks?.[454]];
  if (anchors.some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) return null;
  const [forehead, chin, leftCheek, rightCheek] = anchors;
  const left = Math.min(leftCheek.x, rightCheek.x), right = Math.max(leftCheek.x, rightCheek.x);
  const top = Math.min(forehead.y, chin.y), bottom = Math.max(forehead.y, chin.y);
  if (right <= left || bottom <= top || right - left > 0.8 || bottom - top > 0.95) return null;
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

const feedback = (code, message, diagnostics, tone = "warning") => ({ code, message, tone, aligned: false, diagnostics });

export function evaluateAlignment({ faces, matrix, lighting, width, height }, thresholds = ALIGNMENT) {
  const exactlyOneFace = Array.isArray(faces) && faces.length === 1;
  const bounds = exactlyOneFace ? faceBounds(faces[0]) : null;
  const guide = alignmentGuide(width, height);
  const rotation = headRotationDegrees(matrix);
  const lightValid = !!lighting && [lighting.mean, lighting.darkFraction, lighting.brightFraction].every(Number.isFinite);
  const lightingPass = lightValid && lighting.mean >= thresholds.minLuminance && lighting.mean <= thresholds.maxLuminance &&
    lighting.darkFraction <= thresholds.maxClippedFraction && lighting.brightFraction <= thresholds.maxClippedFraction;
  const posePass = rotation !== null && rotation <= thresholds.maxRotationDegrees;
  const faceSizePass = !!bounds && !!guide &&
    bounds.width / (2 * guide.rx) >= thresholds.minFill && bounds.width / (2 * guide.rx) <= thresholds.maxFill &&
    bounds.height / (2 * guide.ry) >= thresholds.minFill && bounds.height / (2 * guide.ry) <= thresholds.maxFill;
  let centered = false;
  if (bounds && guide) {
    // Landmarks use unmirrored source-frame x coordinates. Mirror only the
    // horizontal offset used for user-facing guidance; the centered test is symmetric.
    const dx = (guide.cx - bounds.cx) / guide.rx;
    const dy = (bounds.cy - guide.cy) / guide.ry;
    centered = Math.abs(dx) <= thresholds.centerTolerance && Math.abs(dy) <= thresholds.centerTolerance;
  }
  const diagnostics = {
    exactlyOneFace,
    centered: bounds && guide ? centered : null,
    faceSize: bounds && guide ? faceSizePass : null,
    lighting: lightValid ? lightingPass : null,
    headPose: rotation !== null ? posePass : null,
  };
  const fail = (code, message, tone) => feedback(code, message, diagnostics, tone);
  if (!Array.isArray(faces)) return fail("invalid", "Face check unavailable. Try again.");
  if (faces.length > 1) return fail("multiple_faces", "Only one face should be in view.");
  if (!lightValid) return fail("invalid", "Unable to check lighting. Try again.");
  if (lighting.mean < thresholds.minLuminance || lighting.darkFraction > thresholds.maxClippedFraction) return fail("too_dark", "Add light in front of your face.");
  if (lighting.mean > thresholds.maxLuminance || lighting.brightFraction > thresholds.maxClippedFraction) return fail("too_bright", "Reduce glare or bright light on your face.");
  if (!faces.length) return fail("no_face", "Position your face inside the oval.", "neutral");
  if (!bounds || !guide) return fail("invalid", "Keep your forehead, cheeks, and chin visible.");
  if (!posePass) return fail("rotation", "Look straight at the camera; keep your head upright.");
  const fillX = bounds.width / (2 * guide.rx), fillY = bounds.height / (2 * guide.ry);
  if (fillX > thresholds.maxFill || fillY > thresholds.maxFill) return fail("too_close", "Move a little farther back so your full face stays inside the oval.");
  if (fillX < thresholds.minFill || fillY < thresholds.minFill) return fail("too_far", "Move a little closer until your face fills more of the oval.");
  const dx = (guide.cx - bounds.cx) / guide.rx, dy = (bounds.cy - guide.cy) / guide.ry;
  // Corrective directions follow the mirrored preview, while centering is symmetric.
  if (dx < -thresholds.centerTolerance) return fail("left", "Move right toward the center.");
  if (dx > thresholds.centerTolerance) return fail("right", "Move left toward the center.");
  if (dy < -thresholds.centerTolerance) return fail("up", "Move down toward the center.");
  if (dy > thresholds.centerTolerance) return fail("down", "Move up toward the center.");
  return { code: "aligned", message: "Hold still for 3 seconds.", tone: "success", aligned: true, diagnostics };
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
