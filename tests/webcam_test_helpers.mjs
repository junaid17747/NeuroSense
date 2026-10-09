import { readFile } from "node:fs/promises";

export async function loadAlignment() {
  const source = await readFile(new URL("../dashboard/static/webcam_alignment.js", import.meta.url), "utf8");
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

export const frontalMatrix = { rows: 4, columns: 4, data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
export const goodLight = { mean: 125, darkFraction: 0, brightFraction: 0 };

export function face({ cx = 0.5, cy = 0.5, fill = 0.85, width = 640, height = 480, leftEAR = .3, rightEAR = .3 } = {}) {
  const ry = Math.min(0.38, 0.40 * width / (0.72 * height));
  const rx = ry * 0.72 * height / width;
  const points = Array.from({ length: 478 }, () => ({ x: cx, y: cy }));
  points[0] = { x: cx - rx * fill, y: cy - ry * fill };
  points[1] = { x: cx + rx * fill, y: cy + ry * fill };
  const eyeWidth = rx * fill * .45;
  for (const [indices, offset, ear] of [[[33,160,158,133,153,144], -.5, leftEAR], [[362,385,387,263,373,380], .5, rightEAR]]) {
    const x = cx + rx * fill * offset, y = cy - ry * fill * .2;
    const dy = eyeWidth * ear * width / height / 2;
    const coords = [[x-eyeWidth/2,y],[x-eyeWidth/4,y-dy],[x+eyeWidth/4,y-dy],[x+eyeWidth/2,y],[x+eyeWidth/4,y+dy],[x-eyeWidth/4,y+dy]];
    indices.forEach((index, i) => { points[index] = { x: coords[i][0], y: coords[i][1] }; });
  }
  return points;
}

export async function loadBehavior() {
  const source = (await Promise.all(["webcam_alignment.js", "webcam_behavior.js"].map(name =>
    readFile(new URL(`../dashboard/static/${name}`, import.meta.url), "utf8")))).join("\n");
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

export function rotation(degrees, axis = "z") {
  const angle = degrees * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
  const values = axis === "x" ? [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]
    : axis === "y" ? [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]
    : [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  return { rows: 4, columns: 4, data: values };
}
