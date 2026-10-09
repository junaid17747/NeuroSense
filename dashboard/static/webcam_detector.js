// Fixed asset versions: downloads contain model/runtime code, never camera data.
export const FACE_ASSETS = Object.freeze({
  module: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs",
  wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
  model: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
});

export async function createFaceDetector() {
  const { FaceLandmarker, FilesetResolver } = await import(FACE_ASSETS.module);
  const fileset = await FilesetResolver.forVisionTasks(FACE_ASSETS.wasm);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: FACE_ASSETS.model, delegate: "CPU" },
    runningMode: "VIDEO",
    numFaces: 2, // Detect a second face so it cannot silently pass calibration.
    minFaceDetectionConfidence: 0.6,
    minFacePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
  });
}
