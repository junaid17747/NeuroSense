# V3 webcam alignment and behavioral monitoring

The authenticated Overview dashboard includes **📷 Start Webcam Monitoring** below
the attention metrics and above the trend chart. It opens **Webcam Monitoring**,
which is also available in the workspace sidebar. **Back to Overview** restores
the Overview view and retains the user's EEG history.

The page reuses the Streamlit v2 component and live browser preview. **Enable Camera** explicitly
requests video-only camera access through the browser's permission system.
Opening or revisiting the page never requests camera access, including when
permission was previously granted. **Disable Camera**, navigation away, and
logout stop the active tracks. A permission request that completes after leaving
the page is also stopped. Permission denial and missing or busy cameras show
retry guidance.

Video, sampled pixels, landmarks, aggregate measurements, and ratings stay in browser memory. The component does
not send component state, frames, images, or landmarks to Python, FastAPI, or the
database. There is no recording, recognition, identification, face matching, or
face enrollment. The page explicitly labels this as a **camera-positioning check**,
not an EEG measurement, neurological assessment, or seizure detector.

## Alignment behavior

The mirrored preview has a centered oval with a fixed physical width/height ratio
of 0.72. Its coordinates are derived from the uncropped video dimensions and used
by both the overlay and the alignment checks, including portrait cameras. Cyan
means adjusting or waiting, amber indicates a correction, and green means aligned.
Text feedback is always shown alongside color.

MediaPipe detects up to two faces to distinguish one face from multiple faces.
The checks cover absence of a face, multiple faces, horizontal/vertical position,
distance, lighting, and rotation away from a frontal upright pose. Directions
refer to movement within the mirrored preview. Light is sampled from the face
region, or the central frame when there is no detected face.

**Ready for Monitoring** requires 3 continuous seconds of successful checks on
fresh frames plus a usable open-eye baseline. A visible progress bar resets when
alignment is lost, video stalls, the tab is hidden, or the camera/model fails.
**Start Monitoring** is a separate explicit action and is disabled until ready.
Keep eyes comfortably open during calibration; occasional natural blinks are
handled by the robust baseline. Glasses glare or unusable eye geometry can prevent
calibration even when the face is centered.

## Phase 2 sessions

Start reuses the existing stream and detector. Pause suspends measurement while
the preview continues. Resume requires fresh alignment and eye calibration and
an explicit click. Stop ends the session and leaves its aggregate summary visible.
There is no automatic resume after tracking recovers. Disable Camera also stops
the session. Navigation, logout, component teardown, or starting another session
clear the previous measurements and ratings.

The live cards show blink count, estimated blinks per valid minute, current and
last eye-closure duration, percentage of valid time eyes appear closed, approximate
yaw/pitch/roll, smoothed angular head movement, tracking confidence, and measurement
coverage. The timeline retains up to 300 one-second aggregate points from the
latest 5 minutes, with local timestamps and gaps for paused/unavailable data.
Instantaneous pose/closure readings become unavailable during pauses; valid
session totals remain visible. No diagnosis, medical alert, or fatigue prediction
is computed.

Before/after tiredness ratings are optional integers from 0 (not tired) to 10
(very tired). The before value is captured at Start; the after field is enabled
only after Stop. They are labeled self-reports and never influence camera metrics.
There is **no save/export feature** in this phase: even aggregate results and
ratings stay local and transient, so no storage consent or database changes are
introduced. Component instances do not share session data across accounts or tabs.

### Measurement method

Eye aspect ratio (EAR) uses six landmarks per eye, with distances in video-pixel
coordinates. Contours are `[33,160,158,133,153,144]` and
`[362,385,387,263,373,380]`. During stable setup, at least 30 eye samples spanning
2.5 seconds are collected; the 80th percentile provides a separate open-eye
baseline for each eye. Median consistency and plausible open EAR bounds reject
unusable calibration. These buffers remain in memory and are cleared on reset.

Both eyes below 65% of their baseline start a closure; reopening requires 82%.
Hysteresis suppresses threshold jitter. An observed open–closed–open sequence
counts as one blink only with at least two closed observations and estimated
closure duration of 60–700 ms. Longer closures contribute to eyes-closed time,
but are not counted as repeated blinks. Transitions are interpolated halfway
between samples. Blink counts are conservative estimates and can miss short
blinks; partial blinks or winks are not counted reliably.

The [EAR method](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
informs the ratio calculation; this implementation uses calibrated thresholds
and temporal checks, not that paper's trained classifier or reported accuracy.

Only consecutive usable samples contribute observed time. Pauses, missing frames,
nonfinite values, and poor tracking cannot join a closure into a false blink.
Blink rate uses valid observed minutes and remains unavailable until 10 valid
seconds have accumulated. Eyes-closed percentage uses valid observed time as its
denominator. Measurement coverage is valid observed time divided by requested
running time, excluding explicit or automatic pauses.

Tracking confidence is explicitly **heuristic**, not a detector probability:
unavailable while paused or without a valid observation, high with at least 90%
coverage, otherwise limited. MediaPipe's landmark result does not supply a
calibrated per-frame face confidence. Lighting, face count, eye geometry,
bilateral consistency, pose and frame freshness gate measurements before they
contribute. Slight face translations are allowed during a session so ordinary
movement need not invalidate measurements. Multiple faces, absence, excessive
rotation, clipping, bad lighting, inconsistent eyes, and stale frames pause it.

## Implementation and assets

`dashboard/webcam.py` composes local ES modules into the existing component:

- `webcam_alignment.js`: pure geometry, thresholds, light/pose checks, and state machine.
- `webcam_behavior.js`: eye calibration, blink events, quality gates, head pose, and isolated session state.
- `webcam_session_ui.js`: aggregate measurement cards and timestamped SVG timeline.
- `webcam_detector.js`: pinned MediaPipe Face Landmarker loader and options.
- `webcam.js`: camera lifecycle, sampling, rendering, explicit Start action, and cleanup.

There are no additional Python dependencies. After Enable Camera succeeds, the
browser downloads `@mediapipe/tasks-vision@0.10.32` JavaScript/WASM from jsDelivr and
the Google Face Landmarker float16 model version 1. These are inbound asset
downloads; no camera content is attached. Initial loading needs internet access
to `cdn.jsdelivr.net` and `storage.googleapis.com`. Versions and URLs live in
`FACE_ASSETS`; no `latest` URLs are used. See the
[MediaPipe web guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js).

Model failures or a 30-second load timeout preserve the camera preview, disable
Start, and offer Retry Alignment. Runtime inference failures also offer retry.
Late model loads are closed after cancellation; active models, animation callbacks,
timers, camera tracks, and the lighting canvas are released on disable or unmount.
Browser camera access requires HTTPS or localhost.

## Thresholds and limitations

All positioning defaults are centralized in `ALIGNMENT` and covered by tests:

| Check | Default |
| --- | --- |
| Stable alignment | 3,000 ms |
| Alignment state-machine gap limit | 600 ms (behavioral checks impose the stricter limit below) |
| Maximum behavioral sample gap / inference duration | 150 ms |
| Detection cadence | Target up to 25 checks/second; fresh video frames only |
| Center tolerance | 18% of the oval radius, per axis |
| Face bounding-box fill | 68–104% of the oval width and height |
| Head rotation | At most 20° from the canonical frontal pose |
| Mean luminance | 45–220 on a 0–255 scale |
| Under/overexposed pixels | At most 45% below 25 or above 245 |

These are adjustable positioning heuristics, not validated health measurements.
Lighting, occlusion, camera quality, skin appearance, and detector confidence can
affect results. Rotation is an approximate combined angle from MediaPipe's face
transformation matrix; it is not a clinical head-pose measurement. This is not a
liveness or anti-spoofing check. CPU/WASM detection is throttled on the browser's
main thread; slow devices may pause the interface or fail the sampling/quality
checks. Gaps or inference times above 150 ms cannot contribute measurements.
Glasses, glare, eyelid/eye shape, occlusion, and landmark hallucination can still
produce missed or incorrect events even after checks pass. This has not been
validated against an annotated adult blink dataset or across physical mobile
cameras. It is an experimental adult wellness research interface, not a medical
measurement system. It cannot measure brainwaves or stress, diagnose fatigue or
neurological disease, or detect seizures.

## Verification

Run the regression and camera lifecycle tests from the project root:

```sh
rtk proxy .venv/bin/python -m unittest discover -s tests -v
rtk proxy node tests/test_webcam_alignment.mjs
rtk proxy node tests/test_webcam_behavior.mjs
rtk proxy node tests/test_webcam_camera.mjs
```

The Streamlit navigation tests use a temporary authentication database and fixed
sensor data, click the actual Overview button, verify page order and the return
path, and check that signed-out or revoked sessions cannot access the webcam.
The JavaScript tests exercise pure alignment logic and the actual component with
mocked browser media APIs and detector outputs. They cover geometry, direction,
distance and pose limits, lighting, calibration timing, stale/hidden frames,
explicit Start, model failures/timeouts, retries, blink events, invalid samples,
session controls, ratings, account/session isolation and resource cleanup without
using a physical camera.

For a browser check, sign in, open the webcam page, verify there is no camera
request, enable and allow the camera, then follow the alignment guidance. Hold
still for 3 seconds and explicitly select Start Monitoring. Move away and verify
measurement pauses. Realign and explicitly Resume, test Pause and Stop, and enter
optional before/after ratings. Disable the camera or return to Overview.
Verify that the browser camera indicator turns off. Re-enter the page and verify
that another Enable Camera click is required. Repeat with permission denied and
with logout while the preview is active.
