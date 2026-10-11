# NeuroSense Webcam rPPG — Phase 1

## Status and intended use

This is an experimental, offline browser implementation of contactless pulse-rate estimation from webcam color changes. It is a wellness and research prototype. It is not a medical heart-rate monitor, ECG, diagnostic tool, seizure detector, or emergency alarm. No real-camera accuracy or clinical validity has been established.

## Scientific principle and selected method

Remote photoplethysmography (rPPG) estimates pulsatile blood-volume changes from small, periodic changes in light reflected by skin and recorded as video. NeuroSense uses Plane-Orthogonal-to-Skin (POS), described by Wang et al. The method normalizes RGB channels, projects the color traces onto a plane intended to suppress common skin-reflectance and motion components, then estimates the dominant pulse frequency. The source paper is [Wang et al., *Algorithmic principles of remote-PPG*, IEEE Transactions on Biomedical Engineering (2017), DOI: 10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282); publication metadata is also available from the [Eindhoven University of Technology research portal](https://research.tue.nl/en/publications/algorithmic-principles-of-remote-ppg/).

The implementation in `dashboard/static/webcam_rppg.js` computes channel means over a 25-second window, forms the POS traces `X = G/Gmean - B/Bmean` and `Y = -2R/Rmean + G/Gmean + B/Bmean`, combines them using `S = X + (std(X)/std(Y))*Y`, removes a linear trend, applies a Hann window, and uses an FFT to search the supported 0.75–3.0 Hz pulse band (45–180 BPM). The effective 25-second frequency-bin spacing at 10 Hz is approximately 0.04 Hz (2.4 BPM); zero padding and parabolic interpolation refine the peak location but do not create new signal information. Frequency energy outside the supported band is checked so an unsupported rate is reported unavailable rather than clamped.

## Browser data flow

The existing webcam component opens one `MediaStream`. When the user checks **Enable experimental pulse estimation**, and only while the existing monitoring session is running, the component reuses its video element and face landmarks. It samples at most once per 100 ms, draws the current frame to a temporary 160×120 canvas, and calculates mean RGB values inside three landmark-relative polygons for forehead and cheeks. The polygons use source-frame normalized coordinates; CSS mirroring does not alter pixel selection. The small canvas and RGB data remain in component memory. The component does not transfer frames, landmarks, RGB traces, or rPPG values to Streamlit or the backend.

Pause, quality-pause, Stop, camera shutdown, page navigation, and component unmount clear the rolling pulse buffer. Resume starts a new 25-second warm-up. Unchecking the opt-in control also immediately clears the buffer. The data source is labeled `WEBCAM_RPPG_EXPERIMENTAL`. The existing synthetic sensor retains the separate `SIMULATED_HEART_RATE` label; webcam rPPG is not included in EEG attention calculations, persistence, or synchronization.

Optional reference-device observations can be entered in the expanded validation panel. They exist only in browser memory until the component is left. The user can explicitly download those observations as a local CSV. The report includes reference source, paired timestamps, BPM values, absolute error, signal quality, duration, coverage, lighting, and motion. MAE, RMSE, bias, valid coverage, and failure rate are summaries of only the manually recorded pairs; they are not accuracy claims. Do not enter identifying information into the reference-source field.

## Sampling and quality gates

The sampler uses `performance.now()` timestamps and the video's advancing `currentTime`, rejects repeated frames and non-increasing timestamps, resets on gaps over 300 ms, and uses linear interpolation to a 10 Hz grid. Interpolation is refused where adjacent samples are separated by more than 300 ms. A rolling buffer is capped at 320 RGB summaries and retains about one second before the 25-second analysis window to bracket interpolation at its left edge.

The estimator is unavailable until 25 seconds have accumulated with at least 90% sample coverage. The display reports estimated-window duration and sample coverage, not calibrated confidence. It also rejects:

- multiple or lost faces, tracking changes, failed alignment, or behavioral quality pauses;
- face movement over 2.5% of normalized frame width/height between rPPG samples;
- an average forehead luminance below 35 or above 225 (8-bit scale), clipped/dark pixels exceeding 20%, or a frame-to-frame luminance change above 20%;
- fewer than 18 usable pixels in any selected skin region;
- a spectral peak-to-next-separated-peak power ratio below 2.2;
- region estimates that differ by more than 8 BPM;
- excessive rate instability across successive rolling estimates;
- estimates outside 45–180 BPM, low coverage, or a sampling interruption.

These are initial engineering thresholds, not clinically validated limits. `Good`, `Limited`, `Poor`, and `Unavailable` are heuristic status labels, not probability or medical quality scores. Skin tone, makeup, facial hair, glasses, camera processing, compression, exposure, light spectrum, pose, and device frame timing can cause failure or bias. A result outside the range means “unsupported by this experimental estimator,” not an abnormal pulse.

## Privacy and lifecycle

Video is processed locally. The page does not record video, recognize identity, upload images, log landmarks, or save raw RGB signals. The temporary RGB buffer exists only while the opt-in sampling state is active and is cleared on pause, stop, camera shutdown, and unmount. The optional reference rows remain temporarily in browser memory and are cleared when leaving the page; a CSV is created only after the user presses **Export local CSV**. The browser may retain a user-selected export on the user's device. The ordinary simulated-heart-rate dashboard and existing persistence are not changed by this rPPG component, and no rPPG value is sent to that dashboard.

The component is mounted within the authenticated workspace, but the rPPG processing itself does not use user identity or backend authorization. Browser security still depends on the hosting page, browser, extensions, and the user's device. Camera access requires browser permission and a secure context (HTTPS or localhost).

## Reproducible reference comparison

Use a pulse oximeter or other suitable reference monitor with a visible/stable BPM reading. Follow the device's directions and manually enter its source and reading at the time an rPPG estimate is displayed. Record observations across repeated sessions, cameras, lighting conditions, motion conditions, and a representative range of skin tones. Include failed/unavailable rPPG readings in the denominator by recording the paired reference when the estimate is unavailable. Export the local CSV and preserve its provenance separately under the applicable research/data-handling approval.

For `e_i = rPPG_i - reference_i`, report MAE as `mean(|e_i|)`, RMSE as `sqrt(mean(e_i²))`, and bias as `mean(e_i)`. Valid measurement coverage is the fraction of attempted paired reference observations with an rPPG BPM; failure rate is one minus that coverage. Do not discard failures when reporting coverage. Analyze repeated measurements by participant/device/condition and do not treat repeated windows from one person as independent participants. No validation data or performance result is included with this implementation.

## Browser and performance considerations

The code uses browser video, canvas, `performance.now()`, and standard JavaScript only; it adds no second camera stream or browser package. The image copied for pixel statistics is only 96×72, and polygon tests are limited to each region's bounding box; POS runs on at most 250 resampled points at 10 Hz. This bounded work is performed on the component's JavaScript thread; unusually slow devices may still stutter, in which case camera sampling gaps fail the quality gate. A worker/GPU implementation would need separate validation before use. Browser canvas security restrictions, denied camera permissions, or absent Canvas APIs result in unavailable estimates.

## Troubleshooting

- **Warm-up stays at 0:** start the camera, finish alignment, Start Monitoring, enable rPPG, and keep the face centered for 25 seconds.
- **Insufficient lighting / saturated:** use steady, diffuse light facing the user and avoid a bright window or direct lamp behind/in front of the face.
- **Excessive movement / face tracking lost:** keep forehead and both cheeks visible and remain still; a tracking interruption starts a fresh window.
- **Coverage too low:** use a camera/browser with a stable frame rate and close other applications using substantial CPU/GPU resources.
- **Regions disagree / ambiguous peak:** improve steady front lighting, remove strong color casts if possible, and repeat; do not interpret a rejected measurement.
- **Camera/detector unavailable:** verify HTTPS or localhost, permission, and network access to the MediaPipe runtime/model assets used by the existing alignment component.

## Manual real-camera checklist

1. Serve NeuroSense on HTTPS or localhost and sign in. Check that the Live EEG/Overview sensor is labeled `SIMULATED_HEART_RATE` when running in simulation mode.
2. Open Webcam Monitoring. Confirm that the rPPG checkbox is initially off and the source says `UNAVAILABLE`.
3. Enable the camera and finish the existing alignment and monitoring start steps. Verify that preview, blinking/eye closure/head tracking, session timeline, and Pause/Resume/Stop remain functional.
4. Opt in to rPPG. Confirm warm-up only begins while monitoring runs and the source reads `WEBCAM_RPPG_EXPERIMENTAL`. Keep still under even light for 25 seconds. A real estimate may still be unavailable; do not count this as a successful measurement without comparing it to a reference device.
5. Test dim light, bright glare, face movement, partial face occlusion, multiple faces, a frozen camera, and detector/camera shutdown. Confirm an explicit unavailable reason, no stale BPM, and no fallback to simulated values.
6. Pause, Stop, disable the camera, navigate away, and log out. Confirm sampling stops and the buffer resets. Confirm EEG views do not show the webcam value.
7. Optionally enter reference measurements, include unavailable attempts, export a local CSV, and compute/report errors and coverage. Do not call the outcome clinically accurate without a suitable independent validation study and specialist review.

## Evidence status

- **Implemented and tested:** browser-local sampling path, POS signal processing, bounded timestamped buffer, algorithmic quality gates, separate source label, temporary reference-entry/export workflow, synthetic algorithm tests, and mocked component lifecycle tests.
- **Not experimentally evaluated:** pulse accuracy against a reference device, performance across demographic/device/lighting subgroups, and sustained real-camera usability.
- **Not implemented:** rPPG transfer or synchronization with EEG, external sensor integration, medical monitoring, and any clinical decision workflow.
- **Clinical status:** not clinically validated; no medical claims are made.
