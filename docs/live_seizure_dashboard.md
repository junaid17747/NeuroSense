# Live EEG Seizure Research dashboard

The page opens with the interactive waveform and controls, followed by four evaluation cards, one compact research-status summary, and a collapsed **Research Details** expander. Dataset overview, age selection, readiness checks, hashes, and provenance remain available in the expander.

The authenticated research page refreshes its status fragment every five seconds while open. It fingerprints configured artifact metadata before consulting a 15-second cache; new/changed files invalidate the snapshot immediately. **Refresh now** clears the cache. Archive status runs full ZIP/checksum verification only after approved archive authorization is validated; file presence alone is never called approval or verification. Approval fields and identities are never displayed.

The synthetic demonstration has a one-second bounded fragment with Start, Pause, Resume, Stop / Reset, elapsed time, and channel selection controls. Its generated samples are deterministic and cannot feed the real evaluation cards. CHB-MIT EDF playback is a separate source, is limited to approved/readiness-checked pilot files and a selected channel/time segment, and is labeled prerecorded playback. The headset source remains unavailable until a compatible device is connected.

The four metrics are loaded only from the existing complete, provenance-validated experimental model/evaluation bundle. Sensitivity displays detected/evaluable reference events and percentage; false alarms include evaluable non-seizure hours; latency is the median onset-to-trigger latency and matched-event count. Missing or invalid results display **Not evaluated** with a brief explanation.

For isolated UI acceptance tests only, `NEUROSENSE_SEIZURE_TEST_ARTIFACT` may point to a `neurosense-dashboard-test-evaluation-v1` JSON artifact. It must contain `artifact_kind: "SYNTHETIC_TEST_ONLY"`, `clinical_validation: false`, `research_performance_claim: false`, the validated metric fields, and a correct SHA-256 digest. These fixtures appear only inside Research Details, labeled **SYNTHETIC TEST ONLY — NOT RESEARCH PERFORMANCE**; they never populate the four research evaluation cards. Keep this variable unset outside isolated tests.

Optional paths:

- `NEUROSENSE_CHBMIT_ARCHIVE_ROOT` (default: `~/NeuroSense-EEG-Data/archive`)
- `NEUROSENSE_CHBMIT_TEMPORARY_ROOT` (default: `~/NeuroSense-EEG-Data/temporary`)
- `NEUROSENSE_CHBMIT_ARCHIVE_APPROVAL` (default: `~/NeuroSense-EEG-Data/manifests/archive-approval.json`)
- `NEUROSENSE_CHBMIT_REVIEW_PACKET` (default: `~/NeuroSense-EEG-Data/reports/annotation-review-packet.json`)
- `NEUROSENSE_CHBMIT_ROOT` and `NEUROSENSE_SEIZURE_RUN_DIR` for the separately gated frozen-pilot readiness/model-artifact loader.

Training artifacts now record UTC training and evaluation timestamps for display in validated provenance. Test-only fixtures do not create a model or research result.

## Simplified dashboard verification — 2026-10-13

The active route in `dashboard/app.py` renders `render_seizure_lab` inside the
`seizure_research_workspace` container. Its scoped styles make control labels
readable on the existing dark background and keep all four “Not evaluated”
values visible without clipping.

Headless Chrome opened the actual entrypoint at `http://127.0.0.1:8501`, signed in
using an isolated temporary test database, and navigated through the authenticated
sidebar. The browser confirmed the waveform-first layout, the four evaluation
cards, one compact status summary, and the collapsed Research Details expander.
Start advanced the elapsed clock from 1.7 to 3.7 seconds and changed the rendered
waveform; Pause held both the clock at 4.3 seconds and the waveform unchanged;
Resume advanced from 6.1 to 7.1 seconds; Stop / Reset returned to zero. The final
style check confirmed readable labels, unclipped metric values, and no rendered
Streamlit exceptions. EDF and headset source selections showed their unavailable
states without substituting synthetic data. No real EEG playback was exercised.

The complete Python suite passed: 298 tests. All four existing JavaScript test
files passed. `git diff --check` passed. Browser checks used a temporary database
only; the normal server uses `data/database/neurosense.db` and the project’s
`.venv/bin/python`. No EEG was downloaded and no real seizure model was trained.
