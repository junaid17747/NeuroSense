# NeuroSense V4 Phase 3 — Offline waveform and preprocessing foundation

## Scope and status

Checkpoint C adds bounded sample decoding and reproducible preprocessing for
locally held, already reviewed EDF recordings. It does not download data, train
a seizure model, generate predictions, create an application endpoint, or add a
dashboard. Passing synthetic tests establishes software behavior only; no
clinical recording has been decoded or validated in this project.

The existing EEG V1/V2 attention path remains separate. Its simulated EEG and
attention models are not used here. No seizure model is registered or enabled.

## Architecture

`seizure_research.edf.inspect_edf()` validates a bounded header, signal
calibration, record count, record duration, payload length, per-channel samples
per record, and an EDF record size limit. It streams a SHA-256 fingerprint and
does not return EDF patient/recording header fields. `seizure_research.waveform.read_edf()`
accepts a dataset root plus a normalized relative path, explicit source-channel
indices, a half-open time range, and a sample limit. It binds its output to the
same source hash used by the reviewed cohort manifest.

`seizure_research.preprocessing` requires both a matching recording-age
configuration and an approved cohort manifest. The output keeps channel order,
timestamps, normalized units, quality windows, aligned reference labels,
content hashes, dataset and annotation provenance, and the manifest/config
versions. `seizure_research.quality` produces channel/window assessments and
exclusion intervals. `seizure_research.splitting.assign_windows_to_splits()`
inherits each window's partition from its recording and canonical patient.
Processing validates any supplied split against the complete reviewed cohort
with `validate_split_manifest()`. Standalone window assignment callers must
perform that same full-cohort validation first. Serialized window manifests
also reject patient/recording conflicts and duplicate window IDs.

No new dependency was added. NumPy and SciPy were already pinned in
`requirements.txt`; the application, SQLite schema, FastAPI routes, and
Streamlit navigation were not changed.

## EDF decoder contract

The current decoder supports ordinary continuous, signed 16-bit EDF with a
positive fixed record count, valid calibration ranges, and exactly sized data.
The physical calibration is:

```text
physical = physical_min
         + (digital - digital_min)
           * (physical_max - physical_min) / (digital_max - digital_min)
```

Supported EDF electrical dimensions are ASCII `V`, `mV`, and `uV` (case
insensitive at the decoder boundary); output samples are in microvolts.
Non-ASCII headers, including literal Unicode micro symbols, are rejected by
the strict legacy header reader even though the internal unit helper recognizes
those spellings. Unknown units fail with an explicit
error. Selected channel indices are preserved as supplied, including duplicate
labels; label-only lookup returns every match, so callers must resolve
ambiguity explicitly. Non-EEG signals such as ECG, EMG, VNS, and auxiliary
channels are not inferred or silently mapped: the caller explicitly selects
the EEG source indices and maps each selected signal to an output channel.
Duplicate source labels require a review reference when preprocessing maps
them to distinct montage channels.

Each channel retains its own sampling rate and source metadata. The decoder
reads only selected channel blocks within `[start_seconds, end_seconds)`. The
default output limit is 2,000,000 samples across selected channels; callers may
raise it only up to the hard 10,000,000-sample ceiling. Header/hash inspection
still scans the complete file with bounded memory. EDF records above 61,440
bytes are rejected by this project resource policy; the EDF specification
recommends this size rather than declaring every larger record invalid. EDF+, EDF+D, BDF, unknown-duration files, EDF annotations,
unknown units, and files with inconsistent or truncated payloads are not
supported. EDF+ can encode event text and discontinuous data records, so it
must not be treated as ordinary continuous EDF; see the [EDF specification](https://www.edfplus.info/specs/edf.html)
and [EDF+ specification](https://www.edfplus.info/specs/edfplus.html).

Relative paths reject traversal, absolute paths, backslashes, and symlinks that
resolve outside the supplied dataset root. The waveform reader additionally
rejects all symlink components, repeated separators, dot segments, and control
characters. File identity, size and modification state are checked before and
after inspection and decoding, including the interval between those operations.
Decoded arrays are read-only; this is accidental-mutation protection, not an
authenticity boundary against code executing in the same Python process. The decoder returns no raw EDF
identity/date header fields and emits only stable rejection codes.

## Population-specific preprocessing

Every configuration names one of the six provisional software age groups and
must match the recording's age at recording. `make_age_config()` requires an
explicit channel map, output rate, and research review reference. The values
below are deliberately provisional engineering examples. They are not
clinical recommendations, validated protocols, or evidence that a model is
eligible for the population.

| Software population | High-pass | Low-pass | Profile ID prefix |
|---|---:|---:|---|
| Neonatal | 0.5 Hz | 30 Hz | `neonatal-research-v1` |
| Infant | 0.5 Hz | 35 Hz | `infant-research-v1` |
| Child | 0.5 Hz | 40 Hz | `child-research-v1` |
| Adolescent | 1.0 Hz | 40 Hz | `adolescent-research-v1` |
| Adult | 1.0 Hz | 40 Hz | `adult-research-v1` |
| Older adult | 0.5 Hz | 35 Hz | `older_adult-research-v1` |

All defaults use a fourth-order Butterworth SOS band-pass and an optional 50 Hz
Q=30 notch. The line frequency must be below the target Nyquist rate. Cutoffs
must satisfy `0 < high-pass < low-pass < Nyquist`. Configurations reject an
age mismatch, an unreviewed montage/channel order, an unexpected source label,
an unreviewed duplicate source label, and source rates that do not match their
explicit channel map. Heterogeneous channel rates are represented in montage order in the cohort
manifest and must match the reviewed per-channel rates. The nominal rate
remains the summary rate; it is not substituted for a different per-channel
rate. Existing single-rate model evidence rejects heterogeneous source rates.
The CHB-MIT adapter version is now `2`: rebuild and revalidate older saved
cohort and split manifests rather than editing their hashes in place.

When explicitly enabled, rate conversion uses SciPy polyphase resampling with
a rational ratio, followed by linear interpolation onto a shared target-rate
time grid. The grid is the intersection of channel coverage and remains within
the requested read interval. It does not invent data outside that intersection.
Average reference requires at least two mapped channels in an explicitly
reviewed referential or average-reference montage; bipolar derivations are
rejected. Resampling cannot recreate bandwidth absent from a lower-rate source;
review the effective bandwidth of every channel before feature extraction. No learned normalization or cohort-fitted transform is
performed in Checkpoint C, so no test-set statistics are fitted or reused.

## Filter and boundary behavior

Filtering uses second-order sections with forward/backward `sosfiltfilt`, so
the operation is zero-phase and non-causal. See the upstream
[`sosfiltfilt` contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
SOS poles must be strictly inside the unit circle. Odd padding uses
`3 * (2 * section_count + 1)` samples. The excluded edge guard is the larger
of that padding, a pole-decay estimate `ceil(log(tolerance)/log(max_pole_radius))`,
and the resampling edge guard when applicable. `transient_tolerance` defaults
to `0.001`. This is a conservative engineering heuristic, not proof that all
transients are removed; review response plots and independent signals.

Too-short requested intervals fail explicitly. Short residual segments after
quality exclusions are excluded with an explicit report reason. Excluded and
edge samples remain NaN. Native digital and physical samples are quality-checked
**before** resampling, so digital rail saturation cannot disappear through unit
conversion or interpolation. Both resampling and filtering operate separately
inside each usable segment. Neither operation spans known exclusions or
independent EDF files. The polyphase conversion follows SciPy
[`resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html);
its numerator and denominator are capped at 1,000. Target sample counts are
checked against a 10,000,000-value aggregate limit before allocation. Ordinary EDF is
continuous by format; files that encode discontinuity through EDF+ are
rejected instead of pretending that their timebase is continuous.

Resampling and filtering are deterministic for the same input bytes,
configuration, SciPy/NumPy runtime, and code version. The report records hashes
and versions but deliberately omits a wall-clock processing timestamp so its
content remains reproducible. The processed signal digest includes the sample
time grid and output array bytes.

## Signal-quality checks

The default configurable quality window is 5 seconds with a 5-second step.
Current initial thresholds are engineering safeguards only:

- Flatline: standard deviation below 0.1 microvolts in a window.
- Extreme amplitude: absolute value over 1,000 microvolts.
- Abrupt discontinuity: adjacent sample jump over 1,000 microvolts.
- Sampling inconsistency: deviation from the nominal sample period beyond
  the configured relative tolerance (default `1e-6`, with a `1e-8` numerical
  allowance), or uncovered window edges exceeding one sample period. The
  processing boundary rejects irregular decoder timebases before interpolation.
- Digital saturation: rail samples above 1% of a window.
- Usable coverage: at least 50% of the requested time after the union of
  exclusions; overlapping windows never double-count duration. Partial trailing
  quality windows are explicitly excluded. After filter edge removal, the same
  minimum is checked again; insufficient processed coverage masks all output.

The report identifies missing, duplicate, and unexpected labels; invalid
timestamps/rates; non-finite and corrupt digital samples; flatlines; extremes;
rail saturation; discontinuities; per-window reasons; usable fraction; and
merged excluded intervals. Global metadata failures exclude the complete
interval; malformed digital array shapes cannot bypass saturation checks.
The raw quality status and final processing status are separate: acceptable
raw data can still become unusable after edge removal. Quality and segmentation
window counts are limited to 100,000 per call. These are bounded-memory
operations, not a constant-cost guarantee for large recordings.
An unusable window is not a seizure-negative
window. Poor quality can make processing unavailable, but never demonstrates
seizure absence. Thresholds must be reviewed against the intended dataset,
device, montage, and age group before any model-development use.

## Annotation alignment policy

Reference annotations and windows use half-open time intervals `[start, end)`
relative to the EDF recording start. A window is:

- **Positive** if it overlaps one or more reviewed seizure intervals.
- **Negative** only if annotation coverage is explicitly complete and it
  overlaps no seizure interval.
- **Ambiguous** if it has no known seizure overlap but coverage is partial or
  unknown. A caller may downgrade coverage but cannot upgrade unknown or partial
  coverage to complete with an override.
- **Excluded** if it overlaps a recording or signal-quality exclusion; this
  takes priority over other labels.

Onset exactly at a window's exclusive end does not overlap that window; an
event offset exactly at the next window's start does not overlap the next
window. Multiple events are retained by event ID. Overlapping positive windows
may refer to the same event and must not be counted as separate seizure events.
Only complete windows are segmented; partial trailing time remains visible in
the processing coverage interval but is not converted into a full window.

## Leakage and provenance safeguards

The existing split manifest groups recording aliases/sessions by reviewed
`patient_group_id`, rejects duplicate recording/source hashes, and is
deterministic. `assign_windows_to_splits()` accepts overlapping windows but
assigns each from the recording's single split; an unknown recording or a
patient identity mismatch is rejected. Callers must validate the full split
manifest against the complete cohort before deriving window assignments.

Each processing report includes the approved cohort-manifest hash, dataset and
release, recording ID, signal and annotation hashes, annotation provenance,
the full effective configuration and its hash (including any threshold
override), channel map, processing/quality/alignment versions, Python/NumPy/SciPy
versions, original event intervals and review references, window labels,
exclusions, output hash, and optional split-manifest hash. `coverage_fraction`
describes the decoded time interval; `usable_fraction` describes retained signal
within it. `coverage_complete=true` does not mean all samples are usable. Reports contain no EDF patient name, birth date, or canonical patient
group ID. Use `process_approved_recording()` for saved manifests: it automatically runs
`validate_manifest()` against the local source files and annotations before
decoding. The lower-level `preprocess_recording()` accepts trusted in-memory
decoder output and checks manifest/configuration bindings, but cannot prove
the origin of arbitrary arrays constructed by a caller. Store cohort,
processing, and audit artifacts outside Git under the approved research data
controls; hashes do not prove source authenticity or permission.

## Synthetic verification and manual validation

`tests/test_seizure_processing.py` and `tests/test_seizure_processing_safety.py`
create small EDF files and fictitious event
labels in temporary directories. It checks sample calibration, microvolt
conversion, time and channel selection, per-channel sample rates, mixed-rate
resampling, bounded reads, unsupported units, path traversal, malformed or
oversized EDF, duplicate channels, filter phase/reference behavior, quality
failure reasons, deterministic processing hashes, annotation boundaries,
unknown coverage, and patient-disjoint window assignments. Regression cases
cover source replacement during inspection, native digital saturation, NaN/Inf,
excluded-interval isolation under resampling, partial quality tails, allocation
limits, annotation-file changes, forged split assignments, the known CHB-MIT
alias constraint, and stateless preprocessing across successive recordings. No clinical files
or identifiers are used. Passing these tests does not validate the decoder
against a clinical dataset or establish clinical performance.

Before any real-data use, a qualified data custodian and institutional reviewer
must confirm data-use permission, privacy/ethics requirements, and the approved
cohort manifest. A technical reviewer should then compare metadata and bounded
samples with an independent EDF reader for authorized files; inspect channel
labels, calibration, units, timestamps, rates, filter responses, edge masking,
quality flags, and annotations; and preserve the validation record outside
Git. A clinical EEG specialist must review the population-specific protocol,
especially for neonatal recordings. Do not train on a dataset until source
permissions, annotation integrity, identity mapping, patient-independent
splits, and preprocessing/quality decisions have passed review.

## Known limitations and next prerequisites

- No approved real EEG cohort is available or included in the repository.
- The initial CHB-MIT adapter still reads summary-text annotations only; it
  does not parse or reconcile separate `.seizure` files.
- The waveform path is a narrow legacy EDF reader, not a general EDF/EDF+/BDF
  library. BDF, EDF+, EDF+D, and EDF annotations are rejected.
- The current reader materializes the selected bounded interval in memory;
  long recordings must be processed in explicitly bounded chunks. Filtering
  across chunk boundaries requires overlap/edge policy review before use.
- Quality thresholds and age-specific profiles are provisional and have no
  clinical validation. No neonatal interpretation or live monitoring exists.
- No seizure model, training workflow, evaluation results, API endpoint, or
  Seizure Research Lab dashboard exists yet.

Checkpoint D remains blocked on authorized, reviewed real EEG data and
completion of the annotation/identity/quality review. Do not use synthetic
fixtures as substitutes for a research cohort.


## Configuration schema and offline handoff

`PreprocessingConfig.model_json_schema()` supplies the complete schema. Required
fields are `config_id`, `age_group`, `research_review_ref`, an ordered `channels`
map (`source_index`, `source_label`, `target_label`, `expected_source_rate_hz`),
`target_sampling_rate_hz`, `highpass_hz`, and `lowpass_hz`. Optional fields select
resampling, line-noise frequency, filter order (2–8), reference mode, window and
step seconds, transient tolerance, and the nested `QualityThresholds`.
A duplicate source label additionally needs `duplicate_label_review_ref`.
No configuration default is clinical validation evidence.

Use already authorized files and externally maintained review/configuration
JSON. This Python example neither downloads data nor writes artifacts:

```python
from pathlib import Path
from seizure_research.ingestion import CohortManifest
from seizure_research.preprocessing import PreprocessingConfig, process_approved_recording
from seizure_research.splitting import SplitManifest

manifest = CohortManifest.model_validate_json(Path("/authorized/cohort_manifest.json").read_text())
config = PreprocessingConfig.model_validate_json(Path("/authorized/preprocessing.json").read_text())
split = SplitManifest.model_validate_json(Path("/authorized/patient_split.json").read_text())
entry = manifest.records[0]  # select explicitly for the real study
processed = process_approved_recording(
    "/authorized/chbmit", manifest, entry.recording.recording_id, config,
    start_seconds=0.0, end_seconds=min(60.0, entry.recording.duration_seconds),
    split_manifest=split,
)
# Review processed.report.status, exclusions and window labels before use.
# Retain source events once; overlapping positive windows are not event counts.
```

The selected config must match the selected record's population and approved
channel selection. Split-less processing is allowed for bounded exploratory
inspection, explicitly recorded as `split_manifest_sha256=null`; it is not a
training/evaluation-ready output. No fitted normalization, feature extractor,
model selection or threshold tuning is implemented. A future normalizer must
fit training patients only and persist that partition provenance. The present
stateless-transform regression verifies that processing a different recording
cannot change another recording's output.

## Remaining checkpoints: prioritized readiness

| Priority/checkpoint | Existing foundation | Required next work / completion evidence |
| --- | --- | --- |
| 1 — D: experimental model | Approved-manifest gate, bounded waveforms, configs, quality exclusions, patient splits; **not ready to train** | Obtain an authorized real cohort; review identity aliases, ages, device/montage and expert annotation completeness; independently verify decoding and exclusions. Freeze splits and preprocessing; implement EEG features and an interpretable baseline, training-only normalization, imbalance strategy and validation-only threshold/event rules. Record experiment/model hashes. Keep unsupported ages unavailable. |
| 2 — E: scientific evaluation | Decision-time event-matching contracts, false alarms/hour, latency and patient/age aggregation | Produce genuine held-out predictions after development is frozen. Prespecify processing exclusions and adjudicate reference events touching them; include seizure-free time. Add quality-stratified error analysis, justified patient-level uncertainty and external-validation protocol. Zero-phase preprocessing cannot support prospective alarm-latency claims. |
| 3 — F: Seizure Research Lab | Existing authenticated Streamlit shell; **no research page** | Add authorized dataset provenance, waveform/reference viewer, quality/coverage, model eligibility, actual predictions and reproducibility panels only after pipeline verification. Clearly distinguish real approved EEG, synthetic demonstrations, references, predictions and unavailable states. No fake metrics or live alerts. |
| 4 — G: integration | Offline file-to-processed-recording helper and regression tests | Integrate frozen model/event processing/evaluation with the dashboard; test user scoping, invalid inputs, privacy, resource limits and browser behavior. Benchmark authorized large files and agree on chunk overlap/edge policy. Preserve separate webcam/rPPG behavior; no EEG–webcam fusion claim. |
| 5 — H: university completion | Phase 1–3 methodology and provenance documentation | Prepare architecture diagram, problem/objectives, literature outline, dataset permissions, methodology, actual results, limitations, reproducible installation/demo, screenshots of real functionality, presentation and future work. Results sections remain pending until experiments exist. |

Neither processing reports nor the existing evaluator automatically resolve
new quality exclusions that intersect seizure events. A reviewed evaluation
domain and compatible processed acquisition profile are prerequisites to D/E;
never silently drop these events or score an unavailable interval as negative.

## Outstanding limitations by severity

- **High, readiness blocker:** no approved real EEG cohort or independent
  clinical waveform/annotation validation; no population profile is validated.
- **High, scientific protocol:** review filter/transient/quality settings,
  montage eligibility, excluded reference events and chunk boundaries before
  training. Acausal filtering supports retrospective research only.
- **Medium, supported scope:** EDF+/BDF/discontinuous and unknown-length formats
  are rejected; summary-text CHB-MIT annotations are not reconciled with other
  formats. Upsampling does not increase source bandwidth.
- **Medium, operational:** full-file checksums cost I/O even for bounded reads;
  real-data memory/time profiling and independent-reader comparison remain.
- **Medium, provenance boundary:** review references are attestations, not
  signatures; hashes do not prove identity, permission or clinical validity.
  Python callers must not forge decoder arrays or reuse fabricated test reviews.
- **Future work:** model training, scientific results, research dashboard and
  live-camera rPPG validation are absent. This checkpoint adds no clinical use.

No unresolved failures remain in the tested synthetic processing path. The
limitations above are not evidence of clinical performance or readiness.

## Verification record — 2026-10-11

Runtime: Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, Pydantic 2.13.5,
Node 22.22.0. No dependency changes or clinical downloads were needed.

| Check | Actual result |
| --- | --- |
| Complete Python unittest discovery | 220 passed, 0 failed, 0 skipped; 42.446 seconds |
| Seizure research subset | 133 passed: 48 foundation, 41 ingestion, 44 waveform/processing/regression cases |
| Webcam alignment | 22 passed |
| Webcam behavior | 16 passed |
| Webcam camera lifecycle | 28 passed |
| Webcam rPPG | 8 passed |
| Git whitespace check | Passed |

The direct JavaScript runs report **74 tests**. Node's combined file runner
reported five successful files including the helper module, so direct suite
counts above are used instead of treating file counts as test counts. Existing
Streamlit bare-mode `ScriptRunContext` warnings appeared during passing Python
tests. No physical camera validation or clinical experiment was performed.

Reproduce from `NeuroSense/`:

```sh
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
rtk proxy node tests/test_webcam_alignment.mjs
rtk proxy node tests/test_webcam_behavior.mjs
rtk proxy node tests/test_webcam_camera.mjs
rtk proxy node tests/test_webcam_rppg.mjs
rtk git diff --check
```

The complete Python run also exercises authentication, user-scoped SQLite,
backend contracts, Streamlit navigation and existing EEG V2 artifacts. Existing
EEG V1 has no populated dedicated test suite; its code/model files are unchanged.
All clinical examples and test approvals remain fictitious.

Files added: `seizure_research/waveform.py`, `preprocessing.py`, `quality.py`,
`tests/test_seizure_processing.py`, `tests/test_seizure_processing_safety.py`,
and this document. Files modified: `seizure_research/edf.py`, `schemas.py`,
`adapters/base.py`, `adapters/chb_mit.py`, `splitting.py`, `eligibility.py`, and
`docs/seizure_research_phase2.md`. No webcam/rPPG, authentication, backend,
dashboard, attention-model, dependency or stored-model files were changed.
Changes remain unstaged and uncommitted; no push was performed.
