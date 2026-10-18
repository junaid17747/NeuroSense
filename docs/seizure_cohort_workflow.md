# Proposed CHB-MIT cohort: training and evaluation runbook

This workflow already exists as `seizure_research.multipatient`. It is separate from `seizure_research.training run-pilot`, which remains limited to the frozen six-patient pilot. No command below grants authorization. Do not run real training until the private reviewed inputs and applicable decisions are complete. No real EEG was downloaded or processed during this implementation.

## Frozen inputs and current limitations

| Input | Location | Content digest |
| --- | --- | --- |
| Official inventory | `reports/checkpoint_d7/official_inventory.json` | `ec686dddc0d49f5c39e1922b4b6f91e4e5572c31ffbd0b54016ef86c31267c85` |
| Proposed 45-recording selection | `reports/checkpoint_d7/selection_proposal.json` | `87defcecf646bbd876a27178dc771a036e4396edd7d010e791891bbd48affa53` |
| Proposed canonical split | `reports/final/proposed_patient_split.json` | `40ddf35ff39f34a15d7f8e36924df2ed9ae7ca06610dd03869288287c9592404` |

These are the manifests' canonical JSON content digests, not hashes of their pretty-printed files. Split version: `chbmit-multipatient-proposed-split-v1`; seed: 42; algorithm: `sha256-json-seed-patient-v1`. The proposal remains 13/4/6 canonical people and 25/8/12 recordings in train/validation/test. Unknown-age chb24 needs a documented exclusion in the current profile; that leaves at most 44 eligible recordings from 22 people, before any other exclusions. Do not silently describe the resulting model as trained on all 45.

The [split audit](../reports/final/proposed_split_audit.json) records class/age coverage and provisional montage compatibility. No adult validation or adolescent validation/test patients are present. The [faculty package](seizure_cohort_faculty_review.md) explains the scientific and governance limits. Do not change assignments based on held-out outcomes.

## Private reviewed inputs

Keep completed inputs under `~/NeuroSense-EEG-Data`, outside Git. Existing archive/temporary directories and the incomplete review packet are retained.

- `manifests/archive-approval.json`: a genuine completion of the existing blank archive approval template. Staging, processing and training are separate decisions. Bind the inventory, selection, exact normalized plan/configuration, review packet, split and run configuration; specify the approved `runs` root.
- `reports/annotation-review-packet.json`: every selected row has a genuine reviewer decision and source binding. Excluded rows need a reason and reviewer evidence. Eligible rows need reconciled events, full-duration review, exact seizure-free and excluded intervals, and quality review. A summary-zero candidate requires explicit seizure-free evidence. Resealing with `seal_review_packet` only recomputes a checksum; it does not attest review.
- `manifests/review-plan.json`: the existing `IngestionPlan` schema, restricted to exactly the eligible recordings. Include reviewed canonical identities, dataset/permission evidence, source hashes, age source, annotation provenance and protocol, all source channels with keep/exclude reasons, quality exclusions, and reviewer evidence.
- `manifests/preprocessing.json`: a mapping from each eligible source EDF path to its reviewed `PreprocessingConfig`. Exactly 18 ordered bipolar channels, 256 Hz, no resampling or rereferencing, 5-second windows/1-second steps; age groups must match the published metadata. Filter and quality settings and channel indices need explicit review.

Authorization digests use `seizure_research.splitting.digest`: normalized `IngestionPlan.model_dump(mode="json")`, normalized per-recording preprocessing mapping, the split's `split_proposal_sha256`, the sealed packet's `packet_sha256`, and `DEFAULT_RUN.model_dump(mode="json")` (or the exact supplied `--run-config`). Do not hash an abbreviated configuration instead of its validated effective settings. The default run is `chbmit-multipatient-xgb-v1`, seed 42. Approval evidence remains a human responsibility; a structurally valid JSON file is not independently authenticated approval.

The existing private annotation packet may predate explicit eligibility and interval-partition fields. A human must add the fields described by `build_review_packet`, preserving source hashes and recording coverage; the software does not auto-complete it.

## Exact future training command

Run from the repository root only after the archive has been genuinely authorized, staged and verified and the separate processing/training decisions are documented:

```bash
rtk proxy .venv/bin/python -m seizure_research.multipatient run \
  --root "$HOME/NeuroSense-EEG-Data/archive" \
  --temporary-root "$HOME/NeuroSense-EEG-Data/temporary" \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json \
  --approval-record "$HOME/NeuroSense-EEG-Data/manifests/archive-approval.json" \
  --split-proposal reports/final/proposed_patient_split.json \
  --review-plan "$HOME/NeuroSense-EEG-Data/manifests/review-plan.json" \
  --review-packet "$HOME/NeuroSense-EEG-Data/reports/annotation-review-packet.json" \
  --preprocessing "$HOME/NeuroSense-EEG-Data/manifests/preprocessing.json" \
  --output-dir "$HOME/NeuroSense-EEG-Data/runs/chbmit-multipatient-xgb-v1" \
  --confirm-authorized-training
```

For a no-training input check use the same inputs with command `check`, omitting `--output-dir` and `--confirm-authorized-training`. It validates review evidence, authorization and full archive integrity without decoding signal samples. It does not substitute for the per-recording header, sidecar and preprocessing checks during `run`. Currently it fails closed because required private inputs are missing. Missing files, unverified backgrounds, conflicting sidecars, unknown eligible ages, incompatible channels, changed input bindings or an incomplete archive prevent training.

The existing archive `plan` remains a no-download storage check; staging commands and their authorization gates are documented in [D8 storage setup](checkpoint_d8_storage_setup.md). Keep the 2 GiB retained archive cap, separate 1 GiB temporary quota and 20 GiB system free-space reserve. Runs have a separate 5 GiB cap across their approved root. The archive manager is reused unchanged.

## Processing and event evaluation

EDF samples are decoded in bounded chunks with a sample cap and filter context; features are written incrementally to per-recording Parquet shards. Train and validation feature tables are then assembled in memory; this is a bounded-source prototype, not an out-of-core XGBoost trainer. Resource feasibility must be checked against admitted recording durations. No memory claim for the full 686-recording dataset is made.

The independent seizure model uses 378 EEG features, training-only scaling and patient weighting. Fitting uses train patients; early stopping and operating-threshold selection use validation patients. Test feature extraction starts only after model/scaler/threshold hashes are frozen in `test_freeze.json`. Test evaluation rejects threshold changes. If no validation threshold meets the prespecified false-alarm/alert-occupancy constraints, training fails instead of selecting on test results.

Event policy: trailing median of 3 scores; open after 2 qualifying windows; close after 3 below the threshold minus 0.10 (floor 0.01); merge gaps of at most 5 seconds only within contiguous scored coverage. Missing/excluded windows reset state. Trigger time is the right edge of the opening window. Match candidates chronologically, one-to-one, with interval overlap and the trigger inside the half-open reference seizure interval; no onset tolerance. A duplicate unmatched candidate counts as a false alarm.

Sensitivity is detected/evaluable reference seizures; missed events are reference minus detected. Reviewed reference seizures missed because the algorithm abstained remain misses. Human-unreviewed recordings are rejected; reviewed exclusions cannot contribute scored windows or candidates. References intersecting exclusions require adjudication before evaluation. Partly seizure-overlapping windows with known reference timing are ambiguous for binary fitting and excluded from fitting; their signal scores remain available for event timing. These boundary windows are distinct from ambiguous or unreviewed annotation intervals, which must be explicitly excluded.

False alarms/hour uses the union of evaluable **non-seizure** window coverage, subtracting seizure time and omitting the first two windows of each contiguous scored segment. Overlapping windows do not multiply duration. Detection latency is trigger minus reference onset for matched events. Zero reference events, zero non-seizure duration and zero matches yield unavailable sensitivity, alarm rate and latency respectively. Filtering is retrospective: latency is recording-relative, not a measured real-time device delay.

Outputs include model/scaler, code and dependency provenance, validation curve/selected threshold, source/cohort/split/config hashes, test predictions, aggregate/per-patient metrics, per-recording and age summaries, and exploratory patient-bootstrap intervals. `complete.json` is written last and hashes all run files; interrupted bundles remain unavailable. Checksums detect changes, not fabricated human decisions.

## Dashboard and software verification

Set `NEUROSENSE_SEIZURE_RUN_DIR` to the approved complete run directory after training. Default archive and review paths already target `~/NeuroSense-EEG-Data`. The authenticated EEG Seizure Research route retains heading → waveform/controls → four event cards → compact status → collapsed Research Details. Live synthetic viewing, approved EDF playback and the unavailable headset source stay separate. Complete bundles are revalidated against current review/authorization evidence and source integrity; all bundle-file changes invalidate the cached result. Missing, changed or incomplete artifacts produce unavailable metrics. Age-filtered pooled metrics do not grant age-specific model eligibility.

Synthetic verification commands (no real EEG or network EEG transport):

```bash
rtk proxy env MPLCONFIGDIR=/tmp/neurosense-d10-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q
rtk proxy node --test tests/test_webcam_alignment.mjs tests/test_webcam_behavior.mjs tests/test_webcam_camera.mjs tests/test_webcam_rppg.mjs
rtk git diff --check
```

The Python suite uses `unittest`; pytest is not installed in the current environment. Generated EDF integration uses six fictitious patients/twelve short recordings through the multipatient entrypoint, with test-only decisions and an in-memory transport confined to generated bytes. It verifies resumable archiving, strict review/authorization gates, feature extraction, XGBoost fitting, validation-only threshold selection, delayed test extraction, bundle validation and bounded playback. The immutable 23-person proposal is independently tested against official public metadata. Synthetic scores are never reported as actual CHB-MIT performance.
