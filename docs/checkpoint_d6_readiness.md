# Checkpoint D6 — end-to-end technical readiness

Audit date: 2026-10-12. **Synthetic workflow verified; real-data execution remains
blocked.** This is an experimental, retrospective EEG seizure research prototype,
not a clinical diagnostic or emergency-alert device. No clinical EEG was
downloaded, decoded or used for training during this checkpoint. No faculty or
institutional approval was created. Nothing was committed or pushed.

## Implemented and tested

| Stage | D6 verification |
| --- | --- |
| Frozen manifest | Six patients, twelve EDFs, six summaries, six seizure sidecars and one checksum document; exact patient partitions and chb01/chb21 exclusion remain enforced. |
| Staging | A mocked transport served 25 generated files through the actual staging code. Size/hash validation, resumed staging, source corruption rejection and approval-before-transfer gates are exercised. |
| Storage | 5 GiB total root cap and 20 GiB free-space reserve, including one-byte boundary tests. Existing files count against the cap; resumed staging reserves only missing transfers. Transfer chunks and generated result JSON are checked before writing. |
| Readiness | Actual source hashing, cohort reconstruction, annotation evidence, age, montage, identity and human-decision validators ran on synthetic fixtures. Failure blocks smoke and training before waveform decode. |
| EDF and features | Eighteen channels at 256 Hz, bounded decoding, filtering, five-second windows at one-second stride, quality exclusions and 378 features per usable window. Synthetic integration uses multiple 30-second cores and 30-second halos; production defaults remain 300/60 seconds with a 2,000,000-sample ceiling. |
| Patient independence | Train chb02/chb05/chb08; validation chb09; test chb06/chb10. Scaler fitted on unambiguous training rows only. Test feature tables are loaded after model/threshold artifacts are frozen. |
| Threshold and events | Model iteration and event threshold use validation only. Held-out test event metrics include sensitivity, missed events, false events per evaluable non-seizure hour, retrospective latency and per-patient results. |
| Provenance and reload | Source cohort, source-file manifest, annotation evidence, preprocessing, split, feature manifest, approval record, model, scaler, validation selection, research-package code and dependency versions are bound. Reloaded model/scaler must reproduce synthetic test probabilities exactly. |
| Dashboard | Loads the artifacts actually emitted by training, using canonical recording IDs. Rechecks hashes, approved cohort/storage, test membership, bounds, threshold and event/metric consistency. No attention model is invoked. |
| Unavailable states | Missing data/model/approval, malformed or changed artifacts, wrong patients, unsupported ages and invented metrics remain unavailable. Synthetic waveform has no fabricated predictions. No clinical alert action exists in this lab. |

The generated fixture uses the twelve allowlisted filenames for compatibility;
its EDF headers, waveforms, annotation text, checksum document and every review
reference are explicitly TEST_ONLY. Its separate file manifest replaces all
clinical source sizes/hashes with synthetic ones. Tests simulate transport,
free-space availability and persistence detection for their disposable temporary
directories. They do **not** mock a passing readiness result for the pipeline,
feature extraction, model fitting, threshold selection or result loading. UI
tests inject the resulting readiness context because the UI normally uses the
real frozen source manifest. Synthetic scores are not evidence of EEG efficacy.

## D5 defects repaired

- The dashboard compared canonical IDs such as `chb_mit:1.0.0:chb06_01.edf`
  against source paths such as `chb06/chb06_01.edf`, rejecting genuine outputs.
  It now uses the saved cohort and split to validate canonical IDs.
- The viewer passed full-record annotations to a 20-second plot and selected
  a training recording while loading test-only predictions. It now displays
  the first 20 seconds of test recording `chb06/chb06_01.edf`, clipping overlays
  to that interval and omitting offscreen events.
- Unsupported age selections previously left the real-data action accessible.
  They now block waveform/result loading and retain unavailable metrics.
- Self-consistent JSON hashes alone did not bind results to the current
  approved cohort or validate patient membership/metrics. Run snapshots and
  strengthened provenance now provide those checks; event conversion and
  metrics are recomputed from saved scores and reviewed annotations.
- Storage approval did not name or require a persistent destination. The
  approval schema now requires `staging_root`, `storage_persistence_confirmed`
  and `storage_policy_ref` at staging/readiness gates. Temporary roots
  `/tmp`, `/var/tmp`, `/dev/shm` and `/run`, including symlink routes, are rejected.
  The persistence field is a custodian attestation, not automatic proof of
  mount durability, backup or institutional approval.
- D3/D4/download instructions placed `faculty-approval.json` above the dataset
  root. All current commands now put it directly **inside the dataset root**,
  matching D5 and the dashboard.
- Resume space planning double-counted existing files; JSON outputs lacked
  exact pre-write byte checks; malformed results could escape as type errors.
  These are corrected. Missing approval lists also keep readiness false.
- The configurable label-occupancy value was ignored by feature extraction.
  The run configuration now explicitly fixes it to the implemented 0.5 policy.

## Ready but awaiting real data

The software can stage the authorized allowlist, audit reviewed metadata,
perform a bounded smoke test, generate features, fit the independent seizure
baseline and load its results into the authenticated lab. Real EDF formats,
montages, duration/summary consistency, usable-window counts, class balance,
operating thresholds, runtime and storage consumption have **not** been
validated against CHB-MIT recordings. Any of these may block the first real run;
synthetic success is not a guarantee that real input passes review or training.

The authoritative file manifest remains unchanged during D6:

`docs/chbmit_pilot_file_manifest.json`

SHA-256: `dc81f5f555e098806ebe977a6e553eee314e7b66a897a5620f037d44e2b79bcd`.

The staged sources total **1,004,239,275 bytes (0.935 GiB)**, including
**1,004,143,360 EDF bytes**. Twenty-four source digests are pinned and compared
with the official checksum document's entries. The checksum document itself
is size-checked and strictly parsed; its complete body digest is not pinned
in the frozen manifest. No remote source was re-fetched for this audit.

## Awaiting human approval — exact current blockers

1. **Persistent storage has not been selected or approved.** At audit,
   `/mnt/approved/neurosense/chbmit-pilot` did not exist. The older
   `/tmp/neurosense-chbmit-pilot` placeholder was empty and is unsuitable.
   A custodian must provide the real location and its access, retention,
   backup and deletion evidence; the template still records no approval.
2. **Dataset-use review, institutional determination and exact faculty cohort
   review remain undocumented.** General faculty encouragement is insufficient
   evidence for these separate fields.
3. **Staging authorization remains missing**, including the approver/date,
   evidence, frozen manifest hash and named persistent destination.
4. **The 25 clinical source files are absent.** No authentic cohort manifest,
   reviewed preprocessing configuration or annotation-evidence manifest exists
   in a configured approved root. Genuine age, identity, quality and complete
   annotation review remain required, including background coverage and
   sidecar reconciliation. Staging permission does not authorize EEG decoding
   for human review; record any necessary review access separately.
5. **Separate processing/training authorization remains missing**, bound to
   the reviewed cohort. The processing flag alone never satisfies this gate.
6. **No seizure-specific model or result bundle trained on real EEG exists.**
   `NEUROSENSE_CHBMIT_ROOT` and `NEUROSENSE_SEIZURE_RUN_DIR` were unset at audit.
   A passing synthetic suite supplies neither data nor approval.

Storage constants are **5,368,709,120 bytes** and **21,474,836,480 bytes**.
The machine's read-only space snapshot is recorded in
[`staging_plan.json`](../reports/checkpoint_d6/staging_plan.json); it describes
the filesystem anchor for an example path, not an approved provisioned volume.
Rerun the plan on the actual storage device before staging. Checks are point-in-time
and do not reserve space against unrelated concurrent writers; serialize pilot
writers and keep the same reserve throughout the run.

## Not yet implemented

- Automated interpretation of seizure sidecars and automatic human evidence
  creation: the supported path is summary parsing plus genuine human
  reconciliation. Evidence creation must not be inferred from test fixtures.
- A generic post-training seizure inference CLI or live EEG inference service;
  D6 verifies retrospective saved test results and model/scaler reload.
- Seizure models for other age groups, external/prospective validation,
  clinical calibration or clinical performance claims. The pilot has one
  validation patient and two test patients and cannot establish broad validity.
- Interactive selection of arbitrary real recording/time ranges in the lab;
  the current UI uses one bounded test segment. Events outside its first
  20 seconds remain offscreen; full test metrics still cover the test inventory.
- Clinical diagnostic or emergency-alert functionality is intentionally absent.

## Exact future commands

Run from `/home/yahyasiddiqui/Neurosense Mini Project/NeuroSense`, using the
existing `.venv`. The `/mnt/approved/...` path below is an **example requiring
real custodian approval**, not an instruction to create or authorize it now.
Replace that prefix consistently if a different persistent location is chosen.
All download, ingestion, smoke and training commands below are future commands;
only the two read-only plans were run during D6.

Read-only plans (no network, decoding or directory creation):

```bash
rtk proxy .venv/bin/python -m seizure_research.staging plan \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --file-manifest docs/chbmit_pilot_file_manifest.json
```

```bash
rtk proxy .venv/bin/python -m seizure_research.training plan-pilot \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --file-manifest docs/chbmit_pilot_file_manifest.json
```

After the actual staging decisions and persistent-location evidence are recorded
in `<root>/faculty-approval.json`, stage exactly the allowlist:

```bash
rtk proxy .venv/bin/python -m seizure_research.staging stage \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --file-manifest docs/chbmit_pilot_file_manifest.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --confirm-authorized-staging
```

After authorized metadata/annotation/quality review, a reviewer supplies the
`IngestionPlan` JSON described by `seizure_research/adapters/base.py`. This
metadata ingestion reads headers and hashes; it does not decode waveforms:

```bash
rtk proxy .venv/bin/python -m seizure_research ingest \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --plan /mnt/approved/neurosense/chbmit-pilot/review-plan.json \
  --output-dir /mnt/approved/neurosense/chbmit-pilot/reviewed-ingestion
```

Ingestion must report `cohort_approved: true`. Its output uses an underscore;
the reviewed runtime input uses a hyphen. Install it at the explicit input path
and record its printed semantic manifest hash in the genuine approval record:

```bash
rtk proxy cp --no-clobber \
  /mnt/approved/neurosense/chbmit-pilot/reviewed-ingestion/cohort_manifest.json \
  /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json
```

The reviewer also supplies `<root>/preprocessing.json` (`PreprocessingConfig`)
and `<root>/annotation-evidence.json` (`AnnotationEvidenceManifest`). They
must reflect actual reviewed channels and sources. After the separate
processing authorization is genuinely recorded, audit readiness:

```bash
rtk proxy .venv/bin/python -m seizure_research.readiness check \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --file-manifest docs/chbmit_pilot_file_manifest.json
```

Require exit code 0, `ready_for_real_data_processing: true`, and no blockers.
Then run the authorized five-second, eighteen-channel smoke test:

```bash
rtk proxy .venv/bin/python -m seizure_research.smoke \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --recording chb02/chb02_16.edf \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --file-manifest docs/chbmit_pilot_file_manifest.json \
  --confirm-authorized-processing
```

After smoke review, run the authorized experiment into a new directory:

```bash
rtk proxy .venv/bin/python -m seizure_research.training run-pilot \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --file-manifest docs/chbmit_pilot_file_manifest.json \
  --output-dir /mnt/approved/neurosense/chbmit-pilot/runs/chbmit-child-xgb-pilot-v1 \
  --approve-real-data-processing
```

Configure the authenticated dashboard to load that result bundle:

```bash
rtk proxy env \
  NEUROSENSE_CHBMIT_ROOT=/mnt/approved/neurosense/chbmit-pilot \
  NEUROSENSE_SEIZURE_RUN_DIR=/mnt/approved/neurosense/chbmit-pilot/runs/chbmit-child-xgb-pilot-v1 \
  .venv/bin/streamlit run dashboard/app.py
```

Log in, open Seizure Research Lab, select Child, and explicitly check approvals
and load the bounded segment. This route uses the separate seizure artifacts;
it does not replace or retrain the attention model.

## What the first successful real-data run produces

Staging produces the verified 25-file source set. Smoke returns source hash,
channel count, duration and sample count (1,280 samples/channel for the frozen
five-second/256-Hz profile), without saving waveform arrays or training a model.
A successful training run produces:

- Twelve compressed feature shards under `features/{train,validation,test}/`,
  with window IDs, canonical patient/recording IDs, labels and feature values.
  The feature manifest records accepted, ambiguous and excluded-window counts.
- `cohort_manifest.json`, `preprocessing.json`, `split_manifest.json` and
  `feature_manifest.json`, tying the experiment to reviewed input metadata.
- `model/xgboost.json`, `model/scaler.npz`, `model/validation_selection.json`
  and `model/provenance.json`: independent seizure baseline, training-only
  normalization, threshold curve/selected iteration and reproducibility data.
- `test_predictions.json` and `test_evaluation.json`: test-window scores,
  experimental candidate events and event/per-patient metrics with hashes.

There is no guaranteed sensitivity, candidate event count or usable operating
point. A model predicting no events may yield zero sensitivity. Missing
classes, failed quality checks or no feasible validation threshold stop the
run. Partial outputs are not a successful experiment and must not be reused
as a new output directory; the loader withholds incomplete bundles. The
dashboard keeps undefined metrics unavailable and always records
`clinical_validation: false`. Artifacts include research metadata and must
remain within approved storage.

## Verification evidence and uncommitted work

Final verification results are recorded in `reports/checkpoint_d6/`:

- `python_baseline.txt`: 258 pre-change Python tests passed.
- `python_tests.txt`: **272 tests passed, zero failures/skips**, 71.117 seconds,
  through `.venv/bin/python`; includes 14 new integration/boundary checks.
- `integration_tests.txt`: initial focused synthetic integration run, 11 passed;
  the three subsequently added checks also passed in the final complete suite.
- `javascript_tests.txt`: all four webcam test files passed.
- `current_readiness.json`: read-only report for the currently absent example root.
- `preservation.json`: hashes checking existing attention/model, webcam/rPPG,
  authentication/backend and dashboard-route files were unchanged during D6
  verification.
- `git_diff_check.txt` and `uncommitted_changes.txt`: final whitespace check
  and complete working-tree inventory, including work already present at start.

Reproduce verification without clinical data:

```bash
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
rtk proxy node --test tests/test_webcam_*.mjs
rtk git diff --check
rtk git status --short --untracked-files=all
```

D6 changed only the seizure research pipeline, seizure lab, synthetic tests and
review documentation/evidence. Existing uncommitted D1–D5 work was preserved;
the final inventory distinguishes files touched by D6 from pre-existing work.
