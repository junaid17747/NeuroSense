# Checkpoint D5: Seizure Research Lab and approved-run workflow

D6 audit, fixes, current blockers and tested commands: [Checkpoint D6](checkpoint_d6_readiness.md).

## Scope and current state

The authenticated Streamlit workspace now has a separate **Seizure Research
Lab** section. It is research-only, has no clinical alerts, and displays no
real metrics when the corresponding verified artifacts are absent. The
waveform shown by default is a deterministic, bounded synthetic signal. The
frozen cohort is six patients and 12 recordings: train `chb02/chb05/chb08`,
validation `chb09`, and test `chb06/chb10`; `chb01`/`chb21` remain recognized
as one patient and are excluded.

The pilot is child-age-group only. That identifies the planned data target; it
does not imply that any individual recording is eligible or that the model is
validated for children. Other age groups are explicitly unsupported by this
frozen pilot. No approved real recordings, model, predictions, or metrics are
currently available. No real EEG was accessed, processed, or trained during
Checkpoint D5.

The optional real waveform action is disabled until an external staged root
is configured with `NEUROSENSE_CHBMIT_ROOT`. On explicit user action it runs
the read-only D4 readiness audit; waveform decoding then requires the audit
to pass and a complete processing approval in the human approval record. It
reads at most 20 seconds from one allowlisted recording and selected channels.
Predicted-event overlays and metrics require the generated test prediction,
evaluation, and model provenance files, valid matching hashes, a passing
readiness report, and processing approval. Reviewed annotations and model
candidate events have separate plot colors.

## Training boundary and reproducibility

`python -m seizure_research.training plan-pilot` reads only the frozen file
manifest and prints the planned stages; it does not read or create files in
the patient-data root. The real `run-pilot` command now requires an explicit
processing flag, a D4 approval record, all cohort/preprocessing/annotation
manifests, and a clean `check_readiness` result. The readiness gate checks the
25-file staged set and hashes before any EDF decode. It also checks the
reviewed annotation coverage, quality metadata, and institutional, faculty,
staging, and processing decisions.

Patient splits are checked again at feature loading, training, threshold
selection, and test scoring. The normalizer is fit on train rows. Model
iteration and event threshold selection use validation only; test patients
are loaded after those are frozen. Model provenance binds source manifest,
annotation evidence, split, feature shards, preprocessing, code, and runtime.
The run writes `features/{train,validation,test}/`, `feature_manifest.json`,
`split_manifest.json`, `cohort_manifest.json`, `preprocessing.json`, `model/`, `test_predictions.json`, and
`test_evaluation.json` below a new output directory inside the approved
external dataset root. The event prediction artifact identifies the model,
split, validation-selected threshold, and test partition. Clinical validation
is always recorded as false.

All generated files must remain outside the Git worktree. The staging and
training code reject worktree destinations and enforce the 5 GiB pilot cap
and 20 GiB minimum free-space reserve.

## Dry-run command

This is safe before approval and does not read EEG:

```bash
python -m seizure_research.training plan-pilot \
  --root /mnt/approved/neurosense/chbmit-pilot
```

## Workflow after genuine approvals

Staging requires faculty cohort review, license/data-use review, institutional
ethics determination and explicit staging authorization against the frozen
manifest and named persistent storage location. Before processing, complete
authorized human annotation/quality review and obtain separate processing and
training approval. See D6 for the complete sequence and storage fields.

First stage only the allowlisted files (5 GiB cap, 20 GiB reserve):

```bash
python -m seizure_research.staging stage \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --confirm-authorized-staging
```

Then run the read-only readiness audit after reviewed cohort, preprocessing,
annotation-evidence, and approval records have been placed in that external
root:

```bash
python -m seizure_research.readiness check \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json
```

After readiness passes and the separate processing authorization is recorded,
decode the bounded smoke window:

```bash
python -m seizure_research.smoke \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --recording chb02/chb02_16.edf \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --file-manifest docs/chbmit_pilot_file_manifest.json \
  --confirm-authorized-processing
```

Only after smoke review should the approved pilot run. Use a new output
directory under the external root:

```bash
python -m seizure_research.training run-pilot \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --file-manifest docs/chbmit_pilot_file_manifest.json \
  --output-dir /mnt/approved/neurosense/chbmit-pilot/runs/chbmit-child-xgb-pilot-v1 \
  --approve-real-data-processing
```

The output is a retrospective research result only. It is not clinical
validation, a diagnostic device, or evidence for care decisions.
