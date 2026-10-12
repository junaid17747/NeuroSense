# Checkpoint D1: seizure pilot pipeline

This is an offline research implementation for a proposed CHB-MIT pilot. The
code does not download recordings, and the synthetic tests do not establish
clinical performance. A model produced by this pipeline would be experimental
and would require separate clinical validation before any clinical use.

## Frozen planning cohort

The pilot gate accepts exactly these twelve EDFs and patient partitions:

| Partition | Patient | Proposed recordings |
| --- | --- | --- |
| Train | chb02 | `chb02/chb02_01.edf`, `chb02/chb02_16.edf` |
| Train | chb05 | `chb05/chb05_01.edf`, `chb05/chb05_06.edf` |
| Train | chb08 | `chb08/chb08_02.edf`, `chb08/chb08_03.edf` |
| Validation | chb09 | `chb09/chb09_01.edf`, `chb09/chb09_06.edf` |
| Test | chb06 | `chb06/chb06_01.edf`, `chb06/chb06_02.edf` |
| Test | chb10 | `chb10/chb10_01.edf`, `chb10/chb10_12.edf` |

The implementation also freezes which six recordings are expected to contain
seizures. At cohort validation it rejects changed event presence. Whole patient
groups are assigned before feature generation. `chb01` and `chb21` remain a
single canonical patient in the shared CHB-MIT identity rules; neither is in
this child-only pilot, and the pilot rejects either alias if included.

## Annotation and window policy

The CHB-MIT summary annotations are the structured reference used by the
existing adapter. For recordings with events, the annotation evidence must
identify and hash the exact `.edf.seizures` sidecar, attest that a reviewer
reconciled it against the summary, and supply reviewed event intervals matching
the summary within 0.5 seconds. The pipeline does not guess or automatically
parse the sidecar text format. A recording with no summary events requires a
review attestation that the *entire* recording was checked for seizures, with
the reviewed coverage end equal to the EDF duration. Missing sidecars, changed
hashes, timing/count mismatches, and incomplete background coverage block the
run. The pilot does not infer that a zero-event summary means reviewed
non-seizure data.

Windows are globally aligned 5-second epochs advanced by one second. Seizure
labels use the union of annotated event overlap: at least 50% positive, zero
overlap negative, and partial overlap ambiguous. Ambiguous rows are retained in
feature shards and excluded from fitting. Quality exclusions and invalid or
degenerate windows are counted and omitted from fitting; missing time remains
missing and resets event state. The current pilot profile requires the
reviewed 18-channel bipolar montage, 256 Hz, and does not silently resample or
apply notch filtering.

## Features, model, and event selection

Each retained channel contributes variance, line length, Hjorth mobility and
complexity, excess kurtosis, absolute and relative delta/theta/alpha/beta/gamma
power, spectral entropy, and median/95th-percentile spectral frequency. Robust
cross-channel summaries add median, maximum, and interquartile range. This
378-feature representation is independent of the attention model.

EDF samples are decoded in bounded 300-second cores with a 60-second filter
halo. One core is processed and emitted at a time to compressed Parquet row
groups; no cohort-wide waveform array is written. The CPU XGBoost baseline uses
hist trees, fixed seed/configuration, early stopping on validation only,
training-only `StandardScaler` state, equal total sample weight per training
patient, and a positive-class ratio computed from training labels. Validation
patients alone select the probability threshold under the configured false
event/hour and alert-occupancy limits. The test partition is not loaded until
the model and threshold have been saved.

The event converter applies a trailing median filter, hysteresis, consecutive
open/close counts, and resets on missing or non-contiguous windows. Detection
time is the right edge of the second qualifying smoothed window. Evaluation
matches candidates one-to-one to annotated events and reports event
sensitivity, false alarms per evaluable non-seizure hour, trigger latency,
missed event IDs, and per-patient results. For offline zero-phase filtering, reported
latency is retrospective and is not a real-time guarantee.

## Storage and artifacts

The pilot root is external to the Git repository. It is capped at 5 GiB total;
every extraction/write stage checks the cap and checks that at least 20 GiB
remain free on the containing filesystem. The proposed raw cohort is about
0.96 GB. Feature size depends on reviewed EDF durations and exclusions; an
upper planning formula is approximately 1.5 kB per one-second window before
Parquet compression, plus small manifests and model files. Peak decoded waveform
memory is bounded by the configured chunk/halo sample ceiling; training loads
feature tables rather than EDF signals. No benchmark runtime or compression
ratio is claimed. Each run records cohort, annotation, split, preprocessing,
feature-shard, model, scaler, code, and dependency provenance. Generated
Parquet/scaler artifacts are ignored by Git, and the runner refuses output
inside the repository.

## Approval gates and later command

The official [PhysioNet CHB-MIT v1.0.0 page](https://physionet.org/content/chbmit/1.0.0/)
lists the dataset as Open Access under the Open Data Commons Attribution License
v1.0 and gives the dataset citation (Guttag, 2010, DOI
10.13026/C2K01R) plus the original seizure-detection publication to cite.
PhysioNet says its open datasets need no authentication but remain subject to
their listed licenses; its credentialing and training steps apply only to
datasets whose project pages require them ([PhysioNet FAQ](https://physionet.org/about/faqs/)).
That access status does not itself decide whether this team's secondary research
needs an institutional ethics/IRB determination or other local review; obtain
and document that determination before processing.

Before a real pilot, the user/research owner must provide: (1) documented
permission for this research use and the required human-subject/data-use review;
(2) the twelve authorized local EDF files in an external storage root; (3) a
reviewed cohort manifest binding their hashes, identities, child eligibility,
montage/channel mapping, permissions, and quality review; (4) the CHB-MIT
summary plus exact seizure sidecars for event recordings; (5) per-recording
reviewer evidence for sidecar reconciliation and complete non-seizure coverage;
(6) the frozen preprocessing JSON; and (7) explicit approval to process those
real recordings and run the experimental model. Approval of this code does not
substitute for any of those data approvals.

After those files exist and the user explicitly authorizes the run, the command
will be:

```bash
python -m seizure_research.training run-pilot \
  --root /external/approved/chbmit-pilot \
  --cohort-manifest /external/approved/cohort-manifest.json \
  --preprocessing-config /external/approved/preprocessing.json \
  --annotation-evidence /external/approved/annotation-evidence.json \
  --output-dir /external/approved/chbmit-pilot/run-001 \
  --approve-real-data-processing
```

This command is provided for review only and was not run during Checkpoint D1.
