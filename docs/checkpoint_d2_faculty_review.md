# NeuroSense Checkpoint D2: Faculty Review Brief

**Decision status:** proposed only. No faculty decision, ethics determination,
recording review, annotation evidence, or real-data processing authorization is
represented by this brief.

## Objective and scope

Evaluate whether a reproducible, patient-independent EEG seizure-event
classifier can be developed from a small, reviewed CHB-MIT subset. The first
baseline is a seizure-specific feature pipeline with CPU XGBoost, separate
from NeuroSense's attention model. This is retrospective engineering research;
the output is not a medical device and must not be used to diagnose, treat,
triage, or guide care. No clinical validation has been performed.

## Dataset and proposed cohort

The source is the [official CHB-MIT Scalp EEG Database v1.0.0 on
PhysioNet](https://physionet.org/content/chbmit/1.0.0/). Its project page lists
Open Access under the Open Data Commons Attribution License v1.0 and requests
the dataset citation (Guttag, 2010, DOI 10.13026/C2K01R) and the original
seizure-detection publication. The [official seizure-recording
list](https://physionet.org/content/chbmit/1.0.0/RECORDS-WITH-SEIZURES) supports
the proposed event-file selection. Open access is not institutional ethics
clearance or faculty approval; the institution must determine applicable
requirements for secondary research involving pediatric EEG.

| Partition | CHB-MIT case | Proposed EDF recordings |
| --- | --- | --- |
| Train | chb02 | `chb02_01.edf`; `chb02_16.edf` (annotated seizure) |
| Train | chb05 | `chb05_01.edf`; `chb05_06.edf` (annotated seizure) |
| Train | chb08 | `chb08_02.edf` (annotated seizure); `chb08_03.edf` |
| Validation | chb09 | `chb09_01.edf`; `chb09_06.edf` (annotated seizure) |
| Test | chb06 | `chb06_01.edf` (annotated seizure); `chb06_02.edf` |
| Test | chb10 | `chb10_01.edf`; `chb10_12.edf` (annotated seizure) |

This is six patients and twelve recordings: three train patients, one
validation patient, and two held-out test patients. All recordings from a
patient stay in one partition. `chb01` and `chb21` are known aliases for the
same participant in the source; the code preserves that canonical identity
constraint and excludes both from this proposed cohort.

The preceding planning estimate was about 0.96 GB of EDFs. Exact byte totals
and expected hashes remain deliberately unset in
[the machine-readable file manifest](chbmit_pilot_file_manifest.json); they
must be checked against official per-patient `SHA256SUMS` files and local file
sizes before any download is approved. Six summary files, six checksum files,
and six `.edf.seizures` sidecars are also listed there. Full-record review is
required for recordings proposed as background, including those with no
summary seizure event.

## Storage, privacy, and reproducibility

The external pilot directory has a **5 GiB hard cap**, including source EDFs,
summaries, sidecars, and generated artifacts. The process must leave at least
**20 GiB free** on the containing filesystem. The user previously reported
45 GB free; space must be rechecked immediately before any approved download
or processing stage. No recording or derivative belongs in Git, cloud storage,
or an external model/API service. Keep access limited to approved project
members, do not attempt re-identification, and follow institutional retention
and deletion rules.

The software checks source hashes against reviewed manifests and official
checksum lists; records the annotation source and reviewer evidence; writes a
patient split manifest before feature extraction; and captures preprocessing,
feature, dependency, and model provenance. Extraction uses bounded EDF reads
and incremental feature shards. The read-only readiness command audits staged
files and review records but does not decode EEG, create artifacts, download,
or train.

## Limitations and risks

This small cohort has only one validation patient and two test patients, so
patient-level estimates will be highly uncertain and may not generalize. The
selected recordings are not a population-representative sample; montage,
signal quality, seizure timing, and event labels may differ across patients.
The sidecar text is not automatically interpreted: a qualified reviewer must
reconcile each seizure sidecar against the summary. Zero-event summaries do
not prove reviewed seizure-free time. Event sensitivity, false alarms/hour,
and latency are retrospective research metrics, not clinical performance
claims. The cohort includes very young children; faculty and institutional
review should explicitly address data sensitivity and permitted use.

## Decisions requested

Faculty and the institution should record decisions on:

1. Whether the research question, non-clinical scope, and six-patient cohort
   are appropriate, including the proposed files and split.
2. Whether this secondary analysis requires ethics/IRB review, is exempt, or
   is determined not to require review; record the responsible office and its
   written determination.
3. Whether the PhysioNet license and citation requirements, local data-use
   terms, storage location, access controls, and retention plan are satisfied.
4. Who is qualified to review annotations and signal quality, and what
   evidence will document full-record coverage and the summary/sidecar
   reconciliation.
5. Whether the 5 GiB cap, 20 GiB free-space reserve, and approximately 0.96 GB
   planning estimate are acceptable after exact file sizes are verified.
6. Whether to authorize downloading/staging the listed files, and separately
   whether to authorize real-recording processing and experimental training
   after readiness reports no blockers.

All decisions remain pending until a faculty/institutional approval record is
completed and bound to the final cohort-manifest hash. Approval of this brief
alone does not authorize downloading, EEG processing, or model training.
