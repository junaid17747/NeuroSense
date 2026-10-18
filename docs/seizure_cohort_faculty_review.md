# NeuroSense seizure cohort — faculty review package

Prepared 17 October 2026. **Proposal only: no staging, annotation, training or ethics approval is implied.** Complete a private copy outside Git, under `~/NeuroSense-EEG-Data/reports/`. All decision fields below remain pending.

## Purpose and source

Develop an experimental, retrospective EEG seizure-event detector with patient-independent evaluation. The prototype includes a separate XGBoost seizure pipeline and an authenticated waveform/evaluation page. Synthetic fixtures verify software behavior; they do not establish performance on patients. Webcam, rPPG and synthetic attention outputs are separate from seizure research. No clinical diagnosis, headset inference or medical alert capability is validated.

Dataset: [CHB-MIT Scalp EEG Database, version 1.0.0](https://physionet.org/content/chbmit/1.0.0/). Citation: Guttag, J. (2010), PhysioNet, DOI [10.13026/C2K01R](https://doi.org/10.13026/C2K01R). Also cite the original work: Ali Shoeb, *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*, MIT PhD thesis, September 2009, [MIT record](http://hdl.handle.net/1721.1/54669), and the PhysioNet citation requested on the dataset page at publication time.

The published license is [Open Data Commons Attribution License v1.0](https://physionet.org/content/chbmit/view-license/1.0.0/). A human reviewer must check the actual license, attribution/notices, intended outputs, and applicable university data-use requirements. Public availability does not supply this project's institutional determination or processing authorization.

## Proposed cohort and patient split

The pinned official listing has 686 EDFs across 24 case directories representing 23 people. chb01 and chb21 are the same person. The selected 45 EDFs represent all 23 people through 23 case directories; no chb21 EDF is selected. Earlier D7/D8 prose claiming 24 selected directories was inaccurate. The inventory and selection themselves are unchanged.

There are 23 seizure-positive **candidates**, 22 summary-zero **background candidates**, and 25 declared but unreconciled events. **Zero recordings are verified seizure-free.** The proposal is separate from the unchanged frozen six-patient pilot.

Seed 42 and split algorithm `sha256-json-seed-patient-v1` are frozen before outcomes. All recordings for one person remain together. Human exclusions retain their original assignment. Do not redraw splits or select thresholds using test outcomes.

| Partition | People | EDFs | Positive candidates | Background candidates | Declared events | Ages by recording |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Train | 13 | 25 | 13 | 12 | 14 | Child 14, adolescent 6, adult 4, unknown 1 |
| Validation | 4 | 8 | 4 | 4 | 4 | Child 8 |
| Test | 6 | 12 | 6 | 6 | 7 | Child 10, adult 2 |

Published ages total 16 child, 3 adolescent, 3 adult and 1 unknown person; by recording: 32 child, 6 adolescent, 6 adult and 1 unknown. chb24 lacks published age and must be explicitly excluded under the current training profile unless a separately sourced and reviewed cohort revision becomes available. No age is inferred from EDF surrogate dates. Neonatal, infant and older-adult coverage is absent. There are no adult validation patients and no adolescent validation/test patients. This permits only a limited pooled research proposal, not age-specific validation. Scientific adequacy remains undetermined pending reviewed event counts, quality and evaluable background duration.

The following list accounts for all 45 proposed recordings. Filenames retain their source case prefix; all background entries require full-duration review.

| Canonical person | Partition | Published age group | Positive candidate EDF | Background candidate EDF |
| --- | --- | --- | --- | --- |
| chb01 | train | child | chb01_26.edf | chb01_27.edf |
| chb02 | validation | child | chb02_16.edf | chb02_01.edf |
| chb03 | train | adolescent | chb03_01.edf | chb03_05.edf |
| chb04 | train | adult | chb04_05.edf | chb04_11.edf |
| chb05 | train | child | chb05_06.edf | chb05_02.edf |
| chb06 | test | child | chb06_18.edf | chb06_16.edf |
| chb07 | train | adolescent | chb07_13.edf | chb07_07.edf |
| chb08 | train | child | chb08_02.edf | chb08_03.edf |
| chb09 | train | child | chb09_19.edf | chb09_17.edf |
| chb10 | validation | child | chb10_12.edf | chb10_02.edf |
| chb11 | test | child | chb11_99.edf | chb11_01.edf |
| chb12 | test | child | chb12_11.edf | chb12_19.edf |
| chb13 | train | child | chb13_40.edf | chb13_05.edf |
| chb14 | train | child | chb14_03.edf | chb14_01.edf |
| chb15 | train | adolescent | chb15_06.edf | chb15_01.edf |
| chb16 | test | child | chb16_18.edf | chb16_19.edf |
| chb17 | train | child | chb17a_03.edf | chb17c_13.edf |
| chb18 | train | adult | chb18_31.edf | chb18_01.edf |
| chb19 | test | adult | chb19_30.edf | chb19_01.edf |
| chb20 | validation | child | chb20_68.edf | chb20_22.edf |
| chb22 | validation | child | chb22_20.edf | chb22_01.edf |
| chb23 | test | child | chb23_06.edf | chb23_07.edf |
| chb24 | train | unknown | chb24_06.edf | None |

## Annotation findings and required review

All 12 known issues remain excluded; **zero were resolved into trustworthy labels by public metadata alone**.

- `chb07_18.edf`: the summary declares zero events, but the seizure-record list includes it; its own sidecar is absent from the official listing.
- `chb07_19.edf`: the summary declares one event and a sidecar is listed, but the seizure-record list omits it. A filename transposition is only a hypothesis; timestamps must not be transferred between files.
- `chb24_02`, `_05`, `_08`, `_10`, `_12`, `_16`, `_18`, `_19`, `_20`, `_22` (all `.edf`): listed recordings lack summary entries and positive list/sidecar evidence. Missing entries do not certify seizure-free intervals.

Sources: [chb07 summary](https://physionet.org/files/chbmit/1.0.0/chb07/chb07-summary.txt), [chb24 summary](https://physionet.org/files/chbmit/1.0.0/chb24/chb24-summary.txt), [seizure-record list](https://physionet.org/files/chbmit/1.0.0/RECORDS-WITH-SEIZURES). Exact source URLs, cached body hashes, official EDF/sidecar checksums and per-file interpretations are in [the discrepancy report](../reports/final/annotation_discrepancies.json). All 56 cached metadata hashes were checked; sidecar content and real EEG were not downloaded or inspected in this task.

The existing selection already substitutes metadata-consistent `chb07_13.edf` (positive candidate) and `chb07_07.edf` (background candidate). `chb24_06.edf` is a positive candidate only; there is no approved replacement background for chb24. These alternatives are not human-reviewed. Official evidence, machine interpretation and human review are separate fields; checksum or metadata agreement is not a human attestation.

After applicable authorization, a reviewer must reconcile summaries and exact sidecar event times, verify full-record coverage, identify ambiguous intervals, attest specific seizure-free intervals, and record exclusions, reviewer identity, date and evidence references. The resulting event, seizure-free and excluded intervals must cover each admitted recording without gaps or overlap. Unresolved reference seizures intersecting exclusions require adjudication before evaluation.

## Signal quality and montage

All 45 summaries contain the required 18 bipolar channel labels. This is provisional name compatibility only. Actual EDF headers, duplicate labels and their source indices, polarity, units, 256 Hz sampling, missing channels, artifacts, clipping and discontinuities remain unreviewed. Each recording needs a reviewed channel mapping and age-appropriate preprocessing profile. The current pipeline uses 5-second windows at 1-second steps and rejects incompatible mappings, resampling and unknown ages. Algorithmic abstention must not remove missed reference seizures from sensitivity.

## Storage and retention proposal

Use the existing `~/NeuroSense-EEG-Data/archive` and separate `temporary` directory. Retained archive plus metadata is bounded at **2,135,611,513 bytes**, below **2,147,483,648 bytes (2 GiB)** by 11,872,135 bytes. This bound covers only the selected cohort, not all 686 EDFs. Actual compression is unknown. The separate temporary quota is 1 GiB; the current staging estimate is 224,669,696 bytes peak. Preserve at least 20 GiB free on the Ubuntu system partition.

The existing archive manager checks original checksums, lossless decompression, quotas and resumable transfers. Temporary originals are removed only under the recorded cleanup permission. Keep review records, EEG, features and trained models outside Git. Training artifacts use a separate approved `runs` directory, currently capped at 5 GiB across that directory; they are additional to the 2 GiB source archive. The custodian must set retention periods, access, backups and deletion rules. No retention duration or deletion permission is assumed.

## Decisions for the actual faculty/university reviewers

| Decision or field | Human entry — currently pending |
| --- | --- |
| University, department, responsible researcher, faculty guide | __________ |
| Applicable policy name/link/version and deciding authority | __________ |
| Ethics/data-use determination: required, not required, or denied; rationale | __________ |
| Actual committee/reference number or documented nonrequirement evidence | __________ |
| Dataset license and citation review; reviewer, date, evidence | __________ |
| Cohort, split, age restrictions and evaluation limitations accepted? | __________ |
| Signal-quality, montage and annotation reviewers and evidence location | __________ |
| Storage custodian, access list, backup policy, retention end and deletion method | __________ |
| **Staging/archive authorization:** decision, scope, inventory/selection hashes, approver, date, evidence | __________ |
| Separate temporary quota and temporary-copy cleanup permission | __________ |
| **Processing authorization:** decision, permitted inspection/preprocessing scope, approver, date, evidence | __________ |
| **Training authorization:** decision, reviewed plan/split/preprocessing/annotation/run hashes, output root, approver, date, evidence | __________ |
| Faculty/reviewer signature or institutional decision record, where applicable | __________ |

No reference number, exemption, signature, approval date or institutional policy has been supplied. The [existing blank approval template](chbmit_d7_archive_approval_template.json) contains the machine-readable fields; complete only a private copy. A valid hash binds a document but does not authenticate its author. Software validation cannot replace institutional verification of genuine decisions.

Supporting materials: [proposed split](../reports/final/proposed_patient_split.json), [split audit](../reports/final/proposed_split_audit.json), [training and evaluation runbook](seizure_cohort_workflow.md). Real staging, reviewed annotations, training, held-out performance and clinical validation remain incomplete.
