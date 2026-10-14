# Checkpoint D7 readiness

## Outcome

The complete public CHB-MIT v1.0.0 metadata inventory and a deterministic, **proposed** 2 GiB multi-patient cohort are ready. Only public text metadata, directory listings, official file sizes and the official checksum index were fetched. No EDF, seizure sidecar, or full dataset archive was downloaded; therefore no real EEG compression benchmark, archive, decoding, processing, or training occurred.

The new proposal remains separate from the frozen six-patient D6 pilot. Its selection manifest is `reports/checkpoint_d7/selection_proposal.json`; the original frozen pilot manifest was not changed. The complete inventory is `reports/checkpoint_d7/official_inventory.json`. Both the inventory and selection contain SHA-256 self-digests and reference the source metadata digests. `reports/checkpoint_d7/source_metadata/` is ignored by Git and contains only fetched public metadata.

## Official inventory and count reconciliation

The cached official root/case listings and `RECORDS`, `RECORDS-WITH-SEIZURES`, `SHA256SUMS.txt`, `SUBJECT-INFO`, summaries and sidecar listings were parsed and cross-checked. The complete listing contains **857 files** (686 EDFs, 141 `.edf.seizures` sidecars, 24 patient summaries, and six root-level files) across **24 case directories**. `SHA256SUMS.txt` contains 856 entries; its own hash is pinned from the fetched official file body. The current `RECORDS` and `RECORDS-WITH-SEIZURES` contents match all 686 EDFs and 141 positive-record entries. There are **23 canonical people** after treating chb01/chb21 as one identity. All 686 EDF listing entries have official SHA-256 entries. The EDF byte total is 45,756,994,304; all listed dataset files total 45,759,184,625 bytes. The PhysioNet page's prose still describes the older counts of 664 files and 129 seizure-containing files, which omit the later chb24 additions; the current directories and record files include chb24. The site reports 42.6 GB total uncompressed and ODC Attribution License v1.0. [Official dataset page](https://physionet.org/content/chbmit/1.0.0/), [current seizure-record list](https://physionet.org/content/chbmit/1.0.0/RECORDS-WITH-SEIZURES), [subject ages](https://physionet.org/content/chbmit/1.0.0/SUBJECT-INFO).

The raw EDF inventory would require about **21.31:1** lossless compression to fit all EDFs alone into 2 GiB. No such ratio is assumed or promised. There was no authorized EDF sample available for a representative benchmark. ZIP DEFLATE is implemented with ZIP STORED fallback and exact original SHA-256 verification; no lossy codec, channel removal, resampling, or quantization is used. Zstandard remains an unbenchmarked alternative for a future authorized sample.

## Deterministic proposal

The selection order first chooses the least-cost annotation-consistent recording for each canonical person, then one official seizure-positive recording per represented person where available, then one summary-zero background candidate per person where available. It is deterministic (byte size, positive-evidence tie-break, path), uses original byte sizes plus a 64 KiB per-EDF ZIP upper-bound allowance, retains all case summaries and core official indices, and reserves 16 MiB for retained index/provenance metadata. It does not create labels or call summary-zero files verified negatives.

| Measure | Proposed value |
| --- | ---: |
| Canonical patients represented | 23 / 23 |
| Case directories represented | 24 / 24 |
| Selected EDF recordings | 45 |
| Selected original EDF bytes | 2,115,690,240 |
| Conservative total retained byte bound | 2,135,611,513 |
| Budget remaining under that bound | 11,872,135 |
| Selected official seizure-positive recordings | 23 |
| Selected summary-zero background candidates | 22 |
| Verified seizure-free recordings | 0 |
| Declared seizure events in selected summaries | 25, not reconciled |
| Annotation-issue recordings in complete inventory | 12 |

The complete per-patient report includes official recording/positive/background counts, selected paths and sizes, age by case, selected declared-event counts, and excluded paths/reasons in `selection_proposal.json`. Every represented patient has a seizure-positive and a summary-zero candidate except chb24, which has an official positive but no selected summary-zero candidate. chb01 and chb21 are both represented as recordings but count as one person. The cohort covers all canonical patients, but its 45-recording size and zero verified negative recordings do **not** establish a scientifically adequate training/evaluation cohort. Annotation reconciliation, channel/quality review, age-specific eligibility and a new canonical patient-disjoint split are outstanding.

Selected case ages available from `SUBJECT-INFO` (age at case, not a newly derived age): child 32 recordings; adolescent 6; adult 6; unknown 1. chb01/chb21 are one person with published case ages 11 and 13. chb24 has no published age and is not imputed. The site categorization maps the 1.5-year value to child because age is only published in year precision; exact subyear category cannot be inferred. There is no verified neonatal, infant or older-adult representation. Although child, adolescent and adult ages are represented, no age-specific model is validated or eligible from this unreviewed proposal.

## Archive software

`seizure_research.archive` enforces the hard 2 GiB retained-directory cap, a separate temporary quota, 20 GiB minimum free space on the system partition, a persistent external root bound to approval evidence, and per-file approval rechecks. It supports selected-source-only HTTPS transfers, Range resume, SHA-256 verification, one-member ZIP archives, exact bounded decompression hashing, atomic metadata/index updates, corruption checks, failure records, and removal of a temporary EDF only after its original checksum, archive checksum/decompression, index update and cleanup permission are satisfied. A verified ZIP can be recovered after interruption; interrupted downloads retain `.part` for safe resume. Temporary processing extraction is exposed only behind separate processing approval. Synthetic tests cover listing/identity/selection, annotation gaps, exact ZIP roundtrip, corruption, resume, quota refusal and authorization refusal.

The future training workflow is **not yet implemented for this new archive cohort**. The current D6 `run-pilot` command is limited to the frozen six-patient pilot and must not be used as though it trains on this D7 selection. New-cohort orchestration must bind reviewed event evidence, decode per recording under the existing bounded preprocessing limits, construct canonical patient-disjoint partitions, train a separate seizure model, select its threshold from validation patients only, and report event sensitivity, missed events, false alarms per evaluable non-seizure hour, and detection latency on held-out test patients. The 2 GiB subset may not generalize strongly.

## Authorization and storage

The faculty guide's endorsement of CHB-MIT is not a staging or institutional determination. No genuine data-use, storage-custodian, staging, archive or processing approvals were supplied. The institution is not identified in this checkout, so its applicable public/de-identified-data policy cannot be determined here. Resolve the university's research/ethics requirements and dataset license attribution/citation; obtain separate genuine written determinations for persistent storage/staging and, later, real-data processing/training. Fill a private copy of `docs/chbmit_d7_archive_approval_template.json`; never check it in. It is deliberately all-pending and grants no permission.

No reviewed persistent storage directory was provided. `/mnt/approved` was not assumed, and no permanent dataset was written. Current system-partition free space at report time was 47,211,855,872 bytes (43.98 GiB), above the 20 GiB reserve; it must be rechecked immediately before future transfers. Do not use `/tmp` for the archive. The approved temporary directory must be separate, persistent enough for resumable temporary downloads, and permitted for cleanup by the actual custodian.

## Dashboard

The authenticated sidebar now presents **EEG Seizure Research** as the primary research workspace. It displays the inventory and proposed-coverage counts, the 2 GiB bound, current system free space, honest unknown compression, case-age availability and provenance. It labels the proposal as not staged, the model as **Training pending**, and all event counts as unreconciled. No real waveforms are shown from the proposal. The D6 waveform demonstration remains explicitly labelled synthetic; real EDF access remains behind its existing approval/readiness gate. Webcam, rPPG, authentication and attention routes are unchanged.

## Status

| Status item | D7 result |
| --- | --- |
| FULL CHB-MIT INVENTORY COMPLETE | Complete for current public directory listings and text metadata; see noted page-description count discrepancy. |
| 2 GiB COHORT SELECTED | Proposed deterministically; 23/23 canonical identities and 45 EDFs; not approved/acquired. |
| LOSSLESS ARCHIVE SOFTWARE TESTED | Synthetic tests pass; no clinical compression benchmark. |
| REAL EEG DOWNLOADED | No. |
| REAL EEG ARCHIVED AND VERIFIED | No. |
| REAL EEG PROCESSING AUTHORIZED | No evidence supplied. |
| SEIZURE MODEL TRAINED | No. |
| HELD-OUT EVALUATION COMPLETED | No. |

Exact next safe command, after choosing an existing reviewed persistent root and a separate approved temporary root, is a no-download plan:

```bash
rtk .venv/bin/python -m seizure_research.archive plan \
  --root '<REVIEWED_PERSISTENT_ARCHIVE_ROOT>' \
  --temporary-root '<APPROVED_TEMPORARY_ROOT>' \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json
```

After actual approvals are recorded in a private JSON file and the plan confirms the 20 GiB reserve, staging would use:

```bash
rtk .venv/bin/python -m seizure_research.archive stage \
  --root '<REVIEWED_PERSISTENT_ARCHIVE_ROOT>' \
  --temporary-root '<APPROVED_TEMPORARY_ROOT>' \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json \
  --approval-record '<PRIVATE_APPROVAL_JSON>' \
  --confirm-authorized-archive
```

Resume with the identical stage command; it reuses verified partial downloads and archives. Verify after completion:

```bash
rtk .venv/bin/python -m seizure_research.archive verify \
  --root '<REVIEWED_PERSISTENT_ARCHIVE_ROOT>' \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json
```

After a distinct processing authorization, the archive's bounded real smoke command is:

```bash
rtk .venv/bin/python -m seizure_research.archive smoke \
  --root '<REVIEWED_PERSISTENT_ARCHIVE_ROOT>' \
  --temporary-root '<APPROVED_TEMPORARY_ROOT>' \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json \
  --approval-record '<PRIVATE_APPROVAL_JSON>' \
  --recording '<SELECTED_RELATIVE_EDF_PATH>' \
  --confirm-authorized-processing
```

This only decodes a bounded first-five-second segment and does not train. A D7 new-cohort training command does not exist yet; do not substitute `run-pilot`.

The first authorized staging run will retain selected EDFs as losslessly compressed ZIP members, all selected seizure sidecars, all official case summaries and the core official metadata/checksum files, plus inventory/selection/index/provenance and a truthful success/failure record. It will output original and compressed byte counts, ratio, exact-decompression verification, support-file checks, archive usage/budget remainder and status. It does not create labels, train a model, or establish seizure-free intervals. Its temporary original copy is deleted only after checksum/archive/index verification and explicit cleanup permission.
