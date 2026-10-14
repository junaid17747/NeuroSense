# Checkpoint D3: CHB-MIT real-data staging plan

**Status:** files and metadata verified from the official PhysioNet v1.0.0
distribution; staging and EEG processing have not been approved or performed.
No faculty decision, ethics determination, human annotation review, or
processing authorization is claimed here.

## Research purpose and boundaries

Prepare a small, age-aware, patient-independent seizure-event research cohort
for a later CPU XGBoost baseline. Outputs are experimental research artifacts,
not clinical decisions or a diagnostic device. Faculty's direction to focus on
seizures is research guidance; it does not itself grant data-use permission,
ethics clearance, human annotation approval, authorization to stage, or
authorization to process/train.

The source is [PhysioNet CHB-MIT Scalp EEG Database v1.0.0](https://physionet.org/content/chbmit/1.0.0/).
Its access page specifies the Open Data Commons Attribution License v1.0 and
requests dataset citation (Guttag, 2010, DOI 10.13026/C2K01R), as well as the
original seizure-detection publication. Open Access remains subject to the
license and university requirements. CHB-MIT contains pediatric EEG; the
responsible institutional office must make and document the ethics/IRB
determination.

## Frozen cohort and destinations

Every local path is `<approved external root>/<path>`. Exact URLs, byte sizes,
SHA-256 digests, annotation sidecars, summary files, and the official checksum
source are recorded in the [machine-readable manifest](chbmit_pilot_file_manifest.json).
All files are on PhysioNet's official `https://physionet.org/files/chbmit/1.0.0/`
host. The global `SHA256SUMS.txt` is the checksum source; PhysioNet does not
publish the per-patient checksum paths used by the earlier D2 draft.

| Split | Patient(s) | EDF paths and verified sizes (bytes) |
| --- | --- | --- |
| Train | chb02 | `chb02/chb02_01.edf` 42,399,744; `chb02/chb02_16.edf` 11,299,328 |
| Train | chb05 | `chb05/chb05_01.edf` 42,517,504; `chb05/chb05_06.edf` 42,399,744 |
| Train | chb08 | `chb08/chb08_02.edf` 42,399,744; `chb08/chb08_03.edf` 42,399,744 |
| Validation | chb09 | `chb09/chb09_01.edf` 94,484,992; `chb09/chb09_06.edf` 176,941,312 |
| Test | chb06 | `chb06/chb06_01.edf` 169,898,496; `chb06/chb06_02.edf` 169,580,544 |
| Test | chb10 | `chb10/chb10_01.edf` 85,028,864; `chb10/chb10_12.edf` 84,793,344 |

The EDF total is **1,004,143,360 bytes**. Stage six patient summaries, six
matching `.edf.seizures` files, and one global `SHA256SUMS.txt` (73,298 bytes).
The entire exact allowlist is 25 files / **1,004,239,275 bytes (0.935 GiB)**.
Hashes and lengths for every one of these 25 files are pinned in the manifest.
The official patient directory listings confirm each selected EDF and sidecar
is present; the checksum document lists each selected EDF, sidecar, and
summary. Metadata-only HTTP HEAD requests confirmed exact Content-Length for
all EDFs and support files on 2026-10-12. No EEG file body was fetched.

Patient identities are partitioned before window generation: train
`chb02/chb05/chb08`, validation `chb09`, test `chb06/chb10`. There is no
cross-split patient overlap. The code and manifest preserve the known
`chb01`/`chb21` same-person alias safeguard, and this cohort excludes both.

Six EDFs are proposed seizure recordings and require event reconciliation
between the `.edf.seizures` sidecar and patient summary. Six proposed
background EDFs require human review of the full recording duration; the
summary's lack of events does not establish seizure-free annotation coverage.
No annotation review is represented as complete.

## Space and handling

The pilot root has a hard **5 GiB** cap, with a **20 GiB** free-space reserve.
At the final check, `/` had 47,780,118,528 bytes free (44.50 GiB). Staging the
manifest's full 25-file set would leave 46,775,879,253 bytes free (about
43.56 GiB), and uses 0.935 GiB of the 5 GiB cap. The D3 CLI rechecks remaining
space before every file and refuses writes if the cap or reserve would be
crossed. Recheck immediately before staging; these are snapshots, not a future
guarantee.

Keep the root outside the Git worktree and use only an approved university
storage location. Do not re-identify patients or send files to third-party
services. Retention, access, backup, and deletion must follow the institution's
written determination. Raw EEG, derived features, and model artifacts remain
outside Git.

## Independent human decisions still required

1. Faculty's seizure research guidance and approval of the non-clinical scope,
   exact cohort, patient split, and annotation-review process.
2. Project/faculty confirmation that PhysioNet's license, citation, and local
   data-use terms permit this project's intended access and use.
3. Written ethics/IRB determination from the responsible institutional office,
   including any required final approval or conditions.
4. Qualified human review of montage, age metadata, signal quality, all
   annotation coverage, all six full-duration backgrounds, and reconciliation
   of the six seizure sidecars with summaries.
5. Explicit authorization to stage the exact manifest, bound by its SHA-256,
   and approval of the external location, access, retention, and deletion.
6. After staging and readiness pass, a separate explicit authorization to
   process EEG, extract features, and train/evaluate the experimental model.

The [human checklist](chbmit_pilot_human_review_checklist.md) and
[approval-record template](chbmit_pilot_approval_record_template.json) separate
these decisions. Blank or pending records are blockers.

## Commands for a later authorized run

Read-only plan; no network request and no filesystem write:

```bash
python -m seizure_research.staging plan \
  --root /mnt/approved/neurosense/chbmit-pilot
```

Real staging; run only after the exact manifest, dataset-use, institutional,
faculty, and staging approvals are recorded:

```bash
python -m seizure_research.staging stage \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --confirm-authorized-staging
```

Read-only readiness after staging and human review:

```bash
python -m seizure_research.readiness check \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json
```

Bounded first-window smoke test; run only after readiness has no blockers and
separate real-data processing authorization is recorded:

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

The smoke command decodes one configured window in memory only and does not
train a model. Checkpoint D3 has not performed any of these real-data commands.
