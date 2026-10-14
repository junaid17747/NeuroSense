# Checkpoint D4: final dataset authorization and staging status

D6 update: the temporary placeholder below is not an approved staging location.
Real staging now requires a named persistent root and custodian evidence. See
[the D6 readiness report](checkpoint_d6_readiness.md) for current blockers and
the canonical approval-record path.

**Status:** cohort and file list are frozen; official metadata checks passed;
the staging directory is prepared and empty. No EDF, summary, or seizure
sidecar was downloaded, and no EEG was processed or used for training.

## Frozen cohort and verification

The authoritative file list is
[`chbmit_pilot_file_manifest.json`](chbmit_pilot_file_manifest.json), whose
SHA-256 is
`dc81f5f555e098806ebe977a6e553eee314e7b66a897a5620f037d44e2b79bcd`.
The split is train `chb02`, `chb05`, `chb08`; validation `chb09`; test
`chb06`, `chb10`. The known `chb01`/`chb21` same-person alias is preserved
and both are excluded from this cohort.

The 25-file allowlist is 12 EDFs, six patient summaries, six matching
`.edf.seizures` sidecars, and the official global `SHA256SUMS.txt`. All 25
official URLs returned HTTP success and their Content-Length matched the
manifest. The 24 EDF/summary/sidecar SHA-256 values matched the entries in the
official checksum document. PhysioNet does not list a checksum for
`SHA256SUMS.txt` inside that document; its 73,298-byte body was independently
hashed as `f1698efeb9fef887e55cedf6eda13386d9a888db663aec48b2aa194b7b753d2b`
for this check. The manifest contains each exact URL, expected size, and
per-file digest where the official checksum source provides one.

The selected EDFs total **1,004,143,360 bytes**. The entire manifest totals
**1,004,239,275 bytes (0.935 GiB)**, within the **5 GiB** cap. At the latest
preflight, the filesystem had **47,812,907,008 bytes** free; staging the full
allowlist would leave **46,808,667,733 bytes (about 43.59 GiB)**, above the
required **21,474,836,480-byte (20 GiB)** reserve. The CLI repeats checks
before every file; check again immediately before an approved transfer.

An empty directory now exists at `/tmp/neurosense-chbmit-pilot`. It is only a
temporary placeholder; for actual authorized staging, use the institution's
approved external storage location outside this Git worktree. The repository
ignores EDF/sidecar files and seizure research outputs, and the staging code
rejects a destination inside the repository.

## What the faculty statement establishes

The project owner reports that the faculty guide said, “CHB-MIT is a good
choice; continue the project.” This is academic support for choosing the
dataset. It does not establish that the faculty reviewed the exact frozen
cohort, approved dataset-use terms, completed annotation review, authorized
download/staging, or authorized processing/training. No approval, ethics
determination, or annotation evidence is claimed by this report.

Before staging, the responsible people still need to document:

1. The project/faculty data-use review of the PhysioNet CHB-MIT v1.0.0 Open
   Data Commons Attribution License v1.0, citation, and university rules.
2. A written institutional ethics/IRB determination for this secondary use
   (approved, exempt, or not required), plus any conditions.
3. Faculty review of the exact cohort, patient split, age-aware non-clinical
   scope, and `chb01`/`chb21` alias safeguard.
4. Qualified human review of the six seizure sidecars against summaries and
   full-duration annotation coverage for all six proposed background files;
   also montage, age metadata, signal quality, and exclusions.
5. Explicit authorization to stage exactly this manifest to a named external
   location, with access, retention, backup, and deletion rules.
6. A separate explicit authorization to process EEG and train/evaluate after
   staging and readiness checks pass.

To document genuine decisions, the project owner should fill
[`chbmit_pilot_approval_record_template.json`](chbmit_pilot_approval_record_template.json)
from the actual signed/email/office determination, record the reviewer or
approver, date, evidence reference and outcome for each applicable decision,
and keep the underlying evidence in the institution-approved location. For
faculty review and staging, bind the approval to the manifest SHA-256 above.
For institutional determination, enter the real office's outcome and
reference; do not infer it from faculty support. Complete the human checklist
only after the corresponding reviews. Keep annotation evidence and approval
records out of Git if they identify people or contain sensitive information.

The staging command was invoked against the blank approval template and
returned `blocked: explicit_staging_authorization_missing` before making any
network request or writing any file. The local directory remains empty.

## Commands after the decisions are documented

Use the approved external root in place of the example below. First inspect
space and the frozen file plan; this is read-only:

```bash
python -m seizure_research.staging plan \
  --root /mnt/approved/neurosense/chbmit-pilot
```

After the real dataset-use, institutional, faculty, and staging approvals are
recorded against this manifest, stage only the allowlisted files:

```bash
python -m seizure_research.staging stage \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --confirm-authorized-staging
```

The staging command checks the 24 per-file digests against the official
checksum document and the frozen manifest as each file is written. Then run
the read-only readiness check; it rechecks file paths, sizes, hashes, split,
annotation evidence, storage, and approval blockers:

```bash
python -m seizure_research.readiness check \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json
```

Readiness passing is not permission to process. The bounded EEG smoke test,
feature extraction, and experimental training each remain behind the separate
processing authorization and readiness gates. The objective is age-aware,
patient-independent seizure-event research; the pilot is not clinical
validation or a diagnostic device.
