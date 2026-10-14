# CHB-MIT pilot staging instructions

These commands are instructions only. No EEG was downloaded or processed for
Checkpoint D3. The exact 25-file allowlist, source URLs, verified byte sizes,
SHA-256 values, relative local destinations, and split are in
[`chbmit_pilot_file_manifest.json`](chbmit_pilot_file_manifest.json). Files go
under `<root>/<manifest path>`; choose an approved root outside the Git
worktree, such as `/mnt/approved/neurosense/chbmit-pilot`.

PhysioNet v1.0.0 exposes one dataset-wide `SHA256SUMS.txt`, not per-patient
`SHA256SUMS` files. The staging tool fetches that file first, verifies its
entries against the pinned cohort manifest, then retrieves only the 12 EDFs,
six patient summaries, and six `.edf.seizures` sidecars. It rejects altered
sizes, hashes, paths, or patient assignments. It performs bounded 1 MiB reads,
checks the 5 GiB cap and 20 GiB free-space reserve before every file, keeps
partial data out of Git, and records no EEG waveform arrays to disk.

The 12 EDFs total **1,004,143,360 bytes**. All 25 staged files, including the
single checksum document, total **1,004,239,275 bytes (0.935 GiB)**. Metadata
was checked on 2026-10-12. The expected bytes are below the 5 GiB cap. Current
`/` free space at the final check was 47,780,118,528 bytes; staging the listed
sources would leave 46,775,879,253 bytes free, above the required
21,474,836,480-byte reserve.
These figures are planning snapshots; rerun the preflight immediately before
each real transfer and processing stage.

## Review sequence

1. Faculty provides seizure-focused guidance and reviews the research scope,
   exact recordings, and frozen patient split.
2. Project owner and faculty document the PhysioNet license/data-use review
   and citation requirements.
3. The responsible institutional office records its ethics/IRB determination
   (approval, exemption, or not required) and evidence.
4. A qualified human reviewer reviews montage, age metadata, quality, and all
   annotation coverage, including full-duration review of each proposed
   background recording and reconciliation of each seizure sidecar with its
   summary.
5. Faculty/institution separately authorizes staging these exact files and
   records the file-manifest SHA-256 in `staging_manifest_sha256`.
6. Only after data are staged, reviewed, and readiness is clear may faculty
   separately authorize real-recording processing/training in the approval
   record. This is not clinical validation.

The blank approval record and review checklist are templates, not evidence of
any approval.

## Read-only preflight

```bash
python -m seizure_research.staging plan \
  --root /mnt/approved/neurosense/chbmit-pilot
```

This prints the exact download plan and current storage calculation. It does
not make network requests or create the root directory. Stop if the command
reports a cap, free-space, manifest, or path blocker.

## Download after explicit staging approval

Complete an approval record outside the repository. It must show approved
dataset terms, a completed institutional ethics determination, faculty review,
an explicit staging authorization, and the exact `staging_manifest_sha256`
printed by the preflight. Record the same absolute dataset directory in
`staging_root`, set `storage_persistence_confirmed` only after the custodian
confirms persistence, and provide `storage_policy_ref` identifying the actual
access, retention, backup and deletion decisions. `/tmp`, `/var/tmp`, `/run`
and `/dev/shm` are rejected for real staging and processing. The example
`/mnt/approved/...` is not an existing or approved location in this checkout.
Keep `faculty-approval.json` directly inside the dataset root; this is also
where the dashboard reads it. Then run:

```bash
python -m seizure_research.staging stage \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json \
  --confirm-authorized-staging
```

The staging command is a real network transfer and must not be run until the
listed human decisions are recorded. It downloads no archive or unrelated
recording. On interruption, it refuses an existing `.part` file and reports a
blocker; have the data custodian review and remove only a clearly incomplete
transfer before retrying.

## Readiness and bounded EEG smoke test

After staging, create reviewed cohort, preprocessing, and annotation-evidence
manifests outside the repository. Readiness remains fail-closed until files,
hashes, full annotation coverage, data-use/ethics/faculty decisions, staging
authorization, and separate processing/training authorization are complete:

```bash
python -m seizure_research.readiness check \
  --root /mnt/approved/neurosense/chbmit-pilot \
  --cohort-manifest /mnt/approved/neurosense/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/neurosense/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/neurosense/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/neurosense/chbmit-pilot/faculty-approval.json
```

Only after that command has no blockers and real-data processing is separately
authorized, decode one configured five-second window without saving signal
arrays:

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

The smoke command checks readiness, the pinned file SHA-256, configured
channel labels/rates, then reads a bounded first window. It does not train or
write generated artifacts. A successful smoke test does not replace the
separate full pilot command and is not clinical validation.
