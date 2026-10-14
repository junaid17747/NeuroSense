# Checkpoint D8: local storage and archive readiness

## Local setup

Resolved with `pathlib.Path.home()` as `/home/yahyasiddiqui`. The dedicated data root is `/home/yahyasiddiqui/NeuroSense-EEG-Data`, outside the Git repository. Its `archive/`, `temporary/`, `manifests/`, `reports/`, and `runs/` subdirectories were created and verified owner-only (`0700`). The idempotent implementation is `seizure_research.archive.prepare_local_storage`; tests exercise location separation and permissions. `reports/annotation-review-packet.json` is a private, blank review checklist generated from D7 metadata only (45 unreviewed recordings, no labels, no EEG). No EDF or other clinical EEG is present in these directories.

## Archive plan

The D7 inventory and selection self-hashes validate. Proposal coverage is unchanged: 24 cases, 23 canonical people, 686 official EDFs, 45 selected EDFs, 23 selected official positive recordings, 22 summary-zero candidates, and 25 declared (unreconciled) seizure events. The selection covers all 23 people. Its conservative retained bound is 2,135,611,513 bytes, leaving 11,872,135 bytes under the 2,147,483,648-byte cap. The archive currently contains zero bytes; no selected recording has been downloaded.

The archive plan passed against the actual local paths. It calculated a 224,669,696-byte peak temporary overlap (two copies of the largest pending recording plus ZIP overhead), with the independent temporary quota at 1,073,741,824 bytes. The plan checks retained archive growth and temporary peak together against available space when they share a device; transfer-time writes also recheck actual free space and preserve the 20 GiB system reserve. Latest observed system free space at report completion was 47,203,725,312 bytes; recheck immediately before a real staging run.

## Annotation and model readiness

There are 12 inventory annotation issues, excluded from the proposal pending review:

- `chb07/chb07_18.edf` and `chb07/chb07_19.edf`: the summary's seizure declaration conflicts with the official seizure list/sidecar presence.
- `chb24/chb24_02.edf`, `_05`, `_08`, `_10`, `_12`, `_16`, `_18`, `_19`, `_20`, and `_22`: the EDF is listed but has no corresponding recording entry in `chb24-summary.txt`.

The selected 22 summary-zero recordings are only candidates. **No selected recording is verified seizure-free**, and no annotations were reconciled or converted to training labels. The frozen six-patient pilot manifest remains unchanged at SHA-256 `dc81f5f555e098806ebe977a6e553eee314e7b66a897a5620f037d44e2b79bcd`. D8 does not train or enable a seizure model.

## Approval blockers

The requested local directory is a storage-location preference, not institutional or dataset authorization. Still required before staging are: an applicable review of the PhysioNet ODC Attribution license and citation; the university's actual determination for this public, de-identified research; faculty cohort review; storage custodian approval of access, retention, backup and deletion policy; written staging/archive authorization bound to the inventory and selection hashes; and permission for temporary-copy cleanup. The institution has not been identified in the project workspace. No approval record was created or inferred. Processing/training needs a separate later approval.

## Dashboard and verification

The authenticated Streamlit navigation label is **EEG Seizure Research** and appears in the primary sidebar. The Overview page now leads with a prominent EEG Seizure Research card and navigation button, ahead of attention metrics. The research page reports the proposal as not staged, shows the model as training pending, and does not present synthetic data as seizure predictions. An AppTest verifies the card and navigation. Existing attention, webcam, rPPG and authentication paths remain unchanged.

The D8–E archive plan is a dry run only. No real EEG was downloaded, archived, processed, or trained. The complete `.venv` Python suite passed (291 tests), all four JavaScript test files passed, and `git diff --check` passed. The Overview card and authenticated research navigation are covered by Streamlit AppTests. The D7 proposal cannot yet support real training: human annotation reconciliation, verified background data, age/channel review, and a new canonical patient-disjoint training workflow remain outstanding. The existing real-training path is restricted to the distinct frozen pilot and must not be treated as support for this proposed 23-person cohort.

The final archive-plan run reported `dry_run_no_downloads`, 45 selected recordings, 2,135,611,513 projected retained bytes, 224,669,696 projected temporary bytes, 0 bytes currently retained, and a passing disk-reserve preflight. The private local review packet is 61,634 bytes and lists 45 recordings as unreviewed, with zero verified background recordings and no labels. Local directory modes are `0700`; the packet mode is `0600`. A new fail-closed review audit checks source hashes, recording coverage, event interval bounds, sidecar reconciliation attestations, full-record coverage, and explicit seizure-free attestations; its report never creates labels or approval. The frozen pilot manifest still hashes to `dc81f5f555e098806ebe977a6e553eee314e7b66a897a5620f037d44e2b79bcd`.

## Authorized staging command for a later step

Run only after the approval decisions above are genuinely documented in a private approval file (never Git), and after rerunning the plan and confirming the 20 GiB reserve:

```bash
rtk .venv/bin/python -m seizure_research.archive stage \
  --root "$HOME/NeuroSense-EEG-Data/archive" \
  --temporary-root "$HOME/NeuroSense-EEG-Data/temporary" \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json \
  --approval-record "$HOME/NeuroSense-EEG-Data/manifests/archive-approval.json" \
  --confirm-authorized-archive
```

This command downloads only explicitly selected EDFs and support files after its authorization gates pass. The approval record does not currently exist; the command is therefore not presently authorized to run. It verifies original checksums, exact ZIP decompression, retained-size limits, and the archive index; it does not make labels or train a model. With genuine authorization present, the same command resumes verified partial downloads after interruption.

The exact dry-run command already used is:

```bash
rtk .venv/bin/python -m seizure_research.archive plan \
  --root "$HOME/NeuroSense-EEG-Data/archive" \
  --temporary-root "$HOME/NeuroSense-EEG-Data/temporary" \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json
```

After a human reviewer completes the private packet, audit it before using any annotations:

```bash
rtk .venv/bin/python -m seizure_research.annotation_review \
  --inventory reports/checkpoint_d7/official_inventory.json \
  --selection reports/checkpoint_d7/selection_proposal.json \
  --audit-packet "$HOME/NeuroSense-EEG-Data/reports/annotation-review-packet.json"
```

The current packet is intentionally incomplete, so this command should exit with status 2 and list the missing attestations. A passing audit is still not processing or training authorization.
