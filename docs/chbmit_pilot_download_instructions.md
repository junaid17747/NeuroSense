# CHB-MIT Pilot: Staging Instructions (Not Executed)

**Do not start until the faculty/institutional checklist is complete and the
project owner explicitly approves staging.** These are instructions only; no
download has been run.

The machine-readable [file manifest](chbmit_pilot_file_manifest.json) lists
all twelve selected EDF URLs, their split and annotation role, the six
patient-summary URLs, six official `SHA256SUMS` URLs, and the six `.edf.seizures`
URLs. All links point to the official PhysioNet v1.0.0 distribution. The
manifest intentionally has no per-file expected digest or exact size yet;
obtain those from the official checksum files and HTTP metadata rather than
copying guessed values.

1. Use a dedicated external directory, for example
   `/mnt/approved/chbmit-pilot`, never a directory inside this Git repository.
   Create it only after approval.
2. Before each transfer, query the exact URL size with `wget --spider
   --server-response <URL>` or another approved HTTPS metadata tool. Record
   its `Content-Length`. Check current root usage and filesystem free space:

   ```bash
   du -sb /mnt/approved/chbmit-pilot
   df -B1 /mnt/approved/chbmit-pilot
   ```

   Do not begin a file transfer unless current root bytes plus that file's
   expected bytes remain at or below 5 GiB, and free bytes minus that file's
   expected bytes remain at or above 20 GiB. Repeat these checks before every
   file; do not download archives or whole patient directories.
3. Download only URLs enumerated by the manifest, preserving their relative
   paths. For each item, the command shape is
   `wget --https-only --continue --directory-prefix=/mnt/approved/chbmit-pilot/<case> <official-URL>`.
   First fetch the six `SHA256SUMS` files, then the six summary files, twelve
   EDFs, and six seizure sidecars. Do not fetch unrelated EDFs.
4. In each case directory, verify every staged manifest file against the
   downloaded official checksum list, for example:

   ```bash
   cd /mnt/approved/chbmit-pilot/chb02
   sha256sum --check --ignore-missing SHA256SUMS
   ```

   Preserve the official checksum files. Record actual byte sizes and
   independently verified hashes in the reviewed cohort manifest. If any
   expected file is missing, mismatched, or differs from the selected
   recording/annotation plan, stop and return to faculty review.
5. Run the read-only readiness command after all review artifacts and files
   are staged. It must report no blockers before anyone separately authorizes
   Checkpoint D3 processing.

The command is read-only and prints its JSON report to the terminal. Use the
human-completed [approval record](chbmit_pilot_approval_record_template.json)
with its cohort hash and evidence references filled in; the pending template
is intentionally invalid and cannot produce a ready result.

```bash
python -m seizure_research.readiness check \
  --root /mnt/approved/chbmit-pilot \
  --cohort-manifest /mnt/approved/chbmit-pilot/cohort-manifest.json \
  --preprocessing-config /mnt/approved/chbmit-pilot/preprocessing.json \
  --annotation-evidence /mnt/approved/chbmit-pilot/annotation-evidence.json \
  --approval-record /mnt/approved/chbmit-pilot/faculty-approval.json \
  --file-manifest docs/chbmit_pilot_file_manifest.json
```

Exit status `0` means the local readiness gates passed. Exit status `2` means
the JSON report lists blockers or missing approvals. Passing readiness does
not itself run the approved-data pipeline.

The previous planning estimate was approximately 0.96 GB of raw EDF data. It
is not a substitute for preflight size checks: this selected subset's exact
size must be calculated from the official file metadata before staging.
