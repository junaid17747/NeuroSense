# CHB-MIT pilot: human review and authorization checklist

This blank checklist records decisions; it is not itself an approval. Do not
mark an item complete without the named reviewer, date, and evidence reference.
The exact allowlisted cohort and source file hashes are in
[`chbmit_pilot_file_manifest.json`](chbmit_pilot_file_manifest.json).

## Faculty guidance for seizure-focused research

- [ ] Faculty guide reviewed and supports the age-aware, seizure-event
  research objective and non-clinical scope.
- [ ] Faculty reviewed the proposed six-patient split and exact twelve EDFs:
  train `chb02/chb05/chb08`, validation `chb09`, test `chb06/chb10`.
- [ ] Faculty confirmed the `chb01`/`chb21` same-patient alias safeguard.
- [ ] Faculty selected/approved the human reviewer(s) for annotation and
  signal-quality review.

Faculty reviewer/date/evidence: __________________________________________

## Dataset-use permission and license review

- [ ] Project owner reviewed the PhysioNet CHB-MIT v1.0.0 Open Data Commons
  Attribution License v1.0, required dataset citation, and any local terms.
- [ ] Responsible project authority determined this research use and access
  satisfy the license and university data-use rules.
- [ ] The exact manifest hash reviewed for staging is recorded:
  _______________________________________________________________________
- [ ] Data custodian approved the external location, named-user access,
  retention, backup, and deletion plan.

Dataset-use reviewer/date/evidence: ______________________________________

## Institutional ethics determination

- [ ] Responsible institutional office made a written determination whether
  the secondary analysis requires ethics/IRB review, is exempt, or is not
  required.
- [ ] Any required review is final and covers this proposed access and use.
- [ ] Conditions relevant to pediatric EEG, access, retention, and sharing are
  documented and assigned.

Determination (approved / exempt / not required / pending): _______________

Office, reviewer/date/evidence: _________________________________________

## Human annotation and signal-quality review

- [ ] Human reviewer checked all six seizure sidecars against their matching
  summary event count and onset/offset times; discrepancies were resolved.
- [ ] Human reviewer checked the full duration of all six proposed background
  recordings; no-event summary text alone was not treated as seizure-free
  evidence.
- [ ] Annotation coverage begins at 0 seconds and ends at each EDF's reviewed
  duration. Reviewer, dates, intervals, source hashes, and evidence references
  are recorded.
- [ ] Age metadata, montage, channel order, units, sampling rates, and quality
  were reviewed for each EDF. Exclusions and reasons are recorded.
- [ ] The official dataset-wide `SHA256SUMS.txt` entries, expected sizes, and
  exact staged file list match the D3 machine-readable manifest.

Reviewers/date/evidence and annotation manifest hash: ____________________

## Explicit authorization to stage real recordings

- [ ] Faculty/institution explicitly approved downloading/staging only the 25
  files in the frozen manifest: twelve EDFs, six summaries, six seizure
  sidecars, and the single global checksum file.
- [ ] Approval record includes `staging_manifest_sha256` matching the reviewed
  manifest and names the staging approver, date, and evidence reference.
- [ ] Approved external root is outside Git. The 5 GiB pilot cap and at least
  20 GiB free-space reserve passed immediately before transfer.
- [ ] `staging_root` names that exact persistent location;
  `storage_persistence_confirmed` and `storage_policy_ref` reflect genuine
  custodian review. No temporary directory is used for long-term EEG storage.

Staging approver/date/evidence: __________________________________________

## Separate authorization to process and train

- [ ] Dataset is staged and all file sizes/hashes pass the read-only readiness
  command.
- [ ] Annotation/quality evidence and reviewed patient split are complete.
- [ ] Faculty/institution explicitly authorized bounded EEG processing,
  feature extraction, and experimental XGBoost training, recorded separately
  from authorization to download.
- [ ] Approval record is bound to the final cohort-manifest hash and contains
  the processing approver, date, and evidence reference.
- [ ] Everyone understands this is non-clinical research and does not claim
  clinical validation or authorize care decisions.

Processing approver/date/evidence: _______________________________________

**Do not stage or process recordings while any applicable decision is pending,
any evidence is missing, or the readiness report contains blockers.**
