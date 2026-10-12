# CHB-MIT Pilot: Human Review and Authorization Checklist

**Nothing is approved by this blank checklist.** Check an item only after the
named person or institutional office has completed it and the cited evidence
has been saved in the approved project location.

## Dataset terms and permission

- [ ] Faculty guide reviewed the research question, non-clinical scope, cohort,
  and patient split.
- [ ] Project owner reviewed the PhysioNet CHB-MIT v1.0.0 Open Data Commons
  Attribution License v1.0, citation requirements, and any applicable local
  use terms.
- [ ] Faculty/institution authorized obtaining and staging these exact twelve
  EDFs, their six patient summary files, six official `SHA256SUMS` files, and
  six seizure sidecars listed in `chbmit_pilot_file_manifest.json`.
- [ ] The source checksums and actual byte sizes were independently checked;
  the planned files fit under the total 5 GiB cap.
- [ ] Data custodian confirmed external storage, named-user access, retention,
  backup, and deletion arrangements.

Decision/evidence reference: ____________________________________________

Reviewer and date: ______________________________________________________

## Ethics and human-subject determination

- [ ] Responsible institutional office determined whether this secondary
  analysis requires ethics/IRB review, is exempt, or does not require review.
- [ ] The written determination and any conditions are recorded. If review is
  required, approval is final and covers this use before data are obtained or
  processed.
- [ ] Faculty reviewed the sensitivity of pediatric EEG data and confirmed the
  permitted research-only scope.

Determination (approved / exempt / not required / pending): _______________

Office, reviewer, date, evidence reference: ______________________________

## Cohort, identity, and source-file review

- [ ] Human reviewer confirmed the exact twelve EDF paths and the train
  (`chb02`, `chb05`, `chb08`), validation (`chb09`), and test (`chb06`,
  `chb10`) patient assignments.
- [ ] Reviewer confirmed the `chb01`/`chb21` alias safeguard remains intact and
  neither alias is in this proposed cohort.
- [ ] Official per-patient SHA256 checksum files were obtained and each staged
  EDF, summary, and sidecar matches its official checksum.
- [ ] CHB-MIT age eligibility, EDF metadata, montage, channel mapping, units,
  sampling rate, and signal quality were reviewed for every recording.
- [ ] All twelve files are within the approved data-use and storage scope.

Cohort-manifest SHA-256: _________________________________________________

Reviewer, date, evidence reference: ______________________________________

## Annotation and non-seizure coverage review

- [ ] Each of the six proposed seizure recordings has its exact
  `.edf.seizures` sidecar hash recorded.
- [ ] Reviewer reconciled each sidecar's event count and onset/offset times
  with the corresponding summary; discrepancies were resolved and documented.
- [ ] Each of the six recordings without summary events was reviewed for its
  **entire** EDF duration; absence of summary events alone was not treated as
  seizure-free evidence.
- [ ] Annotation coverage begins at recording time zero and ends at the EDF
  duration for every recording.
- [ ] Annotation evidence records reviewer, date, summary hash, sidecar hash
  where applicable, reviewed intervals, and reconciliation reference.

Annotation-evidence manifest SHA-256: ___________________________________

Reviewers and evidence references: ______________________________________

## Storage and processing authorization

- [ ] External pilot root has a 5 GiB total cap and is outside the Git worktree.
- [ ] Free space is at least 20 GiB before each download/processing stage.
- [ ] Read-only readiness report has no blockers and is bound to the final
  cohort-manifest hash.
- [ ] Faculty has explicitly authorized processing the real recordings for
  feature extraction and experimental model training.
- [ ] The authorization scope, approver, date, and reference are recorded;
  approval does not claim clinical validation.

Readiness-report reference/date: ________________________________________

Processing approver/date/evidence reference: ____________________________

**Do not proceed to Checkpoint D3 while any required item is unchecked, any
decision is pending, or the readiness report contains blockers.**
