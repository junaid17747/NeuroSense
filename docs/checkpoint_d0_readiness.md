# NeuroSense V4 — Checkpoint D0 readiness report

Prepared 2026-10-11. **D0 planning is complete; the project is not ready to run
seizure-model training.** This report audits the current working tree and
proposes a protocol. Proposed components and settings below are not implemented
or validated merely because they appear here.

| Gate | Current status | Evidence required to advance |
| --- | --- | --- |
| Software foundation | Checkpoint C exists; 133 seizure-research tests passed in this audit | Independent real-file decoder verification, cohort-scale processing, feature/training/event/evaluation gaps below |
| Dataset approval | No approved real-data cohort or review packet was supplied for this run | Dataset-use authorization, reviewed identities/ages/channels/annotations, exact files and source hashes |
| Experimental training | Not run; no seizure-training command currently exists | Approved pilot, frozen protocol and splits, tested D1 implementation, separate authorization to train |
| Clinical validation | None | Not established by synthetic tests, dataset access, a fitted model, or retrospective held-out results; outside this prototype's intended use |

No clinical recordings were downloaded or processed, no model was fitted, and
no commit or push was made. Only this document was added in D0. Existing
uncommitted changes, neonatal research, webcam/rPPG, and EEG attention models
are preserved. The first proposed experiment is **offline child-population
CHB-MIT seizure-event research**, not an attention-model adaptation or a live
alarm. Other populations remain unavailable unless separately supported.

**1. Checkpoint C audit**

The audited implementation is linked below; findings concern the actual code,
not just the prior completion report.

| Area | Implemented and tested | D0 finding / remaining limitation |
| --- | --- | --- |
| EDF header and decoding | [edf.py](../seizure_research/edf.py), [waveform.py](../seizure_research/waveform.py): exact payload size, calibration, per-channel rates, indexed channel order, bounded half-open reads, voltage-to-microvolt conversion, file-change checks, private-header omission, traversal/symlink rejection | Only ordinary continuous ASCII-header, signed 16-bit EDF with positive known duration and blank reserved format field. EDF+/BDF/discontinuous files are rejected. The 61,440-byte record ceiling is project policy. Independent-reader comparison on authorized real files is still missing. |
| Approval/provenance | [base.py](../seizure_research/adapters/base.py), [chb_mit.py](../seizure_research/adapters/chb_mit.py), [ingestion.py](../seizure_research/ingestion.py): reviewed plan, source/summary hashes, full manifest revalidation, adapter version 2, all-or-nothing cohort approval | Human attestations are not signatures. The adapter parses summary text only, not `.seizure` files. One failed recording withholds the cohort; do not quietly train on a partial accepted subset. Older adapter-1 manifests need rebuilding, not hand-edited hashes. |
| Preprocessing | [preprocessing.py](../seizure_research/preprocessing.py): explicit population/configuration binding, reviewed channel mapping, microvolt samples, optional rate conversion, stable SOS filters, native quality assessment before transforms, independent usable segments, excluded edges | Zero-phase filtering is acausal. No tested cohort iterator, fixed recording-wide window grid, overlapping-chunk assembly or chunk-invariance protocol exists. These are training blockers, not permission to process each five-second training window independently. |
| Quality | [quality.py](../seizure_research/quality.py): flatlines, rail saturation, corrupt/nonfinite samples, large amplitude/jumps, sampling gaps, metadata issues, partial tails and union-based usable time | Thresholds are engineering examples. Seizure activity can itself trigger amplitude/jump rules. Review exclusion rates and excluded reference events without relaxing gates to improve test metrics. The report distinguishes raw quality, processed usability and decoded coverage. |
| Alignment | `align_reference_windows()` preserves source event IDs and half-open times; any overlap is currently positive; incomplete coverage is never silently negative; exclusions take precedence | Training-specific boundary ambiguity/occupancy policy proposed below is not implemented. Overlapping positive windows cannot be counted as separate reference events. |
| Splits | [splitting.py](../seizure_research/splitting.py): deterministic canonical-patient splits, duplicate source/hash checks, known alias rejection, windows inherit patient partitions; supplied processing splits are fully revalidated | Splitter uses patient-hash order, not age/event-burden stratification. It requires at least three patients but that is not an adequacy claim. Adding patients can change existing assignments; freeze cohort and split together. Standalone window assignment has no duration/processed-artifact verification. |
| Eligibility | [eligibility.py](../seizure_research/eligibility.py): no registered seizure model; unsupported populations/acquisition combinations and mixed rates without matching evidence fail closed | Training permission and dataset suitability are separate from permission to use a finished model. Training success must not automatically create reviewed eligibility evidence. |
| Evaluation | [evaluation.py](../seizure_research/evaluation.py): complete held-out recording coverage, decision-time matching, false alarms/hour, missed events, latency, patient/age summaries | It scores decision timestamps, not candidate intervals. It has no processing-derived evaluation domain, duration/occupancy metric, interval matcher or patient bootstrap. `require_research_ready()` rejects reference events intersecting stored quality exclusions. |
| CLI/dependencies | [__main__.py](../seizure_research/__main__.py) offers metadata inspection/ingestion; NumPy, SciPy, XGBoost and scikit-learn are installed | No seizure-specific feature extractor, training CLI, model-artifact contract or probability-to-event converter exists. Existing attention-training scripts must not be used. |

Two scale limits matter immediately. The decoder defaults to 2,000,000 aggregate
samples, with a hard 10,000,000-sample ceiling. One hour of 18 channels at 256 Hz
contains 16,588,800 samples, exceeding even the ceiling. Bounded chunks are
required. Also, `process_approved_recording()` invokes `validate_manifest()`
for the entire cohort on **every call**, then `read_edf()` hashes the selected
file again. This is suitable for a small integrity check, not an efficient
whole-cohort training extractor.

**2. Official source, terms and provenance**

The pinned source is [PhysioNet CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/),
DOI `10.13026/C2K01R`. It lists open access subject to ODC-By 1.0, 42.6 GB
uncompressed, 664 EDF files, 129 seizure-containing files and 198 seizures.
Its original-cohort description gives 23 cases from 22 subjects, ages 1.5–22;
the later `chb24` addition lacks a `SUBJECT-INFO` entry. These differently scoped
counts must not be treated as an approved local cohort count. `chb21` and
`chb01` are one person recorded 1.5 years apart. The release describes 256 Hz,
16-bit recordings, usually 23 EEG signals, with montage variation, dummy and
occasional auxiliary channels, inter-file gaps and surrogate dates. It provides
summary-text and `.seizure` annotations. Do not infer recording-specific age or
annotation completeness from a filename, surrogate date or missing sidecar.
The source requests dataset, original-work and PhysioNet citations.

The [release's actual ODC-By 1.0 text](https://physionet.org/content/chbmit/view-license/1.0.0/)
covers database rights. Sections 4.2–4.3 require appropriate license/notices when
publicly conveying a database/derivative or publicly using a produced work.
Sections 2.3–2.4 distinguish other rights, including rights in individual
contents and privacy. Review intended use, attribution and any redistribution
with the responsible institution; this report does not grant those rights.
The release does not advertise credentialed access or a signed PhysioNet DUA
as a prerequisite. NeuroSense's human-review fields and institutional ethics
determination are **project requirements**, not additional license terms being
attributed to PhysioNet.

Source metadata identifies clinical contributors, but a public annotation file
does not independently establish every recording's label quality. A reviewer
must document annotation provenance, timing conventions and complete coverage,
and resolve disagreements. Do not fabricate expert review or silently label
an unannotated recording seizure-free. No subject/recording has been selected
or approved by this report.

**3. Exactly what the user must approve or provide**

The concrete prerequisite is an external review packet and an explicit scope of
work. A chat message saying “approved” cannot substitute for missing evidence.
Do not put patient identifiers, raw EDFs or sensitive review notes in Git.

| User/reviewer action | Required evidence/input | Existing contract |
| --- | --- | --- |
| Authorize a specific offline pilot | Purpose, dataset/release, permitted file list, processing scope, responsible reviewer; separate future authorization if acquisition or model fitting is wanted | D0 authorizes neither downloading nor fitting; preserve this boundary |
| Record access, terms, privacy and ethics review | Verified right to possess/use the selected files; attribution/distribution plan; de-identification review; institutional approval, exemption or documented not-required determination as applicable | `permissions.status="approved"`, `deidentified=true`, `terms_review_ref`, `ethics_review_ref`, `access_review_ref`, `reviewed_by`, `reviewed_on`; `dataset_review.verified=true` with reviewer/date/evidence |
| Provide authorized local files | Absolute dataset root outside Git; normalized `subject/filename.edf` and `subject/subject-summary.txt`; pinned release, byte sizes and reviewed SHA-256 values; provider checksum comparison where available | `edf_path`, `summary_path`, `expected_signal_sha256`, `expected_summary_sha256`; source identity is revalidated |
| Review patient identity | Explicit `subject_id → canonical_patient_id` mapping, all aliases/sessions, evidence reference | Verified `identities.review`; include both `chb01` and `chb21` with the same canonical ID whenever either is included |
| Verify age at each recording | Chronological age with source and precision sufficient for the intended age category; resolve rounded/boundary-crossing ages | `age.completed_years`, `age.source`; do not derive age by adding a presumed interval to a surrogate date; defer unresolved records |
| Review acquisition metadata | Actual device manufacturer/model, acquisition software version and clinical-EEG status, with evidence | `device` and verified `metadata_review`; these facts may not be recoverable from the public landing page. Do not insert invented values or use an `unknown` string to pretend they were verified |
| Review every channel | Source index/label; keep or exclude with reason; unique output label and alias reason; units, montage/reference and expected rate, including each deliberate rate override | `channels`, `montage`, `expected_sampling_rate_hz`; exclude non-EEG/dummy signals explicitly; do not pick a repeated label by convenience |
| Review annotation provenance/completeness | Protocol and provenance references, annotator source, expert review evidence; cross-check summary timing/counts with supported source evidence, including `.seizure` review when needed | Verified `annotation_review`, `annotation_coverage="complete"`, `annotation_provenance_ref`, `annotation_protocol_ref`, `annotator_ref`; explicit reviewed zero-event count for seizure-free files |
| Review initial signal suitability | Source-level quality assessment with evidence and any known exclusions; independent sample/calibration comparison on a bounded authorized read | `quality.status="acceptable"`, `quality.review_ref`; unresolved events touching known exclusions currently block ingestion |
| Approve the scientific protocol | Population/channel set, filters, quality rules, windows/labels, split seed, event rules, resource cap, validation operating-point objective and test freeze | Versioned proposed experiment/configuration; `research_review_ref` is required for preprocessing but is not clinical validation |

No passwords or patient names are needed. Use pseudonyms and non-sensitive
evidence references. Public source access alone does not verify the fields
above. If device/software provenance or recording age cannot be established,
keep the recording blocked; proposing an explicitly unknown-metadata research
schema would require a separately reviewed change, not false entries here.

An authorized inspection phase can establish initial quality before cohort
acceptance: inspect limited source waveforms under verified data-use permission,
record the review, then ingest. That permission does not itself approve the
cohort or enable a model. Current low-level `read_edf()` is a decoder, not an
access-control system; use it only inside the authorized scope.

**4. Patient-independent study design**

The first proposed model is `chbmit_child_xgb_v1`, limited to verified ages
1–12 and one reviewed montage/acquisition/preprocessing profile. This is a
scope choice, not evidence of pediatric generalization. Do not pool adult or
adolescent recordings merely to increase sample size. A future adolescent
experiment needs its own reviewed protocol; the neonatal track is unchanged.

1. Define the eligible cohort using permissions and prespecified metadata and
   quality criteria before viewing model performance. Include whole approved
   recordings with seizures and confirmed seizure-free recordings. Document
   every rejected file, excluded duration and affected reference event.
2. Canonicalize patients before splitting. `chb01`/`chb21` count as one patient,
   even if their ages differ or one session falls outside this experiment.
   No session, file, channel, overlapping window or alias may cross partitions.
3. Freeze cohort contents and hashes. Use the existing deterministic split,
   seed **42**, target **60% train / 20% validation / 20% test** by patients.
   Actual counts follow its integer rounding; publish patients, recordings,
   hours, positive/negative windows and reference events in every partition.
4. Review counts before development: each partition must have both usable
   classes and reference events; validation/test must include reviewed
   seizure-free exposure. The current splitter does not enforce this. If it
   fails, stop and revise the cohort/split design under a new documented
   protocol before any fitting; do not search seeds for favorable results.
5. Train on training patients only. Validation patients select boosting length
   and the operating threshold. Test labels and test features remain sealed
   from the training/selection process. No subject-specific tuning on held-out
   patients and no random row/window split or ordinary row-wise cross-validation.
6. Freeze weights, best iteration, features, thresholds, event rules and
   evaluation protocol before the first test evaluation. Do not refit using
   validation patients after selecting the first baseline's threshold; that
   changes the score distribution. Any later refit needs a new calibration and
   evaluation plan. A repeatedly inspected test set becomes development data.

No scaler or imputer is needed for this tree baseline. Fixed unit conversions
and mathematical feature constants are configuration, not fitted statistics.
If a later experiment learns normalization, channel selection, imputation or
feature selection, fit it only inside training patients (and training folds
for cross-validation), save provenance and apply it unchanged elsewhere.
Age or coverage tables may be audited before the split is frozen, but neither
test labels nor quality rejection rates may drive post hoc threshold changes.

**5. Proposed preprocessing and seizure-specific features**

Use reviewed 256 Hz native signals without resampling for the first pilot;
preserve bipolar derivations with `reference_mode="none"`. Propose the existing
child engineering bandpass, 0.5–40 Hz, fourth-order Butterworth SOS, subject to
EEG-reviewer acceptance. Do not silently carry the convenience factory's
50 Hz notch into CHB-MIT: start with an explicitly recorded `line_noise_hz=null`
for this 40 Hz low-pass baseline, review spectra, and change only under a
versioned training-side protocol. No source-specific mains frequency has been
verified in this audit.

Require a fixed reviewed set of 18 bipolar derivations if supported by the
selected files: four four-channel chains plus two midline derivations. Candidate
labels are `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, `FP1-F3`, `F3-C3`, `C3-P3`,
`P3-O1`, `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`, `FP2-F8`, `F8-T8`, `T8-P8`,
`P8-O2`, `FZ-CZ`, `CZ-PZ`. This is a proposed canonical feature order, **not a
claim that every source file has these unique channels**. Human-reviewed
mapping must settle labels, polarity and duplicates. If absent, reject the
profile or design a smaller fixed profile on development data before freezing
the study; do not zero-fill missing channels or change feature order per file.

Proposed windows are **5 seconds with a 1-second step**, anchored to recording
sample zero. Use fixed global window IDs and generate each once. Implement
bounded reads of roughly 300 seconds including context, with a halo at least
the actual filter/resampler edge guard; retain only owned core windows. Compare
against a larger authorized reference read before accepting the chunk policy.
Never bridge independent EDF files or quality gaps, and do not invent inter-file
continuity. Chunk processing can change quality coverage decisions; it must be
verified for reproducibility rather than assumed equivalent to whole-file work.

Proposed feature schema `seizure_eeg_features_v1` uses **18 features per channel**:

| Family | Count/channel | Exact proposed definition |
| --- | ---: | --- |
| Amplitude/activity | 1 | `ln(1 + variance(x)/(1 uV^2))`, population variance |
| Line length | 1 | `ln(1 + fs * mean(abs(diff(x)))/(1 uV/s))` |
| Hjorth parameters | 2 | Mobility `sqrt(var(dx/dt)/var(x))`; complexity `mobility(dx/dt)/mobility(x)` using fixed finite differences |
| Distribution | 1 | Excess kurtosis; pin bias correction and handling of degenerate variance |
| Spectral power | 5 | `ln(1 + band_power/(1 uV^2))` in 0.5–4, 4–8, 8–13, 13–30, 30–40 Hz |
| Relative spectral power | 5 | Each band power divided by integrated 0.5–40 Hz power |
| Spectral organization | 3 | Normalized spectral entropy; 50% and 95% cumulative-power frequencies over 0.5–40 Hz |

Define Welch PSD with an explicit periodic Hann window, 2-second segments,
1-second overlap, constant detrending, density scaling and mean averaging;
pin integration/band-edge interpolation and the cumulative-power convention.
These settings are a proposed engineering specification, not measured efficacy.
SciPy documents the underlying [Welch PSD API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
Use fixed small numerical floors with units in the feature specification;
degenerate/nonfinite features are explicit unavailable windows, not imputed
seizure-free examples. No fitted per-record z-scoring.

Concatenate 18 channels × 18 features = 324 features. Add median, maximum and
IQR across the fixed channel set for each feature: 54 more, **378 total**.
Store numeric rows as float32 after finite/range checks; retain timestamps,
window IDs, labels, quality and provenance separately. Validate features on
known synthetic sinusoids, amplitudes and time shifts, then authorized EEG.
Do not reuse `src/features` attention ratios, synthetic attention labels,
attention-model weights or their performance claims. These features may capture
changes relevant to seizures, but can also respond to artifacts; only actual
held-out experiments can establish discrimination.

**6. Labels, imbalance, exclusions and annotation coverage**

The proposed training label rule is separate from Checkpoint C's existing
any-overlap reference alignment. Preserve the original alignment for audit
and add a versioned D1 occupancy rule; do not silently alter C.

| Window condition | Proposed supervised-training treatment | Validation/test event stream |
| --- | --- | --- |
| Complete annotations; seizure overlap covers at least 50% of window | Positive | Score normally |
| Complete annotations; no seizure overlap | Negative | Score normally, retaining natural background exposure |
| Complete annotations; overlap is greater than zero but less than 50% | Boundary-ambiguous; exclude from fitting and binary window metrics | Still score and retain for event matching; never delete the underlying reference event |
| Unknown/partial annotation coverage | No negative inference; not admitted to this approved cohort | No scientific scoring claim; current ingestion rejects the recording |
| Any quality exclusion, filter edge, missing signal or invalid feature | Excluded; no fit row and no probability | Explicit abstention; terminate/reset event state across the gap, retain duration and affected-event accounting |

Union multiple event overlaps for occupancy, while retaining their distinct
IDs. Very short events may have no majority-positive windows; report this
explicitly and retain them in event-level evaluation. If positive training
examples disappear under this policy, fail the protocol adequacy check rather
than treating those events as negatives.

Keep all eligible training windows for the first small pilot. To reduce patient
duration dominance, assign base weight `1 / eligible_window_count_for_patient`
and rescale so mean training weight is one. Compute the positive multiplier
as **sum of negative base weights / sum of positive base weights**, using
training rows only. Use that value as `scale_pos_weight`, alongside the base
sample weights; do not additionally rebalance with duplicated positives or
SMOTE. Fail if either class is absent. This patient-weighted choice is a
proposal; record effective class weights and each patient's contribution.

Validation and test retain all eligible time, natural prevalence and all
reviewed seizure-free recordings. Do not balance their windows. If larger
training data later needs background subsampling, version a deterministic
per-patient/time-block sampler and inclusion probabilities, recalculate weights
on the retained training set, and leave validation/test untouched. Class-weighted
logistic scores are not automatically calibrated clinical probabilities.

Quality exclusions must be label-independent and frozen before test results.
Publish rejection reasons, usable hours, all reference events, events partly
or wholly excluded and per-patient coverage. Do not quietly remove difficult
seizures to improve sensitivity. Extending the evaluation-domain contract for
such events is a prerequisite; copying processing exclusions into the current
`EEGRecording.quality` can violate its event-overlap guard.

**7. XGBoost baseline and reproducibility**

Proposed initial `XGBClassifier` configuration, independent of all attention
artifacts: `objective="binary:logistic"`, `tree_method="hist"`, `device="cpu"`,
`n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `min_child_weight=5`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5`, `reg_alpha=0`,
`max_bin=256`, `random_state=42`, `n_jobs=4`, `eval_metric="aucpr"`,
`early_stopping_rounds=30`, with the training-derived class multiplier above.
These values are starting hypotheses, not optimized parameters.
The official [parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html)
documents the histogram method and weighting controls.

Use validation patients, never final-test patients, in `eval_set`. Save the best
iteration and verify that model reload predicts with that same tree range;
the [estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
documents early stopping and its prediction behavior. Run one prespecified
configuration for the pilot, not an undisclosed search. Subsequent tuning
requires patient-grouped development folds or a clearly recorded validation
budget; validation scores used for selection are not unbiased final results.

Installed/pinned packages checked here: XGBoost 3.4.1, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, Pydantic 2.13.5. Current upstream XGBoost documentation
identifies 3.4.2; do not upgrade the project implicitly. Test the proposed
interface against the pinned environment when implementing D1.

Save native XGBoost JSON/UBJ rather than loading arbitrary pickle/joblib
artifacts. Persist the feature schema/order, complete training config,
canonical-patient split hash, cohort/annotation/processing hashes, row IDs and
training weights, label/event/evaluation versions, best iteration, validation
selection trace, seed, software versions, thread count and code snapshot hash.
A Git commit ID alone is insufficient while the working tree is dirty.
Record and verify the exact source-file content manifest. A seed is not a
promise of bit-identical results across different runtimes or hardware.

**8. Window scores to candidate events and validation-only selection**

The following finite-state policy is proposed and must be implemented/tested
before training results are interpreted:

1. Process each recording's available window scores in increasing right-edge
   time. Apply a trailing median of three consecutive valid scores; abstain
   during warm-up. Never smooth across missing windows, quality gaps or files.
2. Open an episode after two consecutive smoothed scores at or above
   `theta_on`. Store the right edge of the second qualifying window as
   `trigger_sample_seconds`. Separately retain the supporting window interval;
   do not backdate the trigger to its left edge.
3. Continue with hysteresis `theta_off=max(0.01, theta_on-0.10)`. Close after
   three consecutive smoothed scores below `theta_off`; the estimated support
   offset is the end of the last window before that below-threshold run.
   Mark file-end or gap closures as censored where appropriate.
4. Merge episodes separated by at most five seconds only when the intervening
   interval is fully covered. Retain the earliest trigger and all constituent
   episode IDs. Never merge across EDF boundaries or excluded time. Merging is
   retrospective and must not move the original trigger earlier.
5. Keep the smoothing, persistence, hysteresis and merge parameters fixed for
   the first pilot. Evaluate `theta_on` in `{0.10, 0.15, ..., 0.90}` on validation
   patients only. A proposed research operating target is at most **1 false
   event per evaluable hour**, with alert occupancy at most **10%** of evaluable
   time to prevent an always-on solution. These are reviewable experiment
   constraints, not clinically acceptable alarm-rate claims.
6. Among feasible thresholds maximize macro patient sensitivity over validation
   patients with reference events. Break ties by lower pooled false events/hour,
   then higher threshold. Report the complete threshold curve, pooled and
   patient results. If no threshold is feasible, or all feasible thresholds
   detect zero events, report a failed baseline; do not invent a successful
   operating point or consult test results to change the rule.

Freeze the chosen threshold and all event rules before test access. No
per-test-patient threshold, test-set early stopping or prevalence adjustment.
If score calibration is later added, use a separate development calibration
design or grouped out-of-fold predictions, not the final test patients.

Checkpoint C uses `sosfiltfilt` within entire usable segments. A window can
therefore depend on samples after its right edge. Retain the maximum source
time consulted and actual offline processing timing as provenance.
`trigger_sample_seconds - reference_onset` is **retrospective EEG-time trigger
alignment**, not prospective detection latency. True causal latency requires
a separately tested causal/bounded-lookahead pipeline, buffer/compute timing
and a new protocol. Do not subtract an estimated phase delay to claim real-time
performance from the current acausal pipeline.

**9. Event evaluation specification**

Define a new versioned interval-and-trigger protocol; preserve the current
decision-only evaluator for its existing tests. For each recording, sort
candidate episodes by trigger time and stable ID. A candidate is eligible to
match a reference only if their half-open intervals overlap and its original
trigger lies in `[reference_onset, reference_offset)`. Match the first eligible
candidate to that as-yet-unmatched reference, once. All unmatched candidates,
including duplicate episodes within one reference, count as false events.
One long episode cannot detect multiple seizures. Early/late support overlap
without an in-event trigger is unmatched under this strict baseline; do not
add a tolerance after viewing errors. Report onset/offset localization as
secondary signed errors rather than modifying the primary match rule.

| Metric | Prespecified calculation/report |
| --- | --- |
| Event sensitivity | Matched reference events / all reviewed reference events in the scheduled approved evaluation recordings; null if no reference events |
| Missed events | Total references minus matched references, with event IDs; separate classifier misses from events with partial/full abstention |
| False events per hour | Unmatched candidates / union of evaluable seconds × 3,600. Evaluable time includes seizure-free recordings and ictal time, excludes genuine absent/abstained coverage, and is counted once despite overlapping windows. Zero exposure gives null |
| Detection timing | Per-matched-event `trigger_sample_seconds - onset`, median/IQR and distribution; explicitly retrospective alignment with the current filter. Record unavailable end-to-end causal latency as unavailable, not zero |
| Patient results | Counts, reference/detected/missed events, false events/hour, timing, scheduled/evaluable/excluded hours and reasons; include patients with no seizures or no evaluable time |
| Pooled/macro results | Sum counts/time for pooled rates; separately average defined patient sensitivities, stating how many patients contribute. Do not substitute window accuracy |
| Coverage and occupancy | Evaluable/scheduled time, affected reference events, percentage of evaluable time inside predicted episodes, long episodes and boundary closures |
| Secondary discrimination | Window AUPRC/AUROC only where the binary window labels are defined; no conversion of these values into event sensitivity |

For transparent abstention accounting, primary sensitivity retains references
that were not detected because their signal was excluded; label the outcome
as unavailable/undetected due to exclusion rather than a confident negative.
Also report a secondary sensitivity restricted to fully evaluable references,
with its reduced denominator. Exclusions must not hide events from the primary
inventory. This requires the new evaluation-domain contract; the current
schemas intentionally reject some overlapping-event exclusions and cannot
implement this policy unchanged. If reference completeness is uncertain, do
not score that recording at all—report the cohort rejection.

Unavailable probabilities must be explicit, not absent rows or numeric zeros.
After a gap, smoothing/persistence warm-up contributes to abstention time. The
evaluation-domain implementation must distinguish assessed negative intervals
from warm-up, missing chunks and excluded quality; the denominator cannot be
derived from just “number of windows × window length.”

Use patient-cluster bootstrap intervals only when cohort size supports useful
uncertainty reporting; resample patients with all their events/time, not windows.
The six-patient pilot below is an integration check, not a powered validation
study. Report individual counts and insufficient evidence instead of precise
generalization claims. A child-only study supplies no adolescent/adult/neonatal
validation. No result in this report is a measured model metric.

**10. Resource estimates, with assumptions instead of benchmarks**

The release's 42.6 GB is a provider-listed full-download size, not an instruction
to acquire it. The scenarios below use 23 stored channels at 256 Hz/int16,
18 processed channels, 5-second windows at 1-second stride and 378 float32
features. Actual file/channel counts, durations, exclusions and compression
change these figures. GB is decimal; no runtime benchmark was performed.

`raw_bytes = hours * 3600 * stored_channels * 256 * 2`.
For separate one-hour files, `windows_per_hour = floor((3600-5)/1)+1 = 3596`.
`dense_feature_bytes = windows * 378 * 4`; metadata, weights, predictions,
indices, matrix copies and XGBoost structures are additional.

| Illustrative workload | Raw EEG payload | Dense feature values | If all processed 18-channel float64 waveforms were cached |
| --- | ---: | ---: | ---: |
| 12 hours, e.g. 12 one-hour files | 0.509 GB | 0.065 GB / 43,152 windows | 1.593 GB |
| 100 hours | 4.239 GB | 0.544 GB / 359,600 windows | 13.271 GB |
| 1,000 hours (size scenario, not verified eligible cohort duration) | 42.394 GB | 5.437 GB / 3,596,000 windows | 132.710 GB |

Avoid a full waveform cache. Persist approved raw files once, compact features
and provenance outside Git, and small reviewed plots. Allow roughly 2–3 times
the feature-value size for tabular artifacts, metadata and temporary copies;
this is a planning allowance, not measured format overhead. Reserve **5 GB**
free disk for a 12-hour pilot without a full waveform cache. A full-release
workspace could reasonably reserve **100 GB** free, or **150 GB** if keeping
both an archive and extracted raw files; caching all processed float64 samples
would require substantially more. File selection may make the pilot longer
than 12 hours, so calculate from its actual manifest before authorization.

For a 300-second read of 18 channels, there are 1,382,400 selected samples.
The decoder's int16 digital + float64 physical + per-channel float64 timestamps
alone occupy **23.73 MiB**; filter/resampler matrices, output, masks, report
objects and feature buffers increase peak RSS. Budget **0.25–1 GB per
processing worker** until measured, initially one worker. A 300-second read
of 23 channels is 1,766,400 samples and fits the default bound; account for the
halo inside the requested read rather than appending unlimited context.

For the pilot, target **4 CPU threads and 8 GB RAM minimum, 16 GB preferred**.
No GPU is necessary for the proposed histogram baseline. For a million-hour-
scale typo prevention: the large table scenario is **1,000 hours**, not a
million hours. Its training portion alone can need several GB of dense
features; target **32 GB RAM** for an in-memory larger study, verify RSS, and
use a separately tested external-memory workflow or smaller authorized scope
if needed. Avoid nested process/thread oversubscription.

Training-time planning estimate, **not measured**: for the approximately 43k-row
pilot (fewer rows in training), 378 features, depth 3, at most 300 CPU trees and
four threads, allow **1–30 minutes for a single fit plus validation scoring**.
A multi-million-window run might require **0.5–6 hours** under adequate RAM;
these deliberately broad engineering estimates exclude acquisition, feature
extraction, hashing and human review, and may be exceeded. Do not treat them as
a speed claim for this machine or as a scheduling guarantee.

The current repeated-validation path costs approximately
`number_of_chunks * cohort_bytes` in cohort checksum reads, plus repeated
selected-file hashes. Even a 12-file, 12-hour/0.509-GB scenario with 144
five-minute calls scans about **73 GB** at the cohort layer before extra
reads. OS caching does not remove hashing work. Fix the orchestration before
extrapolating runtime. Once separately authorized, measure a training-side
pilot's wall time, RSS, bytes read, features/second and time/boosting round;
forecast the larger job from those measurements with explicit headroom.
This run performs no timing experiment that fits a model.

**11. Minimum approved real-data pilot and exit criteria**

Start with a staged pilot, not the full release:

1. **Decoder/annotation smoke check:** one approved patient, one complete
   seizure-containing EDF and one explicitly reviewed seizure-free EDF, their
   complete case summary and provenance/checksums. Compare bounded samples
   from the beginning/middle/end and near seizure boundaries with an independent
   EDF reader. Check all selected channels, calibration endpoints, polarity,
   sample timestamps, units and event offsets. This validates no model and
   cannot satisfy a patient-disjoint three-way experiment.
2. **Minimum operational training pilot:** at least **six distinct canonical
   patients** in the chosen child profile, each contributing at least one
   seizure-containing recording and one confirmed seizure-free recording.
   Prefer complete ordinary EDFs with a common reviewed montage. If both
   alias cases are present they still count as one patient. No exact filenames
   are nominated without inspecting authorized metadata. At one hour/file this
   is about 12 hours; actual source durations may be longer.
3. The existing 60/20/20 splitter gives **3 train / 1 validation / 2 test** for
   six patients. Require usable positives and negatives in all partitions and
   audited seizure-free exposure. One validation patient is statistically weak:
   call this an engineering pilot, not population validation. At least three
   patients meets only the splitter's mechanical minimum and is not the
   recommended training pilot.
4. Freeze the processing/label/event policy on development records; document
   rejected files, retained hours and events, and independent-reader comparison
   tolerances. Verify chunk stitching/quality coverage, invariant feature order,
   source hashes, alias grouping and model reload behavior. Only after code,
   permission and review gates pass should fitting be separately authorized.

Pilot test patients must not be called an untouched final cohort after their
results or waveforms have been used to debug or choose methods. Keep that pilot
as development evidence and reserve new patients for a later genuine held-out
study. No number of files alone guarantees adequate seizure diversity, subgroup
coverage or statistical power.

**12. Prioritized implementation gaps and proposed command**

| Priority | Gap to close before the first interpreted training run | Required acceptance evidence |
| --- | --- | --- |
| P0 — data | Missing actual review packet, authorized cohort, device/age/annotation evidence and real-file verification | Reviewed manifest and independent decoder/annotation comparison; unresolved fields remain blocked |
| P0 — scale/integrity | No efficient validated cohort reader or tested chunk iterator | Validate cohort once per immutable run, bind cached reads to approved bytes/handles, detect replacements; fixed global windows, halo ownership, no cross-gap filtering, no duplicate rows; do not replace cryptographic validation with an unchecked path cache |
| P0 — features/labels | No dedicated seizure features, occupancy labels or feature manifests | Synthetic known-signal tests, reviewed real examples, fixed units/order, explicit ambiguous/excluded labels, repeatable hashes and partition-specific files |
| P0 — experiment | No seizure training CLI or training-only weight/statistics boundary | Reject non-approved/mismatched manifests; fail absent classes and test partition access; record dirty-tree snapshot, pinned runtime and seeds; native artifact reload reproduces scores |
| P0 — events/evaluation | No stateful event converter or processing-derived evaluation domain | Deterministic threshold/hysteresis tests; first/last event, gaps, short events, duplicate alarms, always-on output, null coverage and excluded references; one-to-one matching; exact exposure accounting; validation-only selection |
| P1 — scientific robustness | Small-patient uncertainty, model calibration, alternative features and generalization remain untested | Grouped development protocol, patient uncertainty when justified, external evaluation later; no clinical inference from pilot scores |

No P0 gap was implemented in D0. Proposed future invocation, **not executable
today because `seizure_research.training` and these arguments do not exist**:

```sh
# PROPOSED D1 interface only — do not run in D0.
rtk proxy .venv/bin/python -B -m seizure_research.training train-xgboost \
  --development-manifest /authorized/neurosense/d1/development_features_manifest.json \
  --split-manifest /authorized/neurosense/d1/patient_split.json \
  --experiment-config /authorized/neurosense/d1/chbmit_child_xgb_v1.json \
  --output-dir /authorized/neurosense/d1/runs/chbmit-child-xgb-v1
```

The development manifest must contain **train and validation feature shards
only**, each traceable to the approved cohort, annotation and processing
artifacts. The split manifest may enumerate all patient partitions but the
trainer must reject test rows and have no test-feature paths. The experiment
JSON must carry the exact settings in this report, hashes, human review
references and resource bounds; placeholders above are not approvals.
No output overwrite or model registration by default. Final test evaluation
must be a separate frozen-artifact command, implemented later. Keep all inputs
and experiment artifacts outside this Git checkout.

**13. D0 verification and next decision**

This audit reran only the synthetic seizure-research suite:

```sh
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_seizure*.py' -v
```

Actual result: **133 passed, 0 failed, 0 skipped**, in 1.538 seconds. The prior
Checkpoint C record of **220 Python and 74 JavaScript tests** remains historical
evidence; D0 did not rerun those full suites or claim a new full-suite result.
Dependency versions were inspected without fitting anything. Existing project
file hashes were captured before the audit and checked after the report.

The next concrete action is to assemble the review packet in section 3 and
authorize the bounded two-recording source-validation pilot, alongside a
separate D1 implementation request for the P0 software gaps. Approving data
access is not approving training; approving an experimental fit is not
clinical validation. Keep all model use unavailable until the relevant
research evidence and acquisition/population compatibility are reviewed.
