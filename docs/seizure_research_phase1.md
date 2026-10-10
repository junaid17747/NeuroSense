# NeuroSense V4 Phase 1: offline seizure research foundation

This package defines metadata, eligibility, dataset discovery, patient splits,
and scoring contracts for **offline, clinically annotated EEG research**. It
contains no seizure detector, fitted seizure weights, signal importer, training
pipeline, live inference, medical alert, or neonatal consumer monitoring.
`REGISTERED_MODELS` is empty. No clinical dataset was downloaded or approved.
The attention classifier is not a seizure detector and is not used here.

For the subsequent local ingestion adapter and source-review workflow, see
[Phase 2: clinical EEG ingestion](seizure_research_phase2.md). The Phase 1
contracts and verification results below describe the original foundation.

## Repository audit and separation

The repository was clean before this work. Relevant existing architecture:

| Existing area | Findings and Phase 1 boundary |
| --- | --- |
| `src/preprocessing/` | EEG V1 applies a 50 Hz notch and 1–40 Hz bandpass to 1-D signals; other attention paths use a bandpass. These filters and assumptions are not imported into seizure research. |
| `src/features/`, `src/model/` | Attention features, synthetic attention datasets, EEG V1/V2 training, calibration, subject manifests, and evaluation already exist. Their labels and synthetic performance provide no seizure validation evidence. |
| `backend/services/` | The existing ML service loads the synthetic attention classifier; sensor inference expects 1024 samples and uses the default 256 Hz processing path. The new package does not call or modify either service. |
| `backend/security/`, `backend/services/auth_service.py` | Argon2 passwords, opaque session tokens with stored digests, expiry, revocation, and user authentication remain unchanged. |
| `backend/storage/` | SQLite holds users, auth sessions, EEG sessions, and user-scoped sensor records. No seizure tables, migrations, writes, or access-control changes are introduced. |
| `backend/api/main.py`, `dashboard/` | Existing FastAPI routes, Streamlit navigation, webcam alignment/behavior, and authenticated sessions have no seizure integration. |
| `tests/` | Python uses standard-library `unittest`; storage tests use temporary databases. JavaScript webcam tests use Node's test runner with mocked camera/detector inputs. |

`seizure_research/` is a separate top-level package. It uses Python's standard
library and the repository's existing Pydantic 2 dependency; requirements are
unchanged. Importing it loads no application services, sensors, model artifacts,
or database connections. rPPG and multimodal synchronization remain unimplemented.

## Provisional age contract

These are project research categories, not a universal clinical taxonomy:

| Group | Chronological age at recording |
| --- | --- |
| Neonatal | 0–28 completed days, inclusive |
| Infant | 29 completed days until the first calendar birthday |
| Child | 1–12 completed years, before the 13th birthday |
| Adolescent | 13–17 completed years, before the 18th birthday |
| Adult | 18–64 completed years, before the 65th birthday |
| Older adult | 65+ completed years |

`AgeAtRecording` requires a metadata source and nonnegative integer completed
calendar years. Below one year, completed days are also mandatory (0–365,
allowing a leap-year birthday interval). At one year or older, only completed
years are stored. Missing age and year-only ages below one are rejected. Unknown
or imprecise ages that cross a boundary must be resolved outside this contract;
they never default to adult. Do not round fractional ages up to a birthday.

`from_dates()` uses verified calendar dates transiently and does not store the
birth date. Its explicit Feb 29 convention is a March 1 anniversary in non-leap
years. Do not derive age from shifted/surrogate EDF dates. A subject's age may
change between recordings while its patient group remains the same.

**Neonatal EEG requires specialized interpretation and cannot be evaluated
using adult model assumptions.** Chronological age alone does not describe
neonatal maturation: gestational age at birth and postmenstrual age at recording
have optional explicit fields that must be assessed in a neonatal protocol.
Neonatal validation also requires a neonatal specialist review reference. The
Helsinki source describes term neonates with expert annotations; that does not
establish performance for preterm infants or a consumer device. See the
[Helsinki dataset record](https://zenodo.org/records/4940267).

## Metadata and permissions

`schemas.py` provides immutable, extra-field-forbidden typed contracts:

- `EEGRecording`: schema version, global recording ID, dataset subject ID,
  globally curated patient group, identity-review reference, age, recording
  duration, and a versioned preprocessing identifier.
- `ChannelMontage`: montage name/kind, exact ordered channel labels, reference,
  and amplitude units. No silent channel reordering, re-referencing, scaling,
  resampling, or substitution occurs.
- `RecordingDevice`: manufacturer, model, acquisition software version, and
  explicit clinical EEG status. Unknown device metadata remains ineligible
  until resolved; a device name is not evidence of clinical validation.
- `DatasetProvenance`: dataset/release, source URL/recording ID, signal and
  annotation SHA-256 digests, and annotation-source reference.
- `SeizureAnnotation`: half-open onset/offset in seconds from recording start,
  event ID, seizure type, annotator reference, and clinical-review reference.
  Empty annotations are explicit; they are not inferred from missing labels.
- `RecordingQuality`: review state/reference, notes, and nonoverlapping excluded
  intervals. This is a reviewed assessment, not an automatic quality detector.
- `ResearchPermissions`: unknown/denied/approved status, de-identification, and
  references to access, terms, and ethics determinations, reviewer, and date.
  Approval requires every reference; unknown/denied status blocks analysis.

Permission records document a decision; they neither grant permission nor prove
compliance. An ethics determination reference may document an authorized
institutional finding that further review is unnecessary; the code does not
make that finding. Existing user authentication does not authorize clinical
dataset use. Clinical approval is not inferred from public download links.

Keep clinical recordings and identity mappings in an authorized external data
location, not in source control, this package, the attention CSVs, or the runtime
SQLite database. Use pseudonyms and references rather than names, birth dates,
or identifying clinical notes. No raw data persistence is added in Phase 1.

Schemas verify hash **format**; no file is read to verify the supplied hashes.
A future importer must calculate and verify them from the actual signal and
annotation bytes and preserve the source release. Hashes identify artifacts,
not their authenticity or clinical correctness.

## Model eligibility

`assess_eligibility(recording, model)` checks a single explicitly supplied model
specification and returns rejection reasons. It does not select, load, train,
or run a model. It revalidates metadata on entry, including Pydantic objects
constructed through unchecked copy operations.

An offline eligibility pass requires all of the following:

1. Complete metadata, approved research use, a clinical EEG device, reviewed
   acceptable quality with positive usable duration, and complete clinical
   annotation coverage. Reference events cannot intersect quality exclusions.
2. The exact age group in the model's declared supported groups.
3. An exact jointly supported acquisition profile: montage (including order and
   units), device/software, sampling rate, and preprocessing ID. Individually
   supported fields cannot be combined into an unvalidated profile.
4. Reviewed evidence for that **age × acquisition profile × model SHA-256**,
   including dataset, split digest, evaluation protocol, metrics report,
   patient/event/hour counts, patient independence, and clinical annotations.
5. For neonatal recordings, a neonatal specialist review reference on matching
   evidence. Adult or pediatric evidence cannot satisfy this requirement.

There is no fallback. Merely declaring pediatric support cannot reuse adult
validation evidence; every mismatched age pair is rejected. Older adults also
need their own evidence. Both live-prediction and medical-alert flags are
always false, including when offline metadata eligibility passes.

Evidence references are declarations for human audit, not independently
verified reports. A pass establishes compatibility with those declarations;
it is **not clinical validation, sufficient sample size, acceptable clinical
performance, regulatory authorization, or permission to deploy**. No numeric
clinical acceptance thresholds are invented. Evidence must ultimately address
the actual ages, conditions, devices, settings, and seizure types studied; one
sample in a broad category cannot establish generalizability to that category.

## Age-aware dataset registry

The following are discovery placeholders. All retain `ready_for_use=False` and
unverified project license/access/ethics fields. Source pages were consulted on
2026-10-11; current terms and access must be reviewed for a pinned release.

| Candidate | Research review target | Outstanding questions |
| --- | --- | --- |
| [Helsinki neonatal EEG](https://zenodo.org/records/4940267) | Neonatal | Exact recording age, neonatal clinical context, expert disagreement, corrected annotation release, device/montage, project permissions. |
| [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) | Child/adolescent; review the limited young-adult subset separately | Recording-specific age, patient aliases, channel variation, recording gaps, devices, and terms. The original cohort includes ages extending to 22; it does not establish under-one-year coverage. |
| [TUH Seizure Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) | Adult and older-adult review priorities | The source describes manual seizure annotations and a custodian access process. Verify release, actual ages/subgroup counts, identity overlap, device/montage, terms, and local access. Older-adult suitability is unverified. |
| Unresolved infant cohort | 29 days to under one year | No suitable cohort has been verified for this project. Do not substitute neonatal, child, or adult data as validation. |

Targets are discovery priorities, not verified coverage. `candidates_for_age()`
returns candidates or a coverage gap; it never determines recording age or
model eligibility. No dataset licenses, local approvals, access credentials,
or availability commitments are asserted by the registry.

## Reproducible patient-independent splitting

`build_split_manifest()` operates on metadata, not windows or raw signals:

1. Resolve all aliases and repeated visits to a globally curated
   `patient_group_id` before splitting, including aliases across datasets.
   Source-dataset + subject-ID mappings must be consistent. CHB-MIT explicitly
   identifies `chb01` and `chb21` as the same patient; `chb24` lacks age metadata
   in the published SUBJECT-INFO file. See the
   [source methods and description](https://physionet.org/content/chbmit/1.0.0/).
2. Reject duplicate recording IDs, identical signal digests, repeated source
   recordings across releases, and conflicting identities. Exact-hash checks
   cannot detect re-encoded duplicates or unknown cross-dataset aliases; those
   require a documented cohort audit.
3. Sort patients by SHA-256 of canonical JSON `[seed, patient_group_id]`, with
   patient ID as tie-breaker. Default seed is 42. Assign nominal 60% train,
   20% validation, remainder test. Floor train/validation counts and bound them
   to reserve at least one patient in each partition. Require three patients;
   this computational minimum is not a scientifically adequate sample size.
4. Keep every recording, session, age transition, and derived window of each
   patient in one partition. Input order does not change assignments. Adding
   patients can change assignments, so freeze the cohort and manifest together.
5. Store algorithm/version, seed/fractions, assignments, age groups, individual
   metadata digests, and full cohort digest. JSON roundtrips preserve them.
   `validate_split_manifest()` rebuilds and compares the entire manifest before
   scoring, rejecting changed metadata, omissions, assignments, and leakage.

Fit preprocessing, feature selection, and model parameters using train patients
only. Choose thresholds, hyperparameters, event-merging/refractory settings, and
calibration on validation patients (or a prespecified subdivision wholly inside
the development cohort). Freeze those choices before evaluating test patients.
Never tune on test detections or borrow test patients for calibration.

This generic splitter is not age-stratified and is not a replacement for an
official benchmark split. Audit age/event/hour/patient coverage in each frozen
partition before model development; an absent or small age subgroup cannot
support validation. Preserve official held-out partitions in future adapters.
Plan external-site validation separately; patient independence alone does not
remove site/device differences or selection bias.

## Offline scoring protocol

`evaluate_test_set()` consumes the full metadata cohort, a validated manifest,
an `EvaluationProvenance`, and externally produced `OfflineDetection` decision
timestamps for **every** test recording, including explicit empty lists. It
does not generate detections or depend on an attention model. A metrics report
is not an eligibility decision: score experimental/failed candidates separately
from reviewing their suitability.

Provenance records model identity/version/hash, manifest hash, prediction-file
hash, frozen protocol reference, and declared training/selection patient IDs.
Training patients must belong to train and selection patients to validation.
The evaluator rejects test-patient leakage in these declarations, but cannot
prove how an externally supplied model was trained. It also hashes the
normalized detection inputs for reproducibility.

Protocol `decision-within-reference-v1` uses half-open `[onset, offset)` reference
events. The first decision timestamp inside an event detects it; each reference
event can match once. Extra decisions, decisions before onset, and decisions
at/after offset are false alarms. No matching tolerance, merging, or refractory
suppression is applied. This is a deliberately explicit provisional protocol,
not a claim of compliance with every clinical EEG benchmark's event scorer.

| Metric | Definition |
| --- | --- |
| Event sensitivity | Detected reference events / all reference events. Missed events remain in the denominator. No reference events gives `None`, not 100%. |
| False alarms per hour | Unmatched decisions / reviewed usable recording hours, including ictal time. Non-seizure recordings remain in the denominator. This is not an interictal-only denominator. |
| Detection latency | Decision-availability time minus reference onset, for matched events only. Report raw latencies and median alongside misses. Include buffering/processing delay; do not backdate decisions to an earlier window start. |
| Patient-independent performance | Test patients disjoint from train/validation, pooled metrics, per-patient metrics, and unweighted mean patient sensitivity among patients with reference events. Report the number of such patients. |
| Age-specific performance | The same event metrics and patient/recording/hour/event counts for all six recording-age groups. Empty groups remain visible with zero counts and `None` rates. |

Quality exclusions reduce the evaluated-hour denominator and are reported
separately. Events or decisions intersecting excluded time cause rejection,
requiring explicit adjudication of the evaluation domain. There is no silent
dropping of missed events or artifact-associated false alarms. Report exclusions
and full-cohort availability separately; evaluate sensitivity to quality
exclusions before making research claims.

A patient's visits in multiple age categories can contribute to multiple
subgroup patient counts; do not add those counts to estimate unique patients.
No uncertainty estimates are implemented yet. Before publishing performance,
prespecify patient-cluster bootstrap intervals (resample whole patients with
all their recordings), stratification, minimum subgroup sample requirements,
annotation consensus, and any alternative matching tolerance. Never bootstrap
individual windows as if independent. Report low/absent support explicitly.

## Local use and verification

Run from the `NeuroSense` repository root using the existing environment:

```sh
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p test_seizure_research.py -v
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
rtk proxy node tests/test_webcam_alignment.mjs
rtk proxy node tests/test_webcam_behavior.mjs
rtk proxy node tests/test_webcam_camera.mjs
```

Minimal metadata-only usage (no data access or predictions):

```python
from seizure_research.schemas import AgeAtRecording, AgeGroup
from seizure_research.registry import candidates_for_age
from seizure_research.eligibility import REGISTERED_MODELS

age = AgeAtRecording(completed_years=0, completed_days=29,
                     source="reference-to-verified-age-metadata")
assert age.group == AgeGroup.INFANT
assert candidates_for_age(age.group)[0].status == "unresolved_gap"
assert REGISTERED_MODELS == ()
```

For an authorized future metadata cohort, validate each dictionary using
`EEGRecording.model_validate()`, build the split, and persist its
`model_dump_json(indent=2)` in the approved research artifact store. Preserve
that exact manifest, code revision, Python/dependency versions, source and
annotation digests, preprocessing protocol, model digest, and detection input
digest with each experiment. Do not store clinical files or identifiable
metadata in this Git repository. This phase adds no automatic persistence.

Tests use tiny fabricated metadata and supplied decision times. All approvals,
clinical-review references, and model evidence in the fixtures are explicitly
`TEST_ONLY`; they are not research assets or clinical validation. Coverage
includes age boundaries/leap years, missing metadata, no adult fallback,
neonatal specialist review, montage/device/rate compatibility, evidence scope,
permissions, patient leakage, deterministic manifests, scoring denominators,
missing test recordings, empty age groups, and import isolation.

Verification on 2026-10-11 with Python 3.12.3 and Pydantic 2.13.5:

| Check | Result |
| --- | --- |
| New seizure research contract suite | 48 tests passed |
| Complete Python regression suite (includes those 48) | 134 tests passed; no skips or failures |
| Webcam alignment / behavior / camera suites | 19 / 15 / 22 tests passed (56 total) |
| New Python syntax and whitespace checks | Passed |
| Existing tracked source/artifact changes | None |

The Python regression run included the existing published EEG V2 dataset and
artifact checks, authentication, SQLite user scoping/migration, and Streamlit
webcam navigation. Streamlit emitted its existing bare-mode ScriptRunContext
warnings during successful navigation tests. Webcam tests remain mocked; no
physical camera, clinical EEG acquisition, or clinical seizure validation was
performed. EEG V1 has no dedicated populated test suite in this repository;
its existing code and artifacts were preserved unchanged.

## Clinical limitations and next boundaries

This is **not a medical diagnosis, clinical decision system, or emergency
seizure alarm**. It must not guide treatment, reassure a person that seizures
are absent, or replace professional interpretation. No live seizure prediction,
medical alert, or neonatal consumer monitoring is enabled.

False negatives remain a central risk: seizures may be missed because of signal
loss, artifacts, incomplete scalp coverage, uncertain annotations, population
differences, age/development, devices, or an inadequate future model. An empty
detection list does not establish absence of seizures. False positives can also
occur. High pooled sensitivity can hide poor neonatal, pediatric, older-adult,
or individual-patient performance. Short latency among detected events says
nothing about undetected events. Offline processing can look ahead in a signal;
offline results cannot establish live latency or emergency reliability.

Clinical EEG cannot be replaced by webcam alignment, blink behavior, an
attention score, heart rate, rPPG, or an unvalidated consumer EEG device. The
current webcam and attention features supply no seizure evidence.

Future work requires approved data acquisition, verified importers and identity
resolution, age-appropriate preprocessing and specialist annotation protocols,
actual seizure-model development and independent validation, uncertainty and
external-cohort assessment. None of those tasks is represented as complete by
this foundation.
