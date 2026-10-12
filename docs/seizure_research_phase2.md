
# NeuroSense V4 Phase 2: local clinical EEG ingestion

Phase 2 adds **offline metadata ingestion and annotation validation**, starting
with a CHB-MIT 1.0.0 candidate adapter. It does not download a dataset, decode or
copy EEG samples, train a model, produce seizure predictions, or change the
application. An approved manifest means the supplied records passed technical
checks and explicitly recorded human reviews; it is not clinical validation.

## Phase 1 inspection and compatibility

The Phase 1 schemas, registry, patient splitter, evaluator, eligibility checks,
documentation, and tests were inspected first. Ingestion returns the existing
`EEGRecording` contract and uses `require_research_ready()` and
`checked_recordings()` to enforce completeness, quality, permissions, and
cohort consistency. The age schemas, eligibility rules, splitting algorithm,
and evaluation protocol are unchanged. The model registry remains empty.

Changes are confined to `seizure_research`, research documentation/tests, and
raw-clinical-file ignore patterns. Existing webcam, authentication, SQLite,
EEG V1/V2, FastAPI, and Streamlit code and model artifacts are untouched. There
is no new dependency, database table, backend route, or dashboard page.

## Candidate datasets and verified source observations

Sources below were checked on 2026-10-11. Published source terms are distinct
from project access, permission, and ethics verification. All registry entries
remain unapproved for this project; local access was not tested. Population
descriptions do not determine individual recording ages.

### Neonatal: Helsinki expert-annotated EEG

- Provenance: Stevenson, Tapani, Lauronen, and Vanhatalo; Helsinki University
  Hospital. The [Zenodo record, DOI 10.5281/zenodo.4940267](https://zenodo.org/records/4940267)
  identifies release v4 and files, including a corrected annotator-A CSV.
- Population and annotations: the [dataset paper](https://www.nature.com/articles/sdata201939)
  describes 79 neonatal recordings, three independent expert annotations,
  referential EDF signals at 256 Hz, and CSV/MAT labels at one-second
  resolution. The paper also describes 19 electrodes on the 10–20 system and
  an 18-derivation bipolar montage used for annotation. Each CSV represents an
  expert, with subjects in columns and seconds in rows. Annotations identify
  onset and duration but not seizure location or nature. A future adapter must
  preserve disagreement and prespecify consensus rules.
- Access and terms: the repository lists open files. A license identifier was
  not visible in the retrieved record, and the metadata API could not be
  verified; **review the actual release license/terms before use**. The paper's
  publication license is not a substitute for dataset terms. The included
  ethics document does not establish this project's authorization.
- Limitations: specialist neonatal interpretation, chronological and maturational
  age, annotation disagreement, signal/label duration, and corrected versions
  need review. Single-centre expert labels do not establish generalizability.
  The dataset contains technician-noted signal artifacts and elevated seizure
  burden from recordings selected during likely seizure periods, so its event
  prevalence may not represent routine continuous monitoring.
  The registry's short population label is a discovery description; review the
  paper's inclusion criteria and individual gestational/postmenstrual ages.
  No neonatal adapter or consumer monitoring is implemented.

### Pediatric and limited young-adult coverage: CHB-MIT

- Provenance/population: the [PhysioNet release 1.0.0](https://physionet.org/content/chbmit/1.0.0/)
  describes clinical scalp EEG from patients with intractable seizures. Published
  original ages span 1.5–22 years, crossing the project's child, adolescent,
  and adult categories. `chb01` and `chb21` are one patient; `chb24` lacks a
  published SUBJECT-INFO entry. Per-recording age and canonical identity must
  be explicitly verified.
- Access: source files are listed as openly accessible under specified terms.
  No files were downloaded into this project and no local authorization is
  assumed.
- Terms: [Open Data Commons Attribution License v1.0](https://physionet.org/content/chbmit/view-license/1.0.0/)
  is the source's listed license. Its attribution/notices and scope of rights
  need review for the intended use, alongside institutional requirements.
- Format: EDF recordings, per-case text summaries with recording-relative event
  times, and separate `.seizure` files. This adapter parses **summary text
  only**, not the separate annotation format. The release documents 256 Hz
  sampling and a 23-channel scalp montage, while individual EDFs can include
  dummy channels, changed channel lists, and per-record sampling metadata that
  must be inspected rather than assumed. Source summaries demonstrate
  [repeated labels and unnumbered events](https://physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt),
  [dummy channels, channel changes, and filename suffixes](https://physionet.org/files/chbmit/1.0.0/chb17/chb17-summary.txt),
  and [plus-sign filenames](https://physionet.org/files/chbmit/1.0.0/chb02/chb02-summary.txt).
- Limitations: selected clinical population, variable channel layouts, gaps
  between files, uncertain device details, and missing or ambiguous metadata.
  Do not infer neonatal/infant support or broad adult validity from this cohort.

### Adult research candidate: TUH Seizure Corpus (TUSZ)

- Provenance: Temple University Hospital / Neural Engineering Data Consortium.
  The [custodian's corpus page](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
  lists TUSZ clinical EEG with manual event start/stop, channel, and seizure-type
  annotations. It currently points to v2.0.6; pin and verify any acquired release.
- Population: adult and older-adult research are review targets, **not verified
  subgroup coverage**. Review the actual patient ages, sample sizes, diagnoses,
  devices, and official partitions before considering suitability.
- Access/terms: a signed request and custodian acceptance precede access.
  Credentials and local access are unverified. Review the current agreement;
  no open license or project permission is asserted here.
- Annotation format: event timing/channel/type fields are documented. Exact
  release-specific serialization, label mapping, background labels, and timing
  units remain **requiring review** against the
  [custodian's annotation documentation](https://isip.piconepress.com/publications/reports/2020/tuh_eeg/annotations/).
  No TUSZ parser is supplied in this phase.
- Limitations: population and device variation, annotation changes, identity
  overlap between releases, and official split integrity require a separate
  audit. Clinical specialist labels are not infallible ground truth.

### Initial adapter recommendation

Use CHB-MIT 1.0.0 to exercise local EDF inspection, summary parsing, explicit
identity review, and deterministic manifest generation because its release
provides EDF files plus recording-relative summary annotations and a published
age table. This is an engineering validation choice only. Until access terms
and project permissions are reviewed, keep validation synthetic; after review,
any real-data smoke test should be metadata-only and limited to authorized
local files. Do not describe CHB-MIT as a representative pediatric, adolescent,
or adult validation cohort. The Helsinki cohort is a more appropriate candidate
for a later neonatal-specific adapter, while TUSZ is a restricted-access adult
research candidate whose age composition and release-level details need review.

The 29-day-to-under-one-year cohort remains an explicit registry gap. A child
dataset must not be used as infant validation evidence. Neonatal EEG requires
specialized interpretation and cannot be evaluated using adult assumptions.

## Adapter architecture

| Module | Responsibility |
| --- | --- |
| `adapters/base.py` | Typed review plan, canonical identity map, channel dispositions, source artifacts, and `DatasetAdapter` protocol. |
| `edf.py` | Local legacy EDF header inspection, file-size validation, streaming SHA-256; no signal arrays returned. |
| `adapters/chb_mit.py` | Strict CHB-MIT summary parsing and consistency checks, reviewed identity/channel selection, conversion to `EEGRecording`. |
| `ingestion.py` | Adapter dispatch, deterministic cohort/audit construction, full-source manifest revalidation, JSON-only artifact writer. |
| `__main__.py` | Local inspection and ingestion commands with explicit inputs and exit status. |
| `registry.py` | Source observations and unresolved review work, separate from local approval fields. |

Future adapters implement the protocol and extend the explicit plan/dispatch
allowlist with their release-specific review requirements. There is no dynamic
plugin loading, automatic downloading, inferred demographic mapping, or model
selection in this architecture.

## Local EDF inspection contract

The reader follows the [ordinary EDF layout](https://www.edfplus.info/specs/edf.html):
fixed header, per-signal headers, and two-byte sample storage. It checks the
header length, bounded channel count, finite positive data-record duration,
known positive data-record count, samples per record, calibration bounds,
channel labels, and exact expected file size. Rates and total duration are
calculated from the header rather than assumed to be 256 Hz or one hour.

Supported: **ordinary continuous 16-bit EDF with blank reserved format field**.
Rejected: EDF+/EDF+C/EDF+D, BDF, unknown record counts, invalid calibration,
truncated/extra payload, empty labels, unsupported annotation channels, or
nonfinite values. These deliberate restrictions avoid treating a discontinuous
recording as a continuous timeline. Future formats require their own tested
reader; they are not silently coerced.

Memory use is bounded: headers are read, then source bytes are streamed for a
checksum without decoding, retaining, or copying signal samples. Size and file
identity/modification state are checked during inspection. EDF patient IDs,
recording IDs, and dates are not returned, printed, or used for age/identity.
Channel metadata still needs a human review for identifying information.

This is structural inspection, not a signal-quality assessment or a general
EDF validator. Unknown software/device metadata cannot be reconstructed from
the source and must remain unresolved until independently verified.

## Human review and recording acceptance

The `IngestionPlan` is an external JSON file. It contains no raw EEG and uses
pseudonyms/references rather than identifying clinical notes. Its JSON Schema
can be obtained locally from `IngestionPlan.model_json_schema()`.

Required inputs:

| Scope | Required information |
| --- | --- |
| Dataset | `adapter_id="chb_mit"`, `dataset_version="1.0.0"`, explicit verified human dataset review, and approved Phase 1 `ResearchPermissions` with access/terms/ethics/de-identification review references. |
| Identities | Verified identity review and one unambiguous `subject_id → canonical_patient_id` entry per source case. Known aliases `chb01` and `chb21` must both be mapped and agree when either is ingested, even if only one case is in this batch. |
| Recording | Normalized relative EDF/summary paths, source subject, explicitly verified age at recording, actual device/software, expected rate, montage, reviewed quality, and metadata review. |
| Channels | Every source index, source label, keep/exclude decision, unique kept output label, and reasons for exclusions or label aliases. |
| Annotations | Verified expert annotation provenance/clinical review, provenance and protocol references, annotator reference, and `annotation_coverage="complete"`. |
| File binding | Reviewed SHA-256 values for both EDF and summary. Actual bytes must match them before acceptance. |

`HumanReview` defaults to unverified. Marking it verified requires reviewer,
date, and evidence reference. A public link, nonempty file, valid syntax, or
permission field alone does not bypass the review gates. These records are
human attestations, not cryptographic signatures; software cannot independently
establish the truth of an approval, a clinical annotation, or an identity map.
Reviewers must confirm that their evidence covers the exact file hashes,
release, scope of use, annotation source, and recording metadata supplied.

There is no birth-date calculation from anonymized EDF fields or automatic
SUBJECT-INFO import. A rounded age that straddles a category boundary must be
resolved through verified recording-specific metadata. Ages below one year
still require the Phase 1 day precision. Missing or conflicting canonical
identities are rejected; filenames are checked against declared case identity
but never used to invent a canonical patient.

Recordings must live under `root/subject/filename.edf`, with the case summary at
`root/subject/subject-summary.txt`. Relative traversal and symlinks escaping the
root are rejected. Supported filename forms include letter case suffixes
before the underscore and a plus sign before `.edf`.

Channel selection retains source order and must account for all indices. Raw
duplicate/dummy labels can be inspected, but dummy `-` channels cannot be kept;
duplicates need explicitly reviewed exclusion or distinct aliases with reasons.
Kept channels must match reviewed units and montage. Each source rate must
match the reviewed channel-specific rate; the per-channel ordered rates are
preserved in recording metadata. No unit conversion, filtering, resampling, or
re-referencing occurs during Phase 2 ingestion. The resulting
`preprocessing_id` is `chb_mit_raw_channel_selection_v1`; downstream eligibility
requires evidence for that exact profile. A channel selection plan does not
rewrite the source EDF or materialize a filtered signal.

Checkpoint C introduces adapter version `2`, ordered per-channel sampling rates,
and calibration metadata for waveform decoding. Rebuild and revalidate saved
cohort/split artifacts made with adapter version `1`; do not manually repair
their hashes. Mixed-rate raw recordings do not match existing single-rate model
evidence. See [the Checkpoint C contract](seizure_research_phase3.md).

## Summary and annotation validation

The parser accepts numbered or single unnumbered seizure start/end pairs and
explicit event counts. It handles summary channel-change blocks and the source's
24+ hour clock notation; conventional midnight rollover is also supported.
Zero/full-day clock ambiguity is rejected. Summary clocks establish duration
only, not age, absolute dates, or cross-file synchronization.

It rejects missing/duplicate recording blocks, absent counts, incomplete pairs,
mixed/duplicate event numbering, mismatched counts, negative/nonfinite times,
reversed/zero-length events, overlap, out-of-order events, and events extending
past recording duration. Unknown lines reject the summary rather than risk
silently ignoring annotation content. A malformed block elsewhere in the same
summary also blocks acceptance: the complete reviewed source must parse.

EDF and summary durations must agree within 1 microsecond; the exact ordered
source labels must agree; the case-level rate must match the summary, and every
kept EEG rate must match its explicit reviewed channel rate (or the case-level
rate when no override is present). Annotation times use seconds relative to recording start
and the Phase 1 half-open interval convention. The source does not provide a
seizure subtype in this summary, so events use `unspecified_in_source`.

**An absent `.seizure` file, missing summary block, or empty event list is not
evidence of a seizure-free recording.** Acceptance of a zero-event recording
requires its explicit summary count of zero, complete annotation coverage, and
verified annotation-provenance review. Reviewers must establish completeness
from the source annotation protocol and any necessary cross-checks, rather than
assuming the count field establishes it. Such recordings remain in the cohort
and subsequent false-alarm evaluation. A future `.seizure` reader and automated
cross-format reconciliation are still needed if a study requires both sources.

## Deterministic manifests and audit reports

`ingest_cohort()` performs no writes. It returns:

- An audit for every requested recording, with stable rejection codes and
  counts by age, patient, and explicitly reviewed seizure-free status.
- A cohort manifest **only if every requested recording passes** and whole-cohort
  duplicate/identity checks pass. A validated row in a rejected batch is an
  audit finding, not a separately approved partial cohort.

The manifest stores the full reviewed plan, adapter/pipeline versions, source
relative paths/sizes and computed digests, annotation-provenance review
references, per-record review digests, and Phase 1 recordings. It stores no
source root, wall-clock run time, raw EDF header identity, signal samples, or
raw summary text. Audit provenance records Python/Pydantic versions. Input
recording and identity-map order are canonicalized; physical channel order is
preserved. Canonical JSON and SHA-256 produce reproducible artifacts for
unchanged files, reviews, code, and runtime. Relocating the same files under
another root does not change output.

`validate_manifest(root, manifest)` re-reads the source files and reapplies all
reviews/checks before comparing the full manifest. Run it before using a saved
manifest with Phase 1 splitting/evaluation. Checksums bind bytes, not source
authenticity; corroborate them against custodian records during human review.
Exact byte checks cannot find re-encoded duplicates or unidentified aliases.

`write_artifacts()` writes `audit.json`, and `cohort_manifest.json` only when
approved, outside this repository. It creates private directories/files,
rejects symlink outputs, and never overwrites a different or incomplete bundle.
An identical rerun is a no-op; a changed/rejected run needs a new directory so
a stale approved manifest cannot be mistaken for its result. Interrupted writes
are rejected on reuse. No raw files are copied. `.gitignore` also excludes EDF,
BDF, and seizure annotation file extensions as accidental-add protection; this
is not a substitute for keeping clinical data and review artifacts outside Git.

Invalid review-plan syntax/schema or conflicting JSON keys is rejected before
ingestion, with a sanitized CLI error and **no cohort artifacts**. Recording
failures from a valid plan produce an audit-only bundle. Plan evidence references
must themselves avoid PHI because they are preserved in the external manifest.

## Commands and Python handoff

Run from the repository root, with already authorized local files. These
commands do not request credentials or download data:

```sh
rtk proxy .venv/bin/python -B -m seizure_research inspect-edf /authorized/chbmit/chbXX/chbXX_YY.edf
rtk proxy .venv/bin/python -B -m seizure_research inspect-summary /authorized/chbmit/chbXX/chbXX-summary.txt --subject chbXX
rtk proxy .venv/bin/python -B -m seizure_research ingest --root /authorized/chbmit --plan /authorized/reviews/chbmit-plan.json --output-dir /authorized/research/run-001
```

Replace symbolic `XX`/`YY` placeholders with verified source identifiers. Inspect
commands report structure and checksums only; they do not grant approval. The
ingest command returns exit code 0 for an approved cohort, 2 for rejection or
invalid input, with a short JSON result. No clinical EEG was required to build
or test these commands. Never copy the fabricated test approvals into a plan.

After approved ingestion, the Python handoff is:

```python
from pathlib import Path
from seizure_research.ingestion import CohortManifest, validate_manifest
from seizure_research.splitting import build_split_manifest

manifest = CohortManifest.model_validate_json(
    Path("/authorized/research/run-001/cohort_manifest.json").read_text()
)
validate_manifest("/authorized/chbmit", manifest)
records = [entry.recording for entry in manifest.records]
split = build_split_manifest(records, seed=42)
```

The Phase 1 splitter requires at least three canonical patients; an ingestion
batch can legitimately contain fewer. All sessions and age changes of a patient
remain together. Preserve official benchmark splits where required instead of
silently substituting this generic split. Model eligibility remains a separate,
strict age/profile/evidence gate; ingestion never enables a model.

## Tests and remaining work

Fixtures are small EDF byte streams generated in temporary directories and
fictitious summary/review data. No real patient recordings or annotations are
embedded in tests. Known source alias names are used only with fabricated data
to test the alias constraint. Run:

```sh
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p test_seizure_ingestion.py -v
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
rtk proxy node tests/test_webcam_alignment.mjs
rtk proxy node tests/test_webcam_behavior.mjs
rtk proxy node tests/test_webcam_camera.mjs
```

Verification on 2026-10-11, using Python 3.12.3 / Pydantic 2.13.5:

| Check | Result |
| --- | --- |
| New ingestion tests | 41 passed within the full run |
| Complete Python regression suite | 175 passed, including all 48 Phase 1 research tests; no skips or failures |
| Webcam alignment / behavior / camera tests | 19 / 15 / 22 passed (56 total) |
| Python syntax and whitespace checks | Passed |
| Phase 1 age, eligibility, split, and evaluation code | Unchanged from the start-of-phase byte hashes |
| Existing application code/model artifacts | Unchanged; `.gitignore` is the only changed previously tracked file |

The full Python run includes existing EEG V2 published dataset/artifact checks,
authentication, SQLite user scoping/migration, and Streamlit navigation. It
emits the existing bare-mode ScriptRunContext warnings during successful
Streamlit tests. No new seizure training, real clinical ingestion, physical
camera check, or clinical validation was performed. EEG V1 still has no dedicated
populated test suite; its source/artifacts remain unchanged. Phase 1 files were
already untracked at this phase's start; no files were staged or committed.

Remaining tasks require real authorization and reviewed data: acquire a pinned
cohort outside Git; independently verify identities, recording ages, devices,
channel plans, annotation completeness, and usage rights; run the adapter on
that cohort and resolve strict-parser rejections; cross-check other annotation
formats; add separately tested neonatal/adult adapters; audit subgroup coverage
and uncertainty. No whole-release CHB-MIT compatibility claim is made from
synthetic tests. The current reader deliberately supports a narrow EDF profile.

No seizure model is trained or clinically validated. False-negative risks,
annotation uncertainty, and age/device generalizability remain unresolved.
This is not diagnosis or an emergency alarm. Live seizure prediction, medical
alerts, and neonatal consumer monitoring remain disabled.
