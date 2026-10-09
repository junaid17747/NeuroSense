# EEG V2 Phase 1: dataset audit and subject split

Phase 1 audits the existing CSV and publishes a reproducible subject manifest.
It does not train, calibrate, tune, or evaluate a model. All pre-existing files
are preserved, including the uncommitted V2 generator, feature extractor and
training script. No dependencies were added.

## Run from the NeuroSense directory

```bash
rtk proxy .venv/bin/python -B -m src.model.audit_eeg_v2
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

The audit reads the CSV and writes `reports/eeg_v2_phase1/dataset_audit.json`
and `reports/eeg_v2_phase1/split_manifest.json`. Identical reruns leave existing
artifacts untouched. A differing artifact is never overwritten: use a new
`--output-dir` inside NeuroSense for an explicitly different run. Neither
command imports the training script or dataset generator. `-B` suppresses
bytecode writes. Test fixtures and temporary files stay inside NeuroSense.

The unit suite uses standard-library `unittest`. Its full-dataset integration
test verifies the published manifest against all CSV rows. That test explicitly
skips if the local CSV or manifest is absent; run the audit first to include it.
The completed Phase 1 run passed **33 tests in 27.186 seconds**, with **0 skips**,
**0 failures**, **0 errors**, and exit code **0**. The original three test files
are empty; all 33 discovered tests belong to the new Phase 1 suite.

## Dataset identity and audit results

- Input: `data/processed/neurosense_eeg_v2_synthetic_1m.csv`
- Bytes: **397,983,105**
- SHA-256: `6c36f1bf2c1a6a6d4ee133d9a1aac0e0ec2324bc2a79bdec65347bbf3ad4f99d`
- Rows: **1,000,000**; subjects: **1,000**; windows per subject: **1,000**.
- Columns: subject ID, sample ID, the **20** ordered features from
  `src/features/eeg_features_v1.py`, and label.
- Classes: Distracted **329,768**, Focused **339,859**, Neutral **330,373**.
- Fatal integrity errors: **0**. Missing values, nonnumeric/nonfinite feature
  values, invalid subject/sample IDs, duplicate subject/sample keys, duplicate
  complete rows, and duplicate feature vectors: **0** each.
- Sample IDs cover integers 1 through 1000 within each subject. Reusing a
  sample ID for a different subject is valid; the window key is the pair.
- Relative-power ranges, sums, formulas, theta/beta and alpha/beta ratios, and
  the activity/variance identity: **0** mismatching rows at `rtol=1e-7`,
  `atol=1e-9`.
- Status: **passed_with_warnings**. The CSV hash matches before and after audit.

The JSON report includes min, max, mean and sample standard deviation for all
20 features, per-column integrity counts, environment versions and source hashes.

## Reproducible partitions

Seed: **42**. For each unique subject, hash the UTF-8 bytes of
`str(seed) + "\0" + subject_id` with SHA-256. Sort by hexadecimal digest, using
the subject ID to break a tie, then take consecutive groups of 700/100/100/100.
Store each group's subject IDs in lexical order. This algorithm depends only
on the subject IDs and seed, not row order, labels, feature values, or a random
number library version. It is not a stratified split; class presence is checked
after assignment without retrying or inspecting model performance.

| Partition | Subjects | Rows | Distracted | Focused | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 700 | 700,000 | 230,634 | 238,100 | 231,266 |
| Validation | 100 | 100,000 | 33,058 | 33,845 | 33,097 |
| Calibration | 100 | 100,000 | 32,938 | 33,855 | 33,207 |
| Test | 100 | 100,000 | 33,138 | 34,059 | 32,803 |

Pairwise subject overlap, row overlap, unassigned rows, and identical feature
vectors crossing partitions: **0**. Every partition contains all three classes.

The manifest records every subject's membership, row/class counts, dataset
path/size/SHA-256, schema, seed, algorithm and partition purposes. It stores
subject IDs instead of fragile positional row lists. Its file hash binds it to
the exact CSV bytes. Reordering CSV bytes requires a new manifest fingerprint,
although subject membership remains the same.

## Recorded synthetic feature inconsistencies

No values were clipped, recalculated, dropped, or rewritten.

1. **500,045 rows** have RMS squared below variance, beyond numerical tolerance.
   Features from the same window obey `RMS² = variance + mean²`. The generator's
   independent RMS noise breaks that constraint in these rows.
2. **6,928 rows** have Fisher kurtosis below −2. Because the frozen extractor
   uses `bias=False`, −2 alone is not the exact finite-sample cutoff. For its
   1,024-sample windows, the lower bound is
   `-2 * (1024 - 1) / (1024 - 3) = -2.0039177277179236`.
   **6,791 rows** fall below that corrected bound. The generator allows values
   down to −2.5.
3. Hjorth activity and variance are exactly equal in **1,000,000 rows**. This
   redundancy is consistent with the extractor's definitions. Both frozen
   schema columns are retained.
4. This CSV was generated directly in feature space. Time-domain and complexity
   features are sampled separately, and variance is not regenerated after
   band-power noise. There are no associated raw windows with which to verify
   spectral entropy, Hjorth derivative statistics, line length, or equivalence
   to the raw-signal pipeline. This audit establishes neither real EEG accuracy
   nor clinical validity.

## Leakage controls and Phase 2 boundary

`load_manifest_dataset(data_file, manifest_file)` in `eeg_v2_data.py` verifies
the CSV fingerprint before loading, checks it again after reading, reconstructs
the deterministic manifest, and returns the data plus positional row indices.
Use `df.iloc[indices[name]]`. Edited seeds, features, memberships, counts or
dataset bytes fail validation. Missing subjects never default to training.
Copied feature vectors across partitions fail even when their IDs/labels differ.

`model_features(df)` selects only the frozen 20 features; IDs, label and extra
metadata cannot enter the feature matrix. Future work must fit learned
preprocessing and the model on train subjects, use validation subjects for
selection/early stopping, fit probability calibration on calibration subjects
after model selection is frozen, and reserve test subjects for final evaluation.
These roles are recorded in the manifest; training integration is not part of
Phase 1.

The existing `src/model/train_eeg_v2.py` remains unchanged. It still implements
its earlier 800/100/100 split and does not consume this manifest. **Do not run it
as the new pipeline.** Adapting training to this manifest requires Phase 2
approval. No XGBoost training, backend/dashboard changes, replay changes,
SQLite changes, commits or pushes were performed in Phase 1.

## Files added and verification records

- `src/model/eeg_v2_data.py`: schema audit, feature audit, subject splitting,
  manifest validation and feature selection.
- `src/model/audit_eeg_v2.py`: standalone read-only audit entry point and
  non-overwriting artifact writer.
- `tests/test_eeg_v2_phase1.py`: integrity, leakage, reproducibility, tamper,
  preservation and full-dataset integration checks.
- `docs/eeg_v2_phase1.md`: this report and run instructions.
- `reports/eeg_v2_phase1/dataset_audit.json`: complete measured audit.
- `reports/eeg_v2_phase1/split_manifest.json`: fixed subject memberships.
- `reports/eeg_v2_phase1/test_results.txt`: exact test command, results and exit code.
- `reports/eeg_v2_phase1/preservation.json`: before/after hashes of the 110
  pre-existing project files, the resulting file inventory, and audit rerun
  reproducibility checks. The inventory excludes `.git` and `.venv` internals.

All additions are inside NeuroSense. The existing uncommitted changes in
`.gitignore`, `src/preprocessing/eeg_filter.py`, `src/features/eeg_features_v1.py`,
`src/model/generate_synthetic_eeg_v2_1m.py`, and `src/model/train_eeg_v2.py` are
preserved byte-for-byte along with the datasets, V1 models and application files.
