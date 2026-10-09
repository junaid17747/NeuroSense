# EEG V2 Phase 3: sigmoid calibration and final held-out evaluation

Phase 3 calibrates the frozen Phase 2 XGBoost model and evaluates it once on
the 100 held-out test subjects. Every number below is a **synthetic EEG
feature-space benchmark result**. It does not establish clinical validity,
real-world attention detection performance, or equivalence to the raw EEG
pipeline.

## Run and safeguards

```bash
rtk proxy .venv/bin/python -B -m src.model.calibrate_eeg_v2_phase3
rtk proxy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

The entry point refuses to reuse an existing versioned output directory and
creates `models/eeg_v2/phase3_test_evaluation.lock.json` before the first test
prediction. If that lock exists, the command refuses to run test inference
again. The saved `test_predictions.npz` is the sole source for all metrics,
plots, per-subject values and bootstrap statistics after the two model calls.
No later analysis calls either model.

Before calibration, the workflow verifies:

- CSV SHA-256
  `6c36f1bf2c1a6a6d4ee133d9a1aac0e0ec2324bc2a79bdec65347bbf3ad4f99d`;
- Phase 1 manifest SHA-256
  `2a427ef2895b9d19c4b297186ce1e8f404337fd6691e5f48c24a5c29b3cf4c29`;
- frozen Phase 2 UBJSON model SHA-256
  `315b603ae4ca56d3d76305337328a548af738aa4f78f65dfe609b65ba1cf6172`;
- byte-identical Phase 1/Phase 2 manifests;
- all 20 feature names and their order;
- Phase 2 source, dataset, manifest and model provenance.

## Calibration protocol

The installed scikit-learn is 1.9.1, which supports
`CalibratedClassifierCV(FrozenEstimator(...))`. The calibrator is configured as:

| Setting | Value |
| --- | --- |
| Method | Sigmoid (Platt-style one-vs-rest multiclass calibration) |
| Base estimator | Frozen Phase 2 XGBoost model |
| CV | 5 folds within calibration rows; frozen base fits are no-ops |
| Ensemble | False; one calibrated frozen estimator |
| Calibration data | 100 subjects, 100,000 rows |
| Excluded from calibrator fit | Train, validation and test subjects |
| Class order | Distracted, Focused, Neutral |
| Feature order | The frozen 20-feature schema from Phase 1 |

The base booster digest was unchanged during calibration. The calibrated wrapper
was wrapped in a second `FrozenEstimator` before serialization. A reload check
produced bit-identical probabilities on all 100,000 calibration rows.

## Single test evaluation

The test partition contains 100 subjects and 100,000 rows. The calibrated and
uncalibrated frozen models each made exactly one prediction call on it. Neither
result was used for tuning or further fitting. Majority-class probabilities were
defined from training counts only: always Focused (238,100 of 700,000 train
rows).

| Metric | Calibrated | Uncalibrated | Majority baseline |
| --- | ---: | ---: | ---: |
| Accuracy | 0.527060 | 0.528230 | 0.340590 |
| Balanced accuracy | 0.525858 | 0.526711 | 0.333333 |
| Macro F1 | 0.524457 | 0.522150 | 0.169373 |
| Log loss | 0.955580 | 0.951908 | 23.767545 |
| Multiclass Brier score | 0.571969 | 0.570595 | 1.318820 |

Calibration improves macro F1 by 0.002307 on this fixed synthetic test set but
has higher log loss and Brier score than the uncalibrated model. This comparison
is descriptive and does not authorize further tuning.

Class order for the following matrices is Distracted / Focused / Neutral. Rows
are true labels and columns are predictions.

Calibrated confusion matrix:

|  | Distracted | Focused | Neutral |
| --- | ---: | ---: | ---: |
| Distracted | 18,558 | 5,396 | 9,184 |
| Focused | 4,605 | 20,859 | 8,595 |
| Neutral | 9,548 | 9,966 | 13,289 |

Uncalibrated confusion matrix:

|  | Distracted | Focused | Neutral |
| --- | ---: | ---: | ---: |
| Distracted | 19,186 | 5,832 | 8,120 |
| Focused | 4,978 | 21,583 | 7,498 |
| Neutral | 10,096 | 10,653 | 12,054 |

Calibrated per-class precision / recall / F1:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Distracted | 0.567332 | 0.560022 | 0.563653 | 33,138 |
| Focused | 0.575881 | 0.612437 | 0.593597 | 34,059 |
| Neutral | 0.427739 | 0.405115 | 0.416120 | 32,803 |

Uncalibrated per-class precision / recall / F1:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Distracted | 0.560012 | 0.578973 | 0.569334 | 33,138 |
| Focused | 0.566959 | 0.633694 | 0.598472 | 34,059 |
| Neutral | 0.435603 | 0.367466 | 0.398644 | 32,803 |

## Subject results, reliability and uncertainty

`per_subject_metrics.csv` contains one row per test subject for each of the
three prediction strategies, including window count, accuracy-derived confusion
cells, per-class precision/recall/F1, log-loss and Brier sums used in aggregation.

`reliability_bins.json` and `reliability_diagrams.png` use 10 fixed equal-width
probability bins for each class and for the top-label confidence. They are
descriptive diagnostics and were not used to choose the calibrator.

`bootstrap_confidence_intervals.json` contains paired percentile intervals from
2,000 replicates with seed 42. Each replicate samples the 100 test subjects with
replacement, retaining all windows for a sampled subject. Calibrated,
uncalibrated and majority strategies share the same subject draws. The intervals
condition on the fixed models and subjects; they do not include model-fitting or
calibration uncertainty. The primary calibrated 95% intervals are:

| Metric | Estimate | 95% interval |
| --- | ---: | ---: |
| Accuracy | 0.527060 | [0.523900, 0.529880] |
| Balanced accuracy | 0.525858 | [0.522867, 0.528653] |
| Macro F1 | 0.524457 | [0.521351, 0.527311] |
| Log loss | 0.955580 | [0.952965, 0.958302] |
| Multiclass Brier | 0.571969 | [0.570170, 0.573800] |

The paired calibrated-minus-uncalibrated intervals are stored in the same JSON
file. They are comparisons on this one frozen test evaluation, not a basis for
model selection.

## Known synthetic feature inconsistencies

The CSV was not modified. Phase 1 measured and the Phase 3 provenance records:

1. **500,045 rows** have RMS squared below variance. For one signal window,
   RMS² should equal variance plus mean²; the generator adds independent RMS
   noise, so this identity is violated.
2. **6,928 rows** have kurtosis below −2; **6,791** are below the unbiased
   Fisher lower bound −2.0039177277179236 for 1,024-sample windows. The
   generator clips synthetic kurtosis at −2.5.
3. Variance is generated as pre-noise summed band power multiplied by a
   lognormal scale near 90, while spectral bands are perturbed afterward.
   Absolute power scales therefore are not a shared raw-signal-derived physical
   scale. Hjorth activity and variance are also duplicate columns in every row.

These limitations are scientific context for the benchmark. They do not change
the frozen feature values or the evaluation protocol.

## Files

- `src/model/calibrate_eeg_v2_phase3.py`: checksum verification, frozen sigmoid
  calibration, single test inference, artifact publication.
- `src/model/eeg_v2_evaluation.py`: metrics, reliability bins, per-subject table
  and paired subject bootstrap implementation.
- `tests/test_eeg_v2_phase3.py`: 12 calibration, leakage, schema, normalization,
  frozen-model, published-bundle and reproducibility tests.
- `models/eeg_v2/phase3_sigmoid_v1_seed42/`: versioned calibrated model bundle.
- `reports/eeg_v2_phase3/run.log`: exact production command and output.
- `reports/eeg_v2_phase3/test_results_final.txt`: final complete Phase 1–3 test
  output; **67 tests passed in 32.630 seconds**, with zero failures, errors or
  skips. `test_results.txt` records the preceding 66-test verification.

No V1 model, dataset, Phase 1/2 artifact, uncommitted change, backend, dashboard,
replay component or SQLite file was changed. No retraining, commit, push or
FastAPI/Streamlit integration was performed. Phase 3 stops here; further work
requires approval.
