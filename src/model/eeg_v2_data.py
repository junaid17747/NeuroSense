"""Read-only data contracts and subject partitions for EEG V2 Phase 1.

This module never fits a transformer, trains a model, or writes the dataset.
Row indices are positional: consumers must select with DataFrame.iloc.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.eeg_features_v1 import FEATURE_NAMES as FROZEN_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/neurosense_eeg_v2_synthetic_1m.csv"
FEATURE_NAMES = tuple(FROZEN_FEATURE_NAMES)
COLUMNS = ("subject_id", "sample_id", *FEATURE_NAMES, "label")
CLASSES = ("Distracted", "Focused", "Neutral")
SEED = 42
SPLIT_COUNTS = {"train": 700, "validation": 100, "calibration": 100, "test": 100}
SPLIT_ALGORITHM = "sha256(str(seed) + NUL + subject_id), ascending hex digest then subject_id"
SPLIT_PURPOSES = {
    "train": "Fit model and any learned preprocessing only on these subjects.",
    "validation": "Choose hyperparameters and early stopping; do not fit preprocessing here.",
    "calibration": "Fit probability calibration only after model selection is frozen.",
    "test": "Final evaluation only after model selection and calibration are frozen.",
}
RTOL = 1e-7
ATOL = 1e-9


def dataset_fingerprint(path):
    path = Path(path).resolve()
    try:
        display_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = str(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": display_path, "size_bytes": path.stat().st_size, "sha256": digest}


def read_dataset(path=DATA_FILE):
    # Keep identifiers verbatim so whitespace/blank IDs cannot be silently normalized.
    return pd.read_csv(path, dtype={"subject_id": str, "label": str})


def structural_audit(df, samples_per_subject=1000):
    """Return fatal integrity failures; synthetic feature warnings are separate."""
    errors = []
    report = {
        "rows": len(df), "columns": list(df.columns),
        "expected_columns": list(COLUMNS), "feature_count": len(FEATURE_NAMES),
        "expected_subjects": sum(SPLIT_COUNTS.values()),
        "expected_samples_per_subject": samples_per_subject,
        "errors": errors,
    }
    if tuple(df.columns) != COLUMNS:
        errors.append("Column names and order must match the frozen 20-feature schema.")
        return report

    missing = df.isna().sum()
    report["missing_values_by_column"] = {k: int(v) for k, v in missing.items()}
    if missing.any():
        errors.append("Missing values detected.")
    if df.empty:
        errors.append("Dataset must not be empty.")
        return report

    invalid_subjects = ~df["subject_id"].map(
        lambda value: isinstance(value, str) and bool(value) and value == value.strip()
    )
    report["invalid_subject_id_rows"] = int(invalid_subjects.sum())
    if invalid_subjects.any():
        errors.append("Subject IDs must be nonempty strings without surrounding whitespace.")
    subject_rows = df.groupby("subject_id", dropna=False).size()
    report["subject_count"] = len(subject_rows)
    report["rows_per_subject"] = {
        "min": int(subject_rows.min()), "max": int(subject_rows.max()),
        "histogram": {str(k): int(v) for k, v in subject_rows.value_counts().sort_index().items()},
    }
    if len(subject_rows) != sum(SPLIT_COUNTS.values()):
        errors.append("Exactly 1000 subjects are required for the 700/100/100/100 split.")
    if not subject_rows.eq(samples_per_subject).all():
        errors.append(f"Every subject must have exactly {samples_per_subject} windows.")

    sample_ids = pd.to_numeric(df["sample_id"], errors="coerce").to_numpy(dtype=float)
    valid_samples = (
        np.isfinite(sample_ids) & (sample_ids >= 1)
        & (sample_ids <= samples_per_subject) & (sample_ids == np.floor(sample_ids))
    )
    report["invalid_sample_id_rows"] = int((~valid_samples).sum())
    if not valid_samples.all():
        errors.append(f"Sample IDs must be integers in 1..{samples_per_subject} within each subject.")
    duplicate_keys = int(df.duplicated(["subject_id", "sample_id"]).sum())
    report["duplicate_subject_sample_keys"] = duplicate_keys
    report["duplicate_complete_rows"] = int(df.duplicated().sum())
    if duplicate_keys:
        errors.append("Duplicate (subject_id, sample_id) windows detected.")

    report["class_counts"] = {
        str(k): int(v) for k, v in df["label"].value_counts(dropna=False).sort_index().items()
    }
    if set(df["label"].dropna()) != set(CLASSES):
        errors.append(f"Labels must contain exactly {list(CLASSES)}.")

    numeric = df[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    report["nonfinite_or_nonnumeric_by_feature"] = {
        name: int(count) for name, count in zip(FEATURE_NAMES, (~finite).sum(axis=0))
    }
    nonnumeric_dtypes = [name for name in FEATURE_NAMES if not pd.api.types.is_numeric_dtype(df[name])]
    report["nonnumeric_feature_dtypes"] = nonnumeric_dtypes
    if not finite.all() or nonnumeric_dtypes:
        errors.append("Every feature must be numeric and finite.")
    report["duplicate_feature_vectors"] = int(df.duplicated(list(FEATURE_NAMES)).sum())
    return report


def require_valid_structure(df, samples_per_subject=1000):
    report = structural_audit(df, samples_per_subject)
    if report["errors"]:
        raise ValueError(" ".join(report["errors"]))
    return report


def model_features(df):
    """Use an explicit allowlist so IDs, labels and split metadata cannot enter X."""
    return df.loc[:, list(FEATURE_NAMES)].copy()


def subject_partition(subjects, seed=SEED):
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    subjects = list(subjects)
    if any(not isinstance(s, str) or not s or s != s.strip() for s in subjects):
        raise ValueError("Invalid subject identifier.")
    if len(subjects) != 1000 or len(set(subjects)) != 1000:
        raise ValueError("Exactly 1000 unique subjects are required.")

    def rank(subject):
        digest = hashlib.sha256(f"{seed}\0{subject}".encode("utf-8")).hexdigest()
        return digest, subject

    ranked = sorted(subjects, key=rank)
    partitions = {}
    offset = 0
    for name, count in SPLIT_COUNTS.items():
        partitions[name] = sorted(ranked[offset:offset + count])
        offset += count
    return partitions


def split_row_indices(df, partitions):
    """Fail closed on missing, overlapping or unknown subjects; never default to train."""
    if set(partitions) != set(SPLIT_COUNTS):
        raise ValueError("Exactly train, validation, calibration and test partitions are required.")
    seen = set()
    indices = {}
    for name, expected_count in SPLIT_COUNTS.items():
        subjects = partitions[name]
        if len(subjects) != expected_count or len(set(subjects)) != expected_count:
            raise ValueError(f"Wrong or duplicate subject count in {name}.")
        if seen.intersection(subjects):
            raise ValueError("Subject leakage between partitions.")
        seen.update(subjects)
        indices[name] = np.flatnonzero(df["subject_id"].isin(subjects).to_numpy())
        if set(df["label"].iloc[indices[name]]) != set(CLASSES):
            raise ValueError(f"The {name} partition must contain all three classes.")
    if seen != set(df["subject_id"]):
        raise ValueError("Partitions must cover every dataset subject exactly once, with no unknown IDs.")
    combined = np.concatenate(list(indices.values()))
    if len(combined) != len(df) or len(np.unique(combined)) != len(df):
        raise ValueError("Every row must belong to exactly one partition.")

    # Identical feature vectors across partitions are suspicious copied windows,
    # even when their metadata or labels differ. Sample IDs alone are not global IDs.
    duplicate_mask = df.duplicated(list(FEATURE_NAMES), keep=False)
    if duplicate_mask.any():
        owner = {subject: name for name, subjects in partitions.items() for subject in subjects}
        duplicates = df.loc[duplicate_mask, list(FEATURE_NAMES)].copy()
        duplicates["partition"] = df.loc[duplicate_mask, "subject_id"].map(owner).to_numpy()
        groups = duplicates.groupby(list(FEATURE_NAMES), sort=False)["partition"].nunique()
        if groups.gt(1).any():
            raise ValueError("Identical feature vectors cross partitions; possible copied-window leakage.")
    return indices


def build_manifest(df, fingerprint, seed=SEED, samples_per_subject=1000):
    require_valid_structure(df, samples_per_subject)
    partitions = subject_partition(df["subject_id"].unique(), seed)
    indices = split_row_indices(df, partitions)
    splits = {}
    for name, subjects in partitions.items():
        labels = df["label"].iloc[indices[name]]
        splits[name] = {
            "subject_count": len(subjects), "subjects": subjects,
            "rows": len(indices[name]),
            "class_counts": {label: int(labels.eq(label).sum()) for label in CLASSES},
            "purpose": SPLIT_PURPOSES[name],
        }
    return {
        "manifest_version": 1, "dataset": fingerprint,
        "seed": seed, "algorithm": SPLIT_ALGORITHM,
        "feature_names": list(FEATURE_NAMES), "classes": list(CLASSES),
        "excluded_model_columns": ["subject_id", "sample_id", "label"],
        "rows": len(df), "subject_count": int(df["subject_id"].nunique()),
        "samples_per_subject": samples_per_subject, "splits": splits,
    }


def validate_manifest(df, manifest, fingerprint, seed=SEED, samples_per_subject=1000):
    """Recompute the entire contract, rejecting edited IDs, counts, seed or dataset."""
    if manifest.get("dataset") != fingerprint:
        raise ValueError("Manifest dataset fingerprint does not match the current CSV.")
    expected = build_manifest(df, fingerprint, seed, samples_per_subject)
    if manifest != expected:
        raise ValueError("Manifest differs from the reproducible split contract.")
    return split_row_indices(df, {name: item["subjects"] for name, item in manifest["splits"].items()})


def load_manifest_dataset(data_file, manifest_file, seed=SEED, samples_per_subject=1000):
    """Future training entry point: verify file bytes before returning any partitions."""
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    before = dataset_fingerprint(data_file)
    if manifest.get("dataset") != before:
        raise ValueError("Manifest dataset fingerprint does not match the current CSV.")
    df = read_dataset(data_file)
    if dataset_fingerprint(data_file) != before:
        raise ValueError("Dataset changed while it was being read.")
    indices = validate_manifest(df, manifest, before, seed, samples_per_subject)
    return df, indices


def feature_audit(df):
    """Describe inconsistencies without replacing, clipping or dropping any values."""
    values = df[list(FEATURE_NAMES)]
    powers = values[list(FEATURE_NAMES[:5])].to_numpy()
    relative = values[list(FEATURE_NAMES[5:10])].to_numpy()
    total = powers.sum(axis=1)
    expected_relative = powers / np.maximum(total[:, None], 1e-12)
    rms_squared = values["rms"].to_numpy() ** 2
    variance = values["variance"].to_numpy()
    delta = rms_squared - variance
    # The extractor uses scipy.stats.kurtosis(fisher=True, bias=False), N=1024.
    # Its finite-sample lower bound is slightly below -2.
    window_samples = 256 * 4
    kurtosis_bound = -2 * (window_samples - 1) / (window_samples - 3)
    counts = {
        "negative_non_kurtosis_features": int((values.drop(columns="kurtosis") < 0).any(axis=1).sum()),
        "relative_power_outside_unit_interval": int(((relative < 0) | (relative > 1)).any(axis=1).sum()),
        "relative_power_sum_mismatch": int((~np.isclose(relative.sum(axis=1), 1, rtol=RTOL, atol=ATOL)).sum()),
        "relative_power_formula_mismatch": int((~np.isclose(relative, expected_relative, rtol=RTOL, atol=ATOL)).any(axis=1).sum()),
        "theta_beta_ratio_mismatch": int((~np.isclose(values["theta_beta_ratio"], powers[:, 1] / np.maximum(powers[:, 3], 1e-12), rtol=RTOL, atol=ATOL)).sum()),
        "alpha_beta_ratio_mismatch": int((~np.isclose(values["alpha_beta_ratio"], powers[:, 2] / np.maximum(powers[:, 3], 1e-12), rtol=RTOL, atol=ATOL)).sum()),
        "hjorth_activity_variance_mismatch": int((~np.isclose(values["hjorth_activity"], variance, rtol=RTOL, atol=ATOL)).sum()),
        "hjorth_activity_variance_exact_duplicates": int(values["hjorth_activity"].eq(values["variance"]).sum()),
        "rms_squared_below_variance": int((delta < -(ATOL + RTOL * np.abs(variance))).sum()),
        "kurtosis_below_minus_two": int(values["kurtosis"].lt(-2).sum()),
        "kurtosis_below_unbiased_1024_sample_bound": int(values["kurtosis"].lt(kurtosis_bound - ATOL).sum()),
    }
    warnings = []
    if counts["rms_squared_below_variance"]:
        warnings.append({
            "code": "rms_variance_inconsistency", "rows": counts["rms_squared_below_variance"],
            "detail": "For one raw window, RMS squared = variance + mean squared. Independent synthetic RMS noise violates this identity when RMS squared < variance.",
        })
    if counts["kurtosis_below_unbiased_1024_sample_bound"]:
        warnings.append({
            "code": "kurtosis_out_of_range", "rows": counts["kurtosis_below_unbiased_1024_sample_bound"],
            "detail": "The synthetic generator clips kurtosis at -2.5. Some values fall below the bias-corrected Fisher kurtosis lower bound for the extractor's 1024-sample windows.",
        })
    if counts["hjorth_activity_variance_exact_duplicates"]:
        warnings.append({
            "code": "redundant_activity_variance", "rows": counts["hjorth_activity_variance_exact_duplicates"],
            "detail": "Hjorth activity and variance are the same statistic in both generator and extractor. This is feature redundancy, not an integrity failure; both schema columns are preserved.",
        })
    for name, count in counts.items():
        if count and (name.endswith("mismatch") or name in {"negative_non_kurtosis_features", "relative_power_outside_unit_interval"}):
            warnings.append({"code": name, "rows": count, "detail": "Feature consistency check failed; values are preserved for review."})
    warnings.append({
        "code": "feature_space_only",
        "detail": "No underlying raw EEG windows exist in this CSV. Time-domain/complexity features are sampled separately; variance is not regenerated after band-power noise. Spectral entropy, Hjorth derivatives, line length and raw-signal equivalence cannot be validated from these columns. No real-EEG accuracy or clinical validity is established.",
    })
    return {
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "counts": counts, "unbiased_kurtosis_lower_bound_1024_samples": kurtosis_bound,
        "rms_squared_minus_variance": {"min": float(delta.min()), "max": float(delta.max()), "mean": float(delta.mean())},
        "feature_statistics": {
            name: {"min": float(values[name].min()), "max": float(values[name].max()),
                   "mean": float(values[name].mean()), "std_ddof_1": float(values[name].std())}
            for name in FEATURE_NAMES
        },
        "warnings": warnings,
    }
