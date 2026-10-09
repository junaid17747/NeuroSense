"""Metrics computed only from saved predictions; no fitting or model inference."""

import numpy as np
import pandas as pd

from src.model.eeg_v2_data import CLASSES


LOG_EPSILON = np.finfo(np.float64).eps
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
RELIABILITY_BINS = 10
PRIMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier_score")


def checked_probabilities(probabilities, rows):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (rows, len(CLASSES)):
        raise ValueError("Probability matrix must have one row per window and three ordered classes.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Probabilities must sum to one.")
    return probabilities


def checked_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(3)).all():
        raise ValueError("Labels must be integer class indices 0, 1, or 2.")
    if not len(labels):
        raise ValueError("Cannot evaluate an empty partition.")
    return labels


def subject_statistics(labels, probabilities, subjects):
    labels = checked_labels(labels)
    probabilities = checked_probabilities(probabilities, len(labels))
    subjects = np.asarray(subjects, dtype=str)
    if subjects.shape != labels.shape or np.any(subjects == ""):
        raise ValueError("One nonblank subject ID is required per row.")
    names, inverse = np.unique(subjects, return_inverse=True)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(names), 3, 3), dtype=np.int64)
    np.add.at(confusion, (inverse, labels, predicted), 1)
    # Explicit float64 clipping, including for the hard majority baseline.
    clipped = np.clip(probabilities, LOG_EPSILON, 1 - LOG_EPSILON)
    clipped /= clipped.sum(axis=1, keepdims=True)
    losses = -np.log(clipped[np.arange(len(labels)), labels])
    briers = np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1)
    return {
        "subjects": names, "confusion": confusion,
        "log_loss_sum": np.bincount(inverse, weights=losses, minlength=len(names)),
        "brier_sum": np.bincount(inverse, weights=briers, minlength=len(names)),
    }


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def metric_arrays(confusion, loss_sum, brier_sum):
    """Vectorized pooled metrics, for one or many subject-cluster samples."""
    confusion = np.asarray(confusion)
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    correct = np.diagonal(confusion, axis1=-2, axis2=-1)
    rows = support.sum(axis=-1)
    precision = divide(correct, predicted)
    recall = divide(correct, support)
    f1 = divide(2 * correct, support + predicted)
    result = {
        "accuracy": correct.sum(axis=-1) / rows,
        "balanced_accuracy": recall.sum(axis=-1) / (support > 0).sum(axis=-1),
        "macro_f1": f1.mean(axis=-1), "log_loss": loss_sum / rows,
        "multiclass_brier_score": brier_sum / rows,
    }
    for i, label in enumerate(CLASSES):
        for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            result[f"{label}_{name}"] = values[..., i]
    return result


def point_metrics(stats):
    confusion = stats["confusion"].sum(axis=0)
    flat = metric_arrays(confusion, stats["log_loss_sum"].sum(), stats["brier_sum"].sum())
    return {
        **{name: float(flat[name]) for name in PRIMARY_METRICS},
        "per_class": {label: {
            **{metric: float(flat[f"{label}_{metric}"]) for metric in ("precision", "recall", "f1")},
            "support": int(confusion[i].sum()),
        } for i, label in enumerate(CLASSES)},
        "confusion_matrix": confusion.tolist(), "class_order": list(CLASSES),
        "rows": int(confusion.sum()), "subjects": len(stats["subjects"]),
    }


def per_subject_table(all_stats):
    records = []
    for model_name, stats in all_stats.items():
        metrics = metric_arrays(stats["confusion"], stats["log_loss_sum"], stats["brier_sum"])
        for i, subject in enumerate(stats["subjects"]):
            row = {"model": model_name, "subject_id": subject,
                   "rows": int(stats["confusion"][i].sum())}
            row.update({name: float(values[i]) for name, values in metrics.items()})
            for a, true_class in enumerate(CLASSES):
                for b, predicted_class in enumerate(CLASSES):
                    row[f"true_{true_class}_predicted_{predicted_class}"] = int(stats["confusion"][i, a, b])
            records.append(row)
    return pd.DataFrame(records)


def cluster_bootstrap(all_stats, repeats=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Resample whole subjects with replacement; use identical draws for all models."""
    if type(repeats) is not int or repeats < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    reference = next(iter(all_stats.values()))["subjects"]
    if any(not np.array_equal(stats["subjects"], reference) for stats in all_stats.values()):
        raise ValueError("Paired bootstrap requires identical ordered subject sets.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(reference), size=(repeats, len(reference)))
    weights = np.array([np.bincount(row, minlength=len(reference)) for row in draws])
    report = {
        "method": "paired subject-cluster percentile bootstrap",
        "confidence_level": 0.95, "replicates": repeats, "seed": seed,
        "resampling_unit": "whole subject with all original windows; replacement",
        "estimand": "pooled window metrics across sampled subjects",
        "limitation": "Conditional on the fixed fitted models and these synthetic subjects; no model refitting or calibration uncertainty is included.",
        "subjects": reference.tolist(), "models": {},
    }
    samples = {}
    for name, stats in all_stats.items():
        confusion = np.einsum("bs,sij->bij", weights, stats["confusion"])
        values = metric_arrays(confusion, weights @ stats["log_loss_sum"], weights @ stats["brier_sum"])
        estimate = metric_arrays(stats["confusion"].sum(axis=0), stats["log_loss_sum"].sum(), stats["brier_sum"].sum())
        report["models"][name] = {
            metric: {"estimate": float(estimate[metric]),
                     "lower": float(np.quantile(value, 0.025)), "upper": float(np.quantile(value, 0.975))}
            for metric, value in values.items()
        }
        report["models"][name]["confusion_matrix_interval"] = {
            "lower": np.quantile(confusion, 0.025, axis=0).tolist(),
            "upper": np.quantile(confusion, 0.975, axis=0).tolist(),
        }
        samples[name] = values
    if {"calibrated", "uncalibrated"} <= set(samples):
        report["paired_calibrated_minus_uncalibrated"] = {
            metric: {
                "estimate": report["models"]["calibrated"][metric]["estimate"] - report["models"]["uncalibrated"][metric]["estimate"],
                "lower": float(np.quantile(samples["calibrated"][metric] - samples["uncalibrated"][metric], 0.025)),
                "upper": float(np.quantile(samples["calibrated"][metric] - samples["uncalibrated"][metric], 0.975)),
            } for metric in PRIMARY_METRICS
        }
    return report, weights


def reliability_bins(labels, probabilities, bins=RELIABILITY_BINS):
    labels = checked_labels(labels)
    probabilities = checked_probabilities(probabilities, len(labels))
    edges = np.linspace(0, 1, bins + 1)
    series = {}
    scores = [(label, probabilities[:, i], (labels == i).astype(float)) for i, label in enumerate(CLASSES)]
    scores.append(("top_label", probabilities.max(axis=1), (probabilities.argmax(axis=1) == labels).astype(float)))
    for name, predicted, observed in scores:
        assignments = np.minimum((predicted * bins).astype(int), bins - 1)
        counts = np.bincount(assignments, minlength=bins)
        sum_predicted = np.bincount(assignments, weights=predicted, minlength=bins)
        sum_observed = np.bincount(assignments, weights=observed, minlength=bins)
        table = []
        ece = 0.0
        for i, count in enumerate(counts):
            p = float(sum_predicted[i] / count) if count else None
            o = float(sum_observed[i] / count) if count else None
            if count:
                ece += count / len(labels) * abs(p - o)
            table.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                          "count": int(count), "mean_probability": p, "observed_frequency": o})
        series[name] = {"bins": table, "expected_calibration_error": float(ece)}
    return {"strategy": "10 fixed equal-width bins; left-closed, final bin includes 1",
            "ece_note": "Descriptive and bin-dependent; not used to select or tune calibration.",
            "series": series}
