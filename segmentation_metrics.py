"""Canonical binary segmentation metrics for the ResUpNet pipeline.

Hard overlap and confusion metrics are computed from one confusion matrix formed
by pooling every pixel in the evaluated population.  This is micro-averaging;
callers must not average per-slice scores to produce a primary result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real

import numpy as np


METRIC_PROTOCOL_VERSION = "1.0"
PIXEL_MICRO_AGGREGATION = "micro_over_all_pixels"
PRIMARY_METRIC_AGGREGATION = PIXEL_MICRO_AGGREGATION
PRIMARY_METRIC_POPULATION = "all pixels in all selected/capped 2D test slices"
TRAINING_METRIC_AGGREGATION = PIXEL_MICRO_AGGREGATION
TRAINING_METRIC_POPULATION = "all pixels processed in the current epoch phase"
HARD_PREDICTION_RULE = "predicted_probability > threshold"


def _validated_count(name: str, value: Real) -> float:
    count = float(value)
    if not np.isfinite(count) or count < 0:
        raise ValueError(f"{name} must be a finite non-negative count, got {value!r}")
    return count


def confusion_counts_numpy(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return TP, FP, FN, and TN after pooling all array pixels."""

    truth = np.asarray(y_true) > 0.5
    prediction = np.asarray(y_pred) > 0.5
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shapes, got {truth.shape} and {prediction.shape}"
        )
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    tn = int(np.logical_and(~truth, ~prediction).sum())
    return tp, fp, fn, tn


def metrics_from_confusion_counts(tp: Real, fp: Real, fn: Real, tn: Real) -> dict[str, float]:
    """Compute canonical micro metrics from pooled binary confusion counts.

    Undefined ratios use an explicit empty-set policy.  Empty truth and empty
    prediction are a perfect result.  If positives exist but none are predicted,
    foreground precision is zero.  If truth is empty, recall is one.
    """

    tp = _validated_count("tp", tp)
    fp = _validated_count("fp", fp)
    fn = _validated_count("fn", fn)
    tn = _validated_count("tn", tn)

    predicted_positive = tp + fp
    actual_positive = tp + fn
    foreground_union = tp + fp + fn
    total = foreground_union + tn

    dice = 1.0 if foreground_union == 0 else (2.0 * tp) / (2.0 * tp + fp + fn)
    iou = 1.0 if foreground_union == 0 else tp / foreground_union
    precision = (
        tp / predicted_positive
        if predicted_positive > 0
        else (1.0 if actual_positive == 0 else 0.0)
    )
    recall = tp / actual_positive if actual_positive > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    accuracy = (tp + tn) / total if total > 0 else 1.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        # For one binary foreground class, pooled-count F1 is exactly Dice.
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(dice),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def micro_metrics_from_arrays(y_true, y_pred) -> dict[str, float | int]:
    """Compute one canonical micro result across all pixels in two arrays."""

    tp, fp, fn, tn = confusion_counts_numpy(y_true, y_pred)
    return {
        **metrics_from_confusion_counts(tp, fp, fn, tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate_confusion_rows(rows: Iterable[Mapping[str, Real]]) -> dict[str, float | int]:
    """Pool row-level confusion counts before computing the micro metrics."""

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    count = 0
    for row in rows:
        count += 1
        for key in totals:
            totals[key] += int(_validated_count(key, row[key]))
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        **totals,
        **metrics_from_confusion_counts(**totals),
    }
