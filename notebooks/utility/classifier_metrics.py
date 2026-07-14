"""Validation/test metric suite for compact downstream classification.

Implemented with numpy + stdlib only (no scipy/sklearn/statsmodels) so it can be imported and
unit-tested in the lightweight `base` conda env, matching this project's existing convention of
keeping widely-reused notebooks/utility modules free of heavy import-time dependencies.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def confusion_counts(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> dict:
    y = _as_array(labels); p = _as_array(probabilities)
    pred = (p >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mann-Whitney U formulation: P(score(positive) > score(negative)), ties count as 0.5."""
    y = _as_array(labels); p = _as_array(probabilities)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("ROC-AUC undefined: need at least one positive and one negative label")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    rank = 1
    while i < len(sorted_p):
        j = i
        while j < len(sorted_p) and sorted_p[j] == sorted_p[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _pr_points(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = _as_array(labels); p = _as_array(probabilities)
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    n_pos = float(np.sum(y == 1))
    if n_pos == 0:
        raise ValueError("PR-AUC undefined: no positive labels")
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    return precision, recall


def pr_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Average precision: sum of precision * delta-recall over the sorted-by-score curve."""
    precision, recall = _pr_points(labels, probabilities)
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum(precision * (recall - recall_prev)))


def youden_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    """Threshold maximizing sensitivity + specificity - 1 over all distinct score cut points."""
    y = _as_array(labels); p = _as_array(probabilities)
    candidates = np.unique(p)
    best = {"threshold": 0.5, "youden_j": -1.0}
    for t in candidates:
        c = confusion_counts(y, p, float(t))
        sens = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        spec = c["tn"] / (c["tn"] + c["fp"]) if (c["tn"] + c["fp"]) else 0.0
        j = sens + spec - 1.0
        if j > best["youden_j"]:
            best = {"threshold": float(t), "youden_j": float(j)}
    return best


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y = _as_array(labels); p = _as_array(probabilities)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(labels: Sequence[int], probabilities: Sequence[float], n_bins: int = 10) -> float:
    y = _as_array(labels); p = _as_array(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(p[mask]))
        bin_acc = float(np.mean(y[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def sensitivity_at_fixed_specificity(labels: Sequence[int], probabilities: Sequence[float],
                                     target_specificity: float = 0.90) -> dict:
    """Highest sensitivity among thresholds meeting the preregistered specificity target."""
    y, p = _as_array(labels), _as_array(probabilities)
    candidates = np.unique(np.concatenate(([1.0 + np.finfo(float).eps], p)))
    feasible = []
    for threshold in candidates:
        counts = confusion_counts(y, p, float(threshold))
        sensitivity = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        specificity = counts["tn"] / (counts["tn"] + counts["fp"]) if counts["tn"] + counts["fp"] else 0.0
        if specificity >= target_specificity:
            feasible.append((sensitivity, specificity, float(threshold)))
    sensitivity, specificity, threshold = max(feasible, default=(0.0, 1.0, 1.0), key=lambda row: (row[0], row[1], row[2]))
    return {"target_specificity": target_specificity, "sensitivity": sensitivity,
            "achieved_specificity": specificity, "threshold": threshold}


def metrics_at_threshold(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> dict:
    c = confusion_counts(labels, probabilities, threshold)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator else 0.0
    return {
        "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity_recall": sensitivity, "specificity": specificity,
        "precision_ppv": precision, "npv": npv, "f1": f1,
        "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "mcc": mcc,
    }


def full_report(labels: Sequence[int], probabilities: Sequence[float], threshold: float | None = None) -> dict:
    """Every metric required by spec section 7.1 / 16, at a given (or Youden-derived) threshold."""
    if threshold is None:
        threshold = youden_threshold(labels, probabilities)["threshold"]
    report = metrics_at_threshold(labels, probabilities, threshold)
    report["roc_auc"] = roc_auc(labels, probabilities)
    report["pr_auc"] = pr_auc(labels, probabilities)
    report["brier_score"] = brier_score(labels, probabilities)
    report["ece"] = expected_calibration_error(labels, probabilities)
    report["sensitivity_at_specificity_0_90"] = sensitivity_at_fixed_specificity(labels, probabilities, 0.90)
    return report


def ensemble_probabilities(per_seed_probabilities: Sequence[Sequence[float]]) -> list[float]:
    """Mean probability across seeds (spec section 6): deterministic, order of seeds irrelevant."""
    arr = np.asarray(per_seed_probabilities, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a (n_seeds, n_samples) array of per-seed probabilities")
    return arr.mean(axis=0).tolist()


def seed_stability(per_seed_metric_values: Sequence[float]) -> dict:
    arr = _as_array(per_seed_metric_values)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)), "max": float(np.max(arr)), "range": float(np.max(arr) - np.min(arr))}
