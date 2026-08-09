"""Final-comparison statistics: paired stratified bootstrap, DeLong, McNemar, Holm (spec 17).

Implemented with numpy + stdlib `math` only (no scipy/statsmodels), matching this project's
convention of keeping notebooks/utility modules importable without heavy optional deps. The
normal and chi-square(df=1) CDFs needed by DeLong/McNemar are both expressible through
`math.erf`, so no numerical-integration dependency is required.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi2_1df_cdf(x: float) -> float:
    """P(chi2_1 <= x) = P(-sqrt(x) <= Z <= sqrt(x)) = 2*Phi(sqrt(x)) - 1, for a 1-dof chi-square."""
    if x < 0:
        return 0.0
    return 2.0 * _normal_cdf(math.sqrt(x)) - 1.0


# --- paired, stratified, patient-level bootstrap -----------------------------------------------

def paired_stratified_bootstrap(labels: Sequence[int], probs_a: Sequence[float], probs_b: Sequence[float],
                                 metric_fn: Callable[[Sequence[int], Sequence[float]], float],
                                 n_bootstrap: int = 2000, seed: int = 42) -> dict:
    """Resample patients with replacement, stratified by class so each resample keeps the same
    positive/negative counts as the original set, using the *same* resampled indices for both
    models A and B (paired) so the difference distribution reflects patient-level correlation,
    not independent noise. Degenerate resamples (a stratum collapsing to one class only) are
    skipped rather than allowed to corrupt metric_fn with an undefined AUC.
    """
    y = np.asarray(labels)
    a = np.asarray(probs_a, dtype=np.float64)
    b = np.asarray(probs_b, dtype=np.float64)
    if len(y) != len(a) or len(y) != len(b):
        raise ValueError("labels, probs_a, probs_b must be the same length (same patient order)")

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("bootstrap requires at least one positive and one negative patient")

    rng = np.random.RandomState(seed)
    diffs, values_a, values_b, skipped = [], [], [], 0
    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        y_s, a_s, b_s = y[idx], a[idx], b[idx]
        if len(set(y_s.tolist())) < 2:
            skipped += 1
            continue
        va = metric_fn(y_s, a_s)
        vb = metric_fn(y_s, b_s)
        values_a.append(va); values_b.append(vb); diffs.append(vb - va)

    diffs_arr = np.asarray(diffs)
    lo, hi = np.percentile(diffs_arr, [2.5, 97.5])
    # two-tailed bootstrap p-value: proportion of resamples on the other side of zero from the
    # observed direction, doubled, capped at 1.
    observed = float(np.mean(values_b) - np.mean(values_a))
    side = np.mean(diffs_arr <= 0) if observed > 0 else np.mean(diffs_arr >= 0)
    p_value = min(1.0, 2.0 * float(side))

    return {
        "n_bootstrap": n_bootstrap, "n_skipped_degenerate": skipped, "seed": seed,
        "mean_a": float(np.mean(values_a)), "mean_b": float(np.mean(values_b)),
        "mean_diff_b_minus_a": float(np.mean(diffs_arr)), "ci_95_low": float(lo), "ci_95_high": float(hi),
        "p_value_two_sided": p_value,
    }


# --- DeLong test for two correlated ROC-AUCs ----------------------------------------------------

def _psi_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """psi[i,j] = 1 if x_i>y_j, 0.5 if equal, 0 if x_i<y_j (Sen/DeLong structural kernel)."""
    diff = x[:, None] - y[None, :]
    return np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))


def _structural_components(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    psi = _psi_matrix(pos_scores, neg_scores)
    v10 = psi.mean(axis=1)  # per-positive component, length m
    v01 = psi.mean(axis=0)  # per-negative component, length n
    auc = float(v10.mean())
    return v10, v01, auc


def delong_test(labels: Sequence[int], probs_a: Sequence[float], probs_b: Sequence[float]) -> dict:
    """Two correlated ROC-AUCs on the *same* patients (same order); errors explicitly on any
    length/label mismatch rather than silently comparing misaligned patients.
    """
    y = np.asarray(labels)
    a = np.asarray(probs_a, dtype=np.float64)
    b = np.asarray(probs_b, dtype=np.float64)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("labels, probs_a, probs_b must be perfectly patient-aligned (same length)")

    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("DeLong test requires at least one positive and one negative patient")

    v10_a, v01_a, auc_a = _structural_components(a[pos], a[neg])
    v10_b, v01_b, auc_b = _structural_components(b[pos], b[neg])

    m, n = pos.sum(), neg.sum()
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1) if n > 1 else np.zeros((2, 2))

    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    diff = auc_b - auc_a
    if var <= 0:
        z, p_value = 0.0, 1.0
    else:
        z = diff / math.sqrt(var)
        p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "diff_b_minus_a": diff, "variance": float(var), "z": z, "p_value": p_value,
            "n_positive": int(m), "n_negative": int(n)}


# --- McNemar test (locked threshold) -------------------------------------------------------------

def mcnemar_test(labels: Sequence[int], preds_a: Sequence[int], preds_b: Sequence[int], exact_threshold: int = 25) -> dict:
    """preds_* are already-thresholded 0/1 predictions at the locked decision threshold — no
    threshold is recomputed here. Uses the exact binomial test for small discordant counts
    (n01+n10 < exact_threshold), the continuity-corrected chi-square approximation otherwise.
    """
    y = np.asarray(labels)
    pa = np.asarray(preds_a)
    pb = np.asarray(preds_b)
    if not (len(y) == len(pa) == len(pb)):
        raise ValueError("labels, preds_a, preds_b must be perfectly patient-aligned")

    correct_a = (pa == y)
    correct_b = (pb == y)
    n10 = int(np.sum(correct_a & ~correct_b))  # a correct, b wrong
    n01 = int(np.sum(~correct_a & correct_b))  # a wrong, b correct
    n_discordant = n10 + n01

    if n_discordant == 0:
        return {"n10": n10, "n01": n01, "n_discordant": 0, "method": "degenerate_no_discordant_pairs", "p_value": 1.0}

    if n_discordant < exact_threshold:
        k = min(n10, n01)
        p_value = min(1.0, 2.0 * sum(math.comb(n_discordant, i) * (0.5 ** n_discordant) for i in range(0, k + 1)))
        method = "exact_binomial"
        statistic = None
    else:
        statistic = (abs(n10 - n01) - 1) ** 2 / n_discordant
        p_value = 1.0 - _chi2_1df_cdf(statistic)
        method = "chi_square_continuity_corrected"

    return {"n10": n10, "n01": n01, "n_discordant": n_discordant, "method": method, "statistic": statistic, "p_value": p_value}


# --- Holm-Bonferroni step-down correction, applied strictly within one family -------------------

def holm_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Family-wise Holm correction. Caller must pass exactly one family's p-values (spec 17.4:
    primary_roc_auc, primary_pr_auc, primary_mcnemar, secondary_* are separate families and must
    never be pooled into one correction).
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    adjusted: dict[str, float] = {}
    reject: dict[str, bool] = {}
    running_max = 0.0
    stopped = False
    for rank, (name, p) in enumerate(items, start=1):
        adj = min(1.0, p * (n - rank + 1))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
        reject[name] = (not stopped) and (running_max <= alpha)
        if not reject[name]:
            stopped = True
    return {"alpha": alpha, "n_comparisons": n, "adjusted_p_values": adjusted, "reject_null": reject}
