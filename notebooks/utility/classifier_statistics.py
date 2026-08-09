"""Statistics used by the final v1.1 comparison: paired bootstrap and Holm.

Implemented with NumPy only, keeping the module importable without SciPy or
Statsmodels. Historical unused DeLong and McNemar helpers were removed from the
frozen release because they are not part of the executable v1.1 protocol.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


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


# --- Holm-Bonferroni step-down correction, applied strictly within one family -------------------

def holm_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Correct one declared family of p-like values with Holm's step-down method.

    The v1.1 caller passes the eight protocol-declared PR-AUC bootstrap tail
    areas as one family. This helper does not infer or combine families.
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
