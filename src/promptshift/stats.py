"""Metrics, stratified paired bootstrap, Holm correction, Krippendorff's alpha.

Everything here is plain numpy so it can be unit-tested without any data.
"""

import numpy as np


# ------------------------------------------------------------------ metrics
def _safe_div(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def confusion(y, p):
    """Counts along the last axis; works for 1-D arrays and (B, N) arrays."""
    y = np.asarray(y)
    p = np.asarray(p)
    tp = np.sum((y == 1) & (p == 1), axis=-1)
    fp = np.sum((y == 0) & (p == 1), axis=-1)
    fn = np.sum((y == 1) & (p == 0), axis=-1)
    tn = np.sum((y == 0) & (p == 0), axis=-1)
    return tp, fp, fn, tn


def macro_f1(y, p):
    tp, fp, fn, tn = confusion(y, p)
    f1_off = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_not = _safe_div(2 * tn, 2 * tn + fn + fp)
    return (f1_off + f1_not) / 2


def recall_offensive(y, p):
    tp, _, fn, _ = confusion(y, p)
    return _safe_div(tp, tp + fn)


def recall_not_offensive(y, p):
    _, fp, _, tn = confusion(y, p)
    return _safe_div(tn, tn + fp)


def precision_offensive(y, p):
    tp, fp, _, _ = confusion(y, p)
    return _safe_div(tp, tp + fp)


def accuracy(y, p):
    return np.mean(np.asarray(y) == np.asarray(p), axis=-1)


def prevalence(p):
    return np.mean(np.asarray(p) == 1, axis=-1)


METRICS = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "offensive_recall": recall_offensive,
    "not_offensive_recall": recall_not_offensive,
    "offensive_precision": precision_offensive,
}


def mean_pairwise_disagreement(columns):
    """Per-post share of disagreeing pairs among the given label columns."""
    columns = [np.asarray(c) for c in columns]
    pairs = [
        (columns[i] != columns[j]).astype(float)
        for i in range(len(columns)) for j in range(i + 1, len(columns))
    ]
    return np.mean(pairs, axis=0)


# ---------------------------------------------------------------- bootstrap
def stratified_indices(y, n_boot, rng):
    """(n_boot, N) resampling indices, resampling within each gold class."""
    y = np.asarray(y)
    blocks = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        blocks.append(members[rng.integers(0, len(members), size=(n_boot, len(members)))])
    return np.concatenate(blocks, axis=1).astype(np.int32)


def summarise(observed, boot, alternative="two-sided", level=0.95):
    """Percentile CI and bootstrap p-value against zero."""
    boot = np.asarray(boot, dtype=float)
    n = len(boot)
    tail = (1 - level) / 2
    low, high = np.quantile(boot, [tail, 1 - tail])
    one_sided_lower = np.quantile(boot, 1 - level)
    one_sided_upper = np.quantile(boot, level)
    p_le = (1 + np.sum(boot <= 0)) / (n + 1)
    p_ge = (1 + np.sum(boot >= 0)) / (n + 1)
    if alternative == "greater":
        p_value = p_le
    elif alternative == "less":
        p_value = p_ge
    else:
        p_value = min(1.0, 2 * min(p_le, p_ge))
    return {
        "estimate": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "one_sided_lower": float(one_sided_lower),
        "one_sided_upper": float(one_sided_upper),
        "p_value": float(p_value),
        "alternative": alternative,
    }


def holm(p_values):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def size_verdict(result, sesoi):
    """Practical-size verdict, kept separate from statistical significance."""
    if result["alternative"] == "greater":
        if result["one_sided_lower"] > sesoi:
            return "materially positive (lower bound above SESOI)"
        if result["one_sided_upper"] < sesoi:
            return "smaller than SESOI (upper bound below SESOI)"
        return "size inconclusive"
    low, high = result["ci_low"], result["ci_high"]
    if low > sesoi:
        return "materially positive"
    if high < -sesoi:
        return "materially negative"
    if -sesoi < low and high < sesoi:
        return "practically equivalent (CI within ±SESOI)"
    return "size inconclusive"


# ------------------------------------------------------- Krippendorff's alpha
def krippendorff_alpha_nominal(data):
    """Nominal alpha. data: (units, coders) array, NaN for missing values."""
    data = np.asarray(data, dtype=float)
    values = np.unique(data[~np.isnan(data)])
    if len(values) < 2:
        return 1.0
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for unit in data:
        present = unit[~np.isnan(unit)]
        m = len(present)
        if m < 2:
            continue
        counts = np.zeros(len(values))
        for value in present:
            counts[index[value]] += 1
        pair_matrix = np.outer(counts, counts) - np.diag(counts)
        coincidence += pair_matrix / (m - 1)
    totals = coincidence.sum(axis=1)
    n = totals.sum()
    observed = (n - np.trace(coincidence)) / n
    expected = (n * n - np.sum(totals ** 2)) / (n * (n - 1))
    return float(1 - observed / expected) if expected > 0 else 1.0
