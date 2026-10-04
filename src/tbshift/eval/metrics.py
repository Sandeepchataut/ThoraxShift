"""Evaluation metrics: ROC AUC with DeLong variance, paired DeLong test, stratified bootstrap,
threshold metrics with Wilson intervals.

The DeLong variance uses the fast midrank formulation (O(n log n) per score vector).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (1-based, ties averaged) of a 1-D array."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _check_binary(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int).ravel()
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y_true must be binary 0/1")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("y_true must contain both classes")
    return y


def delong_auc_cov(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their DeLong covariance matrix for k score vectors on the same samples.

    Parameters
    ----------
    y_true : (n,) binary labels, 1 = positive.
    scores : (n,) or (k, n) array of scores; higher means more likely positive.

    Returns
    -------
    aucs : (k,)
    cov : (k, k)
    """
    y = _check_binary(y_true)
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    if s.shape[1] != len(y):
        raise ValueError("scores must have shape (k, n_samples)")
    pos = s[:, y == 1]
    neg = s[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = s.shape[0]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


@dataclass(frozen=True)
class AUCResult:
    auc: float
    ci_low: float
    ci_high: float
    se: float
    n_pos: int
    n_neg: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def auc_delong_ci(y_true, scores, alpha: float = 0.05) -> AUCResult:
    """ROC AUC with a DeLong normal-approximation confidence interval, clipped to [0, 1]."""
    y = _check_binary(y_true)
    aucs, cov = delong_auc_cov(y, scores)
    auc = float(aucs[0])
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return AUCResult(
        auc=auc,
        ci_low=float(max(0.0, auc - z * se)),
        ci_high=float(min(1.0, auc + z * se)),
        se=se,
        n_pos=int(y.sum()),
        n_neg=int(len(y) - y.sum()),
    )


def delong_test(y_true, scores_a, scores_b) -> dict:
    """Two-sided paired DeLong test for AUC(a) != AUC(b) on the same samples."""
    aucs, cov = delong_auc_cov(y_true, np.vstack([scores_a, scores_b]))
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        # Identical (or perfectly co-ranked) score vectors: no evidence of a difference.
        # Non-zero difference with zero variance is undefined; never report it as p = 0.
        p = 1.0 if diff == 0 else float("nan")
        z = 0.0 if diff == 0 else float("nan")
    else:
        z = diff / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p": p}


def stratified_bootstrap_indices(y_true, n_boot: int, rng: np.random.Generator):
    """Yield index arrays that resample positives and negatives separately (class sizes fixed)."""
    y = _check_binary(y_true)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])


def bootstrap_auc_ci(y_true, scores, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile stratified-bootstrap CI for ROC AUC. Returns (auc, low, high)."""
    from sklearn.metrics import roc_auc_score

    y = _check_binary(y_true)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([roc_auc_score(y[idx], s[idx])
                      for idx in stratified_bootstrap_indices(y, n_boot, rng)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def youden_threshold(y_true, scores) -> float:
    """Threshold maximising sensitivity + specificity - 1. Positive iff score >= threshold."""
    from sklearn.metrics import roc_curve

    y = _check_binary(y_true)
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    t = float(thr[best])
    return t if np.isfinite(t) else float(np.max(scores))


def threshold_metrics(y_true, scores, threshold: float) -> dict:
    """Sensitivity, specificity, accuracy, precision at a fixed threshold, with Wilson CIs."""
    y = _check_binary(y_true)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out = {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    for name, k, n in [("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp),
                       ("accuracy", tp + tn, len(y)), ("precision", tp, tp + fp)]:
        out[name] = k / n if n else float("nan")
        out[f"{name}_ci"] = wilson_ci(k, n)
    return out


# ---------------------------------------------------------------------------------------------
# Paired bootstrap on a shared set of images, transfer gaps, multiple-comparison correction
# ---------------------------------------------------------------------------------------------

def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def paired_bootstrap(y_true, score_vectors, statistic, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Stratified bootstrap of statistic(y, *score_vectors) over a shared set of images.

    Every score vector is resampled with the SAME indices, so correlation between models (or
    between in-domain and cross-domain scores of the same images) is preserved.
    Returns the point estimate, a percentile CI, and a two-sided bootstrap p-value for the
    hypothesis statistic == 0.
    """
    y = _check_binary(y_true)
    vecs = [np.asarray(v, dtype=float) for v in score_vectors]
    if any(len(v) != len(y) for v in vecs):
        raise ValueError("all score vectors must align with y_true")
    point = float(statistic(y, *vecs))
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(y[i], *[v[i] for v in vecs])
                      for i in stratified_bootstrap_indices(y, n_boot, rng)], dtype=float)
    finite = boots[np.isfinite(boots)]
    nan_fraction = float(1 - len(finite) / len(boots))
    if len(finite) == 0:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_boot": float("nan"), "n_boot": n_boot, "nan_fraction": nan_fraction, "seed": seed}
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    # (k + 1) / (B + 1): a bootstrap p-value is never exactly 0.
    b = len(finite)
    p = float(min(1.0, 2 * min(((finite <= 0).sum() + 1) / (b + 1), ((finite >= 0).sum() + 1) / (b + 1))))
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "p_boot": p,
            "n_boot": n_boot, "nan_fraction": nan_fraction, "seed": seed}


def auc_difference(y_true, scores_a, scores_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on the same images: paired DeLong test plus paired bootstrap CI."""
    d = delong_test(y_true, scores_a, scores_b)
    b = paired_bootstrap(y_true, [scores_a, scores_b], lambda y, a, c: _auc(y, a) - _auc(y, c),
                         n_boot, seed)
    return {"auc_a": d["auc_a"], "auc_b": d["auc_b"], "diff": d["diff"], "delong_z": d["z"],
            "delong_p": d["p"], "boot_ci_low": b["ci_low"], "boot_ci_high": b["ci_high"],
            "boot_p": b["p_boot"]}


def transfer_gap(y_target, scores_in_domain, scores_cross, n_boot: int = 2000, seed: int = 0) -> dict:
    """Transfer gap on target T: AUC_in(T) - AUC_cross(S->T), both on the same T images.

    scores_in_domain: out-of-fold scores on T from in-domain CV on T.
    scores_cross: scores on T from a model trained on S.
    Also returns the relative gap (AUC_in - AUC_cross) / (AUC_in - 0.5), i.e. the fraction of
    above-chance in-domain performance lost under transfer.
    """
    absolute = paired_bootstrap(y_target, [scores_in_domain, scores_cross],
                                lambda y, a, c: _auc(y, a) - _auc(y, c), n_boot, seed)

    def rel(y, a, c):
        auc_in = _auc(y, a)
        return (auc_in - _auc(y, c)) / (auc_in - 0.5) if auc_in > 0.5 else np.nan

    relative = paired_bootstrap(y_target, [scores_in_domain, scores_cross], rel, n_boot, seed)
    # Descriptive only: the ratio is unstable near AUC_in = 0.5, so no hypothesis test is reported.
    relative["p_boot"] = None
    relative["descriptive_only"] = True
    return {"auc_in_domain": _auc(y_target, scores_in_domain), "auc_cross": _auc(y_target, scores_cross),
            "gap": absolute, "relative_gap": relative}


def gap_difference(y_target, in_a, cross_a, in_b, cross_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """(gap of model A) - (gap of model B) on the same target images, paired bootstrap."""
    def stat(y, ia, ca, ib, cb):
        return (_auc(y, ia) - _auc(y, ca)) - (_auc(y, ib) - _auc(y, cb))
    return paired_bootstrap(y_target, [in_a, cross_a, in_b, cross_b], stat, n_boot, seed)


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1).

    The family size m is ALWAYS the full length of the input. A NaN p-value (e.g. a test that
    could not be computed) is treated as p = 1 for the correction, so it cannot make the other
    tests more lenient, and its own adjusted value is returned as NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    nan = np.isnan(p)
    pv = np.where(nan, 1.0, p)
    m = len(pv)
    order = np.argsort(pv, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pv[idx])
        adj[idx] = min(1.0, running)
    adj[nan] = np.nan
    return adj
