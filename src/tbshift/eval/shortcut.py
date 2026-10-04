"""Dataset-identification probe (shortcut control).

Within one label class (e.g. normal images only), can a fixed-capacity linear classifier tell
which dataset an image came from? Disease is held constant, so success means the representation
encodes acquisition/site information that a disease classifier could exploit as a shortcut.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tbshift.models.classical import BlockWeighter

PROBE_C = 1.0  # fixed capacity: the same regularisation for every representation; never tuned


def probe_model(block_sizes=None) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("blocks", BlockWeighter(block_sizes)),
        ("lr", LogisticRegression(C=PROBE_C, class_weight="balanced", max_iter=5000)),
    ])


def _oof_scores(X, d, n_splits, seed, block_sizes):
    oof = np.zeros(len(d))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, d):
        m = probe_model(block_sizes).fit(X[tr], d[tr])
        oof[te] = m.decision_function(X[te])
    return oof


def dataset_id_probe(X_a: np.ndarray, X_b: np.ndarray, n_splits: int = 5, repeats: int = 5,
                     seed: int = 0, block_sizes=None, n_perm: int = 200) -> dict:
    """Out-of-fold balanced accuracy and AUC for 'dataset A vs dataset B', repeated CV,
    with a label-permutation null.

    Chance level is 0.5 for both metrics, whatever the class sizes (balanced accuracy and
    class-balanced weights). The permutation null shuffles the dataset labels and reruns the
    first CV repeat, giving the distribution of balanced accuracy under "no site information" for
    THIS representation (its dimension and sample sizes). p = (k + 1) / (n_perm + 1), where k counts
    null values >= the observed first-repeat balanced accuracy.
    """
    X = np.vstack([X_a, X_b])
    d = np.r_[np.zeros(len(X_a), int), np.ones(len(X_b), int)]
    n_splits = min(n_splits, len(X_a), len(X_b))
    if n_splits < 2:
        raise ValueError("each dataset needs at least 2 images for the probe")
    bas, aucs = [], []
    for r in range(repeats):
        oof = _oof_scores(X, d, n_splits, seed + r, block_sizes)
        bas.append(balanced_accuracy_score(d, (oof > 0).astype(int)))
        aucs.append(roc_auc_score(d, oof))
    out = {"n_a": int(len(X_a)), "n_b": int(len(X_b)), "repeats": repeats, "n_splits": n_splits,
           "balanced_accuracy_mean": float(np.mean(bas)),
           "balanced_accuracy_sd": float(np.std(bas, ddof=1)) if repeats > 1 else 0.0,
           "auc_mean": float(np.mean(aucs)),
           "balanced_accuracy_per_repeat": [float(v) for v in bas]}
    if n_perm:
        rng = np.random.default_rng(seed + 99_991)
        null = np.array([balanced_accuracy_score(dp, (_oof_scores(X, dp, n_splits, seed, block_sizes) > 0).astype(int))
                         for dp in (rng.permutation(d) for _ in range(n_perm))])
        out.update({"n_perm": n_perm, "null_ba_mean": float(null.mean()),
                    "null_ba_q95": float(np.quantile(null, 0.95)),
                    "perm_p": float(((null >= bas[0]).sum() + 1) / (n_perm + 1))})
    return out
