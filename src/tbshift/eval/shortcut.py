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


def dataset_id_probe(X_a: np.ndarray, X_b: np.ndarray, n_splits: int = 5, repeats: int = 5,
                     seed: int = 0, block_sizes=None) -> dict:
    """Out-of-fold balanced accuracy and AUC for 'dataset A vs dataset B', repeated CV.

    Chance level is 0.5 for both metrics, whatever the class sizes (balanced accuracy and
    class-balanced weights).
    """
    X = np.vstack([X_a, X_b])
    d = np.r_[np.zeros(len(X_a), int), np.ones(len(X_b), int)]
    n_splits = min(n_splits, len(X_a), len(X_b))
    if n_splits < 2:
        raise ValueError("each dataset needs at least 2 images for the probe")
    bas, aucs = [], []
    for r in range(repeats):
        oof = np.zeros(len(d))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, d):
            m = probe_model(block_sizes).fit(X[tr], d[tr])
            oof[te] = m.decision_function(X[te])
        bas.append(balanced_accuracy_score(d, (oof > 0).astype(int)))
        aucs.append(roc_auc_score(d, oof))
    return {"n_a": int(len(X_a)), "n_b": int(len(X_b)), "repeats": repeats, "n_splits": n_splits,
            "balanced_accuracy_mean": float(np.mean(bas)),
            "balanced_accuracy_sd": float(np.std(bas, ddof=1)) if repeats > 1 else 0.0,
            "auc_mean": float(np.mean(aucs)),
            "balanced_accuracy_per_repeat": [float(v) for v in bas]}
