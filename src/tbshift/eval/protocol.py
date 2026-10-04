"""Evaluation protocols. Every fitted step lives inside the estimator, so it is refit on the
training portion only (no leakage from outer test folds or from the target domain).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from tbshift.models.classical import scores as model_scores
from tbshift.eval.metrics import youden_threshold


def _search(estimator, grid: dict, inner_splits: int, seed: int, n_jobs: int = -1) -> GridSearchCV:
    inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    return GridSearchCV(clone(estimator), grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs,
                        refit=True, error_score="raise")


def nested_cv_oof(X: np.ndarray, y: np.ndarray, estimator, grid: dict,
                  outer_splits: int = 5, inner_splits: int = 5, seed: int = 0,
                  n_jobs: int = -1) -> dict:
    """Out-of-fold scores from nested CV. Each sample is scored exactly once, by a model that
    never saw it, with hyperparameters chosen by inner CV on that model's training portion."""
    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan)
    fold = np.full(len(y), -1, dtype=int)
    chosen = []
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        gs = _search(estimator, grid, inner_splits, seed, n_jobs).fit(X[tr], y[tr])
        oof[te] = model_scores(gs.best_estimator_, X[te])
        fold[te] = k
        chosen.append({"fold": k, **gs.best_params_, "inner_auc": float(gs.best_score_)})
    assert not np.isnan(oof).any()
    return {"scores": oof, "fold": fold, "chosen_params": chosen}


def source_to_target(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, estimator, grid: dict,
                     inner_splits: int = 5, seed: int = 0) -> dict:
    """Fit on the whole source (hyperparameters by inner CV), score the target once.

    The operating threshold is chosen on the SOURCE only: Youden's J on source out-of-fold
    scores from the selected configuration. It is then applied unchanged to the target.
    """
    ys = np.asarray(ys).astype(int)
    gs = _search(estimator, grid, inner_splits, seed).fit(Xs, ys)
    best = gs.best_estimator_
    # Different fold assignment from the grid search, so the threshold is not chosen on the
    # same folds that selected the configuration.
    cv = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + 10_000)
    src_oof = cross_val_predict(clone(best), Xs, ys, cv=cv,
                                method="decision_function" if hasattr(best, "decision_function")
                                else "predict_proba")
    if src_oof.ndim == 2:
        src_oof = src_oof[:, 1]
    return {
        "target_scores": model_scores(best, Xt),
        "source_oof_scores": np.asarray(src_oof, dtype=float),
        "threshold": youden_threshold(ys, src_oof),
        "chosen_params": {**gs.best_params_, "inner_auc": float(gs.best_score_)},
    }
