"""Classical classifiers: standardise -> per-block weighting -> RBF-SVM."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_C = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_GAMMA_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)


class BlockWeighter(BaseEstimator, TransformerMixin):
    """Scale each feature block by 1/sqrt(block dimension) after standardisation.

    With standardised features, a block of d columns contributes ~d to a squared Euclidean
    distance, so without weighting the largest block (e.g. HOG) dominates the RBF kernel.
    After weighting, every block contributes ~1. Stateless: nothing is learned from data.
    """

    def __init__(self, block_sizes: tuple[int, ...] | None = None):
        self.block_sizes = block_sizes

    def fit(self, X, y=None):
        if self.block_sizes is not None and sum(self.block_sizes) != X.shape[1]:
            raise ValueError(f"block sizes {self.block_sizes} do not sum to {X.shape[1]} columns")
        return self

    def transform(self, X):
        if self.block_sizes is None:
            return X
        w = np.concatenate([np.full(d, 1.0 / np.sqrt(d)) for d in self.block_sizes])
        return X * w


def svm_rbf(seed: int = 0, block_sizes: tuple[int, ...] | None = None) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("blocks", BlockWeighter(block_sizes)),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
    ])


def svm_grid(n_features: int, C=DEFAULT_C, gamma_factors=DEFAULT_GAMMA_FACTORS,
             n_blocks: int | None = None) -> dict:
    """Gamma grid relative to the expected squared-distance scale.

    Unweighted standardised features: scale = n_features (sklearn 'scale').
    Block-weighted features: every block contributes ~1, so scale = number of blocks.
    """
    scale = n_blocks if n_blocks else n_features
    return {"svm__C": list(C), "svm__gamma": [f / scale for f in gamma_factors]}


def scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous score, higher = abnormal. Uses decision_function where available."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
