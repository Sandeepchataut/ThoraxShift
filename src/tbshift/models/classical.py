"""Classical classifiers: standardise -> RBF-SVM."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_C = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_GAMMA_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)


def svm_rbf(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
    ])


def svm_grid(n_features: int, C=DEFAULT_C, gamma_factors=DEFAULT_GAMMA_FACTORS) -> dict:
    """Gamma grid relative to 1/n_features (= sklearn 'scale' after standardisation)."""
    return {"svm__C": list(C), "svm__gamma": [f / n_features for f in gamma_factors]}


def scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous score, higher = abnormal. Uses decision_function where available."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
