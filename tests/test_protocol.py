import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tbshift.eval.protocol import nested_cv_oof, source_to_target
from tbshift.models.classical import svm_grid, svm_rbf


def _data(n=120, d=5, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    X = rng.normal(size=(n, d)); X[:, 0] += shift * y
    return X, y


class _RecordingClf(BaseEstimator, ClassifierMixin):
    """Records the row ids it was fitted on (column 0 holds the id)."""
    seen: list = []

    def __init__(self, c=1.0):
        self.c = c

    def fit(self, X, y):
        _RecordingClf.seen.append(set(X[:, 0].astype(int)))
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        return X[:, 1]


def test_nested_cv_scores_every_sample_once_and_never_trains_on_it():
    X, y = _data()
    ids = np.arange(len(y))
    Xi = np.c_[ids, X]
    _RecordingClf.seen = []
    res = nested_cv_oof(Xi, y, _RecordingClf(), {"c": [1.0, 2.0]}, outer_splits=5,
                        inner_splits=3, seed=0, n_jobs=1)
    assert np.isfinite(res["scores"]).all()
    assert sorted(np.bincount(res["fold"])) == [24] * 5
    # Each outer test fold's ids must be absent from every fit made for that fold.
    test_sets = [set(ids[res["fold"] == k]) for k in range(5)]
    fits_per_outer = len(_RecordingClf.seen) // 5
    for k in range(5):
        for fitted in _RecordingClf.seen[k * fits_per_outer:(k + 1) * fits_per_outer]:
            assert not (fitted & test_sets[k])


def test_svm_learns_separable_signal():
    X, y = _data(n=200, shift=3.0)
    res = nested_cv_oof(X, y, svm_rbf(), svm_grid(X.shape[1], C=(1.0,), gamma_factors=(1.0,)),
                        outer_splits=3, inner_splits=3)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, res["scores"]) > 0.9


def test_source_to_target_threshold_uses_source_only():
    Xs, ys = _data(seed=1)
    Xt, yt = _data(seed=2)
    grid = svm_grid(Xs.shape[1], C=(1.0,), gamma_factors=(1.0,))
    r1 = source_to_target(Xs, ys, Xt, svm_rbf(), grid, inner_splits=3)
    # The API never receives target labels; the result must also be deterministic.
    r2 = source_to_target(Xs, ys, Xt, svm_rbf(), grid, inner_splits=3)
    assert r1["threshold"] == r2["threshold"]
    np.testing.assert_array_equal(r1["target_scores"], r2["target_scores"])
    assert len(r1["target_scores"]) == len(yt)
