import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from tbshift.eval.protocol import _FixedConfig, _search, nested_cv_oof, source_to_target
from tbshift.models import heads as hd


def _data(n=160, d=12, shift=1.2, seed=0, tokens=None):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    if tokens:
        X = rng.normal(size=(n, tokens, d))
        X[y == 1, 0, :3] += 3 * shift          # signal in ONE token only (a "lesion")
    else:
        X = rng.normal(size=(n, d))
        X[:, :3] += shift * y[:, None]
    return X, y


@pytest.mark.parametrize("head", ["logreg", "svm_rbf"])
def test_sklearn_heads_run_through_nested_cv(head):
    X, y = _data()
    est, grid, n_jobs = hd.make(head, X.shape[1], seed=0)
    res = nested_cv_oof(X, y, est, grid, outer_splits=4, inner_splits=3, seed=0, n_jobs=1)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, res["scores"]) > 0.8
    assert len(res["chosen_params"]) == 4


def test_hybrid_blocks_are_weighted_and_validated():
    X, y = _data(d=20)
    est, grid, _ = hd.make("svm_rbf", X.shape[1], seed=0, block_sizes=(12, 8))
    assert est.named_steps["blocks"].block_sizes == (12, 8)
    assert max(grid["svm__gamma"]) == pytest.approx(10.0 / 2)      # scale = number of blocks
    with pytest.raises(ValueError):
        hd.make("mlp", X.shape[1], seed=0, block_sizes=(12, 8))


class _Counting(ClassifierMixin, BaseEstimator):
    fits = 0

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        _Counting.fits += 1
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return X[:, 0]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def test_single_point_grid_fits_directly_without_inner_search():
    X, y = _data()
    s = _search(_Counting(), {"seed": [3]}, inner_splits=5, seed=0)
    assert isinstance(s, _FixedConfig)
    _Counting.fits = 0
    res = nested_cv_oof(X, y, _Counting(), {"seed": [3]}, outer_splits=5, seed=0, n_jobs=1)
    assert _Counting.fits == 5                     # one fit per outer fold, no inner CV
    assert all(np.isnan(c["inner_auc"]) for c in res["chosen_params"])
    out = source_to_target(X, y, X[:10], _Counting(), {"seed": [3]}, seed=0, n_jobs=1)
    assert np.isfinite(out["threshold"])


@pytest.mark.parametrize("arch", ["mlp", "attn_mil"])
def test_torch_heads_learn_and_are_deterministic(arch):
    pytest.importorskip("torch")
    X, y = _data(n=200, d=16, tokens=9 if arch == "attn_mil" else None)
    est, grid, n_jobs = hd.make(arch, X.shape[-1], seed=0, device="cpu")
    assert n_jobs == 1 and all(len(v) == 1 for v in grid.values())
    est.set_params(max_epochs=40, patience=40)
    a = clone(est).fit(X[:150], y[:150]).decision_function(X[150:])
    b = clone(est).fit(X[:150], y[:150]).decision_function(X[150:])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[150:], a) > 0.75


def test_torch_head_never_sees_test_rows_in_its_standardisation():
    pytest.importorskip("torch")
    X, y = _data(n=120, d=6)
    X_big = X.copy()
    X_big[100:] += 1000.0                          # rows the head is not fitted on
    m = hd.TorchHead(arch="mlp", max_epochs=2, patience=2, device="cpu").fit(X_big[:100], y[:100])
    assert np.abs(m.mean_).max() < 10


def test_attention_maps_shape():
    pytest.importorskip("torch")
    X, y = _data(n=80, d=8, tokens=5)
    m = hd.TorchHead(arch="attn_mil", max_epochs=3, patience=3, device="cpu").fit(X, y)
    a = m.attention_maps(X[:7])
    assert a.shape == (7, 5)
    np.testing.assert_allclose(a.sum(1), 1.0, atol=1e-5)
