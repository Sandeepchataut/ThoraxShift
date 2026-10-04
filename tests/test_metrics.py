import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tbshift.eval import metrics as M


def _data(n=300, sep=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, y * sep + rng.normal(size=n)


def test_delong_auc_matches_sklearn_with_ties():
    y, s = _data()
    s = np.round(s, 1)  # force ties
    aucs, _ = M.delong_auc_cov(y, s)
    assert aucs[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_delong_se_close_to_bootstrap():
    y, s = _data(n=400, sep=1.0, seed=1)
    se = M.auc_delong_ci(y, s).se
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(y[i], s[i]) for i in M.stratified_bootstrap_indices(y, 2000, rng)]
    assert se == pytest.approx(np.std(boots), rel=0.15)


def test_delong_known_value():
    # Hand-checked: positives {0.9, 0.6}, negatives {0.7, 0.2}: 3 of 4 pairs ordered correctly.
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.7, 0.2])
    assert M.auc_delong_ci(y, s).auc == pytest.approx(0.75)


def test_delong_test_identical_scores():
    y, s = _data()
    r = M.delong_test(y, s, s)
    assert r["diff"] == 0 and r["p"] == 1.0


def test_delong_test_detects_real_difference_and_not_noise():
    y, s = _data(n=500, sep=2.0, seed=2)
    rng = np.random.default_rng(3)
    assert M.delong_test(y, s, rng.normal(size=len(y)))["p"] < 1e-6
    noisy = s + rng.normal(scale=0.01, size=len(y))
    assert M.delong_test(y, s, noisy)["p"] > 0.05


def test_delong_type1_error_is_calibrated():
    # Two independent noisy readings of the same latent score: H0 (equal AUC) is true.
    rejections = 0
    for k in range(300):
        rng = np.random.default_rng(100 + k)
        y = rng.integers(0, 2, 200)
        latent = y * 1.0 + rng.normal(size=200)
        a = latent + rng.normal(size=200)
        b = latent + rng.normal(size=200)
        rejections += M.delong_test(y, a, b)["p"] < 0.05
    assert 0.02 <= rejections / 300 <= 0.09


def test_wilson_ci():
    lo, hi = M.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert M.wilson_ci(0, 10)[0] == 0.0


def test_threshold_metrics_counts():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    r = M.threshold_metrics(y, s, 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["sensitivity"] == 0.5 and r["specificity"] == 0.5


def test_youden_threshold_separates_perfectly():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    t = M.youden_threshold(y, s)
    r = M.threshold_metrics(y, s, t)
    assert r["sensitivity"] == 1 and r["specificity"] == 1


def test_single_class_rejected():
    with pytest.raises(ValueError):
        M.auc_delong_ci(np.ones(5), np.arange(5))
