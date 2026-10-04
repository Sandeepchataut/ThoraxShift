import numpy as np

from tbshift.eval.shortcut import dataset_id_probe


def test_probe_is_at_chance_for_identical_distributions():
    rng = np.random.default_rng(0)
    r = dataset_id_probe(rng.normal(size=(80, 10)), rng.normal(size=(300, 10)), repeats=3)
    assert abs(r["balanced_accuracy_mean"] - 0.5) < 0.08


def test_probe_detects_a_site_offset():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(80, 10)); b = rng.normal(size=(300, 10)); b[:, 0] += 3.0
    r = dataset_id_probe(a, b, repeats=3)
    assert r["balanced_accuracy_mean"] > 0.9 and r["auc_mean"] > 0.95
