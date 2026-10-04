import numpy as np
import pytest

from tbshift.models.classical import BlockWeighter
from tbshift.runs import run_identity, stratified_subsample


def test_block_weighter_equalises_block_contribution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 101))          # blocks of 100 and 1 standardised columns
    Xw = BlockWeighter((100, 1)).fit(X).transform(X)
    contrib_big = (Xw[:, :100] ** 2).sum(axis=1).mean()
    contrib_small = (Xw[:, 100:] ** 2).sum(axis=1).mean()
    assert contrib_big == pytest.approx(contrib_small, rel=0.2)


def test_block_weighter_rejects_wrong_sizes():
    with pytest.raises(ValueError):
        BlockWeighter((3, 3)).fit(np.zeros((5, 7)))


def test_stratified_subsample_keeps_prevalence_and_is_seeded():
    y = np.r_[np.zeros(300), np.ones(100)].astype(int)
    a = stratified_subsample(y, 100, seed=1)
    assert len(a) == 100 and y[a].sum() == 25
    np.testing.assert_array_equal(a, stratified_subsample(y, 100, seed=1))
    assert len(stratified_subsample(y, 0, seed=1)) == 400


def test_run_identity_validates_protocol():
    with pytest.raises(ValueError):
        run_identity("handcrafted", "sc", "indomain", "lung", "a", "b", None, 512)
