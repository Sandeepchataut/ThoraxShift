import numpy as np
import pytest

from tbshift.features import shape_context as sc
from tbshift.features import texture as tx

P = sc.ShapeContextParams()


def _circle_points(n=60, r=50.0, centre=(100.0, 100.0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[centre[0] + r * np.sin(t), centre[1] + r * np.cos(t)]


def test_descriptor_shape_and_normalisation():
    d = sc.shape_context_descriptors(_circle_points(), P)
    assert d.shape == (60, P.n_bins)
    np.testing.assert_allclose(d.sum(axis=1), 1.0)


def test_translation_and_scale_invariance():
    # Generic (random) positions: a regular polygon puts pairwise angles exactly on bin
    # boundaries, where floating-point rounding legitimately flips the bin.
    pts = np.random.default_rng(0).uniform(0, 200, size=(80, 2))
    d0 = sc.shape_context_descriptors(pts, P)
    d1 = sc.shape_context_descriptors(pts * 2.5 + 37.0, P)
    np.testing.assert_allclose(d0, d1, atol=1e-12)


def test_angular_convention_point_to_the_right_is_angle_zero():
    # Two points: q directly to the right of p (same row, larger column) -> theta bin 0 for p.
    p = sc.ShapeContextParams(n_r=1, n_theta=4, r_inner=0.5, r_outer=2.0)
    d = sc.shape_context_descriptors(np.array([[10.0, 10.0], [10.0, 20.0]]), p)
    assert d[0].argmax() == 0  # right
    assert d[1].argmax() == 2  # left
    # A point above (smaller row index) lies at +90 degrees -> bin 1.
    d = sc.shape_context_descriptors(np.array([[20.0, 10.0], [10.0, 10.0]]), p)
    assert d[0].argmax() == 1


def test_degenerate_inputs():
    assert sc.shape_context_descriptors(np.zeros((0, 2)), P).shape == (0, P.n_bins)
    assert sc.shape_context_descriptors(np.zeros((1, 2)), P).shape == (1, P.n_bins)
    assert not sc.shape_context_descriptors(np.zeros((5, 2)), P).any()  # all coincident


def test_extract_on_blank_image_is_finite_and_fixed_length():
    out_blank = sc.extract(np.zeros((128, 128)), None, P, seed=0)
    img = np.zeros((128, 128)); img[30:90, 40:100] = 1.0
    out_box = sc.extract(img, None, P, seed=0)
    assert out_blank["pooled"].shape == out_box["pooled"].shape
    assert np.isfinite(out_blank["pooled"]).all() and out_blank["n_edge_px"] == 0
    assert out_box["n_edge_px"] > 0


def test_extract_is_deterministic_for_a_seed():
    rng = np.random.default_rng(0)
    img = rng.random((128, 128))
    a = sc.extract(img, None, P, seed=7)["pooled"]
    b = sc.extract(img, None, P, seed=7)["pooled"]
    np.testing.assert_array_equal(a, b)


def test_mask_restricts_edges():
    img = np.zeros((128, 128)); img[20:40, 20:40] = 1; img[80:100, 80:100] = 1
    mask = np.zeros_like(img, bool); mask[10:50, 10:50] = True
    p = sc.ShapeContextParams(mask_dilation_px=0)
    edges = sc.thoracic_edge_map(img, mask, p)
    assert edges[60:, 60:].sum() == 0 and edges[:60, :60].sum() > 0


@pytest.mark.parametrize("mask", [None, "box"])
def test_texture_fixed_length_and_finite(mask):
    rng = np.random.default_rng(0)
    p = tx.TextureParams(gabor_frequencies=(0.1,), gabor_n_orient=2)
    m = None
    if mask == "box":
        m = np.zeros((128, 128), bool); m[20:100, 30:90] = True
    a = tx.extract(rng.random((128, 128)), m, p)
    b = tx.extract(np.full((128, 128), 0.5), m, p)  # constant image
    for k in a:
        assert a[k].shape == b[k].shape and np.isfinite(a[k]).all() and np.isfinite(b[k]).all()
