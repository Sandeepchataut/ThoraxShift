"""Thoracic edge maps and shape-context descriptors.

Shape context: for each sampled edge point, a log-polar histogram of the relative positions of
all other sampled points, with radial distances normalised by the mean pairwise distance (scale
invariance). Rotation is NOT normalised, because PA chest radiographs have a canonical
orientation. Edge detector, point count, bin layout and pooling are all parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import feature


@dataclass(frozen=True)
class ShapeContextParams:
    n_points: int = 400          # edge points sampled per image
    n_r: int = 5                 # radial bins
    n_theta: int = 12            # angular bins
    r_inner: float = 0.125       # inner radius (in units of mean pairwise distance)
    r_outer: float = 2.0         # outer radius
    canny_sigma: float = 2.0
    mask_dilation_px: int = 15   # = preprocess.THORACIC_DILATION_PX
    grid: tuple[int, int] = (2, 2)  # spatial cells for pooling (rows, cols)

    @property
    def n_bins(self) -> int:
        return self.n_r * self.n_theta


def thoracic_edge_map(img: np.ndarray, mask: np.ndarray | None, p: ShapeContextParams) -> np.ndarray:
    """Binary edge map restricted to the (dilated) lung region. img is float in [0, 1]."""
    edges = feature.canny(img, sigma=p.canny_sigma)
    if mask is not None:
        region = mask.astype(bool)
        if p.mask_dilation_px > 0:
            region = ndimage.binary_dilation(region, iterations=p.mask_dilation_px)
        edges &= region
    return edges


def sample_edge_points(edges: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Up to n (row, col) points sampled uniformly without replacement from edge pixels."""
    pts = np.argwhere(edges)
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts.astype(float)


def shape_context_descriptors(points: np.ndarray, p: ShapeContextParams) -> np.ndarray:
    """(n_points, n_r * n_theta) log-polar histograms, each L1-normalised (rows of zeros if empty)."""
    n = len(points)
    if n < 2:
        return np.zeros((n, p.n_bins))
    diff = points[None, :, :] - points[:, None, :]          # [i, j] = p_j - p_i
    dist = np.hypot(diff[..., 0], diff[..., 1])
    off_diag = ~np.eye(n, dtype=bool)
    mean_d = dist[off_diag].mean()
    if mean_d == 0:
        return np.zeros((n, p.n_bins))
    r = dist / mean_d
    # Image rows grow downward; negate so angles are in the conventional counter-clockwise frame.
    theta = np.mod(np.arctan2(-diff[..., 0], diff[..., 1]), 2 * np.pi)

    r_edges = np.logspace(np.log10(p.r_inner), np.log10(p.r_outer), p.n_r + 1)
    r_bin = np.digitize(r, r_edges) - 1                    # -1 below inner, n_r beyond outer
    t_bin = np.minimum((theta / (2 * np.pi) * p.n_theta).astype(int), p.n_theta - 1)
    valid = off_diag & (r_bin >= 0) & (r_bin < p.n_r)

    desc = np.zeros((n, p.n_bins))
    rows, cols = np.nonzero(valid)
    np.add.at(desc, (rows, r_bin[rows, cols] * p.n_theta + t_bin[rows, cols]), 1.0)
    sums = desc.sum(axis=1, keepdims=True)
    np.divide(desc, sums, out=desc, where=sums > 0)
    return desc


def grid_mean_pool(points: np.ndarray, desc: np.ndarray, shape: tuple[int, int],
                   grid: tuple[int, int]) -> np.ndarray:
    """Mean descriptor per spatial cell, concatenated, plus the fraction of points per cell.

    Keeps coarse spatial layout (upper/lower, left/right lung) that a global mean would lose.
    Cells with no points contribute zeros.
    """
    gr, gc = grid
    n_bins = desc.shape[1]
    out = np.zeros((gr * gc, n_bins + 1))
    if len(points) == 0:
        return out.ravel()
    ri = np.minimum((points[:, 0] / shape[0] * gr).astype(int), gr - 1)
    ci = np.minimum((points[:, 1] / shape[1] * gc).astype(int), gc - 1)
    cell = ri * gc + ci
    for k in range(gr * gc):
        sel = cell == k
        if sel.any():
            out[k, :n_bins] = desc[sel].mean(axis=0)
            out[k, n_bins] = sel.mean()
    return out.ravel()


def extract(img: np.ndarray, mask: np.ndarray | None, p: ShapeContextParams,
            seed: int) -> dict:
    """Per-image shape-context output: sampled points, descriptors, pooled vector, edge density."""
    rng = np.random.default_rng(seed)
    edges = thoracic_edge_map(img, mask, p)
    pts = sample_edge_points(edges, p.n_points, rng)
    desc = shape_context_descriptors(pts, p)
    region = mask.astype(bool).sum() if mask is not None else edges.size
    return {
        "points": pts,
        "descriptors": desc,
        "pooled": grid_mean_pool(pts, desc, img.shape, p.grid),
        "edge_density": float(edges.sum() / max(region, 1)),
        "n_edge_px": int(edges.sum()),
    }
