"""Figure 3: shape-context illustration on real sampled edge points.

(a) the log-polar bins around one reference point, over the image's sampled thoracic edge points,
with the 2 x 2 spatial pooling grid; (b) that point's 5 x 12 log-polar histogram.
The reference point is chosen by a fixed rule: the sampled point closest to the centre of the
upper-left pooling cell.

    python scripts/figures/fig_shape_context.py --dataset montgomery
"""
import argparse
import zlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style
from tbshift.data import images
from tbshift.features import shape_context as sc

style.apply()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="montgomery")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    p = sc.ShapeContextParams()
    df = images.manifest(args.dataset).sort_values("image_id")
    iid = df[df.label == 1].image_id.iloc[0]
    img = images.read_prepared(args.dataset, iid, args.size)
    mask = images.read_mask(args.dataset, iid, args.size)
    out = sc.extract(img, mask, p, zlib.crc32(iid.encode()))
    pts, desc = out["points"], out["descriptors"]
    target = np.array([args.size / 4, args.size / 4])
    i = int(np.argmin(np.hypot(*(pts - target).T)))
    ref = pts[i]

    off = ~np.eye(len(pts), dtype=bool)
    d = np.hypot(*(pts[None] - pts[:, None]).transpose(2, 0, 1))
    mean_d = d[off].mean()
    radii = np.logspace(np.log10(p.r_inner), np.log10(p.r_outer), p.n_r + 1) * mean_d

    fig = plt.figure(figsize=(style.FULL, 2.9))
    ax = fig.add_axes([0.0, 0.02, 0.48, 0.92])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.35)
    ax.scatter(pts[:, 1], pts[:, 0], s=1.5, c=style.INK_2, linewidths=0)
    th = np.linspace(0, 2 * np.pi, 361)
    for rr in radii:
        ax.plot(ref[1] + rr * np.cos(th), ref[0] - rr * np.sin(th), color=style.FAMILY["handcrafted"], lw=0.7)
    for k in range(p.n_theta):
        a = 2 * np.pi * k / p.n_theta
        ax.plot([ref[1] + radii[0] * np.cos(a), ref[1] + radii[-1] * np.cos(a)],
                [ref[0] - radii[0] * np.sin(a), ref[0] - radii[-1] * np.sin(a)],
                color=style.FAMILY["handcrafted"], lw=0.5)
    ax.scatter([ref[1]], [ref[0]], s=28, c=style.FAMILY["deep"], edgecolors="white", linewidths=0.8, zorder=5)
    ax.axhline(args.size / 2, color=style.INK, lw=0.8, ls="--")
    ax.axvline(args.size / 2, color=style.INK, lw=0.8, ls="--")
    ax.set_xlim(0, args.size); ax.set_ylim(args.size, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) log-polar bins around one edge point; dashed: 2 × 2 pooling grid", fontsize=7.5)

    ax2 = fig.add_axes([0.56, 0.2, 0.42, 0.6])
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + style.SEQ)
    hist = desc[i].reshape(p.n_r, p.n_theta)
    im = ax2.imshow(hist, cmap=cmap, aspect="auto", origin="lower", interpolation="nearest")
    ax2.set_xticks(range(p.n_theta), [f"{int(360 * k / p.n_theta)}°" for k in range(p.n_theta)], fontsize=6)
    ax2.set_yticks(range(p.n_r), [f"{k + 1}" for k in range(p.n_r)])
    ax2.set_xlabel("angle bin (counter-clockwise from +x)")
    ax2.set_ylabel("log-radius bin (inner → outer)")
    ax2.grid(False)
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cb.set_label("fraction of points", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax2.set_title(f"(b) 5 × 12 histogram of the marked point ({style.DATASET_LABEL.get(args.dataset, args.dataset)}, {iid})", fontsize=7.5)
    style.save(fig, "fig3_shape_context")


if __name__ == "__main__":
    main()
