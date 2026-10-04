"""Figure 2: qualitative panel, one image per dataset.

Columns: preprocessed radiograph, lung mask and thoracic region, thoracic edge map, sampled
shape-context points with the 2 x 2 pooling grid. The image per dataset is chosen by a fixed rule
(first abnormal image by image_id), not by visual selection.

    python scripts/figures/fig_qualitative.py --datasets shenzhen montgomery
"""
import argparse
import zlib

import matplotlib.pyplot as plt
import numpy as np

import style
from tbshift import preprocess
from tbshift.data import images
from tbshift.features import shape_context as sc

style.apply()


def pick(dataset: str) -> str:
    df = images.manifest(dataset).sort_values("image_id")
    return df[df.label == 1].image_id.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["shenzhen", "montgomery"])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    p = sc.ShapeContextParams()
    cols = ["Preprocessed", "Lung mask / thoracic region", "Thoracic edge map", "Shape-context points"]
    n = len(args.datasets)
    fig, axes = plt.subplots(n, 4, figsize=(style.FULL, style.FULL / 4 * n + 0.25))
    axes = np.atleast_2d(axes)
    for r, ds in enumerate(args.datasets):
        iid = pick(ds)
        img = images.read_prepared(ds, iid, args.size)
        mask = images.read_mask(ds, iid, args.size)
        region = preprocess.thoracic_region(mask) if mask is not None else np.ones_like(img, bool)
        out = sc.extract(img, mask, p, zlib.crc32(iid.encode()))
        edges = sc.thoracic_edge_map(img, mask, p)
        ax = axes[r]
        ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[1].imshow(img, cmap="gray", vmin=0, vmax=1)
        if mask is not None:
            ax[1].contour(region, levels=[0.5], colors=[style.MASK["lung"]], linewidths=0.6, linestyles="--")
            ax[1].contour(mask, levels=[0.5], colors=[style.MASK["lung"]], linewidths=0.9)
        ax[2].imshow(edges, cmap="gray_r", interpolation="nearest")
        ax[3].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.45)
        pts = out["points"]
        ax[3].scatter(pts[:, 1], pts[:, 0], s=1.2, c=style.FAMILY["handcrafted"], linewidths=0)
        for k in (1,):
            ax[3].axhline(args.size * k / 2, color=style.INK_2, lw=0.6, ls=":")
            ax[3].axvline(args.size * k / 2, color=style.INK_2, lw=0.6, ls=":")
        for c in range(4):
            ax[c].set_xticks([]); ax[c].set_yticks([]); ax[c].grid(False)
            for s in ax[c].spines.values():
                s.set_visible(False)
            if r == 0:
                ax[c].set_title(cols[c], fontsize=7.5)
        ax[0].set_ylabel(f"{style.DATASET_LABEL.get(ds, ds)}\n{iid}", fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    style.save(fig, "fig2_qualitative")


if __name__ == "__main__":
    main()
