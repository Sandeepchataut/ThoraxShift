"""Figure 1: pipeline schematic (no data).

Three tiers: shared data preparation, the two model families, and the evaluation protocols.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import style

style.apply()


def box(ax, x, y, w, h, title, sub="", face="#f6f5f2", edge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=face, edgecolor=edge or style.INK_2, linewidth=0.8,
                                mutation_aspect=1))
    ty = y + h * (0.66 if sub else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=7.5, weight="bold", color=style.INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=6.4,
                color=style.INK_2, linespacing=1.15)
    return x, y, w, h


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                 color=style.INK_2, shrinkA=0, shrinkB=0))


def right(b): return (b[0] + b[2], b[1] + b[3] / 2)
def left(b): return (b[0], b[1] + b[3] / 2)
def bottom(b): return (b[0] + b[2] / 2, b[1])
def top(b): return (b[0] + b[2] / 2, b[1] + b[3])


def main():
    W, H = 100, 52
    fig, ax = plt.subplots(figsize=(style.FULL, style.FULL * H / W))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_aspect("equal")

    # Tier 1: shared data preparation (identical for every dataset and model family)
    y1, h1, w1, gap = 38, 11, 21.5, 4.67
    xs = [i * (w1 + gap) for i in range(4)]
    t1 = [box(ax, xs[0], y1, w1, h1, "Public CXR datasets", "Shenzhen · Montgomery\nTBX11K"),
          box(ax, xs[1], y1, w1, h1, "Audit", "acquisition properties\nduplicate check"),
          box(ax, xs[2], y1, w1, h1, "Preprocessing", "intensity window · pad to square\n512 × 512 · CLAHE"),
          box(ax, xs[3], y1, w1, h1, "Lung segmentation", "one pretrained model\nDice-validated")]
    for a, b in zip(t1, t1[1:]):
        arrow(ax, right(a), left(b))
    ax.text(0, H - 1, "Shared preparation", fontsize=7, color=style.INK_2, style="italic", va="top")

    # Tier 2: the two model families, side by side
    y2, h2 = 20, 11
    hc = box(ax, 0, y2, 30, h2, "Hand-crafted features", "edge-map shape context · LBP\nGLCM · Gabor · HOG",
             face="#e7f0fb", edge=style.FAMILY["handcrafted"])
    hc_c = box(ax, 34, y2, 13, h2, "RBF-SVM", "nested CV", face="#e7f0fb", edge=style.FAMILY["handcrafted"])
    dp = box(ax, 53, y2, 30, h2, "Deep network", "DenseNet-121, ImageNet init\nfull fine-tuning",
             face="#fdebe3", edge=style.FAMILY["deep"])
    dp_c = box(ax, 87, y2, 13, h2, "Classifier", "fixed recipe\n3 seeds", face="#fdebe3", edge=style.FAMILY["deep"])
    arrow(ax, right(hc), left(hc_c)); arrow(ax, right(dp), left(dp_c))
    # lung masks feed both families
    seg = t1[3]
    sx, sy = bottom(seg)
    ax.plot([sx, sx], [sy, 34], color=style.INK_2, lw=0.8)
    ax.plot([top(hc)[0], sx], [34, 34], color=style.INK_2, lw=0.8)
    arrow(ax, (top(hc)[0], 34), top(hc)); arrow(ax, (top(dp)[0], 34), top(dp))

    # Tier 3: evaluation band fed by both classifiers through one bus
    y3, h3, bus = 4, 9, 16.5
    evals = [box(ax, 0, y3, 31, h3, "In-domain", "stratified 5-fold CV"),
             box(ax, 34.5, y3, 31, h3, "Cross-domain", "train on S, test on T · threshold fixed on S"),
             box(ax, 69, y3, 31, h3, "Shortcut probe", "dataset ID within class · masked vs unmasked")]
    xs_bus = [bottom(hc_c)[0], bottom(dp_c)[0]] + [top(e)[0] for e in evals]
    ax.plot([min(xs_bus), max(xs_bus)], [bus, bus], color=style.INK_2, lw=0.8)
    for b in (hc_c, dp_c):
        ax.plot([bottom(b)[0]] * 2, [bottom(b)[1], bus], color=style.INK_2, lw=0.8)
    for e in evals:
        arrow(ax, (top(e)[0], bus), top(e))
    ax.text(W / 2, 1.0, "Every comparison: AUC with 95% CI · paired DeLong test · paired bootstrap · "
            "Holm correction · pre-specified analysis plan", ha="center", va="center", fontsize=6.6,
            color=style.INK_2, style="italic")
    style.save(fig, "fig1_pipeline")


if __name__ == "__main__":
    main()
