"""Figures 4-7 from real runs only, or clearly-labelled layout mock-ups.

    python scripts/figures/fig_results.py --aggregate outputs/runs/<id>_aggregate \
        --probe outputs/runs/<id>_shortcut_probe_... --roc-source montgomery --roc-target shenzhen
    python scripts/figures/fig_results.py --mock      # layout only: watermarked, *_MOCK files

Fig 4  ROC curves on one target: in-domain vs cross-domain, hand-crafted vs deep, with
       stratified-bootstrap 95% bands (vertical averaging).
Fig 5  Source x target AUC matrix, one panel per model family (diagonal = in-domain).
Fig 6  Transfer gap with paired-bootstrap 95% CIs, per (S -> T) and model family.
Fig 7  Shortcut probe: dataset-identification balanced accuracy, masked vs unmasked.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

import style
from tbshift.eval.metrics import stratified_bootstrap_indices
from tbshift.provenance import RUNS_ROOT

style.apply()
FPR = np.linspace(0, 1, 101)


# ------------------------------------------------------------------ data access
def run_predictions(run: str) -> pd.DataFrame:
    p = pd.read_csv(RUNS_ROOT / run / "predictions.csv")
    return p[p["repeat"] == 0]


def roc_band(y, s, n_boot=1000, seed=0):
    def interp(yy, ss):
        f, t, _ = roc_curve(yy, ss)
        return np.interp(FPR, f, t)
    rng = np.random.default_rng(seed)
    boots = np.array([interp(y[i], s[i]) for i in stratified_bootstrap_indices(y, n_boot, rng)])
    return interp(y, s), np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)


def mock_inputs(rng):
    """Synthetic structures with the real schema. Used ONLY to check layout (watermarked)."""
    ds = ["shenzhen", "montgomery", "tbx11k"]
    roc = {}
    for fam in ("handcrafted", "deep"):
        for proto, sep in (("in", 1.6), ("cross", 0.8)):
            y = np.r_[np.zeros(300), np.ones(300)].astype(int)
            roc[(fam, proto)] = (y, y * sep + rng.normal(size=600))
    mat = {fam: pd.DataFrame(rng.uniform(0.6, 0.95, (3, 3)), index=ds, columns=ds) for fam in ("handcrafted", "deep")}
    gap = pd.DataFrame([{"source": s, "target": t, "model_family": f, "gap": g, "gap_ci_low": g - 0.05,
                         "gap_ci_high": g + 0.05}
                        for s in ds for t in ds if s != t for f, g in
                        (("handcrafted", rng.uniform(0, .2)), ("deep", rng.uniform(0, .2)))])
    reps = ["sc", "lbp", "glcm", "gabor", "hog", "sc+lbp+glcm+gabor+hog", "frozen-densenet121"]
    probe = pd.DataFrame([{"dataset_a": "montgomery", "dataset_b": "shenzhen", "representation": r,
                           "mask": m, "label_class": c, "balanced_accuracy_mean": rng.uniform(.5, 1),
                           "balanced_accuracy_sd": .02}
                          for r in reps for m in ("lung", "none") for c in ("normal", "abnormal")])
    return roc, mat, gap, probe


def real_inputs(agg: Path, probe_dir: Path | None, src: str, tgt: str):
    tables = {n: pd.read_csv(agg / "tables" / f"{n}.csv") for n in ("indomain", "cross", "transfer_gap", "primary_endpoint")}
    prim = tables["primary_endpoint"]
    row = prim[(prim.source == src) & (prim.target == tgt)]
    if row.empty:
        raise SystemExit(f"no primary-endpoint row for {src}->{tgt} in {agg}")
    row = row.iloc[0]
    gap = tables["transfer_gap"]
    g = gap[(gap.source == src) & (gap.target == tgt)]
    roc = {("handcrafted", "cross"): row.hc_run, ("deep", "cross"): row.deep_run}
    for fam in ("handcrafted", "deep"):
        roc[(fam, "in")] = g[g.model_family == fam].in_domain_run.iloc[0]
    roc = {k: (lambda p: (p.label.to_numpy(), p.score.to_numpy()))(run_predictions(v)) for k, v in roc.items()}
    mat = {}
    for fam in ("handcrafted", "deep"):
        ind = tables["indomain"][(tables["indomain"].model_family == fam) & (tables["indomain"]["mask"] == "lung")]
        cr = tables["cross"][(tables["cross"].model_family == fam) & (tables["cross"]["mask"] == "lung")
                             & (tables["cross"].source_subsample == 0)]
        ds = sorted(set(ind.dataset) | set(cr.source) | set(cr.target))
        m = pd.DataFrame(np.nan, index=ds, columns=ds)
        for r in ind.itertuples():
            m.loc[r.dataset, r.dataset] = r.auc
        for r in cr.itertuples():
            m.loc[r.source, r.target] = r.auc
        mat[fam] = m
    probe = pd.read_csv(probe_dir / "probe.csv") if probe_dir else None
    return roc, mat, gap, probe


# ------------------------------------------------------------------ figures
def fig4(roc, src, tgt, mock):
    fig, ax = plt.subplots(figsize=(style.COL, style.COL * 0.95))
    ax.plot([0, 1], [0, 1], color=style.GRID, lw=0.8, zorder=0)
    for fam in ("handcrafted", "deep"):
        for proto, ls in (("in", "-"), ("cross", "--")):
            y, s = roc[(fam, proto)]
            mid, lo, hi = roc_band(y, s)
            ax.fill_between(FPR, lo, hi, color=style.FAMILY[fam], alpha=0.12, linewidth=0)
            lab = f"{style.FAMILY_LABEL[fam]}, " + (f"in-domain ({style.DATASET_LABEL.get(tgt, tgt)} CV)"
                                                     if proto == "in" else f"trained on {style.DATASET_LABEL.get(src, src)}")
            ax.plot(FPR, mid, color=style.FAMILY[fam], ls=ls, lw=1.4, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"Target: {style.DATASET_LABEL.get(tgt, tgt)}")
    ax.legend(loc="lower right", fontsize=6.2, handlelength=2.2)
    style.save(fig, "fig4_roc", mock)


def fig5(mat, mock):
    fams = ("handcrafted", "deep")
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL * 0.62, 2.3), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("seq", style.SEQ)
    for ax, fam in zip(axes, fams):
        m = mat[fam]
        im = ax.imshow(m.to_numpy(dtype=float), cmap=cmap, vmin=0.5, vmax=1.0)
        lab = [style.DATASET_LABEL.get(d, d) for d in m.index]
        ax.set_xticks(range(len(lab)), lab, rotation=30, ha="right"); ax.set_yticks(range(len(lab)), lab)
        ax.set_xlabel("Target"); ax.set_ylabel("Source" if fam == "handcrafted" else "")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.8 else style.INK, weight="bold" if i == j else "normal")
        ax.set_title(style.FAMILY_LABEL[fam])
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("AUC (diagonal: in-domain)")
    style.save(fig, "fig5_auc_matrix", mock)


def fig6(gap, mock):
    pairs = gap[["source", "target"]].drop_duplicates().sort_values(["target", "source"]).values.tolist()
    fig, ax = plt.subplots(figsize=(style.COL, 0.42 * len(pairs) + 0.7))
    for k, (s, t) in enumerate(pairs):
        for fam, dy in (("handcrafted", -0.13), ("deep", 0.13)):
            r = gap[(gap.source == s) & (gap.target == t) & (gap.model_family == fam)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.gap, k + dy, xerr=[[r.gap - r.gap_ci_low], [r.gap_ci_high - r.gap]], fmt="o",
                        ms=4, color=style.FAMILY[fam], elinewidth=1.2, capsize=0,
                        label=style.FAMILY_LABEL[fam] if k == 0 else None)
    ax.axvline(0, color=style.INK_2, lw=0.8)
    ax.set_yticks(range(len(pairs)), [f"{style.DATASET_LABEL.get(s, s)} → {style.DATASET_LABEL.get(t, t)}"
                                      for s, t in pairs])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Transfer gap: in-domain AUC − cross-domain AUC on target")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
    style.save(fig, "fig6_transfer_gap", mock)


def fig7(probe, mock):
    pair = probe[["dataset_a", "dataset_b"]].drop_duplicates().iloc[0].tolist()
    p = probe[(probe.dataset_a == pair[0]) & (probe.dataset_b == pair[1])]
    reps = ["sc", "lbp", "glcm", "gabor", "hog", "sc+lbp+glcm+gabor+hog", "frozen-densenet121"]
    names = ["Shape ctx", "LBP", "GLCM", "Gabor", "HOG", "FUSION", "Frozen\nDenseNet"]
    reps = [r for r in reps if r in set(p.representation)]
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL, 2.2), sharey=True)
    w = 0.38
    for ax, cls in zip(axes, ("normal", "abnormal")):
        for j, (mask, dx) in enumerate((("lung", -w / 2), ("none", w / 2))):
            vals = [p[(p.representation == r) & (p["mask"] == mask) & (p.label_class == cls)]
                    .balanced_accuracy_mean.iloc[0] for r in reps]
            x = np.arange(len(reps)) + dx
            ax.bar(x, vals, width=w - 0.04, color=style.MASK[mask], hatch="////" if mask == "none" else None,
                   edgecolor="white", linewidth=0.5, label=style.MASK_LABEL[mask])
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, color=style.INK,
                        rotation=90)
        ax.axhline(0.5, color=style.INK_2, lw=0.8, ls="--", label="Chance (0.5)")
        ax.set_xticks(range(len(reps)), [names[["sc", "lbp", "glcm", "gabor", "hog", "sc+lbp+glcm+gabor+hog",
                                                 "frozen-densenet121"].index(r)] for r in reps], fontsize=6.5)
        ax.set_ylim(0.4, 1.14); ax.grid(axis="x", visible=False)
        ax.set_title(f"{cls.capitalize()} images only: "
                     f"{style.DATASET_LABEL.get(pair[0], pair[0])} vs {style.DATASET_LABEL.get(pair[1], pair[1])}")
    axes[0].set_ylabel("Dataset-ID balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.06))
    style.save(fig, "fig7_shortcut_probe", mock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=Path)
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--roc-source", default="montgomery")
    ap.add_argument("--roc-target", default="shenzhen")
    ap.add_argument("--mock", action="store_true", help="layout mock-up from synthetic data (watermarked)")
    args = ap.parse_args()
    if args.mock:
        roc, mat, gap, probe = mock_inputs(np.random.default_rng(0))
    else:
        if not args.aggregate:
            raise SystemExit("--aggregate is required (or --mock for a labelled layout mock-up)")
        roc, mat, gap, probe = real_inputs(args.aggregate, args.probe, args.roc_source, args.roc_target)
    fig4(roc, args.roc_source, args.roc_target, args.mock)
    fig5(mat, args.mock)
    fig6(gap, args.mock)
    if probe is not None:
        fig7(probe, args.mock)


if __name__ == "__main__":
    main()
