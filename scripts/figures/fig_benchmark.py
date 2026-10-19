"""Benchmark figures from an aggregate_benchmark run (real runs only), or a watermarked mock-up.

    python scripts/figures/fig_benchmark.py --aggregate outputs/runs/<id>_aggregate_benchmark
    python scripts/figures/fig_benchmark.py --mock            # layout check: *_MOCK files

B1  AUC matrix: FUSION and every frozen backbone (RBF-SVM head); in-domain and the six transfers.
B2  Primary endpoint forest plot: FUSION − backbone AUC difference with 95% CI per primary pair.
B3  Head ablation: mean cross-domain and in-domain AUC per backbone x head.
B4  Transfer gap per model for each primary pair.
B5  Pre-specified backbone contrasts (a − b) per primary pair.
B6  Shortcut probe: dataset-ID balanced accuracy, lung-masked vs unmasked, per representation.
B7  ROC on one target: FUSION vs the source-selected deep model and two reference backbones.
B8  Hybrid gain: (backbone + FUSION) − backbone.
Output: outputs/figures/*.pdf and *.svg (vector), via style.save.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

import style
from tbshift import study as S
from tbshift.eval.metrics import stratified_bootstrap_indices
from tbshift.models import backbones as bb
from tbshift.models import heads as hd
from tbshift.provenance import RUNS_ROOT

style.apply()
SEQ = LinearSegmentedColormap.from_list("seq", style.SEQ)
DL = style.DATASET_LABEL
FPR = np.linspace(0, 1, 101)
BACKBONE_ORDER = list(bb.BACKBONES)


def pair_label(s, t):
    return f"{DL.get(s, s)}→{DL.get(t, t)}"


def model_group(model: str, family: str) -> str:
    if family == S.HC_FAMILY:
        return "handcrafted"
    return bb.get(S.parse_frozen_model(model)[0]).group


def model_label(model: str, family: str) -> str:
    if family == S.HC_FAMILY:
        return "FUSION (hand-crafted)" if model == S.HC_PRIMARY else model
    return bb.get(S.parse_frozen_model(model)[0]).label


def group_legend(ax, groups, **kw):
    h = [Line2D([], [], marker=style.GROUP_MARKER[g], ls="", color=style.GROUP[g], ms=5,
                label=style.GROUP_LABEL[g]) for g in groups]
    return ax.legend(handles=h, **kw)


# ------------------------------------------------------------------------------------ inputs
def load_tables(agg: Path) -> dict:
    out = {}
    for f in (agg / "tables").glob("*.csv"):
        try:
            out[f.stem] = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            out[f.stem] = pd.DataFrame()
    return out


def mock_tables(rng) -> dict:
    """Synthetic tables with the real schema. Used ONLY to check layout (watermarked)."""
    hc = S.HC_PRIMARY
    models = [(S.HC_FAMILY, hc)] + [(S.FROZEN_FAMILY, S.frozen_model(b, S.PRIMARY_HEAD)) for b in BACKBONE_ORDER]
    t_in, t_cross = [], []
    for fam, m in models:
        for d in S.DATASETS:
            t_in.append({"model_family": fam, "model": m, "mask": "lung", "dataset": d, "auc": rng.uniform(.82, .97),
                         "hybrid": False})
        for s, t in S.CROSS_PAIRS:
            for mk in ("lung", "none"):
                t_cross.append({"model_family": fam, "model": m, "mask": mk, "source": s, "target": t,
                                "source_subsample": 0, "auc": rng.uniform(.6, .9), "run": ""})
    prim = pd.DataFrame([{"source": s, "target": t, "backbone": b, "role": "primary", "status": "ok",
                          "diff": (d := rng.normal(0, .05)), "boot_ci_low": d - .04, "boot_ci_high": d + .04,
                          "p_holm": rng.uniform(0, 1), "significant": abs(d) > .06}
                         for s, t in S.PRIMARY_PAIRS for b in BACKBONE_ORDER])
    lb = pd.DataFrame([{"model_family": S.FROZEN_FAMILY, "model": S.frozen_model(b, h),
                        "mean_cross_auc": rng.uniform(.65, .85), "mean_indomain_auc": rng.uniform(.85, .95)}
                       for b in BACKBONE_ORDER for h in hd.HEADS])
    head = lb.assign(backbone=lb.model.map(lambda m: S.parse_frozen_model(m)[0]),
                     head=lb.model.map(lambda m: S.parse_frozen_model(m)[1]))
    head_tab = head.pivot_table(index="backbone", columns="head", values="mean_cross_auc").add_prefix("cross_") \
        .join(head.pivot_table(index="backbone", columns="head", values="mean_indomain_auc").add_prefix("indomain_")) \
        .reset_index()
    gap = pd.DataFrame([{"source": s, "target": t, "model_family": f, "model": m, "status": "ok",
                         "gap": (g := rng.uniform(0, .25)), "gap_ci_low": g - .04, "gap_ci_high": g + .04}
                        for s, t in S.PRIMARY_PAIRS for f, m in models])
    con = pd.DataFrame([{"contrast": c, "question": q, "model_a": a, "model_b": b, "source": s, "target": t,
                         "role": "secondary", "status": "ok", "diff": (d := rng.normal(0, .04)),
                         "boot_ci_low": d - .03, "boot_ci_high": d + .03, "significant": abs(d) > .05}
                        for c, q, a, b in bb.CONTRASTS for s, t in S.PRIMARY_PAIRS])
    reps = list(S.HC_FEATURES) + [hc] + [f"frozen-{b}" for b in BACKBONE_ORDER]
    probe = pd.DataFrame([{"dataset_a": a, "dataset_b": b, "representation": r, "mask": mk, "label_class": c,
                           "balanced_accuracy_mean": rng.uniform(.55, 1.0)}
                          for a, b in (("montgomery", "shenzhen"), ("montgomery", "tbx11k"), ("shenzhen", "tbx11k"))
                          for r in reps for mk in ("lung", "none") for c in ("normal", "abnormal")])
    hyb = pd.DataFrame([{"backbone": b, "source": s, "target": t, "role": "secondary", "status": "ok",
                         "diff": rng.normal(.01, .02)} for b in BACKBONE_ORDER for s, t in S.CROSS_PAIRS])
    sel = pd.DataFrame([{"source": s, "selected_model": S.frozen_model("rad_dino", "attn_mil")} for s in S.DATASETS])
    return {"indomain": pd.DataFrame(t_in), "cross": pd.DataFrame(t_cross), "primary_endpoint": prim,
            "head_ablation": head_tab, "transfer_gap": gap, "contrasts": con, "shortcut_probe": probe,
            "hybrid": hyb, "selection": sel, "leaderboard": lb}


# ------------------------------------------------------------------------------------ figures
def b1_matrix(T, mock):
    t_in, t_cross = T["indomain"], T["cross"]
    if t_in.empty and t_cross.empty:
        return
    models = [(S.HC_FAMILY, S.HC_PRIMARY)] + [(S.FROZEN_FAMILY, S.frozen_model(b, S.PRIMARY_HEAD))
                                               for b in BACKBONE_ORDER]
    cols = [("in", d, d) for d in S.DATASETS] + [("x", s, t) for s, t in S.CROSS_PAIRS]
    M = np.full((len(models), len(cols)), np.nan)
    for i, (fam, m) in enumerate(models):
        for j, (kind, s, t) in enumerate(cols):
            if kind == "in":
                r = t_in[(t_in.model == m) & (t_in["mask"] == "lung") & (t_in.dataset == s)]
            else:
                r = t_cross[(t_cross.model == m) & (t_cross["mask"] == "lung") & (t_cross.source == s)
                            & (t_cross.target == t) & (t_cross.source_subsample == 0)]
            if len(r):
                M[i, j] = r.auc.iloc[0]
    fig, ax = plt.subplots(figsize=(style.FULL, 0.24 * len(models) + 1.2))
    im = ax.imshow(M, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v > 0.82 else style.INK)
    ax.set_xticks(range(len(cols)), [f"{DL[s]}\n(in-domain)" if k == "in" else pair_label(s, t).replace("→", "→\n")
                                     for k, s, t in cols], fontsize=6.3)
    ax.set_yticks(range(len(models)), [model_label(m, f) for f, m in models])
    ax.axvline(len(S.DATASETS) - 0.5, color="white", lw=2.5)
    groups = [model_group(m, f) for f, m in models]
    for i in range(1, len(models)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    for lab, g in zip(ax.get_yticklabels(), groups):
        lab.set_fontweight("bold" if g == "handcrafted" else "normal")
    ax.grid(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("AUC (repeat 0)")
    ax.set_title("Frozen backbones with the FUSION classifier (RBF-SVM), lung-masked input")
    style.save(fig, "figB1_auc_matrix", mock)


def b2_primary(T, mock):
    p = T.get("primary_endpoint", pd.DataFrame())
    if p.empty:
        return
    p = p[(p.role == "primary") & (p.status == "ok")]
    pairs = [pr for pr in S.PRIMARY_PAIRS if ((p.source == pr[0]) & (p.target == pr[1])).any()]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(style.FULL, 0.2 * len(BACKBONE_ORDER) + 1.5), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (s, t) in zip(axes, pairs):
        q = p[(p.source == s) & (p.target == t)].set_index("backbone")
        for k, b in enumerate(BACKBONE_ORDER):
            if b not in q.index:
                continue
            r = q.loc[b]
            g = bb.get(b).group
            sig = bool(r.significant)
            ax.errorbar(r["diff"], k, xerr=[[r["diff"] - r.boot_ci_low], [r.boot_ci_high - r["diff"]]],
                        fmt=style.GROUP_MARKER[g], ms=4.5, color=style.GROUP[g], elinewidth=1.1, capsize=0,
                        mfc=style.GROUP[g] if sig else "white", mew=1.0)
        ax.axvline(0, color=style.INK_2, lw=0.8)
        ax.set_title(pair_label(s, t))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(BACKBONE_ORDER)), [bb.get(b).label for b in BACKBONE_ORDER])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("ΔAUC")
    fig.suptitle("Target AUC difference, FUSION − frozen backbone (same RBF-SVM classifier); "
                 "> 0 favours FUSION", fontsize=8)
    handles = [Line2D([], [], marker=style.GROUP_MARKER[g], ls="", color=style.GROUP[g], ms=5,
                      label=style.GROUP_LABEL[g]) for g in ("cnn_in1k", "vit_in1k", "ssl_natural", "cxr")]
    handles += [Line2D([], [], marker="o", ls="", color=style.INK_2, ms=5, label="significant (Holm)"),
                Line2D([], [], marker="o", ls="", color=style.INK_2, mfc="white", ms=5, label="not significant")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=6.3)
    style.save(fig, "figB2_primary_forest", mock)


def b3_heads(T, mock):
    h = T.get("head_ablation", pd.DataFrame())
    if h.empty:
        return
    h = h.set_index("backbone").reindex([b for b in BACKBONE_ORDER if b in set(h.backbone)])
    heads = list(hd.HEADS)
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL * 0.8, 0.22 * len(h) + 1.0), sharey=True,
                             constrained_layout=True)
    for ax, kind, title in zip(axes, ("cross", "indomain"), ("Mean cross-domain AUC (6 transfers)",
                                                              "Mean in-domain AUC (3 datasets)")):
        cols = [f"{kind}_{x}" for x in heads]
        M = h.reindex(columns=cols).to_numpy(dtype=float)
        lo = np.nanmin(M) if np.isfinite(M).any() else 0.5
        im = ax.imshow(M, cmap=SEQ, vmin=max(0.5, np.floor(lo * 20) / 20), vmax=1.0, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                best = np.nanargmax(M[i]) == j
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.2,
                        color="white" if v > 0.85 else style.INK, weight="bold" if best else "normal")
        ax.set_xticks(range(len(heads)), [hd.HEADS[x].label for x in heads], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(False); ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axes[0].set_yticks(range(len(h)), [bb.get(b).label for b in h.index])
    style.save(fig, "figB3_head_ablation", mock)


def b4_gap(T, mock):
    g = T.get("transfer_gap", pd.DataFrame())
    if g.empty:
        return
    g = g[g.status == "ok"]
    models = [(S.HC_FAMILY, S.HC_PRIMARY)] + [(S.FROZEN_FAMILY, S.frozen_model(b, S.PRIMARY_HEAD))
                                               for b in BACKBONE_ORDER]
    pairs = [pr for pr in S.PRIMARY_PAIRS if ((g.source == pr[0]) & (g.target == pr[1])).any()]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(style.FULL, 0.2 * len(models) + 1.4), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (s, t) in zip(axes, pairs):
        q = g[(g.source == s) & (g.target == t)].set_index("model")
        for k, (fam, m) in enumerate(models):
            if m not in q.index:
                continue
            r = q.loc[m]
            grp = model_group(m, fam)
            ax.errorbar(r.gap, k, xerr=[[r.gap - r.gap_ci_low], [r.gap_ci_high - r.gap]],
                        fmt=style.GROUP_MARKER[grp], ms=4.5, color=style.GROUP[grp], elinewidth=1.1, capsize=0)
        ax.axvline(0, color=style.INK_2, lw=0.8)
        hc = q.loc[S.HC_PRIMARY].gap if S.HC_PRIMARY in q.index else None
        if hc is not None:
            ax.axvline(hc, color=style.GROUP["handcrafted"], lw=0.8, ls="--")
        ax.set_title(pair_label(s, t)); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [model_label(m, f) for f, m in models])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("Gap (AUC)")
    fig.suptitle("Transfer gap on the target: in-domain AUC − cross-domain AUC (dashed line: FUSION)",
                 fontsize=8)
    fig.legend(handles=[Line2D([], [], marker=style.GROUP_MARKER[g], ls="", color=style.GROUP[g], ms=5,
                               label=style.GROUP_LABEL[g]) for g in style.GROUP],
               loc="outside lower center", ncol=5, fontsize=6.3)
    style.save(fig, "figB4_transfer_gap", mock)


def b5_contrasts(T, mock):
    c = T.get("contrasts", pd.DataFrame())
    if c.empty:
        return
    c = c[(c.status == "ok") & (c.role == "secondary")]
    ids = [x[0] for x in bb.CONTRASTS if x[0] in set(c.contrast)]
    if not ids:
        return
    labels = {cid: f"{bb.get(a).label} − {bb.get(b).label}" for cid, _, a, b in bb.CONTRASTS}
    fig, ax = plt.subplots(figsize=(style.COL * 1.45, 0.34 * len(ids) + 0.9))
    for pi, (s, t) in enumerate(S.PRIMARY_PAIRS):
        dy = (pi - 1.5) * 0.17
        for k, cid in enumerate(ids):
            r = c[(c.contrast == cid) & (c.source == s) & (c.target == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r["diff"], k + dy, xerr=[[r["diff"] - r.boot_ci_low], [r.boot_ci_high - r["diff"]]],
                        fmt=style.PAIR_MARKERS[pi], ms=3.8, color=style.PAIR_COLORS[pi], elinewidth=0.9,
                        capsize=0, mfc=style.PAIR_COLORS[pi] if r.significant else "white",
                        label=pair_label(s, t) if k == 0 else None)
    ax.axvline(0, color=style.INK_2, lw=0.8)
    ax.set_yticks(range(len(ids)), [labels[i] for i in ids]); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔAUC on target (a − b), RBF-SVM head")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=2, fontsize=6.3)
    style.save(fig, "figB5_contrasts", mock)


def b6_probe(T, mock):
    pr = T.get("shortcut_probe", pd.DataFrame())
    if pr.empty:
        return
    reps = list(S.HC_FEATURES) + [S.HC_PRIMARY] + [f"frozen-{b}" for b in BACKBONE_ORDER]
    reps = [r for r in reps if r in set(pr.representation)]
    names = {**{f: f.upper() if f != "sc" else "Shape ctx" for f in S.HC_FEATURES}, S.HC_PRIMARY: "FUSION",
             **{f"frozen-{b}": bb.get(b).label for b in BACKBONE_ORDER}}
    agg = pr.groupby(["representation", "mask", "label_class"]).balanced_accuracy_mean.mean()
    fig, axes = plt.subplots(2, 1, figsize=(style.FULL, 3.8), sharex=True, constrained_layout=True)
    w = 0.38
    for ax, cls in zip(axes, ("normal", "abnormal")):
        for mk, dx in (("lung", -w / 2), ("none", w / 2)):
            vals = [agg.get((r, mk, cls), np.nan) for r in reps]
            ax.bar(np.arange(len(reps)) + dx, vals, width=w - 0.04, color=style.MASK[mk],
                   hatch="////" if mk == "none" else None, edgecolor="white", linewidth=0.5,
                   label=style.MASK_LABEL[mk])
        ax.axhline(0.5, color=style.INK_2, lw=0.8, ls="--", label="Chance")
        ax.set_ylim(0.4, 1.02); ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"{cls.capitalize()} only\nbalanced accuracy")
        ax.axvline(len(S.HC_FEATURES) + 0.5, color=style.GRID, lw=1.0)
    axes[-1].set_xticks(range(len(reps)), [names[r] for r in reps], rotation=40, ha="right", fontsize=6.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=6.5)
    fig.suptitle("Dataset-identification probe (mean over dataset pairs); lower = less site information",
                 fontsize=8)
    style.save(fig, "figB6_shortcut_probe", mock)


def roc_band(y, s, n_boot=1000, seed=0):
    def interp(yy, ss):
        f, t, _ = roc_curve(yy, ss)
        return np.interp(FPR, f, t)
    rng = np.random.default_rng(seed)
    boots = np.array([interp(y[i], s[i]) for i in stratified_bootstrap_indices(y, n_boot, rng)])
    return interp(y, s), np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)


def b7_roc(T, src, tgt, mock, rng):
    x = T["cross"]
    sel = T.get("selection", pd.DataFrame())
    chosen = sel[sel.source == src].selected_model.iloc[0] if not sel.empty and (sel.source == src).any() else None
    wanted = [("FUSION (hand-crafted)", S.HC_PRIMARY, "handcrafted", "-"),
              ("DenseNet-121 / RBF-SVM", S.frozen_model("densenet121", S.PRIMARY_HEAD), "cnn_in1k", "--"),
              ("RAD-DINO / RBF-SVM", S.frozen_model("rad_dino", S.PRIMARY_HEAD), "cxr", ":")]
    if chosen:
        b, h, _ = S.parse_frozen_model(chosen)
        wanted.append((f"Source-selected: {bb.get(b).label} / {hd.HEADS[h].label}", chosen, bb.get(b).group, "-."))
    fig, ax = plt.subplots(figsize=(style.COL, style.COL * 0.95))
    ax.plot([0, 1], [0, 1], color=style.GRID, lw=0.8, zorder=0)
    drawn = 0
    for lab, model, grp, ls in wanted:
        if mock:
            y = np.r_[np.zeros(300), np.ones(300)].astype(int)
            s = y * rng.uniform(0.6, 1.4) + rng.normal(size=600)
        else:
            r = x[(x.model == model) & (x["mask"] == "lung") & (x.source == src) & (x.target == tgt)
                  & (x.source_subsample == 0)]
            if r.empty:
                continue
            p = pd.read_csv(RUNS_ROOT / r.run.iloc[0] / "predictions.csv")
            p = p[p["repeat"] == 0]
            y, s = p.label.to_numpy(), p.score.to_numpy()
        mid, lo, hi = roc_band(y, s, n_boot=500)
        ax.fill_between(FPR, lo, hi, color=style.GROUP[grp], alpha=0.10, linewidth=0)
        ax.plot(FPR, mid, color=style.GROUP[grp], ls=ls, lw=1.4, label=lab)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"Trained on {DL.get(src, src)}, tested on {DL.get(tgt, tgt)}")
    ax.legend(loc="lower right", fontsize=5.8, handlelength=2.4)
    style.save(fig, f"figB7_roc_{src}-to-{tgt}", mock)


def b8_hybrid(T, mock):
    h = T.get("hybrid", pd.DataFrame())
    if h.empty:
        return
    h = h[h.status == "ok"]
    order = [b for b in BACKBONE_ORDER if b in set(h.backbone)]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(style.COL * 1.3, 0.2 * len(order) + 0.9))
    for k, b in enumerate(order):
        q = h[h.backbone == b]
        g = bb.get(b).group
        ax.barh(k, q["diff"].mean(), color=style.GROUP[g], height=0.6, alpha=0.85)
        ax.scatter(q["diff"], np.full(len(q), k), s=8, color=style.INK, zorder=3, linewidths=0)
    ax.axvline(0, color=style.INK_2, lw=0.8)
    ax.set_yticks(range(len(order)), [bb.get(b).label for b in order]); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔAUC: (backbone + FUSION) − backbone\nbar = mean over transfers, dots = each transfer")
    style.save(fig, "figB8_hybrid_gain", mock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=Path)
    ap.add_argument("--roc-pairs", nargs="+", default=["montgomery:shenzhen", "shenzhen:tbx11k"],
                    help="source:target pairs for the ROC figure")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    if args.mock:
        T = mock_tables(rng)
    else:
        if not args.aggregate:
            raise SystemExit("--aggregate is required (or --mock for a labelled layout mock-up)")
        T = load_tables(args.aggregate)
    for name in ("indomain", "cross"):
        T.setdefault(name, pd.DataFrame())
    b1_matrix(T, args.mock)
    b2_primary(T, args.mock)
    b3_heads(T, args.mock)
    b4_gap(T, args.mock)
    b5_contrasts(T, args.mock)
    b6_probe(T, args.mock)
    for pr in args.roc_pairs:
        s, t = pr.split(":")
        if not T["cross"].empty:
            b7_roc(T, s, t, args.mock, rng)
    b8_hybrid(T, args.mock)


if __name__ == "__main__":
    main()
