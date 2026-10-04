"""Aggregate run directories into the pre-specified results tables.

    python scripts/aggregate.py
    python scripts/aggregate.py --select <run_id> <run_id> ...   # resolve duplicates explicitly

Integrity rules:
  * runs with recipe_modified=True (smoke tests) are never used;
  * if two runs share the same identity (tbshift.runs.RUN_KEYS), aggregation stops and lists
    them; the user must choose with --select. Nothing is picked silently;
  * the primary repeat (repeat 0, seed 0) is used for every paired test, as pre-specified;
    mean +/- SD over repeats is reported alongside.

Writes a new run directory with tables (CSV + LaTeX), results.json and inputs.json (the exact
run ids used). Figures are generated from this directory.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift.eval import metrics as M  # noqa: E402
from tbshift.provenance import RUNS_ROOT, new_run, write_json  # noqa: E402
from tbshift.runs import RUN_KEYS  # noqa: E402


def discover(root: Path, select: set[str]) -> dict:
    found = defaultdict(list)
    for d in sorted(root.iterdir()):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        if m.get("model_family") not in ("handcrafted", "deep"):
            continue
        if m.get("recipe_modified") or d.name.endswith("_SMOKE") or "_SMOKE" in d.name:
            continue
        if select and d.name not in select:
            continue
        found[tuple(m.get(k) for k in RUN_KEYS)].append((d, m))
    dups = {k: v for k, v in found.items() if len(v) > 1}
    if dups:
        lines = [f"  {dict(zip(RUN_KEYS, k))}:\n    " + "\n    ".join(d.name for d, _ in v)
                 for k, v in dups.items()]
        raise SystemExit("Duplicate runs for the same identity; choose with --select:\n" + "\n".join(lines))
    return {k: v[0] for k, v in found.items()}


def ident(**kw) -> tuple:
    base = {"source_subsample": 0}
    base.update(kw)
    return tuple(base.get(k) for k in RUN_KEYS)


def primary_scores(run_dir: Path) -> pd.DataFrame:
    p = pd.read_csv(run_dir / "predictions.csv")
    p = p[p["repeat"] == 0]
    if p.image_id.duplicated().any():
        raise ValueError(f"duplicate image ids in {run_dir}")
    return p[["image_id", "label", "score"]].set_index("image_id")


def joined(a: pd.DataFrame, b: pd.DataFrame, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    if len(j) != len(a) or len(j) != len(b):
        raise ValueError(f"{what}: image sets differ ({len(a)} vs {len(b)}, joined {len(j)})")
    if not (j.label_a == j.label_b).all():
        raise ValueError(f"{what}: labels disagree between runs")
    return j.label_a.to_numpy(), j.score_a.to_numpy(), j.score_b.to_numpy()


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    body = "\n".join(" & ".join(fmt(v).replace("_", r"\_") for v in row) + r" \\" for row in df.itertuples(index=False))
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
            f"\\begin{{tabular}}{{{'l' * len(cols)}}}\n\\toprule\n"
            + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\" + "\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--select", nargs="*", default=[])
    ap.add_argument("--hc-model", default="sc+lbp+glcm+gabor+hog")
    ap.add_argument("--deep-model", default="densenet121")
    ap.add_argument("--mask", default="lung")
    ap.add_argument("--primary-targets", nargs="+", default=["shenzhen", "tbx11k"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = discover(args.runs_root, set(args.select))
    if not runs:
        raise SystemExit("no eligible runs found")
    used = set()

    # T1 in-domain and T2 cross-domain summaries (all models, all masks)
    t_in, t_cross = [], []
    for key, (d, m) in sorted(runs.items(), key=lambda kv: str(kv[0])):
        k = dict(zip(RUN_KEYS, key))
        r0 = m["repeats"][0]
        if k["protocol"] == "indomain":
            a = r0["auc_delong"]
            t_in.append({**{x: k[x] for x in ("model_family", "model", "mask")}, "dataset": k["source"],
                         "n": m["n"], "auc": a["auc"], "ci_low": a["ci_low"], "ci_high": a["ci_high"],
                         "auc_mean_repeats": m["auc_mean_over_repeats"],
                         "auc_sd_repeats": m["auc_sd_over_repeats"], "run": d.name})
        else:
            a = r0["target_metrics"]["auc_delong"]
            th = r0["target_metrics"]["at_source_threshold"]
            t_cross.append({**{x: k[x] for x in ("model_family", "model", "mask", "source", "target",
                                                 "source_subsample")},
                            "n_target": m["n_target"], "auc": a["auc"], "ci_low": a["ci_low"],
                            "ci_high": a["ci_high"], "auc_mean_repeats": m["target_auc_mean_over_repeats"],
                            "auc_sd_repeats": m["target_auc_sd_over_repeats"],
                            "sens_at_src_thr": th["sensitivity"], "spec_at_src_thr": th["specificity"],
                            "run": d.name})
        used.add(d.name)

    # T3 primary endpoint: hand-crafted vs deep on the same target images
    t_primary, t_gap = [], []
    pairs = sorted({(k[RUN_KEYS.index("source")], k[RUN_KEYS.index("target")])
                    for k in runs if k[RUN_KEYS.index("protocol")] == "cross"})
    for s, t in pairs:
        hc = runs.get(ident(model_family="handcrafted", model=args.hc_model, protocol="cross",
                            mask=args.mask, source=s, target=t,
                            input_size=512))
        dp_candidates = [v for k, v in runs.items() if k[:6] == ("deep", args.deep_model, "cross", args.mask, s, t)
                         and k[RUN_KEYS.index("source_subsample")] == 0]
        if hc is None or not dp_candidates:
            continue
        dp = dp_candidates[0]
        y, s_hc, s_dp = joined(primary_scores(hc[0]), primary_scores(dp[0]), f"{s}->{t}")
        r = M.auc_difference(y, s_hc, s_dp, n_boot=args.n_boot, seed=args.seed)
        t_primary.append({"source": s, "target": t, "role": "primary" if t in args.primary_targets
                          else "descriptive", "n_target": len(y), "auc_handcrafted": r["auc_a"],
                          "auc_deep": r["auc_b"], "diff": r["diff"], "boot_ci_low": r["boot_ci_low"],
                          "boot_ci_high": r["boot_ci_high"], "delong_p": r["delong_p"],
                          "hc_run": hc[0].name, "deep_run": dp[0].name})

        # T4 transfer gap per model family, and the difference of gaps
        in_hc = runs.get(ident(model_family="handcrafted", model=args.hc_model, protocol="indomain",
                               mask=args.mask, source=t, target=t, input_size=512))
        in_dp = [v for k, v in runs.items() if k[:6] == ("deep", args.deep_model, "indomain", args.mask, t, t)]
        if in_hc is None or not in_dp:
            continue
        yh, ih, ch = joined(primary_scores(in_hc[0]), primary_scores(hc[0]), f"gap hc {s}->{t}")
        yd, idp, cdp = joined(primary_scores(in_dp[0][0]), primary_scores(dp[0]), f"gap deep {s}->{t}")
        gh = M.transfer_gap(yh, ih, ch, n_boot=args.n_boot, seed=args.seed)
        gd = M.transfer_gap(yd, idp, cdp, n_boot=args.n_boot, seed=args.seed)
        # Difference of gaps needs all four score vectors on the same images.
        a = primary_scores(in_hc[0]).join(primary_scores(hc[0]), rsuffix="_c")
        b = primary_scores(in_dp[0][0]).join(primary_scores(dp[0]), rsuffix="_c")
        ab = a.join(b, rsuffix="_d", how="inner")
        gdiff = M.gap_difference(ab.label.to_numpy(), ab.score.to_numpy(), ab.score_c.to_numpy(),
                                 ab.score_d.to_numpy(), ab.score_c_d.to_numpy(),
                                 n_boot=args.n_boot, seed=args.seed)
        for fam, g, irun in (("handcrafted", gh, in_hc[0]), ("deep", gd, in_dp[0][0])):
            t_gap.append({"source": s, "target": t, "model_family": fam,
                          "auc_in_domain": g["auc_in_domain"], "auc_cross": g["auc_cross"],
                          "gap": g["gap"]["estimate"], "gap_ci_low": g["gap"]["ci_low"],
                          "gap_ci_high": g["gap"]["ci_high"], "relative_gap": g["relative_gap"]["estimate"],
                          "rel_ci_low": g["relative_gap"]["ci_low"], "rel_ci_high": g["relative_gap"]["ci_high"],
                          "gap_diff_hc_minus_deep": gdiff["estimate"], "gap_diff_ci_low": gdiff["ci_low"],
                          "gap_diff_ci_high": gdiff["ci_high"], "in_domain_run": irun.name})
        used.update({in_hc[0].name, in_dp[0][0].name})

    # T5 size-matched transfer gap: cross models trained on a subsample of S matched to the
    # in-domain training size, compared with the in-domain reference on T.
    t_sized = []
    for key, (d, m) in runs.items():
        k = dict(zip(RUN_KEYS, key))
        if k["protocol"] != "cross" or not k["source_subsample"] or k["mask"] != args.mask:
            continue
        ref = [v for kk, v in runs.items() if kk[:6] == (k["model_family"], k["model"], "indomain",
                                                         k["mask"], k["target"], k["target"])]
        if not ref:
            continue
        y, si, sx = joined(primary_scores(ref[0][0]), primary_scores(d), f"sized {d.name}")
        g = M.transfer_gap(y, si, sx, n_boot=args.n_boot, seed=args.seed)
        t_sized.append({"model_family": k["model_family"], "model": k["model"], "source": k["source"],
                        "target": k["target"], "source_subsample": k["source_subsample"],
                        "auc_in_domain": g["auc_in_domain"], "auc_cross": g["auc_cross"],
                        "gap": g["gap"]["estimate"], "gap_ci_low": g["gap"]["ci_low"],
                        "gap_ci_high": g["gap"]["ci_high"], "run": d.name})
        used.update({d.name, ref[0][0].name})

    tp = pd.DataFrame(t_primary)
    if len(tp):
        mask = tp.role == "primary"
        tp["delong_p_holm"] = np.nan
        tp.loc[mask, "delong_p_holm"] = M.holm(tp.loc[mask, "delong_p"].to_numpy())

    out = new_run("aggregate", {**vars(args), "runs_root": str(args.runs_root)})
    tables = {"indomain": (pd.DataFrame(t_in), "In-domain AUC (repeat 0, DeLong 95\\% CI) and mean $\\pm$ SD over repeats."),
              "cross": (pd.DataFrame(t_cross), "Cross-domain AUC on the target, with sensitivity and specificity at the source-chosen threshold."),
              "primary_endpoint": (tp, "Primary endpoint: paired out-of-domain AUC difference (hand-crafted $-$ deep) on identical target images; Holm-corrected over primary targets."),
              "transfer_gap": (pd.DataFrame(t_gap), "Transfer gap (in-domain AUC on T $-$ cross-domain AUC on T) with paired-bootstrap 95\\% CIs."),
              "transfer_gap_size_matched": (pd.DataFrame(t_sized), "Transfer gap with the source training set subsampled to the in-domain training size.")}
    (out / "tables").mkdir()
    for name, (df, cap) in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
        show = df.drop(columns=[c for c in df.columns if c.endswith("run")], errors="ignore")
        (out / "tables" / f"{name}.tex").write_text(to_latex(show, cap, f"tab:{name}"))
    write_json(out / "inputs.json", sorted(used))
    write_json(out / "results.json", {n: df.to_dict("records") for n, (df, _) in tables.items()})
    print(f"aggregated {len(used)} runs -> {out}")


if __name__ == "__main__":
    main()
