'''Aggregate every run into the pre-specified tables of docs/analysis_plan.md.

    python scripts/aggregate_benchmark.py
    python scripts/aggregate_benchmark.py --select <run_id> ...      # resolve duplicates explicitly
    python scripts/aggregate_benchmark.py --allow-incomplete         # interim tables, marked INCOMPLETE

Integrity rules (unchanged from the original plan):
  * smoke runs (recipe_modified / *_SMOKE) are never used;
  * two runs with the same identity (tbshift.runs.RUN_KEYS) stop aggregation until one is chosen
    with --select; nothing is picked silently;
  * every paired test uses repeat 0 (pre-specified); mean +/- SD over repeats is reported alongside;
  * the Holm families have a FIXED size; a comparison that cannot be computed counts as p = 1.

Writes outputs/runs/<stamp>_aggregate_benchmark/ with tables/*.csv + *.tex, results.json,
inputs.json (exact run ids used) and summary.md (the headline numbers in plain text).
'''

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import study as S  # noqa: E402
from tbshift.eval import metrics as M  # noqa: E402
from tbshift.eval.compare import _cmp_auc, _cmp_gap, _cmp_gapdiff, run_parallel  # noqa: E402
from tbshift.models import backbones as bb  # noqa: E402
from tbshift.models import heads as hd  # noqa: E402
from tbshift.provenance import RUNS_ROOT, new_run, write_json  # noqa: E402
from tbshift.runs import RUN_KEYS  # noqa: E402

FAMILIES = (S.HC_FAMILY, S.FROZEN_FAMILY, "deep")
FT_INPUT_SIZE = 384   # legacy fine-tuning recipe


# ------------------------------------------------------------------------------- run discovery
def discover(root: Path, select: set[str]) -> dict:
    found = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        if m.get("model_family") not in FAMILIES:
            continue
        if m.get("recipe_modified") or "_SMOKE" in d.name:
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


def input_size_of(family: str, model: str) -> int:
    if family == S.HC_FAMILY:
        return S.HC_INPUT_SIZE
    if family == S.FROZEN_FAMILY:
        return bb.get(S.parse_frozen_model(model)[0]).input_size
    return FT_INPUT_SIZE


class Runs:
    def __init__(self, runs: dict):
        self.runs = runs
        self.used: set[str] = set()

    def get(self, family, model, protocol, mask, source, target, subsample=0):
        key = {"model_family": family, "model": model, "protocol": protocol, "mask": mask,
               "source": source, "target": target, "source_subsample": subsample,
               "input_size": input_size_of(family, model)}
        hit = self.runs.get(tuple(key[k] for k in RUN_KEYS))
        if hit is not None:
            self.used.add(hit[0].name)
        return hit

    def items(self):
        return self.runs.items()


def diff_row(base: dict, res, a_name: str, b_name: str) -> dict:
    if res is None or isinstance(res, Exception):
        status = "MISSING" if res is None else f"ERROR: {res}"
        return {**base, "status": status, f"auc_{a_name}": np.nan, f"auc_{b_name}": np.nan, "diff": np.nan,
                "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_p": np.nan, "delong_p": np.nan}
    return {**base, "status": "ok", f"auc_{a_name}": res["auc_a"], f"auc_{b_name}": res["auc_b"],
            "diff": res["diff"], "boot_ci_low": res["boot_ci_low"], "boot_ci_high": res["boot_ci_high"],
            "boot_p": res["boot_p"], "delong_p": res["delong_p"]}


def add_holm(df: pd.DataFrame, family_mask: pd.Series, col="delong_p") -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    if family_mask.any():
        df.loc[family_mask, "p_holm"] = M.holm(df.loc[family_mask, col].to_numpy(dtype=float))
    ok = df.get("status", pd.Series("ok", index=df.index)) == "ok"
    df["significant"] = (family_mask & ok & (df.p_holm < 0.05)
                         & ((df.get("boot_ci_low", np.nan) > 0) | (df.get("boot_ci_high", np.nan) < 0)))
    df["robust"] = ~(family_mask & ok & ((df.p_holm < 0.05) != ((df.get("boot_ci_low", np.nan) > 0)
                                                                | (df.get("boot_ci_high", np.nan) < 0))))
    return df


# ------------------------------------------------------------------------------- output helpers
def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    body = "\n".join(" & ".join(fmt(v).replace("_", r"\_") for v in row) + r" \\"
                     for row in df.itertuples(index=False))
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
            f"\\begin{{tabular}}{{{'l' * len(cols)}}}\n\\toprule\n"
            + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\" + "\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def md_table(df: pd.DataFrame, cols: list[str], floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_(no rows)_\n"
    d = df[[c for c in cols if c in df.columns]]
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                       else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def display(model: str, family: str) -> str:
    if family == S.HC_FAMILY:
        return "FUSION" if model == S.HC_PRIMARY else f"HC:{model}"
    if family == "deep":
        return f"{model} (fine-tuned)"
    b, h, hyb = S.parse_frozen_model(model)
    return f"{bb.get(b).label}{' + FUSION' if hyb else ''} / {hd.HEADS[h].label}"


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--select", nargs="*", default=[])
    ap.add_argument("--probe", type=Path, help="shortcut-probe run dir (default: latest)")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    nb, sd, mask = args.n_boot, args.seed, S.PRIMARY_MASK

    R = Runs(discover(args.runs_root, set(args.select)))
    if not R.runs:
        raise SystemExit(f"no eligible runs under {args.runs_root}")
    HC = (S.HC_FAMILY, S.HC_PRIMARY)
    FZ = lambda b, h=S.PRIMARY_HEAD, hyb=False: (S.FROZEN_FAMILY, S.frozen_model(b, h, hyb))  # noqa: E731

    # ---------------- T1 / T2: every run
    t_in, t_cross = [], []
    for key, (d, m) in sorted(R.items(), key=lambda kv: str(kv[0])):
        k = dict(zip(RUN_KEYS, key))
        row = {"model_family": k["model_family"], "model": k["model"],
               "display": display(k["model"], k["model_family"]), "mask": k["mask"],
               "backbone": m.get("backbone", ""), "head": m.get("head", ""), "hybrid": m.get("hybrid", False)}
        r0 = m["repeats"][0]
        if k["protocol"] == "indomain":
            a = r0["auc_delong"]
            t_in.append({**row, "dataset": k["source"], "n": m["n"], "auc": a["auc"], "ci_low": a["ci_low"],
                         "ci_high": a["ci_high"], "auc_mean_repeats": m["auc_mean_over_repeats"],
                         "auc_sd_repeats": m["auc_sd_over_repeats"], "n_repeats": len(m["repeats"]),
                         "run": d.name})
        else:
            a = r0["target_metrics"]["auc_delong"]
            th = r0["target_metrics"].get("at_source_threshold", {})
            src = r0.get("source_oof", r0.get("source_val", {})).get("auc_delong", {})
            t_cross.append({**row, "source": k["source"], "target": k["target"],
                            "source_subsample": k["source_subsample"], "n_target": m["n_target"],
                            "auc": a["auc"], "ci_low": a["ci_low"], "ci_high": a["ci_high"],
                            "auc_mean_repeats": m["target_auc_mean_over_repeats"],
                            "auc_sd_repeats": m["target_auc_sd_over_repeats"], "n_repeats": len(m["repeats"]),
                            "source_auc": src.get("auc", np.nan),
                            "sens_at_src_thr": th.get("sensitivity", np.nan),
                            "spec_at_src_thr": th.get("specificity", np.nan), "run": d.name})
        R.used.add(d.name)
    t_in, t_cross = pd.DataFrame(t_in), pd.DataFrame(t_cross)

    tasks, meta = [], []

    def queue(table, base, fn, fn_args, a_name="a", b_name="b"):
        if any(x is None for x in fn_args):
            meta.append((table, base, None, a_name, b_name))
        else:
            tasks.append((fn, tuple(x[0] if isinstance(x, tuple) else x for x in fn_args) + (nb, sd)))
            meta.append((table, base, len(tasks) - 1, a_name, b_name))

    # ---------------- T3 primary: FUSION vs each backbone (primary head), lung, cross
    for s, t in S.CROSS_PAIRS:
        hc = R.get(*HC, "cross", mask, s, t)
        for b in S.BACKBONES:
            dp = R.get(*FZ(b), "cross", mask, s, t)
            role = "primary" if (s, t) in S.PRIMARY_PAIRS else "descriptive"
            queue("primary", {"source": s, "target": t, "backbone": b, "role": role,
                              "hc_run": hc[0].name if hc else None, "deep_run": dp[0].name if dp else None},
                  _cmp_auc, (hc, dp), "fusion", "deep")

    # ---------------- T4 source-selected best deep configuration (secondary)
    fz_in = t_in[(t_in.model_family == S.FROZEN_FAMILY) & (t_in["mask"] == mask) & (~t_in.hybrid.astype(bool))] \
        if not t_in.empty else t_in
    selection = []
    for s in S.DATASETS:
        cand = fz_in[fz_in.dataset == s] if not fz_in.empty else fz_in
        if cand.empty:
            continue
        best = cand.sort_values(["auc_mean_repeats", "auc"], ascending=False).iloc[0]
        selection.append({"source": s, "selected_model": best.model, "display": best.display,
                          "source_indomain_auc_mean": best.auc_mean_repeats, "n_candidates": len(cand)})
        for t in S.DATASETS:
            if t == s or (s, t) not in S.CROSS_PAIRS:
                continue
            hc = R.get(*HC, "cross", mask, s, t)
            dp = R.get(S.FROZEN_FAMILY, best.model, "cross", mask, s, t)
            role = "secondary" if (s, t) in S.PRIMARY_PAIRS else "descriptive"
            queue("selected", {"source": s, "target": t, "selected_model": best.model, "role": role},
                  _cmp_auc, (hc, dp), "fusion", "deep")

    # ---------------- T5 transfer gaps (FUSION and each backbone with the primary head)
    gap_models = [HC] + [FZ(b) for b in S.BACKBONES]
    for s, t in S.CROSS_PAIRS:
        hc_in, hc_x = R.get(*HC, "indomain", mask, t, t), R.get(*HC, "cross", mask, s, t)
        for fam, model in gap_models:
            i, x = R.get(fam, model, "indomain", mask, t, t), R.get(fam, model, "cross", mask, s, t)
            base = {"source": s, "target": t, "model_family": fam, "model": model}
            queue("gap", base, _cmp_gap, (i, x))
            if fam != S.HC_FAMILY:
                queue("gapdiff", base, _cmp_gapdiff, (hc_in, hc_x, i, x))

    # ---------------- T6 pre-specified backbone contrasts (primary head, lung, cross)
    for cid, question, a, b in bb.CONTRASTS:
        for s, t in S.CROSS_PAIRS:
            ra, rb = R.get(*FZ(a), "cross", mask, s, t), R.get(*FZ(b), "cross", mask, s, t)
            role = "secondary" if (s, t) in S.PRIMARY_PAIRS else "descriptive"
            queue("contrast", {"contrast": cid, "question": question, "model_a": a, "model_b": b,
                               "source": s, "target": t, "role": role}, _cmp_auc, (ra, rb))

    # ---------------- T7 hybrid: does FUSION add to deep features?  (hybrid - deep)
    for b in S.BACKBONES:
        for s, t in S.CROSS_PAIRS:
            hy, dp = R.get(*FZ(b, hyb=True), "cross", mask, s, t), R.get(*FZ(b), "cross", mask, s, t)
            role = "secondary" if (s, t) in S.PRIMARY_PAIRS else "descriptive"
            queue("hybrid", {"backbone": b, "source": s, "target": t, "role": role}, _cmp_auc, (hy, dp),
                  "hybrid", "deep")

    # ---------------- T8 shortcut ablation: lung-masked vs unmasked (same model, same target)
    for fam, model in gap_models:
        for s, t in S.CROSS_PAIRS:
            lu = R.get(fam, model, "cross", mask, s, t)
            no = R.get(fam, model, "cross", S.SHORTCUT_MASK, s, t)
            queue("mask", {"model_family": fam, "model": model, "source": s, "target": t}, _cmp_auc,
                  (lu, no), "lung", "unmasked")

    # ---------------- T9 legacy fine-tuning reference (only if those runs exist)
    for s, t in S.CROSS_PAIRS:
        ft = R.get("deep", "densenet121", "cross", mask, s, t)
        if ft is None:
            continue
        queue("finetune", {"source": s, "target": t}, _cmp_auc,
              (R.get(*FZ("densenet121"), "cross", mask, s, t), ft), "frozen", "finetuned")

    print(f"computing {len(tasks)} paired comparisons ({nb} bootstrap resamples each) ...", flush=True)
    results = run_parallel(tasks, args.jobs)

    tables_rows = defaultdict(list)
    for table, base, idx, a_name, b_name in meta:
        res = results[idx] if idx is not None else None
        if table == "gap":
            if res is None or isinstance(res, Exception):
                row = {**base, "status": "MISSING" if res is None else f"ERROR: {res}"}
            else:
                row = {**base, "status": "ok", "auc_in_domain": res["auc_in_domain"], "auc_cross": res["auc_cross"],
                       "gap": res["gap"]["estimate"], "gap_ci_low": res["gap"]["ci_low"],
                       "gap_ci_high": res["gap"]["ci_high"],
                       "relative_gap_descriptive": res["relative_gap"]["estimate"],
                       "rel_ci_low": res["relative_gap"]["ci_low"], "rel_ci_high": res["relative_gap"]["ci_high"],
                       "rel_nan_fraction": res["relative_gap"]["nan_fraction"]}
        elif table == "gapdiff":
            if res is None or isinstance(res, Exception):
                row = {**base, "status": "MISSING" if res is None else f"ERROR: {res}"}
            else:
                row = {**base, "status": "ok", "gap_diff_fusion_minus_deep": res["estimate"],
                       "ci_low": res["ci_low"], "ci_high": res["ci_high"], "p_boot": res["p_boot"]}
        else:
            row = diff_row(base, res, a_name, b_name)
        tables_rows[table].append(row)

    T = {k: pd.DataFrame(v) for k, v in tables_rows.items()}
    for name in ("primary", "selected", "contrast", "hybrid", "mask", "finetune", "gap", "gapdiff"):
        T.setdefault(name, pd.DataFrame())

    # Holm families (fixed sizes; missing comparisons have p = NaN -> treated as 1)
    if not T["primary"].empty:
        fam = T["primary"].role == "primary"
        assert fam.sum() == len(S.PRIMARY_PAIRS) * len(S.BACKBONES)
        T["primary"] = add_holm(T["primary"], fam)
    for name in ("selected", "contrast", "hybrid"):
        if not T[name].empty:
            T[name] = add_holm(T[name], T[name].role == "secondary")
    missing_primary = (T["primary"][(T["primary"].role == "primary") & (T["primary"].status != "ok")]
                       if not T["primary"].empty else pd.DataFrame())
    if len(missing_primary) and not args.allow_incomplete:
        raise SystemExit(f"{len(missing_primary)} primary comparisons missing or failed (the Holm family is "
                         f"fixed by the plan):\n" + missing_primary[["source", "target", "backbone", "status"]]
                         .to_string(index=False) + "\nRun them, or pass --allow-incomplete for an interim table.")

    # ---------------- Leaderboard (descriptive) and head ablation
    lb = pd.DataFrame()
    head_tab = pd.DataFrame()
    if not t_cross.empty:
        xc = t_cross[(t_cross["mask"] == mask) & (t_cross.source_subsample == 0)]
        piv = xc.pivot_table(index=["model_family", "model", "display"], columns=["source", "target"],
                             values="auc")
        ranks = piv.rank(ascending=False, axis=0)
        lb = pd.DataFrame({"mean_cross_auc": piv.mean(axis=1), "min_cross_auc": piv.min(axis=1),
                           "mean_rank": ranks.mean(axis=1), "n_pairs": piv.notna().sum(axis=1)})
        prim_cols = [c for c in piv.columns if tuple(c) in S.PRIMARY_PAIRS]
        lb["mean_primary_pairs_auc"] = piv[prim_cols].mean(axis=1) if prim_cols else np.nan
        if not t_in.empty:
            ii = t_in[t_in["mask"] == mask].groupby(["model_family", "model", "display"]).auc.mean()
            lb["mean_indomain_auc"] = ii.reindex(lb.index)
        lb = lb.reset_index().sort_values(["mean_rank", "mean_cross_auc"], ascending=[True, False])
        for c in piv.columns:
            lb[f"auc_{c[0]}_to_{c[1]}"] = lb.set_index(["model_family", "model", "display"]).index.map(
                piv[c].to_dict())
        fz = lb[lb.model_family == S.FROZEN_FAMILY].copy()
        if not fz.empty:
            parsed = fz.model.map(S.parse_frozen_model)
            fz["backbone"] = parsed.map(lambda p: p[0]); fz["head"] = parsed.map(lambda p: p[1])
            fz["hybrid"] = parsed.map(lambda p: p[2])
            fz = fz[~fz.hybrid]
            head_tab = fz.pivot_table(index="backbone", columns="head", values="mean_cross_auc")
            head_in = fz.pivot_table(index="backbone", columns="head", values="mean_indomain_auc")
            head_tab = head_tab.add_prefix("cross_").join(head_in.add_prefix("indomain_")).reset_index()

    # ---------------- Size-matched transfer gap (descriptive)
    sized = []
    for key, (d, m) in R.items():
        k = dict(zip(RUN_KEYS, key))
        if k["protocol"] != "cross" or not k["source_subsample"] or k["mask"] != mask:
            continue
        ref = R.get(k["model_family"], k["model"], "indomain", mask, k["target"], k["target"])
        if ref is None:
            continue
        try:
            g = _cmp_gap(ref[0], d, nb, sd)
        except ValueError as e:
            sized.append({"model": k["model"], "source": k["source"], "target": k["target"], "status": str(e)})
            continue
        R.used.add(d.name)
        sized.append({"model_family": k["model_family"], "model": k["model"], "source": k["source"],
                      "target": k["target"], "source_subsample": k["source_subsample"], "status": "ok",
                      "auc_in_domain": g["auc_in_domain"], "auc_cross": g["auc_cross"],
                      "gap": g["gap"]["estimate"], "gap_ci_low": g["gap"]["ci_low"], "gap_ci_high": g["gap"]["ci_high"]})
    sized = pd.DataFrame(sized)

    # ---------------- Shortcut probe
    probe_dir = args.probe
    if probe_dir is None:
        cands = sorted(args.runs_root.glob("*_shortcut_probe_*"))
        probe_dir = cands[-1] if cands else None
    probe = pd.read_csv(probe_dir / "probe.csv") if probe_dir and (probe_dir / "probe.csv").exists() else pd.DataFrame()

    # ---------------- write
    complete = len(missing_primary) == 0
    out = new_run("aggregate_benchmark" + ("" if complete else "_INCOMPLETE"),
                  {**vars(args), "runs_root": str(args.runs_root), "probe": str(probe_dir) if probe_dir else None})
    (out / "tables").mkdir()
    captions = {
        "indomain": (t_in, "In-domain AUC (repeat 0, DeLong 95\\% CI) and mean $\\pm$ SD over repeats."),
        "cross": (t_cross, "Cross-domain AUC on the target with sensitivity/specificity at the source-chosen threshold."),
        "primary_endpoint": (T["primary"], "Primary endpoint: paired out-of-domain AUC difference (FUSION $-$ frozen backbone, RBF-SVM head) on identical target images; Holm over 4 pairs $\\times$ 12 backbones."),
        "source_selected": (T["selected"], "FUSION vs the deep configuration selected on source in-domain AUC only."),
        "selection": (pd.DataFrame(selection), "Deep configuration selected per source (highest source in-domain AUC)."),
        "leaderboard": (lb, "Descriptive leaderboard: mean cross-domain AUC and mean rank over the six transfer pairs (lung-masked)."),
        "head_ablation": (head_tab, "Head ablation: mean cross-domain and in-domain AUC per backbone and head."),
        "transfer_gap": (T["gap"], "Transfer gap (in-domain AUC on T $-$ cross-domain AUC on T), paired bootstrap 95\\% CI."),
        "transfer_gap_difference": (T["gapdiff"], "Difference of transfer gaps (FUSION $-$ backbone) on identical target images."),
        "transfer_gap_size_matched": (sized, "Transfer gap with the source subsampled to the in-domain training size."),
        "contrasts": (T["contrast"], "Pre-specified backbone contrasts (model a $-$ model b), RBF-SVM head."),
        "hybrid": (T["hybrid"], "Hybrid (backbone + FUSION) $-$ backbone alone, RBF-SVM head."),
        "mask_ablation": (T["mask"], "Lung-masked $-$ unmasked input, same model and target."),
        "finetune_reference": (T["finetune"], "Frozen $-$ fully fine-tuned DenseNet-121 (legacy recipe)."),
        "shortcut_probe": (probe, "Dataset-identification probe within label class (balanced accuracy)."),
    }
    for name, (df, cap) in captions.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
        show = df.drop(columns=[c for c in df.columns if c.endswith("run") or c == "question"], errors="ignore")
        (out / "tables" / f"{name}.tex").write_text(to_latex(show, cap, f"tab:{name}"))
    write_json(out / "inputs.json", sorted(R.used))
    write_json(out / "results.json", {"complete": complete,
                                      **{n: df.to_dict("records") for n, (df, _) in captions.items()}})
    write_summary(out, complete, T, lb, head_tab, pd.DataFrame(selection), probe)
    print(f"aggregated {len(R.used)} runs -> {out}")


def write_summary(out: Path, complete: bool, T: dict, lb: pd.DataFrame, head_tab: pd.DataFrame,
                  selection: pd.DataFrame, probe: pd.DataFrame) -> None:
    parts = [f"# ThoraxShift results summary{'' if complete else ' (INCOMPLETE)'}\n",
             "All AUCs are repeat 0 on the lung-masked input unless stated. Full tables in tables/.\n"]
    p = T["primary"]
    if not p.empty:
        pp = p[p.role == "primary"].copy()
        parts += ["## Primary endpoint: FUSION − frozen backbone (RBF-SVM head)\n",
                  f"Holm family m = {len(pp)}; significant = Holm-DeLong p < 0.05 AND bootstrap CI excludes 0. "
                  f"Significant: {int(pp.significant.sum())} of {len(pp)} "
                  f"(FUSION better: {int((pp.significant & (pp['diff'] > 0)).sum())}, "
                  f"deep better: {int((pp.significant & (pp['diff'] < 0)).sum())}).\n",
                  md_table(pp.sort_values(["target", "source", "diff"]),
                           ["source", "target", "backbone", "auc_fusion", "auc_deep", "diff", "boot_ci_low",
                            "boot_ci_high", "p_holm", "significant", "status"])]
    if not lb.empty:
        parts += ["## Leaderboard (descriptive; mean over the six transfer pairs)\n",
                  md_table(lb.head(25), ["display", "mean_cross_auc", "mean_primary_pairs_auc", "min_cross_auc",
                                         "mean_rank", "mean_indomain_auc", "n_pairs"])]
    if not selection.empty:
        parts += ["## Deep configuration selected on the source only\n",
                  md_table(selection, ["source", "display", "source_indomain_auc_mean", "n_candidates"]),
                  md_table(T["selected"], ["source", "target", "selected_model", "auc_fusion", "auc_deep", "diff",
                                           "boot_ci_low", "boot_ci_high", "p_holm", "significant"])]
    if not head_tab.empty:
        parts += ["## Head ablation (mean cross-domain AUC)\n",
                  md_table(head_tab, ["backbone"] + [c for c in head_tab.columns if c.startswith("cross_")])]
    if not T["contrast"].empty:
        c = T["contrast"]
        summ = c[c.status == "ok"].groupby(["contrast", "model_a", "model_b"]).agg(
            mean_diff=("diff", "mean"), n_sig=("significant", "sum"), n=("diff", "size")).reset_index()
        parts += ["## Backbone contrasts (a − b, mean over pairs; n_sig over the primary pairs)\n",
                  md_table(summ, ["contrast", "model_a", "model_b", "mean_diff", "n_sig", "n"])]
    if not T["hybrid"].empty:
        h = T["hybrid"]
        summ = h[h.status == "ok"].groupby("backbone").agg(mean_gain=("diff", "mean"),
                                                            n_sig=("significant", "sum")).reset_index()
        parts += ["## Hybrid gain (backbone + FUSION − backbone)\n", md_table(summ, ["backbone", "mean_gain", "n_sig"])]
    if not probe.empty:
        pr = probe.groupby(["representation", "mask", "label_class"]).balanced_accuracy_mean.mean().unstack("mask")
        parts += ["## Shortcut probe (dataset-ID balanced accuracy, mean over dataset pairs)\n",
                  md_table(pr.reset_index(), ["representation", "label_class", "lung", "none"])]
    (out / "summary.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
