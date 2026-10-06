'''Frozen backbone + trainable head, one protocol per call. Same output contract as
run_classical.py (tbshift.runs), so aggregation treats every model family identically.

In-domain (nested CV, repeated):
    python scripts/run_frozen.py --backbone rad_dino --head svm_rbf --mask lung \
        --protocol indomain --dataset shenzhen
Cross-domain (fit on the source, threshold from the source only, score the target once):
    python scripts/run_frozen.py --backbone rad_dino --head svm_rbf --mask lung \
        --protocol cross --source shenzhen --target montgomery [--size-matched]
Hybrid (deep pooled features + the five hand-crafted blocks, block-weighted; sklearn heads only):
    python scripts/run_frozen.py ... --hybrid

Requires the feature caches from scripts/extract_deep_features.py (and, for --hybrid,
scripts/extract_features.py). --limit marks the run as a smoke test that aggregation ignores.
'''
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tbshift import deep_features as dfc  # noqa: E402
from tbshift import runs  # noqa: E402
from tbshift.eval.protocol import nested_cv_oof, source_to_target  # noqa: E402
from tbshift.models import backbones as bb  # noqa: E402
from tbshift.models import heads as hd  # noqa: E402
from tbshift.provenance import new_run, set_seed, write_json  # noqa: E402

MODEL_FAMILY = "frozen"


def model_name(backbone: str, head: str, hybrid: bool) -> str:
    from tbshift.study import frozen_model
    return frozen_model(backbone, head, hybrid)


def load_inputs(backbone: str, head: str, dataset: str, mask: str, size: int, hybrid: bool):
    '''(X, y, ids, block_sizes, provenance) for one dataset.'''
    kind = hd.HEADS[head].input
    X, y, ids = dfc.load(backbone, dataset, mask, size, kind=kind)
    meta = dfc.meta(backbone, dataset, mask, size)
    prov = {"deep_cache_pooled_sha256": meta.get("pooled_sha256"),
            "backbone_weights_sha256": meta.get("weights_sha256")}
    blocks = None
    if hybrid:
        from run_classical import FEATURE_ORDER, load as load_hc
        Xh, yh, ids_h, hc_blocks, h = load_hc(dataset, mask, size, FEATURE_ORDER)
        pos = {i: k for k, i in enumerate(ids_h.astype(str))}
        if set(pos) != set(ids):
            raise ValueError(f"{dataset}: hand-crafted and deep caches cover different images")
        order = np.array([pos[i] for i in ids])
        if not (yh[order] == y).all():
            raise ValueError(f"{dataset}: labels differ between hand-crafted and deep caches")
        # Column order: the deep block first, then the hand-crafted blocks in canonical order.
        Xd = np.asarray(X, dtype=np.float64)
        X = np.hstack([Xd, Xh[order]])
        blocks = (Xd.shape[1],) + tuple(hc_blocks)
        prov["handcrafted_cache_sha256"] = h
    return X, y, ids, blocks, prov


def stratified_limit(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    return runs.stratified_subsample(y, n, seed) if n and n < len(y) else np.arange(len(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(bb.BACKBONES))
    ap.add_argument("--head", required=True, choices=sorted(hd.HEADS))
    ap.add_argument("--protocol", required=True, choices=["indomain", "cross"])
    ap.add_argument("--mask", required=True, choices=["lung", "none"])
    ap.add_argument("--dataset")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--size-matched", action="store_true",
                    help="cross only: subsample the source to the target's in-domain training size")
    ap.add_argument("--hybrid", action="store_true", help="append the hand-crafted FUSION blocks")
    ap.add_argument("--size", type=int, default=512, help="prepared image size of the caches")
    ap.add_argument("--repeats", type=int, default=5, help="in-domain nested-CV repeats")
    ap.add_argument("--cross-repeats", type=int, default=None,
                    help="cross-domain repeats (default: 1 for sklearn heads, 3 for torch heads)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: stratified subsample per dataset")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_seed(args.seed)

    spec, hspec = bb.get(args.backbone), hd.HEADS[args.head]
    if args.hybrid and not hspec.supports_blocks:
        raise SystemExit(f"--hybrid is only defined for heads {[h for h, s in hd.HEADS.items() if s.supports_blocks]}")
    model = model_name(args.backbone, args.head, args.hybrid)
    cross_repeats = args.cross_repeats or (3 if hspec.trainer == "torch" else 1)
    modified = bool(args.limit)
    extra = {"backbone": args.backbone, "head": args.head, "hybrid": args.hybrid,
             "backbone_group": spec.group, "head_trainer": hspec.trainer}
    slug = f"fz_{args.backbone}{'+fusion' if args.hybrid else ''}_{args.head}_{args.mask}"

    def estimator(n_features, blocks, seed):
        return hd.make(args.head, n_features, seed, blocks, args.device)

    if args.protocol == "indomain":
        if not args.dataset:
            raise SystemExit("--dataset is required for --protocol indomain")
        X, y, ids, blocks, prov = load_inputs(args.backbone, args.head, args.dataset, args.mask,
                                              args.size, args.hybrid)
        if args.limit:
            keep = stratified_limit(y, args.limit, args.seed)
            X, y, ids = X[keep], y[keep], ids[keep]
        ident = runs.run_identity(MODEL_FAMILY, model, "indomain", args.mask, args.dataset,
                                  args.dataset, None, spec.input_size)
        run = new_run(f"{slug}_in-{args.dataset}{'_SMOKE' if modified else ''}{args.tag}",
                      {**vars(args), "model": model, "backbone_spec": spec.to_dict(),
                       "block_sizes": blocks, "recipe_modified": modified, **prov})
        preds, per_repeat = [], []
        for r in range(args.repeats):
            est, grid, n_jobs = estimator(X.shape[-1], blocks, args.seed + r)
            res = nested_cv_oof(X, y, est, grid, seed=args.seed + r, n_jobs=n_jobs)
            preds.append(runs.predictions_frame(ids, y, res["scores"], r, res["fold"]))
            per_repeat.append({"repeat": r, "seed": args.seed + r, **runs.summarise(y, res["scores"]),
                               "chosen_params": res["chosen_params"]})
            print(f"{model} in-{args.dataset} repeat {r}: AUC {per_repeat[-1]['auc_delong']['auc']:.4f}",
                  flush=True)
        metrics = runs.indomain_metrics(ident, y, per_repeat)
    else:
        if not (args.source and args.target) or args.source == args.target:
            raise SystemExit("--source and --target (different) are required for --protocol cross")
        Xs, ys, _, blocks, prov_s = load_inputs(args.backbone, args.head, args.source, args.mask,
                                                args.size, args.hybrid)
        Xt, yt, ids_t, _, prov_t = load_inputs(args.backbone, args.head, args.target, args.mask,
                                               args.size, args.hybrid)
        if args.limit:
            ks, kt = stratified_limit(ys, args.limit, args.seed), stratified_limit(yt, args.limit, args.seed)
            Xs, ys, Xt, yt, ids_t = Xs[ks], ys[ks], Xt[kt], yt[kt], ids_t[kt]
        subsample = runs.size_matched_n(len(ys), len(yt)) if args.size_matched else 0
        ident = runs.run_identity(MODEL_FAMILY, model, "cross", args.mask, args.source, args.target,
                                  subsample, spec.input_size)
        sub = f"_sub{subsample}" if subsample else ""
        run = new_run(f"{slug}_{args.source}-to-{args.target}{sub}{'_SMOKE' if modified else ''}{args.tag}",
                      {**vars(args), "model": model, "backbone_spec": spec.to_dict(),
                       "block_sizes": blocks, "source_subsample": subsample, "cross_repeats": cross_repeats,
                       "recipe_modified": modified,
                       "provenance": {args.source: prov_s, args.target: prov_t}})
        preds, per_repeat = [], []
        for r in range(cross_repeats):
            seed = args.seed + r
            keep = runs.stratified_subsample(ys, subsample, seed)
            Xs_r = Xs if len(keep) == len(ys) else Xs[keep]
            est, grid, n_jobs = estimator(Xs.shape[-1], blocks, seed)
            res = source_to_target(Xs_r, ys[keep], Xt, est, grid, seed=seed, n_jobs=n_jobs)
            preds.append(runs.predictions_frame(ids_t, yt, res["target_scores"], r))
            per_repeat.append({"repeat": r, "seed": seed, "n_source_train": int(len(keep)),
                               "source_threshold": res["threshold"], "threshold_basis": "source_oof",
                               "chosen_params": res["chosen_params"],
                               "source_oof": runs.summarise(ys[keep], res["source_oof_scores"], res["threshold"]),
                               "target_metrics": runs.summarise(yt, res["target_scores"], res["threshold"])})
            print(f"{model} {args.source}->{args.target} repeat {r}: source OOF AUC "
                  f"{per_repeat[-1]['source_oof']['auc_delong']['auc']:.4f} | target AUC "
                  f"{per_repeat[-1]['target_metrics']['auc_delong']['auc']:.4f}", flush=True)
        metrics = runs.cross_metrics(ident, yt, per_repeat)

    metrics.update(extra)
    metrics["recipe_modified"] = modified
    pd.concat(preds).to_csv(run / "predictions.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(f"run saved: {run}")


if __name__ == "__main__":
    main()
