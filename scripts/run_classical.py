"""Run the hand-crafted (feature + RBF-SVM) model for one feature set and one protocol.

In-domain (nested CV, repeated):
    python scripts/run_classical.py --features sc lbp glcm gabor hog --mask lung \
        --protocol indomain --dataset shenzhen
Cross-domain (fit on the source, score the target once):
    python scripts/run_classical.py --features sc lbp glcm gabor hog --mask lung \
        --protocol cross --source shenzhen --target montgomery [--source-subsample 110]

Each call creates outputs/runs/<id>/ (see tbshift.runs for the output contract). Only numbers
from these files may be reported.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import runs  # noqa: E402
from tbshift.eval.protocol import nested_cv_oof, source_to_target  # noqa: E402
from tbshift.models.classical import svm_grid, svm_rbf  # noqa: E402
from tbshift.provenance import DATA_ROOT, file_sha256, new_run, set_seed, write_json  # noqa: E402

FEATURE_KEYS = {"sc": ["sc_pooled", "sc_edge_density"], "lbp": ["tx_lbp"], "glcm": ["tx_glcm"],
                "gabor": ["tx_gabor"], "hog": ["tx_hog"]}
FEATURE_ORDER = ["sc", "lbp", "glcm", "gabor", "hog"]


def feature_dir(dataset: str, mask: str, size: int) -> Path:
    return DATA_ROOT / "features" / f"{dataset}_mask-{mask}_s{size}"


def load(dataset: str, mask: str, size: int, features: list[str]):
    """Feature matrix (blocks in canonical order), labels, ids, block sizes, cache hash."""
    path = feature_dir(dataset, mask, size) / "features.npz"
    f = np.load(path)
    blocks = [np.hstack([f[k] for k in FEATURE_KEYS[name]]) for name in features]
    X = np.hstack(blocks)
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite features in {dataset}")
    return X, f["label"].astype(int), f["image_id"], tuple(b.shape[1] for b in blocks), file_sha256(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, choices=FEATURE_ORDER)
    ap.add_argument("--protocol", required=True, choices=["indomain", "cross"])
    ap.add_argument("--mask", required=True, choices=["lung", "none"],
                    help="lung = primary; none = shortcut ablation")
    ap.add_argument("--dataset")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--size-matched", action="store_true",
                    help="cross only: subsample the source to the target's in-domain training size")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=5, help="in-domain CV repeats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-block-weighting", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_seed(args.seed)
    feats = [f for f in FEATURE_ORDER if f in args.features]  # canonical order
    model = "+".join(feats)

    def estimator_and_grid(n_features, blocks, seed):
        weighted = len(blocks) > 1 and not args.no_block_weighting
        est = svm_rbf(seed, blocks if weighted else None)
        return est, svm_grid(n_features, n_blocks=len(blocks) if weighted else None)

    if args.protocol == "indomain":
        X, y, ids, blocks, h = load(args.dataset, args.mask, args.size, feats)
        ident = runs.run_identity("handcrafted", model, "indomain", args.mask, args.dataset,
                                  args.dataset, None, args.size)
        run = new_run(f"hc_{model}_{args.mask}_in-{args.dataset}{args.tag}",
                      {**vars(args), "feature_cache_sha256": {args.dataset: h}, "block_sizes": blocks})
        preds, per_repeat = [], []
        for r in range(args.repeats):
            est, grid = estimator_and_grid(X.shape[1], blocks, args.seed + r)
            res = nested_cv_oof(X, y, est, grid, seed=args.seed + r)
            preds.append(runs.predictions_frame(ids, y, res["scores"], r, res["fold"]))
            per_repeat.append({"repeat": r, "seed": args.seed + r, **runs.summarise(y, res["scores"]),
                               "chosen_params": res["chosen_params"]})
            print(f"repeat {r}: AUC {per_repeat[-1]['auc_delong']['auc']:.4f}")
        metrics = runs.indomain_metrics(ident, y, per_repeat)
    else:
        Xs, ys, _, blocks, hs = load(args.source, args.mask, args.size, feats)
        Xt, yt, ids_t, _, ht = load(args.target, args.mask, args.size, feats)
        args.source_subsample = runs.size_matched_n(len(ys), len(yt)) if args.size_matched else 0
        keep = runs.stratified_subsample(ys, args.source_subsample, args.seed)
        Xs, ys = Xs[keep], ys[keep]
        ident = runs.run_identity("handcrafted", model, "cross", args.mask, args.source,
                                  args.target, args.source_subsample, args.size)
        sub = f"_sub{args.source_subsample}" if args.source_subsample else ""
        run = new_run(f"hc_{model}_{args.mask}_{args.source}-to-{args.target}{sub}{args.tag}",
                      {**vars(args), "block_sizes": blocks,
                       "feature_cache_sha256": {args.source: hs, args.target: ht}})
        est, grid = estimator_and_grid(Xs.shape[1], blocks, args.seed)
        res = source_to_target(Xs, ys, Xt, est, grid, seed=args.seed)
        preds = [runs.predictions_frame(ids_t, yt, res["target_scores"], 0)]
        per_repeat = [{"repeat": 0, "seed": args.seed, "n_source_train": int(len(ys)),
                       "source_threshold": res["threshold"], "threshold_basis": "source_oof",
                       "chosen_params": res["chosen_params"],
                       "source_oof": runs.summarise(ys, res["source_oof_scores"], res["threshold"]),
                       "target_metrics": runs.summarise(yt, res["target_scores"], res["threshold"])}]
        metrics = runs.cross_metrics(ident, yt, per_repeat)
        print(f"source OOF AUC {per_repeat[0]['source_oof']['auc_delong']['auc']:.4f} | "
              f"target AUC {per_repeat[0]['target_metrics']['auc_delong']['auc']:.4f}")

    pd.concat(preds).to_csv(run / "predictions.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(f"run saved: {run}")


if __name__ == "__main__":
    main()
