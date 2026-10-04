"""Run the classical (feature + RBF-SVM) grid for one feature set and one protocol.

In-domain (nested CV, repeated):
    python scripts/run_classical.py --features sc --protocol indomain --dataset shenzhen
Cross-domain (fit on the source, score the target once):
    python scripts/run_classical.py --features sc lbp glcm gabor hog --protocol cross \
        --source shenzhen --target montgomery

Each call creates outputs/runs/<id>/ containing config, meta (git commit, environment),
predictions.csv and metrics.json. Only numbers from these files may be reported.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift.eval import metrics as M  # noqa: E402
from tbshift.eval.protocol import nested_cv_oof, source_to_target  # noqa: E402
from tbshift.models.classical import svm_grid, svm_rbf  # noqa: E402
from tbshift.provenance import DATA_ROOT, new_run, set_seed, write_json  # noqa: E402

FEATURE_KEYS = {"sc": ["sc_pooled", "sc_edge_density"], "lbp": ["tx_lbp"], "glcm": ["tx_glcm"],
                "gabor": ["tx_gabor"], "hog": ["tx_hog"]}


def load(dataset: str, mask: str, size: int, features: list[str]):
    f = np.load(DATA_ROOT / "features" / f"{dataset}_mask-{mask}_s{size}" / "features.npz")
    X = np.hstack([f[k] for name in features for k in FEATURE_KEYS[name]])
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite features in {dataset}")
    return X, f["label"].astype(int), f["image_id"]


def summarise(y, s, threshold=None) -> dict:
    out = {"auc_delong": M.auc_delong_ci(y, s).as_dict()}
    if threshold is not None:
        out["at_source_threshold"] = M.threshold_metrics(y, s, threshold)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, choices=sorted(FEATURE_KEYS))
    ap.add_argument("--protocol", required=True, choices=["indomain", "cross"])
    ap.add_argument("--dataset")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--mask", default="none")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_seed(args.seed)
    feat = "+".join(args.features)

    if args.protocol == "indomain":
        X, y, ids = load(args.dataset, args.mask, args.size, args.features)
        run = new_run(f"classical_{feat}_in-{args.dataset}{args.tag}", vars(args))
        preds, per_repeat = [], []
        for r in range(args.repeats):
            res = nested_cv_oof(X, y, svm_rbf(args.seed + r), svm_grid(X.shape[1]),
                                seed=args.seed + r)
            preds.append(pd.DataFrame({"image_id": ids, "label": y, "repeat": r,
                                       "fold": res["fold"], "score": res["scores"]}))
            per_repeat.append({"repeat": r, **summarise(y, res["scores"]),
                               "chosen_params": res["chosen_params"]})
            print(f"repeat {r}: AUC {per_repeat[-1]['auc_delong']['auc']:.4f}")
        aucs = [p["auc_delong"]["auc"] for p in per_repeat]
        metrics = {"protocol": "indomain", "dataset": args.dataset, "features": feat,
                   "n": len(y), "n_pos": int(y.sum()), "auc_mean_over_repeats": float(np.mean(aucs)),
                   "auc_sd_over_repeats": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                   "repeats": per_repeat}
    else:
        Xs, ys, _ = load(args.source, args.mask, args.size, args.features)
        Xt, yt, ids_t = load(args.target, args.mask, args.size, args.features)
        run = new_run(f"classical_{feat}_{args.source}-to-{args.target}{args.tag}", vars(args))
        res = source_to_target(Xs, ys, Xt, svm_rbf(args.seed), svm_grid(Xs.shape[1]), seed=args.seed)
        preds = [pd.DataFrame({"image_id": ids_t, "label": yt, "score": res["target_scores"]})]
        metrics = {"protocol": "cross", "source": args.source, "target": args.target,
                   "features": feat, "n_target": len(yt), "n_target_pos": int(yt.sum()),
                   "source_threshold": res["threshold"], "chosen_params": res["chosen_params"],
                   "source_oof": summarise(ys, res["source_oof_scores"], res["threshold"]),
                   "target": summarise(yt, res["target_scores"], res["threshold"])}
        print(f"source OOF AUC {metrics['source_oof']['auc_delong']['auc']:.4f} | "
              f"target AUC {metrics['target']['auc_delong']['auc']:.4f}")

    pd.concat(preds).to_csv(run / "predictions.csv", index=False)
    write_json(run / "metrics.json", metrics)
    print(f"run saved: {run}")


if __name__ == "__main__":
    main()
