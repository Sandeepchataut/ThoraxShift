"""Deep baseline runs (fixed recipe, several seeds) with the same output contract as run_classical.

In-domain (5-fold CV per seed; early stopping on a validation split of each training fold):
    python scripts/run_deep.py --arch densenet121 --mask lung --protocol indomain --dataset shenzhen
Cross-domain (train on the source, threshold from the source validation split, score the target):
    python scripts/run_deep.py --arch densenet121 --mask lung --protocol cross \
        --source shenzhen --target montgomery
Resume an interrupted run (completed folds/seeds are skipped):
    python scripts/run_deep.py ... --resume outputs/runs/<run_id>

Smoke-test overrides (--epochs, --input-size, --limit, --no-pretrained) mark the run as
recipe_modified; the aggregation step refuses such runs.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import runs  # noqa: E402
from tbshift.data.images import manifest  # noqa: E402
from tbshift.eval.metrics import youden_threshold  # noqa: E402
from tbshift.models import deep  # noqa: E402
from tbshift.provenance import new_run, write_json  # noqa: E402


def rows_of(dataset: str, limit: int = 0):
    df = manifest(dataset)
    if limit:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.head(limit // 2))
    return [(dataset, r.image_id, int(r.label)) for r in df.itertuples()]


def split_train_val(rows, frac: float, seed: int):
    y = [r[2] for r in rows]
    tr, va = train_test_split(np.arange(len(rows)), test_size=frac, stratify=y, random_state=seed)
    return [rows[i] for i in tr], [rows[i] for i in va]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="densenet121", choices=["densenet121", "resnet50", "efficientnet_b0"])
    ap.add_argument("--protocol", required=True, choices=["indomain", "cross"])
    ap.add_argument("--mask", required=True, choices=["lung", "none"])
    ap.add_argument("--dataset")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--source-subsample", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prepared-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--tag", default="")
    # smoke-test overrides (mark the run as recipe_modified)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--input-size", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int)
    args = ap.parse_args()

    recipe = deep.DeepRecipe(arch=args.arch)
    overrides = {k: v for k, v in {"epochs": args.epochs, "input_size": args.input_size,
                                   "pretrained": False if args.no_pretrained else None}.items()
                 if v is not None}
    if args.num_workers is not None:
        recipe = replace(recipe, num_workers=args.num_workers)  # performance only, not a recipe change
    recipe = replace(recipe, **overrides)
    modified = bool(overrides) or bool(args.limit)
    device = deep.pick_device(args.device)

    src = args.dataset if args.protocol == "indomain" else args.source
    tgt = args.dataset if args.protocol == "indomain" else args.target
    ident = runs.run_identity("deep", args.arch, args.protocol, args.mask, src, tgt,
                              args.source_subsample if args.protocol == "cross" else None,
                              recipe.input_size)
    config = {**vars(args), "resume": str(args.resume) if args.resume else None,
              "recipe": recipe.to_dict(), "recipe_modified": modified, "device": str(device)}
    if args.resume:
        run = args.resume
        old = json.loads((run / "config.json").read_text())
        new = json.loads(json.dumps(config, default=str))  # same JSON normalisation as on disk
        for c in (old, new):
            c.get("recipe", {}).pop("num_workers", None)  # performance only
        for k in ("arch", "protocol", "mask", "dataset", "source", "target", "source_subsample",
                  "seeds", "folds", "recipe"):
            if old.get(k) != new.get(k):
                raise SystemExit(f"--resume config mismatch on {k!r}: {old.get(k)} vs {new.get(k)}")
    else:
        name = (f"deep_{args.arch}_{args.mask}_in-{src}" if args.protocol == "indomain"
                else f"deep_{args.arch}_{args.mask}_{src}-to-{tgt}"
                + (f"_sub{args.source_subsample}" if args.source_subsample else ""))
        run = new_run(name + ("_SMOKE" if modified else "") + args.tag, config)
    partial = run / "partial"
    partial.mkdir(exist_ok=True)
    log_f = open(run / "train_log.jsonl", "a")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n"); log_f.flush()

    preds, per_repeat = [], []
    if args.protocol == "indomain":
        rows = rows_of(args.dataset, args.limit)
        y = np.array([r[2] for r in rows]); ids = np.array([r[1] for r in rows])
        for r_idx, seed in enumerate(args.seeds):
            oof = np.full(len(rows), np.nan); fold_of = np.full(len(rows), -1)
            outer = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
            for k, (tr, te) in enumerate(outer.split(ids, y)):
                part = partial / f"r{r_idx}_f{k}.csv"
                if part.exists():
                    p = pd.read_csv(part); oof[te] = p.score.to_numpy(); fold_of[te] = k
                    continue
                tr_rows, va_rows = split_train_val([rows[i] for i in tr], recipe.val_fraction, seed)
                log(f"seed {seed} fold {k}: train {len(tr_rows)} val {len(va_rows)} test {len(te)}")
                model, _ = deep.train(tr_rows, va_rows, args.prepared_size, args.mask, recipe, seed,
                                      device, run / "ckpt" / f"r{r_idx}_f{k}", log)
                s = deep.predict(model, [rows[i] for i in te], args.prepared_size, args.mask, recipe, device)
                pd.DataFrame({"image_id": ids[te], "score": s}).to_csv(part, index=False)
                oof[te] = s; fold_of[te] = k
            preds.append(runs.predictions_frame(ids, y, oof, r_idx, fold_of))
            per_repeat.append({"repeat": r_idx, "seed": seed, **runs.summarise(y, oof)})
            log(f"seed {seed}: in-domain AUC {per_repeat[-1]['auc_delong']['auc']:.4f}")
        metrics = runs.indomain_metrics(ident, y, per_repeat)
    else:
        s_rows = rows_of(args.source, args.limit)
        t_rows = rows_of(args.target, args.limit)
        yt = np.array([r[2] for r in t_rows]); ids_t = np.array([r[1] for r in t_rows])
        for r_idx, seed in enumerate(args.seeds):
            keep = runs.stratified_subsample(np.array([r[2] for r in s_rows]), args.source_subsample, seed)
            src_rows = [s_rows[i] for i in keep]
            tr_rows, va_rows = split_train_val(src_rows, recipe.val_fraction, seed)
            part = partial / f"r{r_idx}.json"
            if part.exists():
                saved = json.loads(part.read_text())
                s_t, s_va, thr = np.array(saved["target"]), np.array(saved["val"]), saved["threshold"]
            else:
                log(f"seed {seed}: source train {len(tr_rows)} val {len(va_rows)} -> target {len(t_rows)}")
                model, _ = deep.train(tr_rows, va_rows, args.prepared_size, args.mask, recipe, seed,
                                      device, run / "ckpt" / f"r{r_idx}", log)
                s_va = deep.predict(model, va_rows, args.prepared_size, args.mask, recipe, device)
                thr = youden_threshold([r[2] for r in va_rows], s_va)
                s_t = deep.predict(model, t_rows, args.prepared_size, args.mask, recipe, device)
                part.write_text(json.dumps({"target": s_t.tolist(), "val": s_va.tolist(), "threshold": thr}))
            y_va = np.array([r[2] for r in va_rows])
            preds.append(runs.predictions_frame(ids_t, yt, s_t, r_idx))
            per_repeat.append({"repeat": r_idx, "seed": seed, "n_source_train": len(tr_rows),
                               "n_source_val": len(va_rows), "source_threshold": float(thr),
                               "threshold_basis": "source_validation_split",
                               "source_val": runs.summarise(y_va, s_va, thr),
                               "target_metrics": runs.summarise(yt, s_t, thr)})
            log(f"seed {seed}: target AUC {per_repeat[-1]['target_metrics']['auc_delong']['auc']:.4f}")
        metrics = runs.cross_metrics(ident, yt, per_repeat)

    metrics["recipe_modified"] = modified
    pd.concat(preds).to_csv(run / "predictions.csv", index=False)
    write_json(run / "metrics.json", metrics)
    log(f"run saved: {run}")


if __name__ == "__main__":
    main()
