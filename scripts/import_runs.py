"""Import run directories produced on another machine (e.g. a Kaggle notebook output) into
outputs/runs/, after checking that they are complete, unmodified-recipe runs from published code.

    python scripts/import_runs.py <downloaded>/runs [--keep-checkpoints] [--dry-run]

Checks per run directory:
  * the output contract exists (config.json, meta.json, pip_freeze.txt, predictions.csv, metrics.json);
    a run without metrics.json is still in progress and is skipped;
  * metrics.json carries every RUN_KEYS field and recipe_modified is false;
  * every code state that wrote into the run (meta.json and sessions.jsonl) is a clean commit that
    is an ancestor of the local HEAD (i.e. published code you have locally);
  * predictions are finite and every repeat scores the same images;
  * a run with the same name is never overwritten.
Checkpoints (ckpt/) are dropped unless --keep-checkpoints.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift.provenance import ROOT, RUNS_ROOT  # noqa: E402
from tbshift.runs import RUN_KEYS  # noqa: E402

REQUIRED = ("config.json", "meta.json", "pip_freeze.txt", "predictions.csv", "metrics.json")


def is_ancestor(commit: str) -> bool:
    r = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT,
                       capture_output=True)
    return r.returncode == 0


def check(run: Path) -> list[str]:
    problems = []
    missing = [f for f in REQUIRED if not (run / f).exists()]
    if missing:
        return [f"missing {missing}"]
    m = json.loads((run / "metrics.json").read_text())
    absent = [k for k in RUN_KEYS if k not in m]
    if absent:
        problems.append(f"metrics.json lacks {absent}")
    if m.get("recipe_modified"):
        problems.append("recipe_modified run (smoke test)")
    states = [json.loads((run / "meta.json").read_text())]
    if (run / "sessions.jsonl").exists():
        states += [json.loads(line) for line in (run / "sessions.jsonl").read_text().splitlines() if line.strip()]
    for st in states:
        c = st.get("git_commit", "unavailable")
        if st.get("git_dirty"):
            problems.append(f"written from a dirty tree at {c}")
        if c == "unavailable" or not is_ancestor(c):
            problems.append(f"commit {c} is not an ancestor of local HEAD (pull first?)")
    p = pd.read_csv(run / "predictions.csv")
    if not np.isfinite(p.score).all():
        problems.append("non-finite scores")
    sets = {r: frozenset(g.image_id) for r, g in p.groupby("repeat")}
    if len(set(sets.values())) > 1:
        problems.append("repeats score different image sets")
    return sorted(set(problems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="directory containing run directories")
    ap.add_argument("--dest", type=Path, default=RUNS_ROOT)
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    imported = skipped = 0
    for run in sorted(d for d in args.source.iterdir() if d.is_dir() and not d.name.startswith("_")):
        if not (run / "metrics.json").exists():
            print(f"IN PROGRESS  {run.name} (no metrics.json; resume it, do not import)")
            skipped += 1
            continue
        problems = check(run)
        if (args.dest / run.name).exists():
            problems.append("a run with this name already exists in the destination")
        if problems:
            print(f"REJECTED     {run.name}: " + "; ".join(problems))
            skipped += 1
            continue
        print(f"OK           {run.name}")
        if not args.dry_run:
            ignore = None if args.keep_checkpoints else shutil.ignore_patterns("ckpt")
            shutil.copytree(run, args.dest / run.name, ignore=ignore)
        imported += 1
    print(f"{imported} imported, {skipped} skipped" + (" (dry run)" if args.dry_run else ""))
    if skipped and not imported:
        sys.exit(1)


if __name__ == "__main__":
    main()
