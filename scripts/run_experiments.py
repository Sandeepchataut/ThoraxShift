'''End-to-end driver: every run the analysis plan needs, then aggregation and figures.

    python scripts/run_experiments.py --dry-run                 # list the jobs, run nothing
    python scripts/run_experiments.py                           # everything, resumable
    python scripts/run_experiments.py --stages frozen probe     # selected stages only
    # multi-GPU: build caches once, shard the runs, then finish
    python scripts/run_experiments.py --stages check hc_features deep_features
    CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --stages handcrafted frozen --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments.py --stages handcrafted frozen --shard 1/2 &
    wait; python scripts/run_experiments.py --stages probe aggregate figures

Stages (in order): check, hc_features, deep_features, handcrafted, frozen, finetune (only with
--with-finetune), probe, aggregate, figures.

Resumable: a run job is skipped when a finished, non-smoke run with the same identity
(tbshift.runs.RUN_KEYS) already exists under TBSHIFT_RUNS. Cache-building stages skip existing
caches. Each job's output goes to outputs/logs/<job>.log; a failed job is reported and the
driver continues (use --stop-on-error to halt). The prerequisites are the manifests, prepared
images and lung masks (scripts/prepare_images.py, scripts/segment_lungs.py).
'''
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbshift import deep_features as dfc  # noqa: E402
from tbshift import study as S  # noqa: E402
from tbshift.data import images  # noqa: E402
from tbshift.models import backbones as bb  # noqa: E402
from tbshift.models import heads as hd  # noqa: E402
from tbshift.provenance import DATA_ROOT, RUNS_ROOT  # noqa: E402
from tbshift.runs import RUN_KEYS  # noqa: E402

STAGES = ["check", "hc_features", "deep_features", "handcrafted", "frozen", "finetune", "probe",
          "aggregate", "figures"]
PY = sys.executable
LOGS = ROOT / "outputs" / "logs"


@dataclass
class Job:
    stage: str
    name: str
    cmd: list[str]
    identity: tuple | None = None          # run identity, for skip-if-done
    done_marker: Path | None = None        # cache file, for skip-if-done
    meta: dict = field(default_factory=dict)


def ident(family, model, protocol, mask, source, target, input_size, subsample=0) -> tuple:
    d = {"model_family": family, "model": model, "protocol": protocol, "mask": mask,
         "source": source, "target": target, "source_subsample": subsample, "input_size": input_size}
    return tuple(d[k] for k in RUN_KEYS)


def finished_identities() -> set[tuple]:
    out = set()
    if not RUNS_ROOT.exists():
        return out
    for d in RUNS_ROOT.iterdir():
        mf = d / "metrics.json"
        if not mf.exists() or "_SMOKE" in d.name:
            continue
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("recipe_modified"):
            continue
        out.add(tuple(m.get(k) for k in RUN_KEYS))
    return out


def dataset_sizes(datasets) -> dict[str, int]:
    return {d: len(images.manifest(d)) for d in datasets}


# ------------------------------------------------------------------------------------ job lists
def jobs_hc_features(datasets, masks):
    out = []
    for ds in datasets:
        for m in masks:
            marker = DATA_ROOT / "features" / f"{ds}_mask-{m}_s{S.PREPARED_SIZE}" / "features.npz"
            out.append(Job("hc_features", f"hcfeat_{ds}_{m}",
                           [PY, "scripts/extract_features.py", "--dataset", ds, "--mask", m],
                           done_marker=marker))
    return out


def jobs_deep_features(datasets, backbones, args):
    out = []
    for spec in backbones:
        missing = [(ds, m) for ds in datasets for m in (S.PRIMARY_MASK, S.SHORTCUT_MASK)
                   if not dfc.exists(spec.name, ds, m, S.PREPARED_SIZE)]
        cmd = [PY, "scripts/extract_deep_features.py", "--backbones", spec.name, "--datasets", *datasets,
               "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)]
        out.append(Job("deep_features", f"deepfeat_{spec.name}", cmd,
                       meta={"done": not missing}))
    return out


def _protocol_jobs(stage, prefix, base_cmd, family, model, input_size, mask, datasets, sizes,
                   size_matched: bool, repeats_flag: str | None = "--repeats"):
    out = []
    for ds in datasets:
        cmd = base_cmd + ["--mask", mask, "--protocol", "indomain", "--dataset", ds]
        if repeats_flag:
            cmd += [repeats_flag, str(S.indomain_repeats(sizes[ds]))]
        out.append(Job(stage, f"{prefix}_{mask}_in-{ds}", cmd,
                       ident(family, model, "indomain", mask, ds, ds, input_size)))
    for s, t in S.CROSS_PAIRS:
        if s in datasets and t in datasets:
            out.append(Job(stage, f"{prefix}_{mask}_{s}-to-{t}",
                           base_cmd + ["--mask", mask, "--protocol", "cross", "--source", s, "--target", t],
                           ident(family, model, "cross", mask, s, t, input_size)))
    if size_matched:
        for s, t, n in S.size_matched_pairs(sizes):
            if s in datasets and t in datasets:
                out.append(Job(stage, f"{prefix}_{mask}_{s}-to-{t}_sub{n}",
                               base_cmd + ["--mask", mask, "--protocol", "cross", "--source", s,
                                           "--target", t, "--size-matched"],
                               ident(family, model, "cross", mask, s, t, input_size, n)))
    return out


def jobs_handcrafted(datasets, sizes):
    out = []
    fusion = [PY, "scripts/run_classical.py", "--features", *S.HC_FEATURES]
    out += _protocol_jobs("handcrafted", "hc_fusion", fusion, S.HC_FAMILY, S.HC_PRIMARY,
                          S.HC_INPUT_SIZE, S.PRIMARY_MASK, datasets, sizes, size_matched=True)
    out += _protocol_jobs("handcrafted", "hc_fusion", fusion, S.HC_FAMILY, S.HC_PRIMARY,
                          S.HC_INPUT_SIZE, S.SHORTCUT_MASK, datasets, sizes, size_matched=False)
    for f in S.HC_FEATURES:                     # secondary: individual families
        out += _protocol_jobs("handcrafted", f"hc_{f}", [PY, "scripts/run_classical.py", "--features", f],
                              S.HC_FAMILY, f, S.HC_INPUT_SIZE, S.PRIMARY_MASK, datasets, sizes,
                              size_matched=False)
    return out


def jobs_frozen(datasets, sizes, backbones, heads):
    out = []
    for spec in backbones:
        for head in heads:
            base = [PY, "scripts/run_frozen.py", "--backbone", spec.name, "--head", head]
            model = S.frozen_model(spec.name, head)
            out += _protocol_jobs("frozen", f"fz_{spec.name}_{head}", base, S.FROZEN_FAMILY, model,
                                  spec.input_size, S.PRIMARY_MASK, datasets, sizes, size_matched=True)
        if S.PRIMARY_HEAD in heads:
            base = [PY, "scripts/run_frozen.py", "--backbone", spec.name, "--head", S.PRIMARY_HEAD]
            # shortcut ablation: unmasked input, primary head
            out += _protocol_jobs("frozen", f"fz_{spec.name}_{S.PRIMARY_HEAD}", base, S.FROZEN_FAMILY,
                                  S.frozen_model(spec.name, S.PRIMARY_HEAD), spec.input_size,
                                  S.SHORTCUT_MASK, datasets, sizes, size_matched=False)
            # hybrid: deep + hand-crafted blocks, primary head
            out += _protocol_jobs("frozen", f"fz_{spec.name}+fusion_{S.PRIMARY_HEAD}", base + ["--hybrid"],
                                  S.FROZEN_FAMILY, S.frozen_model(spec.name, S.PRIMARY_HEAD, True),
                                  spec.input_size, S.PRIMARY_MASK, datasets, sizes, size_matched=False)
    return out


def jobs_finetune(datasets, sizes):
    '''Optional ablation: the legacy full fine-tuning path (DenseNet-121) for reference.'''
    base = [PY, "scripts/run_deep.py", "--arch", "densenet121"]
    return _protocol_jobs("finetune", "ft_densenet121", base, "deep", "densenet121", 384,
                          S.PRIMARY_MASK, datasets, sizes, size_matched=False, repeats_flag=None)


def probe_done(datasets, backbones) -> bool:
    '''A finished probe run over exactly these datasets and backbones already exists.'''
    for d in RUNS_ROOT.glob("*_shortcut_probe_*") if RUNS_ROOT.exists() else []:
        try:
            cfg = json.loads((d / "config.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if ((d / "probe.csv").exists() and cfg.get("datasets") == list(datasets)
                and cfg.get("backbones") == list(backbones) and not cfg.get("no_deep")):
            return True
    return False


def build_jobs(args) -> list[Job]:
    datasets = args.datasets
    backbones = bb.resolve(args.backbones)
    sizes = dataset_sizes(datasets) if set(args.stages) & {"handcrafted", "frozen", "finetune"} else {}
    jobs: list[Job] = []
    if "hc_features" in args.stages:
        jobs += jobs_hc_features(datasets, (S.PRIMARY_MASK, S.SHORTCUT_MASK))
    if "deep_features" in args.stages:
        jobs += jobs_deep_features(datasets, backbones, args)
    if "handcrafted" in args.stages:
        jobs += jobs_handcrafted(datasets, sizes)
    if "frozen" in args.stages:
        jobs += jobs_frozen(datasets, sizes, backbones, args.heads)
    if "finetune" in args.stages and args.with_finetune:
        jobs += jobs_finetune(datasets, sizes)
    if "probe" in args.stages:
        names = [b.name for b in backbones]
        jobs.append(Job("probe", "shortcut_probe",
                        [PY, "scripts/run_shortcut_probe.py", "--datasets", *datasets, "--backbones", *names],
                        meta={"done": probe_done(datasets, names)}))
    return jobs


# ------------------------------------------------------------------------------------ execution
def run_job(job: Job, log_dir: Path) -> tuple[bool, float]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{job.name.replace('/', '_')}.log"
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job.cmd)}\n")
        f.flush()
        rc = subprocess.run(job.cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    return rc == 0, time.time() - t0


def latest_run(prefix: str) -> Path | None:
    cands = sorted(d for d in RUNS_ROOT.glob(f"*_{prefix}*") if d.is_dir())
    return cands[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--datasets", nargs="+", default=list(S.DATASETS))
    ap.add_argument("--backbones", nargs="+", default=["all"])
    ap.add_argument("--heads", nargs="+", default=list(S.HEADS), choices=list(S.HEADS))
    ap.add_argument("--with-finetune", action="store_true", help="also run the legacy full fine-tuning ablation")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i (run stages only)")
    ap.add_argument("--batch-size", type=int, default=32, help="feature extraction batch size")
    ap.add_argument("--num-workers", type=int, default=4, help="feature extraction data workers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="aggregate even if primary runs are missing (tables marked INCOMPLETE)")
    args = ap.parse_args()
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    if shard_n > 1 and set(args.stages) & {"check", "hc_features", "deep_features", "probe",
                                            "aggregate", "figures"}:
        raise SystemExit("--shard applies to the run stages only (handcrafted, frozen, finetune). Build the "
                         "caches first without sharding:\n  python scripts/run_experiments.py --stages "
                         "check hc_features deep_features\nthen launch shards with --stages handcrafted frozen, "
                         "and finish with --stages probe aggregate figures.")

    if "check" in args.stages:
        for ds in args.datasets:
            print("check:", images.check_caches(ds, S.PREPARED_SIZE, need_masks=True))

    jobs = build_jobs(args)
    done = finished_identities()
    todo, skipped = [], 0
    for k, j in enumerate(jobs):
        is_done = ((j.identity is not None and j.identity in done)
                   or (j.done_marker is not None and j.done_marker.exists())
                   or j.meta.get("done", False))
        if is_done:
            skipped += 1
            continue
        # Cache stages are not sharded (other shards depend on them); run stages are.
        if j.stage in ("handcrafted", "frozen", "finetune") and k % shard_n != shard_i:
            continue
        todo.append(j)

    by_stage = {}
    for j in todo:
        by_stage[j.stage] = by_stage.get(j.stage, 0) + 1
    print(f"{len(jobs)} jobs planned, {skipped} already done, {len(todo)} to run in shard "
          f"{shard_i}/{shard_n}: {by_stage}")
    if args.dry_run:
        for j in todo:
            print(f"[{j.stage}] {' '.join(j.cmd[1:])}")
        return

    failures = []
    t_start = time.time()
    for n, j in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {j.stage}: {j.name} ...", end=" ", flush=True)
        ok, sec = run_job(j, LOGS)
        print(("ok" if ok else "FAILED") + f" ({sec / 60:.1f} min, total {(time.time() - t_start) / 3600:.2f} h)",
              flush=True)
        if not ok:
            failures.append(j.name)
            if args.stop_on_error:
                break

    if shard_n == 1 and (not failures or args.allow_incomplete):
        if "aggregate" in args.stages:
            print("aggregate ...", flush=True)
            ok, _ = run_job(Job("aggregate", "aggregate_benchmark",
                                [PY, "scripts/aggregate_benchmark.py"]
                                + (["--allow-incomplete"] if args.allow_incomplete else [])), LOGS)
            if not ok:
                failures.append("aggregate_benchmark")
        if "figures" in args.stages and "aggregate_benchmark" not in failures:
            agg = latest_run("aggregate_benchmark")
            if agg is None:
                failures.append("figures (no aggregate run found)")
            else:
                print(f"figures from {agg.name} ...", flush=True)
                ok, _ = run_job(Job("figures", "fig_benchmark",
                                    [PY, "scripts/figures/fig_benchmark.py", "--aggregate", str(agg)]), LOGS)
                if not ok:
                    failures.append("fig_benchmark")
    elif shard_n > 1:
        print("sharded run: start aggregation once every shard has finished:\n"
              "  python scripts/run_experiments.py --stages aggregate figures")

    if failures:
        print(f"\n{len(failures)} job(s) failed; see {LOGS}/<job>.log:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("all done")


if __name__ == "__main__":
    main()
