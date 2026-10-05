# Compute: where the deep models run, and how results come back

The hand-crafted pipeline, segmentation of the NLM sets, the shortcut probe and aggregation run on
CPU. Only deep-model training needs a GPU. Every GPU run uses a repository script at a pushed
commit; the runner notebook only clones, installs, runs and saves.

## 1. Compute estimate

Assumptions (stated so they can be checked against the first real run's `train_log.jsonl`):

| Quantity | Value |
|---|---|
| Training throughput, DenseNet-121, 384 × 384, AMP, batch 16, one T4 | 60 img/s (planning value); 30 img/s (pessimistic) |
| Epochs per training | 25 on average (maximum 30, early stopping with patience 5) |
| Images per epoch | in-domain: 0.8 · N · 0.85 (outer fold, minus 15% validation); cross: 0.85 · N_S |
| Validation and inference overhead | +10% |
| TBX11K primary task (TB vs healthy, train+val) | about 4,600 images; to be confirmed after download |

Primary endpoint (DenseNet-121, lung-masked, 3 seeds):

| Block | Training images processed |
|---|---|
| In-domain, Shenzhen (5 folds × 3 seeds) | 0.17 M |
| In-domain, Montgomery | 0.04 M |
| In-domain, TBX11K | 1.17 M |
| Cross, 6 source→target pairs × 3 seeds | 0.69 M |
| Size-matched cross (3 defined pairs × 3 seeds) | 0.05 M |
| Total incl. 10% overhead | 2.3 M |

That is about **11 T4-hours** at the planning throughput and **22 T4-hours** at the pessimistic one.
TBX11K accounts for about 75% of it. Budget one full rerun: **22–44 T4-hours**.

Secondary experiments, at the same assumptions:

| Experiment | T4-hours |
|---|---|
| ResNet-50, same protocol | 11–22 |
| EfficientNet-B0, same protocol | 7–14 |
| CXR-pretrained DenseNet-121 (224 × 224 input) | 4–7 |
| Deep unmasked ablation, cross pairs only | 4–8 |
| Shortcut probe (frozen features) and TBX11K lung segmentation | < 1 |
| Total | 25–50 |

**Input size.** The deep recipe uses 384 × 384, not 512 × 512. Cost scales with pixel count:
512 would cost 1.78× as much (about +9–17 h on the primary), and 224 would cost 0.34×. At this
scale the difference is hours, not days. 384 keeps small lesions better resolved than 224 and is
already fixed in the analysis plan. It is a recipe choice for the deep model; it does not need to
match the 512 grid of the hand-crafted features.

## 2. Where to run

| Option | Capacity | Fit for this project |
|---|---|---|
| **Kaggle Notebooks (recommended)** | 30 GPU-hours per week, 12 h per session, T4 × 2 or P100, background "Save & Run All" | The primary work with one rerun fits in one to two weeks. Background execution suits evening work. Outputs are versioned with the notebook. Running two commands in parallel uses both T4s. |
| Colab Pro | 100 compute units per month; a T4 uses about 1.2–2 units per hour, i.e. roughly 50–80 T4-hours | Sessions are tied to an open browser and disconnect when idle. Poor for unattended runs. No advantage over Kaggle here. |
| University HPC | Depends on an active allocation | Use it only if a collaborator can provide access quickly. Do not plan around it. |
| Pay-per-hour cloud GPU | No quota; billed by the hour | The fallback if the Kaggle quota runs out before a deadline. The whole primary costs roughly tens of dollars; check current prices. |

Decision: **Kaggle for everything, with a pay-per-hour GPU as the deadline fallback.**

## 3. Setup checklist (Kaggle)

- [ ] Kaggle account: verify the phone number (needed for GPU and internet access in notebooks).
- [ ] Local Kaggle CLI: `pip install kaggle` in the project venv. Create an API token
      (Kaggle → Settings → API → Create New Token) and save it to `%USERPROFILE%\.kaggle\kaggle.json`.
- [ ] TBX11K: download the official release from the authors' page (mmcheng.net/tb, Google Drive)
      and read its terms of use. Implement the TBX11K manifest builder, then run
      `prepare_images.py` and `segment_lungs.py` for TBX11K, either locally (CPU, overnight) or on
      the Kaggle GPU with the runner.
- [ ] Build the data package from the local caches, not from raw data:
      `data/manifests/`, `data/prepared/`, `data/masks/`, each with its `_provenance.json`.
      Shenzhen and Montgomery are about 150 MB. Upload it as a **private** dataset:
      `kaggle datasets init -p pkg`, edit the metadata (title `thoraxshift-data`), then
      `kaggle datasets create -p pkg --dir-mode zip`. Add `kaggle datasets version` for updates.
- [ ] Runner notebook: on Kaggle, choose New Notebook → File → Import Notebook →
      `notebooks/kaggle_runner.ipynb`. Settings: Accelerator = GPU T4 × 2, Internet = On.
      Add Input → your `thoraxshift-data` dataset.
- [ ] Configuration cell: set `COMMIT` to a pushed 40-character SHA and list `COMMANDS`.
- [ ] Save Version → **Save & Run All**. The run continues with the browser closed.

## 4. Checkpointing and resume

- `run_deep.py` writes, inside the run directory:
  - `partial/` after each completed fold or seed;
  - `ckpt/<fold>/last.pt` after every epoch (model, optimiser, scheduler, scaler, RNG states; written atomically);
  - `ckpt/<fold>/best.pt` at every validation improvement.

  When a fold finishes, its `last.pt` is deleted and its `best.pt` is kept.
- A run killed at the 12-hour limit has lost at most one epoch. To resume:
  - [ ] From the finished notebook version's output, add it as an input to the next version
        (Add Input → Notebook Output Files, or Output → New Dataset).
  - [ ] Put the run directory's path in `RESUME_FROM`, and add `--resume {RUNS}/<run_id>` to the same command.
  - [ ] Keep the same `COMMIT`. Resuming under different code is refused (`sessions.jsonl`
        records every session), as is resuming with different settings.
- The longest command is the TBX11K in-domain run with 3 seeds, at about 5–11 h. If a command
  exceeds a session, resume it; do **not** split its seeds into separate runs. Repeat 0 must be
  seed 0 for the pre-specified primary rule, and separate runs would share one identity.

## 5. Sync results back

- [ ] Locally: `kaggle kernels output <user>/<notebook-slug> -p outputs/incoming/<version>`
- [ ] `python scripts/import_runs.py outputs/incoming/<version>/runs --dry-run`, then without
      `--dry-run`. It imports only complete runs, from clean commits that are ancestors of your
      local HEAD, with an unmodified recipe. It never overwrites, drops checkpoints, and reports
      in-progress runs, which need resuming rather than importing.
- [ ] `python scripts/aggregate.py` reads `outputs/runs/` exactly as it does for local runs.
