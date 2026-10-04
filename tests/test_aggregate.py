"""End-to-end test of scripts/aggregate.py on synthetic run directories."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = dict(source_subsample=0)


def _write_run(root: Path, name: str, ident: dict, ids, y, scores_by_repeat, folds=None):
    d = root / name
    d.mkdir()
    frames = []
    reps = []
    for r, s in enumerate(scores_by_repeat):
        f = pd.DataFrame({"image_id": ids, "label": y, "score": s, "repeat": r})
        frames.append(f)
        auc = {"auc": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        if ident["protocol"] == "indomain":
            reps.append({"repeat": r, "auc_delong": auc})
        else:
            reps.append({"repeat": r, "target_metrics": {"auc_delong": auc,
                         "at_source_threshold": {"sensitivity": 0.5, "specificity": 0.5}}})
    pd.concat(frames).to_csv(d / "predictions.csv", index=False)
    m = {**KEYS, **ident, "repeats": reps}
    if ident["protocol"] == "indomain":
        m.update(n=len(y), auc_mean_over_repeats=0, auc_sd_over_repeats=0)
    else:
        m.update(n_target=len(y), target_auc_mean_over_repeats=0, target_auc_sd_over_repeats=0)
    (d / "metrics.json").write_text(json.dumps(m))


def _fixture(root: Path, duplicate=False, smoke=False):
    rng = np.random.default_rng(0)
    n = 120
    ids = np.array([f"T_{i:03d}" for i in range(n)])
    y = (np.arange(n) % 2).astype(int)
    sig = y * 1.5 + rng.normal(size=n)
    common = dict(mask="lung", target="tbx11k")
    hc = dict(model_family="handcrafted", model="sc+lbp+glcm+gabor+hog", input_size=512, **common)
    dp = dict(model_family="deep", model="densenet121", input_size=384, **common)
    _write_run(root, "a_hc_in", {**hc, "protocol": "indomain", "source": "tbx11k"}, ids, y, [sig, sig])
    _write_run(root, "b_hc_x", {**hc, "protocol": "cross", "source": "shenzhen"}, ids, y, [sig + rng.normal(size=n)])
    _write_run(root, "c_dp_in", {**dp, "protocol": "indomain", "source": "tbx11k"}, ids, y, [sig * 2])
    _write_run(root, "d_dp_x", {**dp, "protocol": "cross", "source": "shenzhen"}, ids, y, [sig + 2 * rng.normal(size=n)])
    _write_run(root, "g_hc_x_sub", {**hc, "protocol": "cross", "source": "shenzhen", "source_subsample": 60},
               ids, y, [sig + rng.normal(size=n)])
    if duplicate:
        _write_run(root, "e_dp_x_dup", {**dp, "protocol": "cross", "source": "shenzhen"}, ids, y, [sig])
    if smoke:
        _write_run(root, "f_dp_x_SMOKE", {**dp, "protocol": "cross", "source": "shenzhen", "recipe_modified": True}, ids, y, [sig])


def _run(root: Path, *extra):
    env = {**os.environ, "TBSHIFT_RUNS": str(root)}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate.py"), "--n-boot", "200", *extra],
                          capture_output=True, text=True, env=env)


def test_aggregate_refuses_when_primary_pairs_are_missing(tmp_path):
    _fixture(tmp_path)
    r = _run(tmp_path)
    assert r.returncode != 0 and "Primary pairs missing" in (r.stderr + r.stdout)


def test_aggregate_incomplete_keeps_full_holm_family(tmp_path):
    _fixture(tmp_path, smoke=True)
    r = _run(tmp_path, "--allow-incomplete")
    assert r.returncode == 0, r.stderr
    out = sorted(tmp_path.glob("*_aggregate_INCOMPLETE"))[-1]
    prim = pd.read_csv(out / "tables" / "primary_endpoint.csv")
    assert len(prim) == 6 and (prim.role == "primary").sum() == 4
    ok = prim[prim.status == "ok"].iloc[0]
    assert (ok.source, ok.target) == ("shenzhen", "tbx11k")
    # Three primary tests missing count as p = 1, so m stays 4: adjusted p = min(1, 4 p).
    assert ok.delong_p_holm == pytest.approx(min(1.0, 4 * ok.delong_p), rel=1e-9)
    assert prim[prim.role == "descriptive"].delong_p_holm.isna().all()
    res = json.loads((out / "results.json").read_text())
    assert res["complete"] is False and len(res["missing_pairs"]) == 5
    gap = pd.read_csv(out / "tables" / "transfer_gap.csv")
    assert set(gap.model_family) == {"handcrafted", "deep"}
    used = json.loads((out / "inputs.json").read_text())
    assert "f_dp_x_SMOKE" not in used
    sized = pd.read_csv(out / "tables" / "transfer_gap_size_matched.csv")
    assert len(sized) == 1 and sized.source_subsample[0] == 60


def test_aggregate_never_uses_a_deep_run_with_another_input_size(tmp_path):
    _fixture(tmp_path)
    rng = np.random.default_rng(5)
    n = 120
    ids = np.array([f"T_{i:03d}" for i in range(n)])
    y = (np.arange(n) % 2).astype(int)
    _write_run(tmp_path, "0_dp_x_512", dict(model_family="deep", model="densenet121", input_size=512,
                                            mask="lung", target="tbx11k", protocol="cross", source="shenzhen"),
               ids, y, [rng.normal(size=n)])
    r = _run(tmp_path, "--allow-incomplete")
    assert r.returncode == 0, r.stderr
    out = sorted(tmp_path.glob("*_aggregate_INCOMPLETE"))[-1]
    prim = pd.read_csv(out / "tables" / "primary_endpoint.csv")
    assert prim[prim.status == "ok"].deep_run.iloc[0] == "d_dp_x"
    cross = pd.read_csv(out / "tables" / "cross.csv")
    assert set(cross.input_size) == {384, 512}


def test_aggregate_refuses_duplicates_until_selected(tmp_path):
    _fixture(tmp_path, duplicate=True)
    r = _run(tmp_path, "--allow-incomplete")
    assert r.returncode != 0 and "Duplicate runs" in (r.stderr + r.stdout)
    r = _run(tmp_path, "--allow-incomplete", "--select", "a_hc_in", "b_hc_x", "c_dp_in", "d_dp_x", "g_hc_x_sub")
    assert r.returncode == 0, r.stderr
