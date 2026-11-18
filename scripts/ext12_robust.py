"""EXT12 stage R (protocol: EXT12_PROTOCOL.md): robust-predictor baseline.

For each source run, retrain the 3-member ensemble with the same configuration, split, normaliser and member seeds, on
clean fit windows plus one EXT10-perturbed copy (true RUL labels; generator seed = run seed + 60000). Writes a mirror
run directory (config.json with robust_predictor=true, splits.json, normalizer.joblib, model_*.pt, complete.json) that
ext12_main.py can evaluate unchanged.
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext10_rpt import cmapss_windows, external_windows  # noqa: E402
from ext11_ablate import perturb  # noqa: E402

from selective_rul.data import SENSORS  # noqa: E402
from selective_rul.model import train_member  # noqa: E402


def process(run, dest, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, _, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import load_dataset

        n_ch = len(load_dataset(cfg["dataset"], ext_dir)[1])
    else:
        windows, _, model_cfg = cmapss_windows(run, cfg, raw)
        n_ch = len(SENSORS)
    tr, va = windows("fit", "clean"), windows("validation", "clean")
    xa = perturb(tr.x, n_ch, cfg["seed"] + 60000)
    aug = type("W", (), {"x": np.concatenate([tr.x, xa]), "y": np.concatenate([np.asarray(tr.y)] * 2),
                         "meta": pd.concat([tr.meta, tr.meta], ignore_index=True)})()
    dest.mkdir(parents=True, exist_ok=True)
    for m in range(cfg["ensemble_size"]):
        model, _ = train_member(aug, va, model_cfg, device, cfg["seed"] + m * 100)
        torch.save({"input_dim": aug.x.shape[-1], "state_dict": model.cpu().state_dict()}, dest / f"model_{m}.pt")
    for f in ("splits.json", "normalizer.joblib"):
        shutil.copy2(run / f, dest / f)
    (dest / "config.json").write_text(json.dumps(dict(cfg, robust_predictor=True), indent=2) + "\n")
    (dest / "complete.json").write_text(json.dumps({"source_run": str(run)}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/CMAPSSData"))
    ap.add_argument("--ext-data-dir", type=Path, default=Path("data/external/processed"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--models", nargs="+", default=["gru", "tcn"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("model") in a.models and cfg.get("point_target", "capped") != "raw" and \
                    cfg.get("head", "point") == "point" and (a.seeds is None or cfg["seed"] in a.seeds) and \
                    (a.datasets is None or cfg.get("dataset", "cmapss") in a.datasets):
                runs.append(done.parent)
    runs = runs[i0::n][: a.limit or None]
    print(f"{len(runs)} runs in shard {a.shard}", flush=True)
    for k, run in enumerate(runs, 1):
        dest = a.out / run.parent.name / run.name
        if (dest / "complete.json").exists():
            continue
        t = time.time()
        try:
            process(run, dest, a.data_dir, a.ext_data_dir, a.device)
            print(f"[{k}/{len(runs)}] {run.parent.name}/{run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT12R_DONE", flush=True)


if __name__ == "__main__":
    main()
