"""EXT16 B (protocol: EXT16_PROTOCOL.md): 5-member ensembles. For each source run, train members 3 and 4 with the same
configuration, data and seed pattern (run seed + 100*m), and write a mirror run (ensemble_size=5, members 0-2 copied)."""
import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext10_rpt import cmapss_windows, external_windows  # noqa: E402

from selective_rul.model import train_member  # noqa: E402


def process(run, dest, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, _, model_cfg = external_windows(run, cfg, ext_dir)
    else:
        windows, _, model_cfg = cmapss_windows(run, cfg, raw)
    tr, va = windows("fit", "clean"), windows("validation", "clean")
    dest.mkdir(parents=True, exist_ok=True)
    for m in range(cfg["ensemble_size"]):
        shutil.copy2(run / f"model_{m}.pt", dest / f"model_{m}.pt")
    for m in range(cfg["ensemble_size"], 5):
        model, _ = train_member(tr, va, model_cfg, device, cfg["seed"] + m * 100)
        torch.save({"input_dim": tr.x.shape[-1], "state_dict": model.cpu().state_dict()}, dest / f"model_{m}.pt")
    for f in ("splits.json", "normalizer.joblib"):
        shutil.copy2(run / f, dest / f)
    (dest / "config.json").write_text(json.dumps(dict(cfg, ensemble_size=5), indent=2) + "\n")
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
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("model") in a.models and cfg.get("head", "point") == "point":
                runs.append(done.parent)
    runs = runs[i0::n]
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
    print("EXT16M5_DONE", flush=True)


if __name__ == "__main__":
    main()
