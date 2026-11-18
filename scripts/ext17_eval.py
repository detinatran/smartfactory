"""EXT17 (EXT17_PROTOCOL.md): evaluate the EXT12 scores on bearings with physical raw-signal faults. Wraps ext12_main
(v2): scenario 'raw_<kind>' returns windows built from RAWDIR/<kind>/{femto,xjtu_sy}.csv with the run's own splits and
normaliser; everything else is unchanged."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import ext12_main_v2 as M  # alternative module name used on the training host
except ImportError:
    import ext12_main as M  # noqa: N812

KINDS = ("raw_gain", "raw_clip", "raw_detach", "raw_noise", "raw_spike")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ext-data-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    orig = M.external_windows

    def wrapped(run, cfg, ext_dir):
        base, eval_part, model_cfg = orig(run, cfg, ext_dir)
        alts = {k: orig(run, cfg, a.raw_dir / k)[0] for k in KINDS}

        def windows(part, scenario):
            return alts[scenario](part, "clean") if scenario in alts else base(part, scenario)
        return windows, eval_part, model_cfg

    M.external_windows = wrapped
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("dataset") == "bearings" and cfg.get("model") in ("gru", "tcn"):
                runs.append(done.parent)
    runs = runs[i0::n]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs in shard {a.shard}", flush=True)
    for k, run in enumerate(runs, 1):
        dest = a.out / f"{run.parent.name}__{run.name}.csv"
        if dest.exists():
            continue
        t = time.time()
        try:
            M.process(run, None, a.ext_data_dir, "cpu", a.passes, KINDS).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT17_DONE", flush=True)


if __name__ == "__main__":
    main()
