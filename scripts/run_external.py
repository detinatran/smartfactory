"""Run the external-data matrix in ONE process (skips runs that already have complete.json).

    python scripts/run_external.py --data-dir <processed> --out runs/external_v1 \
        --datasets ncmapss bearings --models gru tcn --seeds 42 43 44 45 46
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from selective_rul.external import folds_for, load_dataset, run_external

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/external.json")
    parser.add_argument("--datasets", nargs="+", default=["ncmapss", "bearings"])
    parser.add_argument("--models", nargs="+", default=["gru", "tcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    base = json.loads(args.config.read_text())
    if args.device:
        base["device"] = args.device
    if args.epochs:
        base["epochs"] = args.epochs
    jobs = []
    for dataset in args.datasets:
        frame, _, _ = load_dataset(dataset, args.data_dir)
        for model in args.models:
            for seed in args.seeds:
                for fold in folds_for(dataset, frame):
                    jobs.append(dict(base, dataset=dataset, model=model, seed=seed, fold=fold))
    print(f"{len(jobs)} jobs -> {args.out}", flush=True)
    failed = 0
    for i, cfg in enumerate(jobs, 1):
        out = args.out / f"{cfg['dataset']}_{cfg['model']}_{cfg['fold']}_seed{cfg['seed']}"
        if (out / "complete.json").exists():
            continue
        start = time.time()
        try:
            run_external(cfg, args.data_dir, out)
            print(f"[{i}/{len(jobs)}] done {out.name} in {time.time() - start:.0f}s", flush=True)
        except Exception:
            failed += 1
            print(f"[{i}/{len(jobs)}] FAILED {out.name}\n{traceback.format_exc()}", flush=True)
    print(f"EXTERNAL_DONE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
