"""Run and preserve model/head/subset/seed experiments sequentially."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from selective_rul.experiment import check_config
from selective_rul.model import HEAD_NAMES, MODEL_NAMES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["FD001", "FD004"],
        choices=["FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--heads", nargs="+", default=list(HEAD_NAMES), choices=HEAD_NAMES)
    parser.add_argument("--windows", nargs="+", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume and args.batch_dir is None:
        parser.error("--resume requires the original --batch-dir")
    cfg = json.loads(args.config.read_text())
    cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    batch = args.batch_dir or ROOT / "runs" / f"matrix_{datetime.now():%Y%m%d_%H%M%S_%f}"
    jobs = []
    for model in dict.fromkeys(args.models):
        for head in dict.fromkeys(args.heads):
            for window in dict.fromkeys(args.windows or [cfg["window"]]):
                for subset in dict.fromkeys(args.subsets):
                    for seed in dict.fromkeys(args.seeds):
                        job_cfg = dict(
                            cfg, model=model, head=head, window=window, subset=subset, seed=seed
                        )
                        check_config(job_cfg)
                        jobs.append(
                            {
                                "name": f"{model}_{head}_w{window}_{subset}_seed{seed}",
                                "config": job_cfg,
                            }
                        )
    sources = sorted((ROOT / "src/selective_rul").glob("*.py"))
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    plan = {"source_sha256": source_hash, "jobs": jobs}
    print(f"{len(jobs)} jobs; sequential; official test stays unused. Batch: {batch}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["name"])
        return
    if args.resume:
        if json.loads((batch / "plan.json").read_text()) != plan:
            raise SystemExit("Plan/code differs from saved batch; use a new batch directory.")
    else:
        batch.mkdir(parents=True, exist_ok=False)
        (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    failed = 0
    for job in jobs:
        folder = batch / job["name"]
        folder.mkdir(exist_ok=True)
        if list(folder.glob("attempt_*/complete.json")):
            print(f"Already complete: {job['name']}", flush=True)
            continue
        attempt = len(list(folder.glob("attempt_*/"))) + 1
        out = folder / f"attempt_{attempt:03d}"
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(job["config"], indent=2) + "\n")
        command = [
            sys.executable,
            "-m",
            "selective_rul.cli",
            "train",
            "--config",
            str(config_path.resolve()),
            "--out",
            str(out.resolve()),
        ]
        print(shlex.join(command), flush=True)
        with (folder / f"attempt_{attempt:03d}.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        (folder / f"attempt_{attempt:03d}_status.json").write_text(
            json.dumps(
                {"exit_code": code, "finished_at": datetime.now(timezone.utc).isoformat()}, indent=2
            )
            + "\n"
        )
        failed += int(code != 0)
    print(
        f"Batch finished: {len(jobs)} planned, {failed} failed this invocation. {batch}", flush=True
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
