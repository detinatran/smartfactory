import argparse
import json
from datetime import datetime
from pathlib import Path

from .data import validate_data
from .experiment import ROOT, evaluate_official, train_experiment
from .model import HEAD_NAMES, MODEL_NAMES


def main():
    parser = argparse.ArgumentParser(description="Selective RUL research experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List available local backbones and heads")
    comparison = sub.add_parser("compare", help="Aggregate development metrics across models/seeds")
    comparison.add_argument("--runs", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--include-smoke", action="store_true")
    validate = sub.add_parser("validate-data")
    validate.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/CMAPSSData")
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=ROOT / "configs/fd001.json")
    train.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/CMAPSSData")
    train.add_argument("--out", type=Path)
    train.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"])
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--model", choices=MODEL_NAMES)
    train.add_argument("--head", choices=HEAD_NAMES)
    train.add_argument("--window", type=int)
    train.add_argument("--hidden", type=int)
    train.add_argument("--layers", type=int)
    train.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"])
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/CMAPSSData")
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"])
    evaluate.add_argument(
        "--official-test",
        action="store_true",
        required=True,
        help="Explicitly unlock the official test only after freezing the protocol",
    )
    args = parser.parse_args()
    if args.command == "models":
        print(
            json.dumps(
                {
                    "models": MODEL_NAMES,
                    "heads": HEAD_NAMES,
                    "mamba_ref": "Local Mamba-1 reference scan; no official fused kernels",
                    "patch_transformer": "Channel-mixing patch encoder; not a PatchTST reproduction",
                },
                indent=2,
            )
        )
    elif args.command == "compare":
        from .comparison import compare_runs

        compare_runs(args.runs, args.out, args.include_smoke)
    elif args.command == "validate-data":
        summary = validate_data(args.data_dir)
        path = ROOT / "data/validation.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    elif args.command == "train":
        cfg = json.loads(args.config.read_text())
        for key in (
            "subset",
            "seed",
            "epochs",
            "device",
            "model",
            "head",
            "window",
            "hidden",
            "layers",
        ):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        out = (
            args.out
            or ROOT
            / "runs"
            / f"{cfg.get('model', 'tcn')}_{cfg.get('head', 'point')}_{cfg['subset']}_seed{cfg['seed']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        train_experiment(cfg, args.data_dir, out)
    else:
        evaluate_official(args.run, args.data_dir, args.device)


if __name__ == "__main__":
    main()
