"""Add SelectiveNet / MC dropout / deep-ensemble NLL / CQR scores to EXISTING runs.

Writes a mirrored run directory (config.json, thresholds.json, predictions with all old + new
score columns, complete.json) under --out, so summarize_extension.py / summarize_external.py
work unchanged on it. Shared point-ensemble predictions are taken from the saved CSVs.

  python scripts/extra_baselines.py --runs runs/ext_v1_capped_gru_tcn --out runs/extra_v1 \
      --data-dir data/raw/CMAPSSData --ext-data-dir data/external/processed --shard 0/4
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from selective_rul.baselines import baseline_scores, train_head_model
from selective_rul.model import build_model
from selective_rul.selection import apply_threshold, fit_threshold, tie_breakers

NEW = ("selectivenet", "mc_dropout", "deep_ens_nll", "cqr_lower")


def cmapss_windows(run, cfg, raw):
    from selective_rul.data import read_frame
    from selective_rul.experiment import windows_for

    frame = read_frame(raw / f"train_{cfg['subset']}.txt")
    splits = json.loads((run / "splits.json").read_text())
    parts = {k: frame[frame.unit.isin(v)].copy() for k, v in splits.items()}
    normalizer = joblib.load(run / "normalizer.joblib")

    def windows(part, scenario):
        return windows_for(parts[part], normalizer, cfg, scenario)

    return windows, "development", cfg


def external_windows(run, cfg, data_dir):
    from selective_rul.external import corrupt_external, load_dataset, make_external_windows

    frame, features, _ = load_dataset(cfg["dataset"], data_dir)
    splits = json.loads((run / "splits.json").read_text())
    parts = {k: frame[frame.unit.isin(v)].reset_index(drop=True) for k, v in splits.items()}
    normalizer = joblib.load(run / "normalizer.joblib")
    window, stride = cfg["window"][cfg["dataset"]], cfg["stride"][cfg["dataset"]]

    def windows(part, scenario):
        altered = corrupt_external(parts[part], features, normalizer.raw_scale, scenario,
                                   cfg["seed"] + 10000)
        return make_external_windows(altered, normalizer, window, stride)

    model_cfg = dict(cfg, rul_cap=1.0, head="point", point_readout="sigmoid",
                     point_target="capped", window=window)
    return windows, "evaluation", model_cfg


def process(run, out, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    external = "dataset" in cfg
    if external:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        pred_path = {s: run / f"predictions_{s}.csv" for s in cfg["eval_scenarios"]}
        out_pred = {s: out / f"predictions_{s}.csv" for s in cfg["eval_scenarios"]}
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)
        pred_path = {s: run / f"development/predictions_{s}.csv" for s in cfg["eval_scenarios"]}
        out_pred = {s: out / f"development/predictions_{s}.csv" for s in cfg["eval_scenarios"]}
    cap = float(model_cfg["rul_cap"])
    train, validation = windows("fit", "clean"), windows("validation", "clean")
    fitted = {
        "selectivenet": [train_head_model("selectivenet", train, validation, model_cfg, device,
                                          cfg["seed"] + 700)],
        "deep_ens_nll": [train_head_model("deep_ens_nll", train, validation, model_cfg, device,
                                          cfg["seed"] + 800 + 100 * m) for m in range(3)],
        "cqr_lower": [train_head_model("cqr_lower", train, validation, model_cfg, device,
                                       cfg["seed"] + 900 + 100 * m) for m in range(3)],
    }
    ensemble = []
    for member in range(cfg["ensemble_size"]):
        ckpt = torch.load(run / f"model_{member}.pt", map_location="cpu", weights_only=True)
        model = build_model(ckpt["input_dim"], model_cfg)
        model.load_state_dict(ckpt["state_dict"])
        ensemble.append(model.to(device).eval())

    def new_scores(w, shared_prediction):
        return baseline_scores(fitted, ensemble, w.x, shared_prediction, cap, device,
                               cfg["seed"] + 30000)

    # thresholds: old ones + new ones fitted on the clean calibration windows
    thresholds = json.loads((run / "thresholds.json").read_text())
    calib = windows("calibration", "clean")
    with torch.no_grad():
        shared_cal = np.mean([np.concatenate([
            m(torch.from_numpy(calib.x[i:i + 1024]).to(device)).cpu().numpy()
            for i in range(0, len(calib.x), 1024)]) for m in ensemble], axis=0)
    ties_cal = tie_breakers(calib.meta, cfg["seed"] + 20000)
    cal_units = calib.meta.unit.to_numpy()
    for method, score in new_scores(calib, shared_cal).items():
        thresholds[method] = {str(c): fit_threshold(score, ties_cal, cal_units, c)
                              for c in cfg["coverages"]}
    out.mkdir(parents=True, exist_ok=True)
    for scenario in cfg["eval_scenarios"]:
        saved = pd.read_csv(pred_path[scenario])
        w = windows(eval_part, scenario)
        if not (np.array_equal(w.meta.unit.to_numpy(), saved.unit.to_numpy())
                and np.array_equal(w.meta.cycle.to_numpy(), saved.cycle.to_numpy())):
            raise RuntimeError(f"window order mismatch in {run} / {scenario}")
        for method, score in new_scores(w, saved.prediction.to_numpy()).items():
            saved[f"score_{method}"] = score
        out_pred[scenario].parent.mkdir(parents=True, exist_ok=True)
        saved.to_csv(out_pred[scenario], index=False)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    shutil.copy2(run / "config.json", out / "config.json")
    (out / "complete.json").write_text(json.dumps({"source_run": str(run)}) + "\n")


def matched_check(out):
    """Sanity: matched 0.9 acceptance of every new score has ~90% unit-weighted coverage."""
    cfg = json.loads((out / "config.json").read_text())
    path = (out / "predictions_clean.csv") if "dataset" in cfg else (
        out / "development/predictions_clean.csv")
    p = pd.read_csv(path)
    for m in NEW:
        acc = apply_threshold(p[f"score_{m}"].to_numpy(), p.tie_breaker.to_numpy(),
                              fit_threshold(p[f"score_{m}"].to_numpy(), p.tie_breaker.to_numpy(),
                                            p.unit.to_numpy(), 0.9))
        if not 0.8 < acc.mean() < 1.0:
            print(f"WARNING {out.name} {m}: window acceptance {acc.mean():.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/CMAPSSData"))
    parser.add_argument("--ext-data-dir", type=Path,
                        default=Path("data/external/processed"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    shard, n_shards = map(int, args.shard.split("/"))
    runs = []
    for root in args.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("point_target") == "raw":
                continue  # pre-registered: capped C-MAPSS variant only
            runs.append((root, done.parent))
    runs = runs[shard::n_shards]
    for i, (root, run) in enumerate(runs, 1):
        out = args.out / root.name / run.relative_to(root)
        if (out / "complete.json").exists():
            continue
        start = time.time()
        try:
            process(run, out, args.data_dir, args.ext_data_dir, args.device)
            matched_check(out)
            print(f"[{i}/{len(runs)}] {run.name} {time.time() - start:.0f}s", flush=True)
        except Exception as error:  # keep going; failures are listed at the end
            import traceback

            print(f"[{i}/{len(runs)}] FAILED {run}: {error}\n{traceback.format_exc()}", flush=True)
    print("EXTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
