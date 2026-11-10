"""Pool out-of-fold external results per (dataset, model, seed); mean/SD over seeds."""

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rul.external import pooled_acceptance_metrics

KEEP = ["coverage", "optimistic_error_rate", "critical_coverage",
        "critical_optimistic_error_rate", "rmse_raw"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.9)
    args = parser.parse_args()
    groups = {}
    for done in sorted(args.runs.glob("*/complete.json")):
        cfg = json.loads((done.parent / "config.json").read_text())
        groups.setdefault((cfg["dataset"], cfg["model"], cfg["seed"]), []).append(done.parent)
    rows = []
    for (dataset, model, seed), runs in groups.items():
        cfg = json.loads((runs[0] / "config.json").read_text())
        frame = pooled_acceptance_metrics(runs, args.target, cfg["risk_delta"], cfg["critical_rul"])
        frame.insert(0, "n_folds", len(runs))
        frame.insert(0, "seed", seed)
        frame.insert(0, "model", model)
        frame.insert(0, "dataset", dataset)
        rows.append(frame)
    per_seed = pd.concat(rows, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.out / "per_seed.csv", index=False)
    keys = ["dataset", "model", "scenario", "method", "protocol"]
    summary = per_seed.groupby(keys)[KEEP].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_seeds"] = per_seed.groupby(keys).seed.nunique()
    summary.reset_index().to_csv(args.out / "summary.csv", index=False)
    ref = per_seed[per_seed.method == "asymmetric_risk"].set_index(keys[:-2] + ["protocol", "seed"])
    paired = []
    for method, g in per_seed[~per_seed.method.isin(["asymmetric_risk", "always"])].groupby("method"):
        j = ref.join(g.set_index(keys[:-2] + ["protocol", "seed"]), rsuffix="_o", how="inner")
        j["d"] = j.optimistic_error_rate - j.optimistic_error_rate_o
        j["dc"] = j.coverage - j.coverage_o
        for k, h in j.reset_index().groupby(["dataset", "model", "scenario", "protocol"]):
            paired.append(dict(zip(["dataset", "model", "scenario", "protocol"], k),
                               comparator=method, risk_diff_mean=h.d.mean(),
                               risk_diff_sd=h.d.std(), coverage_diff_mean=h.dc.mean(),
                               seeds_asym_lower=int((h.d < 0).sum()), n_seeds=len(h)))
    pd.DataFrame(paired).to_csv(args.out / "paired_vs_asymmetric.csv", index=False)
    print(json.dumps({"groups": len(groups), "rows": len(per_seed)}, indent=2))


if __name__ == "__main__":
    main()
