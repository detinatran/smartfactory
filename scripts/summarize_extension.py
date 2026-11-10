"""Summarize extension runs (uncapped ablation + extended selectors) on DEVELOPMENT engines.

Never reads official-test files. Macro = equal mean over FD001-FD004 within a seed,
then mean/SD over seeds. Subgroup "rul_lt105" re-scores the SAME acceptances
(thresholds fitted on all windows) on windows with raw RUL < 105, where a capped
predictor can still overestimate by more than delta.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selective_rul.metrics import selective_metrics
from selective_rul.selection import apply_threshold, fit_threshold

KEEP = ("coverage", "optimistic_error_rate", "critical_coverage", "critical_optimistic_error_rate",
        "rmse_raw", "rmse_capped")


def variant_of(cfg):
    target = cfg.get("point_target", "capped")
    return "capped125" if target == "capped" else "raw_softplus"


def run_rows(run, target, subgroup_max):
    cfg = json.loads((run / "config.json").read_text())
    key = dict(model=cfg["model"], head=cfg.get("head", "point"), variant=variant_of(cfg),
               subset=cfg["subset"], seed=cfg["seed"])
    thresholds = json.loads((run / "thresholds.json").read_text())
    rows = []
    for scenario in cfg["eval_scenarios"]:
        p = pd.read_csv(run / f"development/predictions_{scenario}.csv")
        pred, raw, capped = p.prediction.to_numpy(), p.rul_raw.to_numpy(), p.rul_capped.to_numpy()
        units, ties = p.unit.to_numpy(), p.tie_breaker.to_numpy()
        groups = {"all": np.ones(len(p), bool), f"rul_lt{subgroup_max:g}": raw < subgroup_max}
        methods = [c[len("score_"):] for c in p.columns if c.startswith("score_")]
        accept = {("always", "always"): np.ones(len(p), bool)}
        for m in methods:
            score = p[f"score_{m}"].to_numpy()
            accept[(m, "frozen_calibration")] = apply_threshold(
                score, ties, thresholds[m][str(target)])
            accept[(m, "matched_coverage_diagnostic")] = apply_threshold(
                score, ties, fit_threshold(score, ties, units, target))
        for group, mask in groups.items():
            for (method, protocol), acc in accept.items():
                metrics = selective_metrics(pred[mask], raw[mask], capped[mask], units[mask],
                                            acc[mask], cfg["risk_delta"], cfg["critical_rul"])
                rows.append(dict(key, scenario=scenario, group=group, method=method,
                                 protocol=protocol, **{k: metrics[k] for k in KEEP}))
    return rows


def macro(per_run):
    keys = ["model", "head", "variant", "scenario", "group", "method", "protocol"]
    within = per_run.groupby(keys + ["seed"])
    counts = within.subset.nunique()
    seed_mean = within[list(KEEP)].mean()
    seed_mean = seed_mean[counts == 4].reset_index()  # only seeds with all four subsets
    out = seed_mean.groupby(keys)[list(KEEP)].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = seed_mean.groupby(keys).seed.nunique()
    return out.reset_index(), seed_mean


def paired(seed_mean, reference="asymmetric_risk"):
    keys = ["model", "head", "variant", "scenario", "group", "protocol", "seed"]
    ref = seed_mean[seed_mean.method == reference].set_index(keys)
    rows = []
    for method, other in seed_mean[seed_mean.method != reference].groupby("method"):
        joined = ref.join(other.set_index(keys), rsuffix="_o", how="inner")
        diff = (joined.optimistic_error_rate - joined.optimistic_error_rate_o).rename("d")
        cov = (joined.coverage - joined.coverage_o).rename("dc")
        frame = pd.concat([diff, cov], axis=1).reset_index()
        for k, g in frame.groupby(keys[:-1]):
            rows.append(dict(zip(keys[:-1], k), comparator=method,
                             risk_diff_mean=g.d.mean(), risk_diff_sd=g.d.std(),
                             coverage_diff_mean=g.dc.mean(),
                             seeds_reference_lower=int((g.d < 0).sum()), n_seeds=len(g)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--subgroup-max", type=float, default=105.0)
    args = parser.parse_args()
    rows, seen = [], 0
    for root in args.runs:
        for done in sorted(root.rglob("complete.json")):
            run = done.parent
            if (run / "official_test").exists():
                raise SystemExit(f"Refusing: official-test output present in {run}")
            rows += run_rows(run, args.target, args.subgroup_max)
            seen += 1
    per_run = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out / "per_run.csv", index=False)
    summary, seed_mean = macro(per_run)
    summary.to_csv(args.out / "macro_summary.csv", index=False)
    seed_mean.to_csv(args.out / "macro_per_seed.csv", index=False)
    paired(seed_mean).to_csv(args.out / "paired_vs_asymmetric.csv", index=False)
    print(json.dumps({"runs": seen, "rows": len(per_run), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
