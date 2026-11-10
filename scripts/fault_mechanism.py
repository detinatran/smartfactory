"""Fault-induced conditional bias: does uncertainty see what a sensor fault does to predictions?

Post hoc analysis (ANALYSIS_V2_PROTOCOL.md). For every mirrored run that holds clean AND corrupted
predictions on the SAME windows (fault_v1/*, battery_fault_v1/*, extra_v1/* for dropout/drift):
  delta     = prediction_fault - prediction_clean      (pure fault effect; the label is unchanged)
  d_spread  = ensemble_std_fault - ensemble_std_clean
Per run x scenario, over affected windows (|delta| > 1e-6 x label scale):
  frac_affected, median delta / clean-RMSE, share of affected windows with delta > 0 (optimistic shift),
  median |delta| / ensemble_std_fault, Spearman(delta, d_spread), Spearman(delta, d_score) per score,
  and for the NEW optimistic events (event under fault, not under clean on that window): the fraction
  each selector REJECTS at matched 90% coverage (higher = it catches fault-induced errors).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from selective_rul.selection import apply_threshold, fit_threshold

SCORES = ("asymmetric_risk", "signed_error_q90", "cqr_lower", "ensemble_std", "mc_dropout",
          "deep_ens_nll", "selectivenet", "symmetric_risk", "random", "asym_aug")


def csv_path(run, scenario):
    p = run / f"predictions_{scenario}.csv"
    return p if p.exists() else run / "development" / f"predictions_{scenario}.csv"


def analyse(run, scenario, delta_cfg, meta):
    clean, fault = pd.read_csv(csv_path(run, "clean")), pd.read_csv(csv_path(run, scenario))
    if not (np.array_equal(clean.unit, fault.unit) and np.array_equal(clean.cycle, fault.cycle)):
        raise RuntimeError(f"window mismatch {run} {scenario}")
    std_c = clean.get("score_ensemble_std", clean.get("ensemble_std")).to_numpy()
    std_f = fault.get("score_ensemble_std", fault.get("ensemble_std")).to_numpy()
    y = fault.rul_raw.to_numpy()
    d = fault.prediction.to_numpy() - clean.prediction.to_numpy()
    scale = np.sqrt(np.mean((clean.prediction.to_numpy() - y) ** 2))
    aff = np.abs(d) > 1e-6 * max(np.abs(y).max(), 1e-9)
    ev_c = clean.prediction.to_numpy() - y > delta_cfg
    ev_f = fault.prediction.to_numpy() - y > delta_cfg
    new = ev_f & ~ev_c
    row = dict(meta, scenario=scenario, n=len(d), frac_affected=float(aff.mean()),
               event_rate_clean=float(ev_c.mean()), event_rate_fault=float(ev_f.mean()),
               frac_new_events=float(new.mean()))
    if aff.sum() >= 10:
        row.update(median_shift_over_rmse=float(np.median(d[aff]) / scale),
                   share_optimistic_shift=float((d[aff] > 0).mean()),
                   median_abs_shift_over_spread=float(np.median(np.abs(d[aff]) / np.maximum(std_f[aff], 1e-9))),
                   spearman_shift_vs_dspread=float(spearmanr(d[aff], (std_f - std_c)[aff]).statistic))
    ties = fault.tie_breaker.to_numpy()
    units = fault.unit.to_numpy()
    for name in SCORES:
        col = [c for c in fault.columns if c == f"score_{name}" or c.startswith(f"score_{name}__")]
        if not col:
            continue
        s_f = fault[col[0]].to_numpy()
        acc = apply_threshold(s_f, ties, fit_threshold(s_f, ties, units, 0.9))
        if new.sum() >= 5:
            row[f"reject_new_{name}"] = float((~acc[new]).mean())
        if col[0] in clean.columns and aff.sum() >= 10:
            row[f"spearman_shift_vs_dscore_{name}"] = float(
                spearmanr(d[aff], (s_f - clean[col[0]].to_numpy())[aff]).statistic)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    args = parser.parse_args()
    rows = []
    for root in args.runs:
        for done in sorted(root.rglob("complete.json")):
            run = done.parent
            cfg = json.loads((run / "config.json").read_text())
            if cfg.get("model") not in ("gru", "tcn"):
                continue
            meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=cfg["seed"],
                        fold=cfg.get("fold", cfg.get("subset")))
            for sc in args.scenarios:
                if csv_path(run, sc).exists() and csv_path(run, "clean").exists():
                    rows.append(analyse(run, sc, float(cfg["risk_delta"]), meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(json.dumps({"rows": len(rows)}))


if __name__ == "__main__":
    main()
