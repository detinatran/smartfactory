"""Per-unit accepted / optimistic masses at matched 90% coverage (REVIEW_V4, for hierarchical bootstrap).

For each mirrored run (fault_v1, battery_fault_v1) and scenario, and each score column, writes per unit:
a = fraction of the unit's windows accepted, b = fraction accepted AND optimistic (pred - y > delta).
Unit-balanced risk of a set of units = sum(b) / sum(a).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rul.selection import apply_threshold, fit_threshold

METHODS = ("asymmetric_risk", "ensemble_std", "mc_dropout", "deep_ens_nll", "selectivenet", "symmetric_risk",
           "cqr_lower", "random", "asym_aug")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            run = done.parent
            cfg = json.loads((run / "config.json").read_text())
            if cfg.get("model") not in ("gru", "tcn"):
                continue
            delta = float(cfg["risk_delta"])
            for sc in cfg["eval_scenarios"]:
                path = run / f"predictions_{sc}.csv"
                path = path if path.exists() else run / "development" / f"predictions_{sc}.csv"
                p = pd.read_csv(path)
                units, ties = p.unit.to_numpy(), p.tie_breaker.to_numpy()
                bad = (p.prediction - p.rul_raw).to_numpy() > delta
                for m in METHODS:
                    col = [c for c in p.columns if c == f"score_{m}" or c.startswith(f"score_{m}__")]
                    if not col:
                        continue
                    s = p[col[0]].to_numpy()
                    acc = apply_threshold(s, ties, fit_threshold(s, ties, units, 0.9))
                    g = pd.DataFrame({"unit": units, "a": acc.astype(float), "b": (acc & bad).astype(float)})
                    g = g.groupby("unit")[["a", "b"]].mean().reset_index()
                    g.insert(0, "method", m)
                    g.insert(0, "scenario", sc)
                    for k, v in reversed(list(dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"],
                                                   seed=cfg["seed"], fold=cfg.get("fold", cfg.get("subset"))).items())):
                        g.insert(0, k, v)
                    rows.append(g)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(a.out, index=False)
    print(sum(len(r) for r in rows))


if __name__ == "__main__":
    main()
