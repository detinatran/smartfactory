"""Summarise the EXT2 pilot (EXT2_PILOT_PROTOCOL.md). Input: all_ext2_v1.csv.gz (merged per-run CSVs of ext2_pilot.py).

Aggregation: mean over the six unseen faults; C-MAPSS subsets averaged within a seed (macro). Pairs = seeds (5).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

src = Path(sys.argv[1])
out = src.parent
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
SC = ["ensemble_std", "mc_dropout", "mc_ensemble", "laplace", "gauss_unbiased", "gauss_bias_aware", "conformal", "full"]
BB = ["gru", "film_gru", "lipschitz"]
d = pd.read_csv(src)
print("runs per dataset/backbone:", d.groupby(["dataset", "backbone"]).apply(lambda g: g[["seed", "fold"]].drop_duplicates().shape[0]).to_dict())
M = ["risk_matched", "risk_frozen", "coverage_frozen", "reject_new", "rho_shift", "abs_shift", "shift_over_spread",
     "rmse", "conf_coverage"]

fa = d[d.scenario.isin(FA)]
# per (dataset, backbone, score, seed): mean over faults then over subsets
# bearing folds pooled with weight = number of evaluation units (as in the paper); others equal weights
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}


def fold_pool(g, cols):
    g = g.reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    return g.groupby(["dataset", "backbone", "score", "seed"]).apply(
        lambda h: pd.Series({c: np.average(h[c][np.isfinite(h[c])], weights=h._w[np.isfinite(h[c])])
                             if np.isfinite(h[c]).any() else np.nan for c in cols}))


x = fold_pool(fa.groupby(["dataset", "backbone", "score", "seed", "fold"])[M].mean(), M)
CL = ["rmse", "conf_coverage", "risk_matched", "coverage_frozen"]
cl = fold_pool(d[d.scenario == "clean"].groupby(["dataset", "backbone", "score", "seed", "fold"])[CL].mean(), CL)
summ = x.groupby(["dataset", "backbone", "score"]).mean()
summ["rmse_clean"] = cl.rmse.groupby(["dataset", "backbone", "score"]).mean()
summ["conf_coverage_clean"] = cl.conf_coverage.groupby(["dataset", "backbone", "score"]).mean()
summ["coverage_clean"] = cl.coverage_frozen.groupby(["dataset", "backbone", "score"]).mean()
for c in ["risk_matched", "risk_frozen", "coverage_frozen", "reject_new", "conf_coverage", "conf_coverage_clean", "coverage_clean"]:
    summ[c] *= 100
summ.reset_index().to_csv(out / "summary_ext2_v1.csv", index=False)
pd.set_option("display.width", 250)
for col in ["risk_matched", "reject_new", "rho_shift", "risk_frozen", "coverage_frozen"]:
    print(f"\n== {col} (unseen faults)")
    t = summ[col].unstack(["dataset", "backbone"])
    print(t.reindex(SC).round(2).to_string())
print("\n== backbone properties (unseen faults; rmse_clean on clean eval; abs_shift / cap)")
bp = summ.xs("ensemble_std", level="score")[["rmse_clean", "abs_shift", "shift_over_spread", "conf_coverage_clean", "conf_coverage"]]
print(bp.round(4).to_string())

rows = []
w = x.risk_matched.unstack("score")
for ds in d.dataset.unique():
    for bb in BB:
        for a, b in [("gauss_bias_aware", "ensemble_std"), ("conformal", "ensemble_std"), ("full", "ensemble_std"),
                     ("laplace", "ensemble_std"), ("full", "gauss_bias_aware"), ("conformal", "gauss_bias_aware")]:
            try:
                dif = (w.loc[(ds, bb)][a] - w.loc[(ds, bb)][b]).dropna()
            except KeyError:
                continue
            rows.append(dict(kind="score", dataset=ds, backbone=bb, a=a, b=b, n=len(dif), mean_pts=100 * dif.mean(),
                             wins_a=int((dif < 0).sum()), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    xb = x.xs("ensemble_std", level="score")
    for metric in ["abs_shift", "shift_over_spread", "reject_new", "rmse"]:
        for bb in ["film_gru", "lipschitz"]:
            try:
                dif = (xb.loc[(ds, bb)][metric] - xb.loc[(ds, "gru")][metric]).dropna()
            except KeyError:
                continue
            rows.append(dict(kind="backbone", dataset=ds, backbone=bb, a=metric, b="gru", n=len(dif), mean_pts=dif.mean(),
                             wins_a=int((dif < 0).sum()), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
pt = pd.DataFrame(rows)
pt.to_csv(out / "paired_ext2_v1.csv", index=False)
print("\n== paired (score: risk_matched points, a-b; backbone: metric difference vs gru)")
print(pt.round(4).to_string(index=False))
