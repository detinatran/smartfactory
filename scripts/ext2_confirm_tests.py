"""Confirmatory tests of EXT2_CONFIRM_PROTOCOL.md (section A).

usage: ext2_confirm_tests.py CONFIRM_CSV PILOT_CSV
C1/C2: FiLM + score vs GRU + score, risk at 90% matched coverage, mean over the six unseen faults, two-sided Wilcoxon,
Holm over the 4 tests. Pairs: C-MAPSS (seed, subset) = 20; N-CMAPSS seed = 5 (direction only; pooled with the pilot
seeds as a secondary, partly-seen analysis). H1 replication is descriptive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
conf, pilot = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
out = Path(sys.argv[1]).parent


def per_pair(d):
    return d[d.scenario.isin(FA)].groupby(["dataset", "backbone", "score", "seed", "fold"]).mean(numeric_only=True)


def holm(p):
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


rows = []
x = per_pair(conf)
for test, score in [("C1", "gauss_bias_aware"), ("C2", "full")]:
    for ds in ["cmapss", "ncmapss"]:
        f = x.loc[(ds, "film_gru", score)].risk_matched
        g = x.loc[(ds, "gru", score)].risk_matched
        dif = (f - g).dropna()
        rows.append(dict(test=test, score=score, dataset=ds, n=len(dif), film=100 * f.mean(), gru=100 * g.mean(),
                         mean_diff_pts=100 * dif.mean(), film_better=int((dif < 0).sum()), p=wilcoxon(dif).pvalue))
res = pd.DataFrame(rows)
res["p_holm"] = holm(res.p.to_numpy())
print("== confirmatory family (seeds 47-51)")
print(res.round(4).to_string(index=False))

# secondary: N-CMAPSS pilot + confirm, 10 seeds (pilot half already seen)
both = pd.concat([per_pair(pilot), x])
sec = []
for score in ["gauss_bias_aware", "full", "conformal"]:
    for ds in ["cmapss", "ncmapss"]:
        dif = (both.loc[(ds, "film_gru", score)].risk_matched - both.loc[(ds, "gru", score)].risk_matched)
        if ds == "cmapss":
            dif = dif.groupby(level="seed").mean()
        dif = dif.dropna()
        sec.append(dict(score=score, dataset=ds, n_seeds=len(dif), mean_diff_pts=100 * dif.mean(),
                        film_better=int((dif < 0).sum()), p=wilcoxon(dif).pvalue))
sec = pd.DataFrame(sec)
print("\n== secondary: pilot + confirmation seeds (42-51), per-seed pairs, unadjusted, half seen")
print(sec.round(4).to_string(index=False))

# H1 replication (descriptive): spread of FiLM vs GRU
h = x.xs("ensemble_std", level="score").groupby(["dataset", "backbone"])[["shift_over_spread", "reject_new", "rmse"]].mean()
print("\n== H1 replication (confirmation seeds)")
print(h.round(4).to_string())
res.to_csv(out / "confirm_tests.csv", index=False)
sec.to_csv(out / "confirm_secondary.csv", index=False)
h.to_csv(out / "confirm_h1.csv")
