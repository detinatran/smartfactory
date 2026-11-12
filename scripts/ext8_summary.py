"""Summarise the EXT8 development stage (EXT8_PROTOCOL.md). usage: ext8_summary.py all_ext8_v1.csv.gz

Seeds 42-56 (30 (backbone, seed) pairs per dataset), exploratory. Each candidate minus full: risk at 90% matched coverage,
mean over six unseen faults; also by era (old 42-51, new 52-56). Applies the default selection rule: safe = mean diff
<= +0.10 points on every dataset and pessimistic rise <= 1 point; pick the safe candidate with the most negative total.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

src = Path(sys.argv[1])
out = src.parent
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
REF = ["ensemble_std", "impact", "gauss_bias_aware", "gba_recon", "full", "full_recon", "rank_recon", "stack"]
CAND = ["stack", "stack_plus", "blend", "stack_cv", "select_cv", "full_recon", "rank_recon", "gba_recon"]
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
d = pd.read_csv(src)
d = d[~d.score.str.startswith("predictor")]
d["era"] = np.where(d.seed >= 52, "new", "old")
pd.set_option("display.width", 250)


def pool(sub, cols, keys=("dataset", "score", "model", "seed")):
    g = sub.groupby(list(keys) + ["fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(list(keys))[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


print("runs per dataset:", d.groupby("dataset").apply(lambda g: g.groupby(["model", "seed", "fold"]).ngroups).to_dict())
x = pool(d[d.scenario.isin(FA)], M)
xc = pool(d[d.scenario == "clean"], ["risk_matched"])
order = REF + [c for c in CAND if c not in REF]
s = x.groupby(["dataset", "score"]).mean() * 100
for col in M:
    print(f"\n== {col} (unseen faults, %, seeds 42-56)")
    print(s[col].unstack("dataset").reindex(columns=DS).reindex(order).round(2).to_string())
print("\n== clean risk_matched (%)")
print((xc.groupby(["dataset", "score"]).mean() * 100).risk_matched.unstack("dataset").reindex(columns=DS)
      .reindex(order).round(2).to_string())
s.reset_index().to_csv(out / "summary_ext8_v1.csv", index=False)

runs = d.groupby(["dataset", "model", "seed", "fold"]).first().reset_index()
print("\n== fallbacks / choices (share of runs)")
print(runs.groupby("dataset")[["stack_fallback", "cv_use_stack"]].mean().reindex(DS).round(2).to_string())
print(pd.crosstab(runs.dataset, runs.select_choice, normalize="index").reindex(DS).round(2).to_string())

w = x.risk_matched.unstack("score")
wp = x.pess_matched.unstack("score")
seed_of = w.index.get_level_values("seed")
rows = []
for c in CAND:
    for ds in DS:
        for era, msk in [("all", np.ones(len(w), bool)), ("old", seed_of < 52), ("new", seed_of >= 52)]:
            if ds not in w[msk].index.get_level_values("dataset"):
                continue
            sub = w[msk].loc[ds]
            dif = (sub[c] - sub["full"]).dropna()
            rows.append(dict(cand=c, dataset=ds, era=era, n=len(dif), mean_pts=100 * dif.mean(),
                             wins=int((dif < 0).sum()), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                             pess_pts=100 * (wp[msk].loc[ds][c] - wp[msk].loc[ds]["full"]).mean()))
t = pd.DataFrame(rows)
t.to_csv(out / "cand_vs_full_ext8_v1.csv", index=False)
print("\n== candidate - full, risk @90% matched (points); era=all is 30 pairs (unadjusted, exploratory)")
print(t[t.era == "all"].pivot(index="cand", columns="dataset", values="mean_pts").reindex(columns=DS).round(3)
      .assign(total=lambda q: q.sum(1)).to_string())
print("\n-- wins out of 30")
print(t[t.era == "all"].pivot(index="cand", columns="dataset", values="wins").reindex(columns=DS).to_string())
print("\n-- p (two-sided Wilcoxon, unadjusted)")
print(t[t.era == "all"].pivot(index="cand", columns="dataset", values="p").reindex(columns=DS).round(4).to_string())
print("\n-- by era (old 20 pairs / new 10 pairs), mean points")
print(t[t.era != "all"].pivot(index=["cand", "era"], columns="dataset", values="mean_pts").reindex(columns=DS)
      .round(3).to_string())

a = t[t.era == "all"].groupby("cand").agg(worst=("mean_pts", "max"), total=("mean_pts", "sum"),
                                          pess=("pess_pts", "max"))
a["safe"] = (a.worst <= 0.10) & (a.pess <= 1.0)
print("\n== default selection rule")
print(a.sort_values("total").round(3).to_string())
safe = a[a.safe & (a.total < 0)]
print("\nselected:", safe.total.idxmin() if len(safe) else "full (no safe candidate improves the total)")
