"""EXT8 holdout test (EXT8_PROTOCOL.md, addendum 2026-11-12). usage: ext8_holdout.py all_ext8_hold.csv.gz

Seeds 57-61, 10 (backbone, seed) pairs per dataset. Primary: rank_recon - full, risk at 90% matched coverage (mean over
six unseen faults), two-sided Wilcoxon, Holm over 4. Criterion: >= 1 significant improvement, no significant loss,
pessimistic rise <= 1 point and mean diff <= +0.3 points on every dataset. Secondary: 8 candidates x 4 datasets, Holm over 32.
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
CAND = ["rank_recon", "stack", "stack_plus", "blend", "stack_cv", "select_cv", "full_recon", "gba_recon"]
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
d = pd.read_csv(src)
d = d[~d.score.str.startswith("predictor")]
assert d.seed.between(57, 61).all(), "holdout file must contain only seeds 57-61"
pd.set_option("display.width", 250)


def pool(sub, cols):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


print("runs per dataset:", d.groupby("dataset").apply(lambda g: g.groupby(["model", "seed", "fold"]).ngroups).to_dict())
x = pool(d[d.scenario.isin(FA)], M)
xc = pool(d[d.scenario == "clean"], ["risk_matched"])
s = x.groupby(["dataset", "score"]).mean() * 100
order = ["ensemble_std", "impact", "gauss_bias_aware", "full"] + CAND
for col in M:
    print(f"\n== {col} (unseen faults, %, holdout seeds 57-61)")
    print(s[col].unstack("dataset").reindex(columns=DS).reindex(order).round(2).to_string())
print("\n== clean risk_matched (%)")
print((xc.groupby(["dataset", "score"]).mean() * 100).risk_matched.unstack("dataset").reindex(columns=DS)
      .reindex(order).round(2).to_string())
s.reset_index().to_csv(out / "summary_ext8_hold.csv", index=False)

w = x.risk_matched.unstack("score")
wp = x.pess_matched.unstack("score")
rows = []
for c in CAND:
    for ds in DS:
        dif = (w.loc[ds][c] - w.loc[ds]["full"]).dropna()
        rows.append(dict(cand=c, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins=int((dif < 0).sum()),
                         p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                         pess_pts=100 * (wp.loc[ds][c] - wp.loc[ds]["full"]).mean()))
t = pd.DataFrame(rows)
t["p_holm32"] = holm(t.p)


def verdict(sub, pcol):
    better = ((sub.mean_pts < 0) & (sub[pcol] < 0.05)).any()
    worse = ((sub.mean_pts > 0) & (sub[pcol] < 0.05)).any()
    return bool(better and not worse and (sub.pess_pts <= 1.0).all() and (sub.mean_pts <= 0.3).all())


prim = t[t.cand == "rank_recon"].copy()
prim["p_holm4"] = holm(prim.p)
print("\n== PRIMARY (holdout): rank_recon - full, risk @90% matched (points), Holm over 4")
print(prim.drop(columns="p_holm32").round(4).to_string(index=False))
print(f"\nCONFIRMED (rank_recon better than ESANN): {verdict(prim, 'p_holm4')}")
print("\n== SECONDARY: all candidates - full, Holm over 32")
print(t.round(4).to_string(index=False))
print("\ncandidates meeting the criterion at Holm-32:",
      [c for c in CAND if verdict(t[t.cand == c], "p_holm32")] or "none")
prim.to_csv(out / "primary_ext8_hold.csv", index=False)
t.to_csv(out / "secondary_ext8_hold.csv", index=False)
