"""Confirmatory test for search rounds (EXT9_PROTOCOL.md section 4).
usage: conf_test.py all.csv.gz METHOD ALPHA SEED_LO SEED_HI

METHOD - full, risk at 90% matched coverage (mean over six unseen faults), per dataset over (backbone, seed) pairs
(C-MAPSS subsets averaged; bearings folds unit-weighted). Two-sided Wilcoxon, Holm over 4 at level ALPHA.
Criterion: >= 1 significant improvement, no significant loss, mean diff <= +0.3 points and pessimistic rise <= 1 point on
every dataset. Other scores are printed descriptively only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

src, method, alpha, lo, hi = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out = src.parent
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
d = pd.read_csv(src)
d = d[~d.score.str.startswith("predictor")]
assert d.seed.between(lo, hi).all(), f"file must contain only seeds {lo}-{hi}"
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


print("runs per dataset:", d.groupby("dataset").apply(lambda g: g.groupby(["model", "seed", "fold"]).ngroups).to_dict())
x = pool(d[d.scenario.isin(FA)], M)
xc = pool(d[d.scenario == "clean"], ["risk_matched"])
s = x.groupby(["dataset", "score"]).mean() * 100
for col in M:
    print(f"\n== {col} (unseen faults, %, seeds {lo}-{hi})")
    print(s[col].unstack("dataset").reindex(columns=DS).round(2).to_string())
print("\n== clean risk_matched (%)")
print((xc.groupby(["dataset", "score"]).mean() * 100).risk_matched.unstack("dataset").reindex(columns=DS).round(2).to_string())
s.reset_index().to_csv(out / "summary_conf.csv", index=False)

w = x.risk_matched.unstack("score")
wp = x.pess_matched.unstack("score")
rows = []
for ds in DS:
    dif = (w.loc[ds][method] - w.loc[ds]["full"]).dropna()
    rows.append(dict(dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins=int((dif < 0).sum()),
                     p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                     pess_pts=100 * (wp.loc[ds][method] - wp.loc[ds]["full"]).mean()))
t = pd.DataFrame(rows)
adj, run = np.empty(len(t)), 0.0
for r, i in enumerate(np.argsort(t.p.to_numpy())):
    run = max(run, (len(t) - r) * t.p.iloc[i])
    adj[i] = min(1.0, run)
t["p_holm"] = adj
better = ((t.mean_pts < 0) & (t.p_holm < alpha)).any()
worse = ((t.mean_pts > 0) & (t.p_holm < alpha)).any()
ok = bool(better and not worse and (t.mean_pts <= 0.3).all() and (t.pess_pts <= 1.0).all())
print(f"\n== CONFIRMATORY: {method} - full, seeds {lo}-{hi}, Holm over 4 at alpha = {alpha}")
print(t.round(5).to_string(index=False))
print(f"\nCONFIRMED: {ok}  (better={better}, worse={worse}, max_mean={t.mean_pts.max():.3f}, max_pess={t.pess_pts.max():.3f})")
t.assign(method=method, alpha=alpha, confirmed=ok).to_csv(out / f"confirm_{method}.csv", index=False)
