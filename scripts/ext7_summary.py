"""Summarise EXT7 (EXT7_PROTOCOL.md). usage: ext7_summary.py all_ext7_v1.csv.gz

Primary family: NEW seeds 52-56, stack - full, risk at 90% matched coverage (mean over six unseen faults), 10 pairs per
dataset, two-sided Wilcoxon, Holm over 4; criterion also requires the pessimistic rate to rise by <= 1 point.
Secondary (unadjusted): rank_recon - full, gba_recon - gauss_bias_aware, full_recon - full on new seeds; all on old seeds.
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
SC = ["ensemble_std", "impact", "gauss_bias_aware", "gba_recon", "full", "full_recon", "rank_recon", "stack"]
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
d = pd.read_csv(src)
d = d[d.score.isin(SC)]
d["era"] = np.where(d.seed >= 52, "new", "old")
pd.set_option("display.width", 250)


def pool(sub, cols):
    g = sub.groupby(["era", "dataset", "score", "model", "seed", "fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(["era", "dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


print(d.groupby(["era", "dataset"]).apply(lambda g: g.groupby(["model", "seed", "fold"]).ngroups).unstack().to_string(), "\n")
x = pool(d[d.scenario.isin(FA)], M)
xc = pool(d[d.scenario == "clean"], ["risk_matched"])
for era in ["new", "old"]:
    s = x.loc[era].groupby(["dataset", "score"]).mean() * 100
    for col in M:
        print(f"== [{era} seeds] {col} (unseen faults, %)")
        print(s[col].unstack("dataset").reindex(columns=DS).reindex(SC).round(2).to_string(), "\n")
    sc = xc.loc[era].groupby(["dataset", "score"]).mean() * 100
    print(f"== [{era} seeds] clean risk_matched (%)")
    print(sc.risk_matched.unstack("dataset").reindex(columns=DS).reindex(SC).round(2).to_string(), "\n")
    s.reset_index().to_csv(out / f"summary_{era}_ext7_v1.csv", index=False)
coef = d[d.era == "new"].groupby("dataset")[["coef_full_recon", "coef_gba_recon", "coef_impact"]].mean()
print("== stacker coefficients (standardised inputs), new seeds\n", coef.reindex(DS).round(2).to_string(), "\n")


def test(era, a, b, ds, col="risk_matched"):
    w = x.loc[era][col].unstack("score")
    dif = (w.loc[ds][a] - w.loc[ds][b]).dropna()
    wp = x.loc[era]["pess_matched"].unstack("score")
    return dict(era=era, a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                pess_diff_pts=100 * (wp.loc[ds][a] - wp.loc[ds][b]).mean())


main = pd.DataFrame([test("new", "stack", "full", ds) for ds in DS])
order = np.argsort(main.p.to_numpy())
adj, run = np.empty(len(main)), 0.0
for r, i in enumerate(order):
    run = max(run, (len(main) - r) * main.p.iloc[i])
    adj[i] = min(1.0, run)
main["p_holm"] = adj
print("== PRIMARY FAMILY (new seeds 52-56): stack - full, risk at 90% matched coverage (points), Holm over 4")
print(main.round(4).to_string(index=False))
worse = ((main.mean_pts > 0) & (main.p_holm < 0.05)).any()
better = ((main.mean_pts < 0) & (main.p_holm < 0.05)).any()
ok = bool(better and not worse and (main.pess_diff_pts <= 1.0).all())
print(f"\ncriterion 'better than ESANN': {ok}\n")
pairs = [("rank_recon", "full"), ("gba_recon", "gauss_bias_aware"), ("full_recon", "full")]
sec = pd.DataFrame([test("new", a, b, ds) for a, b in pairs for ds in DS] +
                   [test("old", a, b, ds) for a, b in [("stack", "full")] + pairs for ds in DS])
print("== secondary (unadjusted)")
print(sec.round(4).to_string(index=False))
main.to_csv(out / "primary_ext7_v1.csv", index=False)
sec.to_csv(out / "secondary_ext7_v1.csv", index=False)
