"""Summarise EXT3 (EXT3_PROTOCOL.md). usage: ext3_summary.py all_ext3_v1.csv.gz

Aggregation as in the paper: mean over the six unseen faults; C-MAPSS subsets macro-averaged; bearing folds pooled with
weight = number of evaluation units. Pairs = (backbone, seed), 20 per dataset. Main family: Holm over 12 tests.
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
SC = ["ensemble_std", "laplace_ev", "full", "gauss_bias_aware", "rank_avg", "conformal", "conformal_w", "mondrian"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
M = ["risk_matched", "risk_frozen", "coverage_frozen", "bound_coverage", "bound_coverage_aff", "risk_certify",
     "coverage_certify", "weight_ess", "laplace_prior"]

d = pd.read_csv(src)
print("runs:", d.groupby("dataset").apply(lambda g: g[["model", "seed", "fold"]].drop_duplicates().shape[0]).to_dict())


def pool(sub):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[M].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    res = {}
    for m in M:
        ok = np.isfinite(g[m])
        gg = g[ok].assign(_v=g[m][ok] * g._w[ok])
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[m] = s._v / s._w
    return pd.DataFrame(res)


x = pool(d[d.scenario.isin(FA)])
cl = pool(d[d.scenario == "clean"])
summ = x.groupby(["dataset", "score"]).mean()
summ_cl = cl.groupby(["dataset", "score"]).mean()
pd.set_option("display.width", 250)
pct = ["risk_matched", "risk_frozen", "coverage_frozen", "bound_coverage", "bound_coverage_aff", "risk_certify",
       "coverage_certify"]
summ[pct] *= 100
summ_cl[pct] *= 100
for col in ["risk_matched", "risk_frozen", "coverage_frozen"]:
    print(f"\n== {col} (unseen faults, %)")
    print(summ[col].unstack("dataset").reindex(columns=DS).reindex(SC).round(2).to_string())
print("\n== conformal bounds: coverage (target 90) clean | faults all windows | faults affected windows; certify rule")
cb = pd.concat({"cov_clean": summ_cl.bound_coverage, "cov_fault": summ.bound_coverage,
                "cov_fault_aff": summ.bound_coverage_aff, "certify_risk_fault": summ.risk_certify,
                "certify_cov_fault": summ.coverage_certify, "certify_risk_clean": summ_cl.risk_certify,
                "certify_cov_clean": summ_cl.coverage_certify}, axis=1)
print(cb.dropna(how="all").round(2).to_string())
print("\n== weight ESS fraction (conformal_w) and Laplace prior (geometric mean)")
print(summ[["weight_ess", "laplace_prior"]].xs("full", level="score").round(3).to_string())

w = x.risk_matched.unstack("score")


def test(a, b, ds):
    if ds not in w.index.get_level_values(0):
        return dict(a=a, b=b, dataset=ds, n=0, mean_pts=np.nan, wins_a=0, p=1.0)
    dif = (w.loc[ds][a] - w.loc[ds][b]).dropna()
    return dict(a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0)


main = pd.DataFrame([test(a, "full", ds) for a in ["conformal_w", "mondrian", "rank_avg"] for ds in DS])
order = np.argsort(main.p.to_numpy())
adj, run = np.empty(len(main)), 0.0
for r, i in enumerate(order):
    run = max(run, (len(main) - r) * main.p.iloc[i])
    adj[i] = min(1.0, run)
main["p_holm"] = adj
print("\n== MAIN FAMILY: score - full (risk at 90% matched coverage, points), Holm over 12")
print(main.round(4).to_string(index=False))
sec = pd.DataFrame([test(a, b, ds) for a, b in [("laplace_ev", "ensemble_std"), ("conformal_w", "conformal"),
                                                ("mondrian", "conformal"), ("gauss_bias_aware", "full")] for ds in DS])
print("\n== secondary (unadjusted)")
print(sec.round(4).to_string(index=False))
summ.reset_index().to_csv(out / "summary_ext3_v1.csv", index=False)
cb.reset_index().to_csv(out / "bounds_ext3_v1.csv", index=False)
main.to_csv(out / "main_family_ext3_v1.csv", index=False)
sec.to_csv(out / "secondary_ext3_v1.csv", index=False)
