"""Summarise EXT5 (EXT5_PROTOCOL.md). usage: ext5_summary.py all_ext5_v1.csv.gz

Aggregation as in the paper: mean over the six unseen faults; C-MAPSS subsets macro-averaged; bearings pooled with
unit-count fold weights; pairs = (backbone, seed), 20 per dataset. Main family: full_loco - full, Holm over 4.
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
SC = ["ensemble_std", "loco_up", "gauss_bias_aware", "gba_loco", "full", "full_loco"]
M = ["risk_matched", "risk_frozen", "coverage_frozen", "reject_new_frozen"]
d = pd.read_csv(src)
pd.set_option("display.width", 250)


def pool(sub, cols):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        ok = g[col].notna()
        gg = g[ok].assign(_v=g[col] * g._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


x = pool(d[d.scenario.isin(FA)], M)
xc = pool(d[d.scenario == "clean"], ["risk_matched", "risk_frozen", "coverage_frozen"])
summ = x.groupby(["dataset", "score"]).mean() * 100
summ_c = xc.groupby(["dataset", "score"]).mean() * 100
for col in M:
    print(f"== {col} (unseen faults, %)")
    print(summ[col].unstack("dataset").reindex(columns=DS).reindex(SC).round(2).to_string(), "\n")
print("== clean: risk_matched / risk_frozen / coverage_frozen (%)")
print(summ_c.unstack("dataset").reindex(SC).round(2).to_string(), "\n")

loc = d[(d.score == "full") & d.scenario.isin(FA)]
lt = loc.groupby(["dataset", "scenario"]).loco_localisation.mean().unstack("dataset").reindex(columns=DS)
chance = loc.groupby("dataset").apply(lambda g: (g.channels_hit / g.n_channels).mean()).reindex(DS)
print("== LOCO localisation on affected windows (share whose top-|d_c| channel is faulty); chance ~ hit/n_channels")
print(lt.round(3).to_string())
print("chance:", chance.round(3).to_dict(), "\n")

w = x.risk_matched.unstack("score")


def test(a, b, ds):
    dif = (w.loc[ds][a] - w.loc[ds][b]).dropna()
    return dict(a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                ties=int((dif == 0).sum()), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0)


main = pd.DataFrame([test("full_loco", "full", ds) for ds in DS])
order = np.argsort(main.p.to_numpy())
adj, run = np.empty(len(main)), 0.0
for r, i in enumerate(order):
    run = max(run, (len(main) - r) * main.p.iloc[i])
    adj[i] = min(1.0, run)
main["p_holm"] = adj
print("== MAIN FAMILY: full_loco - full (risk at 90% matched coverage, points), Holm over 4")
print(main.round(4).to_string(index=False))
worse = ((main.mean_pts > 0) & (main.p_holm < 0.05)).any()
better = ((main.mean_pts < 0) & (main.p_holm < 0.05)).any()
print(f"\ncriterion 'better than ESANN' (no significant loss, >=1 significant win): {bool(better and not worse)}")
sec = pd.DataFrame([test(a, b, ds) for a, b in [("gba_loco", "gauss_bias_aware"), ("loco_up", "ensemble_std"),
                                                 ("gba_loco", "full")] for ds in DS])
print("\n== secondary (unadjusted)")
print(sec.round(4).to_string(index=False))
summ.reset_index().to_csv(out / "summary_ext5_v1.csv", index=False)
summ_c.reset_index().to_csv(out / "summary_clean_ext5_v1.csv", index=False)
lt.to_csv(out / "localisation_ext5_v1.csv")
main.to_csv(out / "main_family_ext5_v1.csv", index=False)
sec.to_csv(out / "secondary_ext5_v1.csv", index=False)
