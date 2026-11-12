"""Summarise EXT6 (EXT6_PROTOCOL.md). usage: ext6_summary.py all_ext6_v1.csv.gz

Aggregation as in the paper; pairs = (backbone, seed), 20 per dataset. Main family (Holm over 8):
A full_recon - full, B repair_full (repaired predictor) - full (original predictor). B additionally requires that the
pessimistic-error rate at 90% matched coverage rises by <= 1 point on every dataset.
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
SC = ["ensemble_std", "impact", "gauss_bias_aware", "full", "full_recon", "repair_full"]
d = pd.read_csv(src)
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


M = ["risk_matched", "pess_matched", "rmse_matched", "risk_frozen", "coverage_frozen"]
sel = d[d.score.isin(SC)]
x = pool(sel[sel.scenario.isin(FA)], M)
xc = pool(sel[sel.scenario == "clean"], M)
summ, summ_c = x.groupby(["dataset", "score"]).mean(), xc.groupby(["dataset", "score"]).mean()
for col in M:
    k = 1 if col == "rmse_matched" else 100
    print(f"== {col} (unseen faults{'' if k == 1 else ', %'})")
    print((summ[col] * k).unstack("dataset").reindex(columns=DS).reindex(SC).round(3 if k == 1 else 2).to_string(), "\n")
print("== clean risk_matched (%)")
print((summ_c.risk_matched * 100).unstack("dataset").reindex(columns=DS).reindex(SC).round(2).to_string(), "\n")

P = ["opt_full", "pess_full", "rmse_full"]
pr = d[d.score.str.startswith("predictor_")]
for label, scen in (("unseen faults", FA), ("clean", ["clean"])):
    pp = pool(pr[pr.scenario.isin(scen)], P).groupby(["dataset", "score"]).mean()
    pp[["opt_full", "pess_full"]] *= 100
    print(f"== predictor at 100% coverage, {label}: optimistic %, pessimistic %, RMSE (capped)")
    print(pp.unstack("score").reindex(DS).round(3).to_string(), "\n")
det = pr[pr.score == "predictor_orig"].groupby(["dataset", "scenario"])[["false_alarm", "sensitivity"]].mean()
print("== detection: false-alarm rate on unchanged channels, sensitivity on faulty channels")
print(det.unstack("dataset").round(3).to_string(), "\n")

w = x.risk_matched.unstack("score")
wp = x.pess_matched.unstack("score")


def test(a, b, ds, tab=w):
    dif = (tab.loc[ds][a] - tab.loc[ds][b]).dropna()
    return dict(a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                ties=int((dif == 0).sum()), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0)


main = pd.DataFrame([test(a, "full", ds) for a in ["full_recon", "repair_full"] for ds in DS])
order = np.argsort(main.p.to_numpy())
adj, run = np.empty(len(main)), 0.0
for r, i in enumerate(order):
    run = max(run, (len(main) - r) * main.p.iloc[i])
    adj[i] = min(1.0, run)
main["p_holm"] = adj
main["pess_diff_pts"] = [100 * (wp.loc[ds][a] - wp.loc[ds]["full"]).mean() for a, ds in zip(main.a, main.dataset)]
print("== MAIN FAMILY (risk at 90% matched coverage, points), Holm over 8")
print(main.round(4).to_string(index=False))
for a in ["full_recon", "repair_full"]:
    m = main[main.a == a]
    worse = ((m.mean_pts > 0) & (m.p_holm < 0.05)).any()
    better = ((m.mean_pts < 0) & (m.p_holm < 0.05)).any()
    ok = bool(better and not worse)
    if a == "repair_full":
        ok = ok and bool((m.pess_diff_pts <= 1.0).all())
    print(f"criterion 'better than ESANN' for {a}: {ok}")
sec = pd.DataFrame([test("impact", "ensemble_std", ds) for ds in DS] +
                   [test("repair_full", "full", ds, wp) | {"a": "repair_full(pess)"} for ds in DS])
print("\n== secondary (unadjusted)")
print(sec.round(4).to_string(index=False))
summ.reset_index().to_csv(out / "summary_ext6_v1.csv", index=False)
summ_c.reset_index().to_csv(out / "summary_clean_ext6_v1.csv", index=False)
det.reset_index().to_csv(out / "detection_ext6_v1.csv", index=False)
main.to_csv(out / "main_family_ext6_v1.csv", index=False)
sec.to_csv(out / "secondary_ext6_v1.csv", index=False)
