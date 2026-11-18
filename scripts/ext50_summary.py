"""EXT50 analysis (EXT50_PROTOCOL.md). usage: ext50_summary.py ALL_EXT50.csv.gz ALL_EXT40.csv.gz OUT.txt"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

fj, fd, out = sys.argv[1:4]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
DS = ["cmapss", "bearings", "battery_cv"]
L = []


def load(f):
    d = pd.read_csv(f)
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    return d[d.dataset.isin(DS) & d.seed.between(162, 171)]


def pool(d, scen, col, scores):
    g = d[d.scenario.isin(scen) & d.score.isin(scores)].groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0); g["v"] = g[col] * g.w
    s = g.groupby(["dataset", "score", "model", "seed"])[["v", "w"]].sum()
    return ((s.v / s.w).unstack("score")).groupby(["dataset", "seed"]).mean()


J, D0 = load(fj), load(fd)
DS = [x for x in DS if x in set(J.dataset)]
rmJ, rmD = pool(J, ["clean"], "rmse_full", ["predictor"]), pool(D0, ["clean"], "rmse_full", ["predictor"])
opJ, opD = pool(J, FA, "opt_full", ["predictor"]) * 100, pool(D0, FA, "opt_full", ["predictor"]) * 100
L.append("== predictor: clean RMSE and optimistic rate at full coverage under faults (%), mean over seeds")
for ds in DS:
    a, b = rmJ.loc[ds].predictor, rmD.loc[ds].predictor.reindex(rmJ.loc[ds].index)
    c, e = opJ.loc[ds].predictor, opD.loc[ds].predictor.reindex(opJ.loc[ds].index)
    L.append(f"  {ds:10s} RMSE default {b.mean():.3f} JEPA {a.mean():.3f} (JEPA lower {int((a < b).sum())}/{len(a)}) | opt under faults default {e.mean():.2f} JEPA {c.mean():.2f}")
SC = ["full", "rpt_L", "rpt_auto_d", "gauss_bias_aware", "err_reg"]
WJ = pool(J, FA, "risk_matched", SC) * 100
WD = pool(D0, FA, "risk_matched", ["full", "rpt_auto_d"]) * 100
L.append("\n== risk (%), six faults, 90% matched coverage")
tab = WJ.groupby("dataset").mean().reindex(DS).round(2)
tab["default_full"] = WD.groupby("dataset").mean().full.reindex(DS).round(2)
tab["default_rpt_auto_d"] = WD.groupby("dataset").mean().rpt_auto_d.reindex(DS).round(2)
L.append(tab.to_string())
for name, A, a, B, b in [("JEPA system (rpt_auto_d) - default system (rpt_auto_d)", WJ, "rpt_auto_d", WD, "rpt_auto_d"),
                         ("within JEPA: rpt_auto_d - full", WJ, "rpt_auto_d", WJ, "full"),
                         ("JEPA full - default full", WJ, "full", WD, "full")]:
    L.append(f"\n-- {name}")
    for ds in DS:
        x = A.loc[ds][a]; y = B.loc[ds][b].reindex(x.index)
        dif = (x - y).dropna()
        L.append(f"  {ds:10s} {dif.mean():+.3f} points, better {int((dif < 0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1:.2g}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
