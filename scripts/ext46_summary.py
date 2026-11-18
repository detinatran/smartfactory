"""EXT46 analysis (EXT46_PROTOCOL.md). usage: ext46_summary.py ALL_EXT46.csv.gz OUT.txt"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

fe, out = sys.argv[1:3]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
DS = ["cmapss", "ncmapss_cv", "bearings", "battery_cv", "hdd"]
d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)


def pool(sub, col, scores):
    g = sub[sub.score.isin(scores)].groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["v"] = g[col] * g.w
    s = g.groupby(["dataset", "score", "seed"])[["v", "w"]].sum()
    return (s.v / s.w).unstack("score") * 100


SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_auto_d", "repair_lconf"]
W = pool(d[d.scenario.isin(FA)], "risk_matched", SC)
C = pool(d[d.scenario == "clean"], "risk_matched", SC)
P = pool(d[d.scenario.isin(FA)], "opt_full", ["predictor", "predictor_repaired"])
L = ["== risk (%), six faults, mean over seeds", W.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string(),
     "\n== clean risk (%)", C.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string(),
     "\n== predictor optimistic rate at full coverage under faults (%): original vs repaired",
     P.groupby("dataset").mean().reindex(DS).round(2).to_string()]
rows = []
for ds in DS:
    dif = (W.loc[ds]["rpt_auto_d"] - W.loc[ds]["repair_lconf"]).dropna()
    rows.append(dict(dataset=ds, mean=dif.mean(), better=int((dif < 0).sum()), n=len(dif), p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
t = pd.DataFrame(rows)
p = t.p.to_numpy(); adj, run = np.empty(len(p)), 0.0
for r, i in enumerate(np.argsort(p)):
    run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
t["p_holm"] = adj
L += ["\n== rpt_auto_d - repair_lconf (points; seed level; Holm over 5)", t.round(4).to_string(index=False)]
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
