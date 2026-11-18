"""EXT36 analysis (EXT36_PROTOCOL.md). usage: ext36_summary.py ALL_EXT36.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
d["w"] = np.where(d.dataset == "bearings", d.fold.map(BW), 1.0)
cols = ["brier_P_full", "brier_P_rpt", "brier_Q_full", "brier_Q_rpt", "auc_osc", "auc_rpt", "chi2_osc", "chi2_rpt"]
g = d.assign(**{c: d[c] * d.w for c in cols}).groupby(["dataset", "model", "seed"])[cols + ["w"]].sum()
for c in cols:
    g[c] = g[c] / g.w
say(f"runs: {d.groupby('dataset').size().to_dict()}")
say("\n== means per family")
m = g.groupby("dataset")[cols].mean().reindex(DS)
m[["brier_P_full", "brier_P_rpt", "brier_Q_full", "brier_Q_rpt"]] *= 100
say(m.round(3).to_string())


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


for name, a, b in [("H-est (Brier on P, x100)", "brier_P_rpt", "brier_P_full"), ("H-Q (Brier on Q, x100)", "brier_Q_rpt", "brier_Q_full"),
                   ("H-cov chi2", "chi2_rpt", "chi2_osc"), ("H-cov AUC", "auc_rpt", "auc_osc")]:
    rows = []
    for ds in DS:
        dif = (g.loc[ds][a] - g.loc[ds][b]).dropna()
        k = 100 if "brier" in a else 1
        rows.append(dict(dataset=ds, mean=k * dif.mean(), rel=dif.mean() / g.loc[ds][b].mean(), lower=int((dif < 0).sum()), n=len(dif),
                         p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== {name}: rpt - osc (Holm over 4)")
    say(t.round(5).to_string(index=False))
(out / "ext36_output.txt").write_text("\n".join(lines) + "\n")
print("EXT36_SUMMARY_DONE")
