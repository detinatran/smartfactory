"""EXT39 analysis (EXT39_PROTOCOL.md, exploratory). usage: ext39_summary.py ALL_EXT39.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery", "hdd4"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
SC = ["gauss_bias_aware", "err_reg", "err_reg_rptL", "full", "sens", "rpt_L", "rpt_sens", "rpt_auto_d"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d = d[~d.score.str.startswith("predictor")]
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
say(f"runs: {d.groupby('dataset').apply(lambda g: g.groupby(['model', 'seed', 'fold']).ngroups, include_groups=False).to_dict()}")


def pool(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w).unstack("score") * 100


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


W = pool(d[d.scenario.isin(FA)], "risk_matched")
WC = pool(d[d.scenario == "clean"], "risk_matched")
say("\n== risk (%), six faults")
say(W.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
say("\n== clean risk (%)")
say(WC.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
for name, a, b in [("M1", "sens", "full"), ("M2", "rpt_L", "sens"), ("M3", "rpt_sens", "rpt_L"), ("extra", "sens", "err_reg_rptL")]:
    rows = []
    for ds in DS:
        dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
        rows.append(dict(dataset=ds, mean=dif.mean(), better=int((dif < 0).sum()), n=len(dif),
                         p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== {name}: {a} - {b} (points; Holm over 5; exploratory)")
    say(t.round(4).to_string(index=False))
    t.to_csv(out / f"ext39_{name}.csv", index=False)
(out / "ext39_output.txt").write_text("\n".join(lines) + "\n")
print("EXT39_SUMMARY_DONE")
