"""EXT19 D: area under the risk-coverage curve on [0.5, 0.95] (trapezoid over 0.5,0.6,0.7,0.8,0.85,0.9,0.95), six unseen
faults, seeds 72-81 (ext12_conf). rpt_L vs baselines, Wilcoxon over pairs, Holm.
usage: ext19_aurc.py all_ext12conf.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

src, out = Path(sys.argv[1]), Path(sys.argv[2])
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
COV = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
COLS = ["risk_c50", "risk_c60", "risk_c70", "risk_c80", "risk_c85", "risk_matched", "risk_c95"]
d = pd.read_csv(src)
d = d[d.scenario.isin(FA) & d.score.isin(["ensemble_std", "full", "gauss_bias_aware", "err_reg", "rpt_L"])]
d["aurc"] = np.trapezoid(d[COLS].to_numpy(), COV, axis=1) / (COV[-1] - COV[0])
g = d.groupby(["dataset", "score", "model", "seed", "fold"]).aurc.mean().reset_index()
g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
g["_v"] = g.aurc * g._w
s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
W = (s._v / s._w).unstack("score") * 100
lines = ["== normalised AURC on [0.5, 0.95] (%, six faults, seeds 72-81)",
         W.groupby("dataset").mean().T.reindex(columns=DS).round(2).to_string()]
rows = []
for b in ["ensemble_std", "full", "gauss_bias_aware", "err_reg"]:
    for ds in DS:
        dif = (W.loc[ds]["rpt_L"] - W.loc[ds][b]).dropna()
        rows.append(dict(b=b, dataset=ds, mean=dif.mean(), wins=int((dif < 0).sum()), p=stats.wilcoxon(dif).pvalue))
t = pd.DataFrame(rows)
adj, run = np.empty(len(t)), 0.0
for r, i in enumerate(np.argsort(t.p.to_numpy())):
    run = max(run, (len(t) - r) * t.p.iloc[i]); adj[i] = min(1.0, run)
t["p_holm"] = adj
lines += ["\n== rpt_L - baseline AURC (points; Holm over 16)",
          t.assign(c=lambda r: r["mean"].map("{:+.2f}".format) + " (" + r.wins.astype(str) + ", p=" +
                   r.p_holm.map("{:.1g}".format) + ")").pivot(index="b", columns="dataset", values="c")
          .reindex(columns=DS).to_string()]
print("\n".join(lines))
(out / "aurc_output.txt").write_text("\n".join(lines) + "\n")
t.to_csv(out / "aurc_tests.csv", index=False)
