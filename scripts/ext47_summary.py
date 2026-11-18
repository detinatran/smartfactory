"""EXT47 analysis (EXT47_PROTOCOL.md). usage: ext47_summary.py ALL_EXT47.csv.gz ALL_EXT40.csv.gz OUT.txt"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

fs, fd, out = sys.argv[1:4]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto_d"]
L = []


def load(f, tag):
    d = pd.read_csv(f)
    d = d[(d.dataset == "cmapss") & d.seed.between(162, 171)]
    rm = d[(d.score == "predictor") & (d.scenario == "clean")].groupby("seed").rmse_full.mean()
    w = d[d.scenario.isin(FA) & d.score.isin(SC)].groupby(["seed", "score"]).risk_matched.mean().unstack() * 100
    return rm, w


rs, ws = load(fs, "strong")
rd, wd = load(fd, "default")
L.append(f"predictor RMSE (clean, all eval windows): default {rd.mean():.2f}, strong {rs.mean():.2f}; strong lower in {int((rs < rd.reindex(rs.index)).sum())}/{len(rs)} seeds")
L.append("\n== risk (%), six faults, C-MAPSS, mean over 10 seeds")
L.append(pd.DataFrame({"default": wd.mean(), "strong": ws.mean()}).reindex(SC).round(2).to_string())
for a, b in [("rpt_auto_d", "full"), ("rpt_L", "full"), ("rpt_auto_d", "gauss_bias_aware"), ("rpt_auto_d", "err_reg")]:
    for tag, w in (("default", wd), ("strong", ws)):
        dif = w[a] - w[b]
        L.append(f"{tag:8s} {a} - {b}: {dif.mean():+.3f} points, better {int((dif < 0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue:.2g}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
