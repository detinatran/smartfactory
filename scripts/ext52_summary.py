"""EXT52 analysis: ELPIS-J variants on bearings vs ELPIS-J (EXT50), ELPIS-J-noJEPA (EXT51) and default (EXT40). usage: ext52_summary.py VARIANT_FILES..."""
import sys
import numpy as np
import pandas as pd
from scipy import stats
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
R = "extensions/results/"


def load(f):
    d = pd.read_csv(f)
    d = d[d.dataset.str.startswith("bearings") & d.seed.between(162, 171)]
    return d


def pool(d, scen, col, sc):
    g = d[d.scenario.isin(scen) & (d.score == sc)].groupby(["model", "seed", "fold"])[col].mean().reset_index()
    g["w"] = g.fold.map(BW); g["v"] = g[col] * g.w
    s = g.groupby(["model", "seed"])[["v", "w"]].sum()
    return (s.v / s.w).groupby("seed").mean()


src = {"default": R + "ext40/all_ext40.csv.gz", "ELPIS-J": R + "ext50/all_ext50.csv.gz", "ELPIS-J-noJEPA": R + "ext51/all_ext51.csv.gz"}
for f in sys.argv[1:]:
    src[f.split("_")[-1].split(".")[0]] = f
rows = {}
for k, f in src.items():
    d = load(f)
    rows[k] = dict(rmse=pool(d, ["clean"], "rmse_full", "predictor"), opt=100 * pool(d, FA, "opt_full", "predictor"),
                   osc=100 * pool(d, FA, "risk_matched", "full"), rptd=100 * pool(d, FA, "risk_matched", "rpt_auto_d"))
print(pd.DataFrame({k: {m: v.mean() for m, v in r.items()} for k, r in rows.items()}).round(3).to_string())
for k in rows:
    if k in ("default",):
        continue
    dif = (rows[k]["rptd"] - rows["default"]["rptd"].reindex(rows[k]["rptd"].index)).dropna()
    print(f"{k:12s} RPT-D vs default RPT-D: {dif.mean():+.3f} (better {int((dif < 0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue:.2g})")
