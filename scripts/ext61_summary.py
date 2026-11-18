"""EXT61 analysis (protocol: EXT61_PROTOCOL.md): bearing ELPIS-J variants V16 (optimism-matched) and V17 (pooled
default+ELPIS-J-V13 ensemble) vs default (EXT40) and V13 (EXT56), seeds 162-171. usage: ext61_summary.py [FILES...]"""
import sys
import pandas as pd
from scipy import stats
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
R = "extensions/results/"


def pool(d, scen, col, sc):
    g = d[d.scenario.isin(scen) & (d.score == sc)].groupby(["model", "seed", "fold"])[col].mean().reset_index()
    g["w"] = g.fold.map(BW); g["v"] = g[col] * g.w
    s = g.groupby(["model", "seed"])[["v", "w"]].sum()
    return (s.v / s.w).groupby("seed").mean()


src = {"default": R + "ext40/all_ext40.csv.gz", "V13": R + "ext56/all_ext56_v13.csv.gz"}
for f in sys.argv[1:]:
    src[f.split("_")[-1].split(".")[0]] = f
rows = {}
for k, f in src.items():
    d = pd.read_csv(f)
    d = d[d.dataset.str.startswith("bearings") & d.seed.between(162, 171)]
    rows[k] = dict(rmse=pool(d, ["clean"], "rmse_full", "predictor"), opt=100 * pool(d, FA, "opt_full", "predictor"),
                   pess=100 * pool(d, FA, "pess_full", "predictor"), osc=100 * pool(d, FA, "risk_matched", "full"),
                   rptd=100 * pool(d, FA, "risk_matched", "rpt_auto_d"),
                   rptd_pess=100 * pool(d, FA, "pess_matched", "rpt_auto_d"))
print(pd.DataFrame({k: {m: v.mean() for m, v in r.items()} for k, r in rows.items()}).round(3).to_string())
for k in rows:
    if k == "default":
        continue
    for m in ("rptd", "rptd_pess"):
        dif = (rows[k][m] - rows["default"][m].reindex(rows[k][m].index)).dropna()
        p = stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
        print(f"{k:8s} {m:9s} vs default: {dif.mean():+.3f} (lower {int((dif < 0).sum())}/{len(dif)}, p={p:.2g})")
    dif = (rows[k]["rptd"] - rows[k]["osc"]).dropna()
    print(f"{k:8s} ELPIS-G - OSC (same predictor): {dif.mean():+.3f} (lower {int((dif < 0).sum())}/{len(dif)})")
