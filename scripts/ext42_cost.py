"""EXT42 (post hoc, exploratory): expected per-window cost under an inspection/failure cost ratio, with each selector
operating at its own cost-minimising coverage on the grid {0.5,...,0.95,1}. cost(c) = (1-c)*1 + R * c * risk(c),
i.e. inspection cost 1 per abstained window and R per accepted optimistic error. usage: ext42_cost.py ALL.csv.gz OUT.txt"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

fe, out = sys.argv[1:3]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
RATIOS = [2, 5, 10, 20, 50]
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto_d"]
d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
d = d[d.scenario.isin(FA + ["clean"])]
key = ["dataset", "model", "seed", "fold", "scenario"]
acc = d[d.score == "predictor"].set_index(key).opt_full
s = d[d.score.isin(SC)].set_index(key + ["score"])
cols = {c: f"risk_c{int(round(100 * c))}" for c in GRID}
cols[0.9] = "risk_matched"
lines = []
for setting, scen in (("six faults", FA), ("clean", ["clean"])):
    lines.append(f"\n######## {setting}")
    for R in RATIOS:
        rows = []
        for sc in SC:
            x = s.xs(sc, level="score")
            x = x[x.index.get_level_values("scenario").isin(scen)]
            costs = np.column_stack([(1 - c) + R * c * x[cols[c]].to_numpy() for c in GRID] +
                                    [R * acc.reindex(x.index).to_numpy()])
            best = costs.min(1)
            r = pd.DataFrame({"cost": best}, index=x.index).reset_index()
            r["score"] = sc
            rows.append(r)
        r = pd.concat(rows)
        r = r.groupby(["dataset", "model", "seed", "fold", "score"]).cost.mean().reset_index()
        r["w"] = np.where(r.dataset == "bearings", r.fold.map(BW), 1.0)
        r["v"] = r.cost * r.w
        p = r.groupby(["dataset", "seed", "score"])[["v", "w"]].sum()
        p = (p.v / p.w).unstack("score") * 100  # cost per 100 windows
        lines.append(f"-- R = {R}: min expected cost per 100 windows (mean over seeds); rpt_auto_d - full (seeds better, Wilcoxon p)")
        for ds, g in p.groupby(level="dataset"):
            dif = g.rpt_auto_d - g.full
            pv = stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
            lines.append(f"  {ds:11s} " + " ".join(f"{sc}={g[sc].mean():.2f}" for sc in SC) +
                         f" | diff {dif.mean():+.3f} ({int((dif < 0).sum())}/{len(dif)}, p={pv:.2g}, rel {100 * dif.mean() / g.full.mean():+.1f}%)")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
