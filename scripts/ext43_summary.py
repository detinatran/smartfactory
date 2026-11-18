"""EXT43 analysis (EXT43_PROTOCOL.md, exploratory). usage: ext43_summary.py ALL_EXT43.csv.gz OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
lc = d[d.kind == "lc"].copy()
# fraction label per run: rank of n_sel within run (1/4, 1/2, all)
lc["rank"] = lc.groupby(["dataset", "model", "seed", "fold"]).n_sel.rank(method="dense")
nmax = lc.groupby(["dataset", "model", "seed", "fold"]).n_sel.transform("max")
lc["f"] = np.where(lc.n_sel == nmax, "1", np.where(lc["rank"] == 1, "1/4", "1/2"))
# runs where 1/4 and 1/2 coincide get only one subsample row; label kept by rank
lc["gain"] = (lc.risk_rpt - lc.risk_full) * 100
lc["w"] = np.where(lc.dataset == "bearings", lc.fold.map(BW), 1.0)
lc["gw"] = lc.gain * lc.w
lc["nw"] = lc.n_sel * lc.w
g = lc.groupby(["dataset", "f", "seed"])[["gw", "nw", "w"]].sum()
g["gain"], g["n"] = g.gw / g.w, g.nw / g.w
lines = ["== rpt_L - full (points) by fraction of selector units; mean over seeds (n = mean selector units per run)"]
rows = []
for (ds, f), x in g.groupby(level=["dataset", "f"]):
    v = x.gain.to_numpy()
    rows.append(dict(dataset=ds, f=f, n_units=x.n.mean(), mean=v.mean(), better=int((v < 0).sum()), seeds=len(v),
                     p=stats.wilcoxon(v).pvalue if (v != 0).any() else 1.0))
t = pd.DataFrame(rows)
t["f"] = pd.Categorical(t.f, ["1/4", "1/2", "1"], ordered=True)
t = t.sort_values(["dataset", "f"])
lines.append(t.round(4).to_string(index=False))
dg = d[d.kind == "diag"]
lines.append("\nD by family (points): " + str((100 * dg.groupby("dataset").D.mean()).round(2).to_dict()))
t.to_csv(out / "ext43_lc.csv", index=False)
(out / "ext43_output.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
