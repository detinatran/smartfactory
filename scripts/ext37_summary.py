"""EXT37 analysis (EXT37_PROTOCOL.md). usage: ext37_summary.py ALL_EXT37.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
d["family"] = np.where((d.dataset == "hdd") & (d.seed >= 112), "hdd (112-121)", np.where(d.dataset == "hdd", "hdd (92-101)", d.dataset))
dg = d[d.kind == "diag"].copy()
dg["gain"] = (dg.risk_rpt - dg.risk_full) * 100
dg["w"] = np.where(dg.dataset == "bearings", dg.fold.map(BW), 1.0)
say(f"diag runs: {dg.groupby('family').size().to_dict()}")
say("\n== diagnostic D (points) and gain rpt_L - full (points), mean per family")
say(dg.groupby("family").apply(lambda g: pd.Series({"D": 100 * np.average(g.D, weights=g.w), "share_D>0": (g.D > 0).mean(),
                                                    "gain": np.average(g.gain, weights=g.w)}), include_groups=False).round(3).to_string())
rho, p = stats.spearmanr(dg.D, dg.gain)
say(f"\nSpearman(D, gain) over all runs: rho={rho:.3f}, p={p:.2g} (negative = larger D, larger RPT benefit)")
for fam, g in dg.groupby("family"):
    if g.D.nunique() > 2:
        r, pp = stats.spearmanr(g.D, g.gain)
        say(f"  {fam}: rho={r:.3f}, p={pp:.2g}")
say("\n== rule 'use RPT iff D > 0': gain vs full per family (points)")
dg["rule"] = np.where(dg.D > 0, dg.gain, 0.0)
say(dg.groupby("family").apply(lambda g: pd.Series({"rule_gain": np.average(g.rule, weights=g.w), "always_rpt": np.average(g.gain, weights=g.w)}),
                               include_groups=False).round(3).to_string())
lc = d[d.kind == "lc"].copy()
if len(lc):
    lc["gain"] = (lc.risk_rpt - lc.risk_full) * 100
    lc["n"] = np.where(lc.n_sel > 150, "all", lc.n_sel.astype(int).astype(str))
    say("\n== learning curve on hard drives: rpt_L - full (points) by number of selector units")
    rows = []
    for (fam, n), g in lc.groupby(["family", "n"]):
        dif = g.gain.to_numpy()
        rows.append(dict(family=fam, n=n, mean=dif.mean(), better=int((dif < 0).sum()), N=len(dif),
                         p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    say(pd.DataFrame(rows).round(4).to_string(index=False))
(out / "ext37_output.txt").write_text("\n".join(lines) + "\n")
print("EXT37_SUMMARY_DONE")
