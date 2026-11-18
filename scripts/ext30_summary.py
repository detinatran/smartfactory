"""EXT30 analysis (EXT30_PROTOCOL.md). usage: ext30_summary.py ALL_EXT30.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["set"] = np.where(d.dataset == "hdd2", "hdd2 (102-111)", np.where(d.seed <= 101, "hdd (92-101)", "hdd (112-121)"))
d["violation"] = d.violation.astype(str).str.lower().eq("true")
d["cert"] = d.level.notna()
say(f"runs: {d.groupby('set').apply(lambda g: g.groupby(['model', 'seed']).ngroups, include_groups=False).to_dict()}")
say(f"calibration units: {d.groupby('set').n_cal.agg(['min', 'max']).to_dict()}")
c = d[d.scenario == "clean"]
for st, g in c.groupby("set"):
    say(f"\n== {st}, CLEAN test: certified% / mean level / coverage% / risk% / violation% (of all runs)")
    t = g.groupby(["bound", "f", "score"]).apply(lambda h: pd.Series({
        "cert": 100 * h.cert.mean(), "level": h.level.mean(), "cov": 100 * h.coverage.mean(),
        "risk": 100 * h.risk.mean(), "viol": 100 * h.violation.mean(), "r_star": 100 * h.r_star.mean()}),
        include_groups=False)
    say(t.round(2).to_string())
f = d[d.scenario.isin(FA)]
say("\n== under the six faults (no guarantee): violation% / coverage% / risk%, wsr bound")
t = f[f.bound == "wsr"].groupby(["set", "f", "score"]).apply(lambda h: pd.Series({
    "viol": 100 * h.violation.mean(), "cov": 100 * h.coverage.mean(), "risk": 100 * h.risk.mean()}), include_groups=False)
say(t.round(2).to_string())
say("\n== exploratory: certified coverage, rpt_auto - full (clean, wsr, f=1.0), Wilcoxon over pairs")
for st, g in c[(c.bound == "wsr") & (c.f == 1.0)].groupby("set"):
    w = g.pivot_table(index=["model", "seed"], columns="score", values="coverage")
    dif = (w.rpt_auto - w.full) * 100
    p = stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
    say(f"{st}: {dif.mean():+.2f} points (better {(dif > 0).sum()}/{len(dif)}), p={p:.3g}")
(out / "ext30_output.txt").write_text("\n".join(lines) + "\n")
print("EXT30_SUMMARY_DONE")
