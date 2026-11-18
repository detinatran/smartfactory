"""EXT31 analysis (EXT31_PROTOCOL.md). usage: ext31_summary.py ALL_EXT31.csv.gz OUTDIR
Pair = (backbone, seed); folds cv0..cv4 averaged with weights = number of test cells (from the per-run unit counts)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
NCELL = {"cv0": 31, "cv1": 21, "cv2": 28, "cv3": 18, "cv4": 26}
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d = d[~d.score.str.startswith("predictor")]
say(f"runs: {d.groupby(['model', 'seed', 'fold']).ngroups}; folds: {sorted(d.fold.unique())}")


def pool(sub):
    g = sub.groupby(["score", "model", "seed", "fold"]).risk_matched.mean().reset_index()
    g["w"] = g.fold.map(NCELL)
    g["v"] = g.risk_matched * g.w
    s = g.groupby(["score", "model", "seed"])[["v", "w"]].sum()
    return (s.v / s.w).unstack("score") * 100


W, Wc = pool(d[d.scenario.isin(FA)]), pool(d[d.scenario == "clean"])
say("\n== risk (%), six faults / clean")
say(pd.DataFrame({"faults": W.mean(), "clean": Wc.mean()}).reindex(SC).round(2).to_string())
signs = np.array(list(itertools.product([-1, 1], repeat=10)))
rows = []
for a in ("rpt_L", "rpt_auto"):
    dif = (W[a] - W["full"]).dropna()
    p = stats.wilcoxon(dif).pvalue
    sd = dif.groupby(level="seed").mean().to_numpy()
    psf = (np.abs((signs * sd).mean(1)) >= abs(sd.mean()) - 1e-12).mean() if len(sd) == 10 else np.nan
    rows.append(dict(a=a, mean=dif.mean(), better=int((dif < 0).sum()), n=len(dif), p=p, seeds_better=int((sd < 0).sum()),
                     p_signflip=psf))
t = pd.DataFrame(rows)
pv = t.p.to_numpy()
adj, run = np.empty(len(pv)), 0.0
for r, i in enumerate(np.argsort(pv)):
    run = max(run, (len(pv) - r) * pv[i])
    adj[i] = min(1.0, run)
t["p_holm"] = adj
say("\n== PRIMARY: minus full (points), Holm over 2")
say(t.round(5).to_string(index=False))
say("\n== descriptive: rpt_L minus directional estimators")
for b in ("gauss_bias_aware", "err_reg"):
    dif = (W["rpt_L"] - W[b]).dropna()
    say(f"rpt_L - {b}: {dif.mean():+.2f} ({(dif < 0).sum()}/{len(dif)}, p={stats.wilcoxon(dif).pvalue:.3g})")
say("\n== per fold: rpt_L - full (points)")
pf = d[d.scenario.isin(FA)].groupby(["fold", "score", "model", "seed"]).risk_matched.mean().unstack("score") * 100
say((pf.rpt_L - pf.full).groupby(level="fold").mean().round(2).to_string())
say(f"\nrpt_auto chose rpt_L: {(d.drop_duplicates(['model', 'seed', 'fold']).choice2 == 'rpt_L').sum()}/{d.groupby(['model', 'seed', 'fold']).ngroups}")
(out / "ext31_output.txt").write_text("\n".join(lines) + "\n")
print("EXT31_SUMMARY_DONE")
