"""EXT13 analysis: OURS (full_recon_rpt, from EXT10) vs six new baselines, Holm over 24 (EXT13_PROTOCOL.md).
usage: ext13_summary.py EXT10_all_conf62.csv.gz EXT13_all.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

f10, f13, out = map(Path, sys.argv[1:4])
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
NEW = ["laplace", "mahalanobis", "knn_feat", "iforest", "ar_residual", "tta_std"]
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
cols = ["dataset", "model", "seed", "fold", "scenario", "score"] + M
d10 = pd.read_csv(f10)[cols]
d = pd.concat([d10[d10.score.isin(["full_recon_rpt", "full"])], pd.read_csv(f13)[cols]], ignore_index=True)
lines = []


def say(s):
    print(s)
    lines.append(str(s))


def pool(sub, cols):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


x = pool(d[d.scenario.isin(FA)], M)
mean = x.groupby(["dataset", "score"]).mean() * 100
for col in M:
    say(f"\n== {col} (%, six unseen faults, seeds 62-71)")
    say(mean[col].unstack("dataset").reindex(index=NEW + ["full", "full_recon_rpt"], columns=DS).round(2).to_string())
xc = pool(d[d.scenario == "clean"], ["risk_matched"])
say("\n== clean risk_matched (%)")
say((xc.groupby(["dataset", "score"]).mean() * 100).risk_matched.unstack("dataset")
    .reindex(index=NEW + ["full", "full_recon_rpt"], columns=DS).round(2).to_string())
W = x.risk_matched.unstack("score")
rows = []
for b in NEW:
    for ds in DS:
        dif = (W.loc[ds]["full_recon_rpt"] - W.loc[ds][b]).dropna()
        rows.append(dict(b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins=int((dif < 0).sum()),
                         p=wilcoxon(dif).pvalue))
t = pd.DataFrame(rows)
adj, run = np.empty(len(t)), 0.0
for r, i in enumerate(np.argsort(t.p.to_numpy())):
    run = max(run, (len(t) - r) * t.p.iloc[i])
    adj[i] = min(1.0, run)
t["p_holm"] = adj
t["cell"] = t.mean_pts.map("{:+.2f}".format) + " (" + t.wins.astype(str) + "/" + t.n.astype(str) + ", p=" + \
    t.p_holm.map("{:.1g}".format) + ")"
say("\n== OURS - baseline (Holm over 24, alpha 0.05)")
say(t.pivot(index="b", columns="dataset", values="cell").reindex(index=NEW, columns=DS).to_string())
t.to_csv(out / "familyEXT13.csv", index=False)
mean.reset_index().to_csv(out / "summary_ext13.csv", index=False)
(out / "summary_output.txt").write_text("\n".join(lines) + "\n")
