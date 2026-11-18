"""EXT15 analysis (EXT15_PROTOCOL.md).
usage: ext15_summary.py EXT15_ALL.csv.gz EXT15_AUX.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fm, fx, out = map(Path, sys.argv[1:4])
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
HELD = ["clip", "lag", "quant", "invert"]
ALLCH = ["bias@1#99", "stuck#99", "gain@1.25#99", "noise@1#99"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
F = "rpt_L"
d = pd.read_csv(fm)
ax = pd.read_csv(fx)
lines = []


def say(s):
    print(s)
    lines.append(str(s))


def pool(sub, col="risk_matched"):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w).unstack("score")


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(W, comps, title, fname):
    rows = []
    for b in comps:
        for ds in DS:
            dif = 100 * (W.loc[ds][F] - W.loc[ds][b]).dropna()
            rows.append(dict(b=b, dataset=ds, mean=dif.mean(), wins=int((dif < 0).sum()), n=len(dif),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== {title} ({F} - baseline, points; Holm over {len(t)})")
    say(t.assign(c=lambda r: r["mean"].map("{:+.2f}".format) + " (" + r.wins.astype(str) + ", p=" +
                 r.p_holm.map("{:.1g}".format) + ")").pivot(index="b", columns="dataset", values="c")
        .reindex(index=comps, columns=DS).to_string())
    t.to_csv(out / fname, index=False)
    return t


W6 = pool(d[d.scenario.isin(FA)])
say("== risk (%), six faults, seeds 72-81")
say((W6.groupby("dataset").mean() * 100).T.reindex(columns=DS).round(2).to_string())
family(W6, ["oe_label", "det_rpt", "symmetric_rpt", "err_reg_rptL", "gba_rptL"], "1. novelty controls (six faults)",
       "ext15_novelty.csv")
Wc = pool(d[d.scenario == "clean"])
say("\n-- clean risk (%)")
say((Wc.groupby("dataset").mean() * 100).T.reindex(columns=DS).round(2).to_string())
WA = pool(d[d.scenario.isin(ALLCH)])
say("\n== 2. all-channel faults, risk (%)")
say((WA.groupby("dataset").mean() * 100).T.reindex(columns=DS).round(2).to_string())
family(WA, ["ensemble_std", "full", "gauss_bias_aware", "err_reg"], "2. all-channel faults", "ext15_allch.csv")
for scn in ALLCH:
    w = pool(d[d.scenario == scn])
    m = w.groupby("dataset").mean() * 100
    say(f"-- {scn}: {F} - full = " + str((m[F] - m["full"]).reindex(DS).round(2).to_dict()))

# 3. overlap
ov = ax[ax.kind == "overlap"].copy()
ov["_w"] = np.where(ov.dataset == "bearings", ov.fold.map(BW), 1.0)
auc = ov.groupby(["dataset", "scenario"]).apply(lambda g: np.average(g.auc.dropna(), weights=g._w[g.auc.notna()])
                                                 if g.auc.notna().any() else np.nan, include_groups=False).unstack(0)
say("\n== 3. generator-vs-fault separability AUC (0.5 = full overlap, 1 = separable)")
say(auc.reindex(index=FA + HELD + ALLCH, columns=DS).round(3).to_string())
gain = {}
for scn in FA + HELD + ALLCH:
    w = pool(d[d.scenario == scn])
    m = w.groupby("dataset").mean() * 100
    gain[scn] = (m[F] - m["full"])
G = pd.DataFrame(gain).T
pairs = [(auc.loc[s, ds], G.loc[s, ds]) for s in G.index for ds in DS if s in auc.index and np.isfinite(auc.loc[s, ds])]
a_, g_ = np.array(pairs).T
rho = stats.spearmanr(a_, g_)
say(f"Spearman(AUC, gain of {F} over full) across fault x dataset cells: rho={rho.statistic:.2f}, p={rho.pvalue:.2g}, "
    f"n={len(a_)}")
say("gain (points) table:")
say(G.reindex(index=FA + HELD + ALLCH, columns=DS).round(2).to_string())

# 4. mu vs sigma
ms = ax[ax.kind == "musigma"].copy()
ms["ratio"] = ms.med_abs_dmu / ms.med_abs_dsd.clip(lower=1e-9)
r = ms.groupby(["dataset", "scenario"])[["ratio", "frac_new_opt_low_sd"]].median().unstack(0)
say("\n== 4. premise of Prop. 1: median |d mu| / |d sigma| on changed windows (median over runs)")
say(r["ratio"].reindex(index=FA + HELD + ALLCH, columns=DS).round(1).to_string())
say("-- fraction of newly optimistic windows whose spread stays below the clean median spread")
say(r["frac_new_opt_low_sd"].reindex(index=FA + HELD + ALLCH, columns=DS).round(2).to_string())
(out / "ext15_output.txt").write_text("\n".join(lines) + "\n")
print("EXT15_SUMMARY_DONE")
