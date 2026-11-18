"""EXT25 analysis (EXT25_PROTOCOL.md). usage: ext25_summary.py ALL_EXT25.csv.gz EXT22_CONF.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, f22, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery", "hdd"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
SC = ["ensemble_std", "pca_spe", "pca_t2", "eq90", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
f = d[d.scenario.isin(FA)]
e = pd.read_csv(f22)
e["dataset"] = np.where(e.dataset.str.startswith("bearings"), "bearings", e.dataset)
k = ["dataset", "model", "seed", "fold", "scenario", "score"]
j = f[f.score.isin(["full", "rpt_L", "rpt_auto"])].merge(e[k + ["risk_matched"]], on=k, suffixes=("", "_22"))
mx = (100 * (j.risk_matched - j.risk_matched_22)).abs().max()
say(f"consistency vs EXT22: {len(j)} rows, max |diff| = {mx:.4f} points -> {'OK' if mx <= 0.01 else 'FAILED'}")


def pool(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g = g[g[col].notna()]
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w).unstack("score") * 100


M = {c: pool(f, c) for c in ("risk_matched", "risk_A", "risk_A_clean", "excess_A", "cov_A", "share_A",
                             "pred_opt_A", "pred_opt_A_clean")}
for c in ("risk_matched", "risk_A", "risk_A_clean", "excess_A", "cov_A"):
    say(f"\n== {c} (%), mean over pairs")
    say(M[c].groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
say("\n== affected share and predictor optimistic rate on A (fault / clean), %")
say(pd.DataFrame({"share_A": M["share_A"].groupby("dataset").mean()["full"],
                  "pred_opt_A": M["pred_opt_A"].groupby("dataset").mean()["full"],
                  "pred_opt_A_clean": M["pred_opt_A_clean"].groupby("dataset").mean()["full"]}).reindex(DS).round(2).to_string())
say("\n== per fault: predictor optimistic rate on A, fault - clean (points)")
pf = f[f.score == "full"].assign(d=lambda t: 100 * (t.pred_opt_A - t.pred_opt_A_clean))
say(pf.groupby(["dataset", "scenario"]).d.mean().unstack().reindex(index=DS, columns=FA).round(2).to_string())


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(tests, title, fname):
    rows = []
    for a, b, col in tests:
        for ds in DS:
            W = M[col]
            if ds not in W.index.get_level_values(0):
                continue
            dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
            rows.append(dict(a=a, b=b, metric=col, dataset=ds, mean=dif.mean(), n=len(dif), better=int((dif < 0).sum()),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== {title} (Holm over {len(t)})")
    say(t.assign(c=lambda r: r["mean"].map("{:+.2f}".format) + " (" + r.better.astype(str) + "/" + r.n.astype(str) +
                 ", p=" + r.p_holm.map("{:.2g}".format) + ")").pivot(index=["a", "b", "metric"], columns="dataset",
                                                                     values="c").reindex(columns=DS).to_string())
    t.to_csv(out / fname, index=False)


family([(a, "full", c) for a in ("rpt_auto", "rpt_L") for c in ("risk_A", "excess_A")], "F1: vs OSC on affected windows",
       "ext25_F1.csv")
family([("rpt_auto", b, "risk_A") for b in ("ensemble_std", "gauss_bias_aware", "err_reg")],
       "F2: rpt_auto vs baselines, risk on affected windows", "ext25_F2.csv")
family([("rpt_auto", b, "risk_matched") for b in ("eq90", "pca_spe", "pca_t2")],
       "F3: rpt_auto vs new baselines, risk_matched", "ext25_F3.csv")
(out / "ext25_output.txt").write_text("\n".join(lines) + "\n")
print("EXT25_SUMMARY_DONE")
