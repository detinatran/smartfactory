"""EXT27 analysis (EXT27_PROTOCOL.md). usage: ext27_summary.py ALL_EXT27.csv.gz EXT22_CONF.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, f22, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


def norm(d):
    d = d.copy()
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    return d


d = norm(pd.read_csv(fe))
say(f"runs per dataset: {d.groupby('dataset').apply(lambda g: g.groupby(['seed', 'fold']).ngroups, include_groups=False).to_dict()}")
say(f"seeds: {sorted(d.seed.unique())}")


def pool(sub, col="risk_matched"):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w).unstack("score") * 100


x = d[~d.score.str.startswith("predictor")]
W, Wc, Wp = pool(x[x.scenario.isin(FA)]), pool(x[x.scenario == "clean"]), pool(x[x.scenario.isin(FA)], "pess_matched")
say("\n== risk (%) six faults, Patch-Transformer, seeds 72-91")
say(W.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
say("\n== clean risk (%)")
say(Wc.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
say(f"\nrpt_auto chose rpt_L: {d.drop_duplicates(['dataset', 'seed', 'fold']).groupby('dataset').choice2.apply(lambda s: (s == 'rpt_L').sum()).to_dict()}")


def rmse(dd):
    p = dd[(dd.score == "predictor") & (dd.scenario == "clean")]
    g = p.groupby(["dataset", "model", "seed", "fold"]).rmse_full.mean().reset_index()
    return g.groupby(["dataset", "model"]).rmse_full.mean().unstack("model")


e = norm(pd.read_csv(f22))
say("\n== clean RMSE of the predictor (Patch-Transformer seeds 72-91 vs GRU/TCN seeds 92-101)")
say(pd.concat([rmse(d), rmse(e)], axis=1).reindex(DS).round(3).to_string())


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(a, bs, title, fname):
    rows = []
    for b in bs:
        for ds in DS:
            dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
            rows.append(dict(a=a, b=b, dataset=ds, mean=dif.mean(), n=len(dif), better=int((dif < 0).sum()),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                             pess=(Wp.loc[ds][a] - Wp.loc[ds][b]).mean()))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== {title} (Holm over {len(t)})")
    say(t.assign(c=lambda r: r["mean"].map("{:+.2f}".format) + " (" + r.better.astype(str) + "/" + r.n.astype(str) +
                 ", p=" + r.p_holm.map("{:.2g}".format) + ")").pivot(index="b", columns="dataset", values="c")
        .reindex(index=bs, columns=DS).to_string())
    t.to_csv(out / fname, index=False)


family("rpt_L", ["full", "gauss_bias_aware", "err_reg"], "B1: rpt_L - baseline", "ext27_B1.csv")
family("rpt_auto", ["full"], "B2: rpt_auto - full", "ext27_B2.csv")
ex = W - Wc
say("\n== excess (faults - clean), rpt_L - full: " + str({ds: round((ex.loc[ds]['rpt_L'] - ex.loc[ds]['full']).mean(), 2) for ds in DS}))
(out / "ext27_output.txt").write_text("\n".join(lines) + "\n")
print("EXT27_SUMMARY_DONE")
