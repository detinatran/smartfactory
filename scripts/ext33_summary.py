"""EXT33 analysis (EXT33_PROTOCOL.md). usage: ext33_summary.py ALL_EXT33.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery", "hdd"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
d["scenario"] = np.where(d.scenario.astype(str).str.startswith("tbias"), "tbias", d.scenario)
if "kind" in d:
    d = d[d.kind != "maint"]
t = d[d.scenario == "tbias"]
say(f"runs: {t.groupby('dataset').apply(lambda g: g.groupby(['model', 'seed', 'fold']).ngroups, include_groups=False).to_dict()}")


def pool(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g = g[g[col].notna()]
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w).unstack("score") * 100


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(W, pairs_, title, fname):
    rows = []
    for a, b in pairs_:
        for ds in DS:
            if ds not in W.index.get_level_values(0):
                continue
            dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
            rows.append(dict(a=a, b=b, dataset=ds, mean=dif.mean(), n=len(dif), better=int((dif < 0).sum()),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p)
    say(f"\n== {title} (Holm over {len(r)})")
    say(r.round(5).to_string(index=False))
    r.to_csv(out / fname, index=False)


say("\n== targeted fault: affected share / predictor optimistic rate on A, clean -> fault (%)")
say(pd.DataFrame({k: pool(t, k).groupby("dataset").mean()["full"] for k in ("share_A", "pred_opt_A_clean", "pred_opt_A")})
    .reindex(DS).round(2).to_string())
EX, RM = pool(t, "excess_A"), pool(t, "risk_matched")
say("\n== excess_A (%)")
say(EX.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
say("\n== risk_matched under the targeted fault (%)")
say(RM.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
family(EX, [("rpt_auto", "full"), ("rpt_L", "full")], "T1: delta excess_A", "ext33_T1.csv")
family(RM, [("rpt_auto", b) for b in ("full", "gauss_bias_aware", "err_reg")], "T2: risk_matched, rpt_auto - baseline", "ext33_T2.csv")
(out / "ext33_output.txt").write_text("\n".join(lines) + "\n")
print("EXT33_SUMMARY_DONE")
