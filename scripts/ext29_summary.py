"""EXT29 analysis (EXT29_PROTOCOL.md), confirmation round 5. usage: ext29_summary.py ALL_EXT29.csv.gz OUTDIR
Primary: delta excess_A (rpt_auto - full), mean of the six original faults, 5 families, Holm over 5, alpha_5 = 0.0015625.
Success: significant reduction on >= 1 family and no significant increase."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery", "hdd"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
A5 = 0.0015625
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
say(f"runs per dataset: {d.groupby('dataset').apply(lambda g: g.groupby(['model', 'seed', 'fold']).ngroups, include_groups=False).to_dict()}")
say(f"seeds: {sorted(d.seed.unique())}")
if "kind" not in d:
    d["kind"] = np.nan
a = d[d.kind != "maint"]
m = d[d.kind == "maint"]


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


def family(W, pairs, title, fname, alpha=0.05, datasets=DS):
    rows = []
    for x, y in pairs:
        for ds in datasets:
            if ds not in W.index.get_level_values(0):
                continue
            dif = (W.loc[ds][x] - W.loc[ds][y]).dropna()
            rows.append(dict(a=x, b=y, dataset=ds, mean=dif.mean(), n=len(dif), better=int((dif < 0).sum()),
                             worse=int((dif > 0).sum()), p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t["sig"] = np.where(t.p_holm < alpha, np.where(t["mean"] < 0, "better", "WORSE"), "")
    say(f"\n== {title} (Holm over {len(t)}, alpha={alpha})")
    say(t.round(5).to_string(index=False))
    t.to_csv(out / fname, index=False)
    return t


fa = a[a.scenario.isin(FA)]
EX = pool(fa, "excess_A")
say("\n== excess_A (%), six original faults")
say(EX.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
say("\n== affected share / predictor optimistic rate on A, fault vs clean (full), six faults")
say(pd.DataFrame({k: pool(fa, k).groupby("dataset").mean()["full"] for k in ("share_A", "pred_opt_A", "pred_opt_A_clean")})
    .reindex(DS).round(2).to_string())
t = family(EX, [("rpt_auto", "full")], "PRIMARY round 5: delta excess_A, rpt_auto - full", "ext29_primary.csv", alpha=A5)
better = (t.sig == "better").any()
worse = (t.sig == "WORSE").any()
say(f"\nCONFIRMED (>=1 significant reduction, no significant increase): {bool(better and not worse)}")

mk = a[a.scenario == "mask"]
say("\n== mask fault: affected share / predictor optimistic rate on A (fault vs clean), full")
say(pd.DataFrame({k: pool(mk, k).groupby("dataset").mean()["full"] for k in ("share_A", "pred_opt_A", "pred_opt_A_clean")})
    .reindex(DS).round(2).to_string())
EXM = pool(mk, "excess_A")
say("\n== mask: excess_A (%)")
say(EXM.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
family(EXM, [("rpt_auto", "full")], "S1: mask, delta excess_A, rpt_auto - full", "ext29_S1.csv")
RM = pool(fa, "risk_matched")
say("\n== risk_matched (%), six faults")
say(RM.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
family(RM, [("rpt_auto", "full")], "S2: risk_matched, rpt_auto - full", "ext29_S2.csv")
family(EX, [("rpt_L", "full")], "S3: delta excess_A, rpt_L - full", "ext29_S3.csv")
say(f"\nrpt_auto chose rpt_L: {d.drop_duplicates(['dataset', 'model', 'seed', 'fold']).groupby('dataset').choice2.apply(lambda s: (s == 'rpt_L').sum()).to_dict()}")

# S4 maintenance (frozen thresholds), descriptive + tests on cost
if len(m):
    mm = m[m.scenario.isin(FA)].copy()
    for ci in (0.01, 0.05):
        mm[f"cost_{ci}"] = mm.fail_rate + ci * mm.insp_per_unit + 0.5 * mm.waste
    for col in ("fail_rate", "insp_per_unit", "waste", "cost_0.01", "cost_0.05"):
        W = pool(mm, col) / 100
        say(f"\n== maintenance, six faults: {col} (mean over pairs)")
        say(W.groupby("dataset").mean().reindex(index=DS, columns=SC).round(4).to_string())
    for ci in (0.01, 0.05):
        W = pool(mm, f"cost_{ci}") / 100
        family(W, [("rpt_auto", b) for b in ("full", "gauss_bias_aware", "ensemble_std")],
               f"S4: maintenance cost (c_i={ci}), rpt_auto - baseline", f"ext29_S4_{ci}.csv")
(out / "ext29_output.txt").write_text("\n".join(lines) + "\n")
print("EXT29_SUMMARY_DONE")
