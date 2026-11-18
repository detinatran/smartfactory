"""EXT24 analysis (EXT24_PROTOCOL.md). usage: ext24_summary.py ALL_EXT24.csv.gz EXT22_CONF.csv.gz OUTDIR
Descriptive table per family x score x pool x pi (violation %, coverage %, risk %), D1 (rpt_auto vs full, CP, risk
target, pi=0.3; Holm over 10), D2 (rpt_auto, CP/CS vs C0; Holm over 20). Pair level = (backbone, seed); C-MAPSS
subsets averaged within a pair, bearing folds unit-weighted. Exploratory, nominal alpha 0.05.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, f22, out = map(Path, sys.argv[1:4])
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
say(f"runs per dataset: {d.groupby('dataset').apply(lambda g: g.groupby(['model', 'seed', 'fold']).ngroups, include_groups=False).to_dict()}")

# consistency with EXT22
c = d[d.kind == "consistency"].groupby(["dataset", "model", "seed", "fold", "score"]).risk_matched.mean()
e = pd.read_csv(f22)
e["dataset"] = np.where(e.dataset.str.startswith("bearings"), "bearings", e.dataset)
e = e[e.scenario.isin(["stuck", "bias", "noise", "intermittent", "spike", "gain"]) & e.score.isin(["full", "rpt_L", "rpt_auto"])]
e = e.groupby(["dataset", "model", "seed", "fold", "score"]).risk_matched.mean()
j = pd.concat([c.rename("a"), e.rename("b")], axis=1, join="inner")
mx = (100 * (j.a - j.b)).abs().max()
say(f"consistency vs EXT22: {len(j)} cells, max |diff| = {mx:.4f} points -> {'OK' if mx <= 0.01 else 'FAILED'}")

x = d[d.kind == "deploy"].copy()
x["violation"] = x.violation.astype(str).str.lower().eq("true").astype(float)
x["w"] = np.where(x.dataset == "bearings", x.fold.map(BW), 1.0)

# descriptive (deployment = run, bearing folds weighted by units)
def wmean(g, col):
    v = g[col].to_numpy(float)
    m = ~np.isnan(v)
    return np.average(v[m], weights=g.w.to_numpy()[m]) if m.any() else np.nan


desc = x.groupby(["dataset", "score", "pool", "target", "pi"]).apply(
    lambda g: pd.Series({"viol": 100 * wmean(g, "violation"), "cov": 100 * wmean(g, "coverage"),
                         "risk": 100 * wmean(g, "risk"), "r_star": 100 * wmean(g, "r_star"),
                         "pess": 100 * wmean(g, "pess")}), include_groups=False).reset_index()
desc.to_csv(out / "ext24_descriptive.csv", index=False)
for tgt in ("risk", "cov90"):
    for pi in (0.0, 0.1, 0.3, 0.5):
        say(f"\n== target={tgt}, pi={pi}: violation% / coverage% / risk% (r* mean in header)")
        t = desc[(desc.target == tgt) & (desc.pi == pi)]
        cell = t.assign(c=t.viol.map("{:.0f}".format) + "/" + t["cov"].map("{:.1f}".format) + "/" + t.risk.map("{:.2f}".format))
        piv = cell.pivot_table(index=["pool", "score"], columns="dataset", values="c", aggfunc="first")
        say(piv.reindex(columns=DS).reindex([(p, s) for p in ("C0", "CS", "CP") for s in SC]).to_string())
say("\nr* (%): " + desc[desc.target == "risk"].groupby("dataset").r_star.mean().round(2).to_dict().__repr__())


def pairs(sub, col):
    g = sub.groupby(["dataset", "score", "pool", "model", "seed"]).apply(lambda h: wmean(h, col), include_groups=False)
    return g


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def test(sub, a, b, key, title, fname):
    rows = []
    for col in ("excess", "coverage"):
        P = pairs(sub, col)
        for ds in DS:
            try:
                da, db = P.loc[(ds,) + a], P.loc[(ds,) + b]
            except KeyError:
                continue
            dif = (da - db).dropna() * 100
            rows.append(dict(metric=col, dataset=ds, mean=dif.mean(), n=len(dif), pos=int((dif > 0).sum()),
                             neg=int((dif < 0).sum()),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


s3 = x[(x.target == "risk") & (x.pi == 0.3)]
t1 = test(s3, ("rpt_auto", "CP"), ("full", "CP"), None, "", "")
t1.to_csv(out / "ext24_D1.csv", index=False)
say("\n== D1: rpt_auto - full (CP, risk target, pi=0.3), points; Holm over %d" % len(t1))
say(t1.round(4).to_string(index=False))
t2a = test(s3, ("rpt_auto", "CP"), ("rpt_auto", "C0"), None, "", "")
t2b = test(s3, ("rpt_auto", "CS"), ("rpt_auto", "C0"), None, "", "")
t2 = pd.concat([t2a.assign(cmp="CP-C0"), t2b.assign(cmp="CS-C0")], ignore_index=True)
t2["p_holm"] = holm(t2.p)
t2.to_csv(out / "ext24_D2.csv", index=False)
say("\n== D2: rpt_auto, pool vs C0 (risk target, pi=0.3), points; Holm over %d" % len(t2))
say(t2.round(4).to_string(index=False))
(out / "ext24_output.txt").write_text("\n".join(lines) + "\n")
print("EXT24_SUMMARY_DONE")
