"""EXT48 analysis (EXT48_PROTOCOL.md), confirmation round 9, seed level. usage: ext48_summary.py ALL_EXT48.csv.gz OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

fe, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss_cv", "bearings", "battery_cv", "hdd"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
A9 = 0.00009765625
L = []


def say(s=""):
    print(s); L.append(str(s))


def holm(p):
    p = np.asarray(p, float); adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
mt = d[d.get("kind", pd.Series(index=d.index, dtype=object)).eq("maint")] if "kind" in d else d.iloc[:0]
d = d[d.get("kind", pd.Series(index=d.index, dtype=object)).ne("maint")] if "kind" in d else d
nr = d.groupby(["dataset", "model", "seed"]).fold.nunique().unstack("model")
full_folds = d.groupby("dataset").fold.nunique()
ok = nr.apply(lambda r: (r == full_folds[r.name[0]]).all(), axis=1)
keep = ok[ok].reset_index()[["dataset", "seed"]]
say(f"complete seeds per family: {keep.groupby('dataset').size().to_dict()}")
d = d.merge(keep, on=["dataset", "seed"])


def pool_seed(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g = g[g[col].notna()]
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return ((s._v / s._w).unstack("score") * 100).groupby(["dataset", "seed"]).mean()


def family(W, a, b, alpha, title, dss=DS):
    rows = []
    for ds in dss:
        if ds not in W.index.get_level_values(0):
            continue
        dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
        rows.append(dict(dataset=ds, mean=dif.mean(), better=int((dif < 0).sum()), worse=int((dif > 0).sum()), n=len(dif),
                         p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p)
    t["sig"] = np.where(t.p_holm < alpha, np.where(t["mean"] < 0, "better", "WORSE"), "")
    say(f"\n== {title}"); say(t.round(6).to_string(index=False))
    return t


fa = d[d.scenario.isin(FA)]
EX = pool_seed(fa, "excess_A")
t = family(EX, "rpt_auto_d", "full", A9, f"CONFIRMATORY round 9: delta excess_A, rpt_auto_d - full (seed level, Holm over 5, alpha_9={A9})")
okc = (t.sig == "better").any() and not (t.sig == "WORSE").any() and (t["mean"] <= 0.3).all()
say(f"CONFIRMED: {bool(okc)}")
t.to_csv(out / "ext48_confirm.csv", index=False)
say("\n== excess_A (%), mean over seeds"); say(EX.groupby("dataset").mean().reindex(DS).round(2).to_string())
W = pool_seed(fa, "risk_matched")
family(W, "rpt_auto_d", "full", 0.05, "(a) whole-test risk, rpt_auto_d - full (Holm over 5, alpha 0.05)")
MK = pool_seed(d[d.scenario == "mask"], "excess_A")
family(MK, "rpt_auto_d", "full", 0.05, "(b) mask-fault excess_A, rpt_auto_d - full (Holm over 5)")
rows = []
for b in ("err_reg_rptL", "gauss_rptL"):
    for ds in [x for x in DS if x in W.index.get_level_values(0)]:
        dif = (W.loc[ds]["rpt_auto_d"] - W.loc[ds][b]).dropna()
        rows.append(dict(vs=b, dataset=ds, mean=dif.mean(), better=int((dif < 0).sum()), n=len(dif), p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
tc = pd.DataFrame(rows); tc["p_holm"] = holm(tc.p)
say("\n== (c) risk, rpt_auto_d - perturbation-trained directional estimators (Holm over 10)"); say(tc.round(5).to_string(index=False))
if len(mt):
    m = mt[mt.scenario.isin(FA)].groupby(["dataset", "score", "model", "seed", "fold"])[["fail_rate", "insp_per_unit", "waste"]].mean().reset_index()
    m = m.groupby(["dataset", "score", "seed"])[["fail_rate", "insp_per_unit", "waste"]].mean()
    say("\n== (d) maintenance simulation, six faults, frozen thresholds (mean over seeds)")
    say(m.groupby(["dataset", "score"]).mean().unstack("score").reindex(DS).round(4).to_string())
    for col in ("fail_rate", "insp_per_unit", "waste"):
        u = m[col].unstack("score")
        dif = (u.rpt_auto_d - u.full)
        say(f"  {col}: rpt_auto_d - full by family: " + str(dif.groupby(level="dataset").mean().reindex(DS).round(4).to_dict()))
u = d.drop_duplicates(["dataset", "model", "seed", "fold"])
say("\nswitch share (rpt_auto_d chose rpt_L): " + str(u.groupby("dataset").choice_d.apply(lambda s: round((s == "rpt_L").mean(), 2)).to_dict()))
(out / "ext48_output.txt").write_text("\n".join(L) + "\n")
print("EXT48_SUMMARY_DONE")
