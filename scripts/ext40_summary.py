"""EXT40 analysis (EXT40_PROTOCOL.md), confirmation round 8, seed-level. usage: ext40_summary.py ALL_EXT40.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss_cv", "bearings", "battery_cv", "hdd"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
A8 = 0.0001953125
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "err_reg_rptL", "full", "rpt_L", "rpt_auto", "rpt_auto_d"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
pred = d[d.score == "predictor"]
d = d[~d.score.str.startswith("predictor")]
nr = d.groupby(["dataset", "model", "seed"]).fold.nunique().unstack("model")
say(f"runs: {d.groupby('dataset').apply(lambda g: g.groupby(['model', 'seed', 'fold']).ngroups, include_groups=False).to_dict()}")
# complete seeds only: both backbones, all folds of the family
full_folds = d.groupby("dataset").fold.nunique()
ok = nr.apply(lambda r: (r == full_folds[r.name[0]]).all(), axis=1)
keep = ok[ok].reset_index()[["dataset", "seed"]]
say(f"complete seeds per family: {keep.groupby('dataset').size().to_dict()}")
d = d.merge(keep, on=["dataset", "seed"])


def pool_seed(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    s = (s._v / s._w).unstack("score") * 100
    return s.groupby(["dataset", "seed"]).mean()  # GRU and TCN averaged -> one value per seed


fa = d[d.scenario.isin(FA)]
W, WP, W70 = pool_seed(fa, "risk_matched"), pool_seed(fa, "pess_matched"), pool_seed(fa, "risk_c70")
say("\n== risk (%), six faults, mean over seeds")
say(W.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
rows = []
rng = np.random.default_rng(4040)
for ds in DS:
    dif = (W.loc[ds]["rpt_auto_d"] - W.loc[ds]["full"]).dropna()
    bs = np.array([rng.choice(dif.to_numpy(), len(dif)).mean() for _ in range(20000)])
    rows.append(dict(dataset=ds, mean=dif.mean(), better=int((dif < 0).sum()), worse=int((dif > 0).sum()), n=len(dif),
                     p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                     ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), ci90_hi=np.percentile(bs, 95),
                     pess=(WP.loc[ds]["rpt_auto_d"] - WP.loc[ds]["full"]).mean()))
t = pd.DataFrame(rows)
p = t.p.to_numpy()
adj, run = np.empty(len(p)), 0.0
for r, i in enumerate(np.argsort(p)):
    run = max(run, (len(p) - r) * p[i])
    adj[i] = min(1.0, run)
t["p_holm"] = adj
t["sig"] = np.where(t.p_holm < A8, np.where(t["mean"] < 0, "better", "WORSE"), "")
say(f"\n== CONFIRMATORY round 8 (seed level): rpt_auto_d - full (Holm over 5, alpha_8 = {A8})")
say(t.round(5).to_string(index=False))
okc = (t.sig == "better").any() and not (t.sig == "WORSE").any() and (t["mean"] <= 0.3).all() and (t.pess <= 1.0).all()
say(f"CONFIRMED: {bool(okc)}")
h = t.set_index("dataset").loc["hdd"]
say(f"(a) hdd non-inferiority at +0.15: mean {h['mean']:+.3f}, 90% CI upper {h.ci90_hi:+.3f} -> {'NON-INFERIOR' if h.ci90_hi < 0.15 else 'not shown'}")
say(f"(b) benchmark families significant at alpha_8: {int((t[t.dataset != 'hdd'].sig == 'better').sum())} of 4")
say("\n== descriptive (seed level, mean diff points; better seeds of n)")
for a, b in [("rpt_L", "full"), ("rpt_auto", "full"), ("rpt_auto_d", "gauss_bias_aware"), ("rpt_auto_d", "err_reg"), ("rpt_auto_d", "err_reg_rptL")]:
    say(f"{a} - {b}: " + "  ".join(f"{ds} {(W.loc[ds][a] - W.loc[ds][b]).mean():+.2f} ({int(((W.loc[ds][a] - W.loc[ds][b]) < 0).sum())}/{len(W.loc[ds])})" for ds in DS))
say("risk at 70% coverage, rpt_auto_d - full: " + "  ".join(f"{ds} {(W70.loc[ds]['rpt_auto_d'] - W70.loc[ds]['full']).mean():+.2f}" for ds in DS))
u = d.drop_duplicates(["dataset", "model", "seed", "fold"])
say("switch share (rpt_auto_d chose rpt_L): " + str(u.groupby("dataset").choice_d.apply(lambda s: round((s == "rpt_L").mean(), 2)).to_dict()))
say("D by family (points): " + str((100 * u.groupby("dataset").D.mean()).round(2).to_dict()))
t.to_csv(out / "ext40_confirm.csv", index=False)
(out / "ext40_output.txt").write_text("\n".join(lines) + "\n")
print("EXT40_SUMMARY_DONE")
