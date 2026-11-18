"""EXT35 analysis (EXT35_PROTOCOL.md), confirmation round 6. usage: ext35_summary.py ALL_EXT35.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery", "hdd3"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
A6 = 0.00078125
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto", "rpt_auto_m"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d = d[~d.score.str.startswith("predictor")]
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
say(f"runs: {d.groupby('dataset').apply(lambda g: g.groupby(['model', 'seed', 'fold']).ngroups, include_groups=False).to_dict()}")
say(f"seeds: {sorted(d.seed.unique())}")


def pool(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w).unstack("score") * 100


fa = d[d.scenario.isin(FA)]
W, WP, WC = pool(fa, "risk_matched"), pool(fa, "pess_matched"), pool(d[d.scenario == "clean"], "risk_matched")
say("\n== risk (%), six faults")
say(W.groupby("dataset").mean().reindex(index=DS, columns=SC).round(2).to_string())
rows = []
for ds in DS:
    dif = (W.loc[ds]["rpt_auto_m"] - W.loc[ds]["full"]).dropna()
    rows.append(dict(dataset=ds, mean=dif.mean(), better=int((dif < 0).sum()), worse=int((dif > 0).sum()), n=len(dif),
                     p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                     pess=(WP.loc[ds]["rpt_auto_m"] - WP.loc[ds]["full"]).mean()))
t = pd.DataFrame(rows)
p = t.p.to_numpy()
adj, run = np.empty(len(p)), 0.0
for r, i in enumerate(np.argsort(p)):
    run = max(run, (len(p) - r) * p[i])
    adj[i] = min(1.0, run)
t["p_holm"] = adj
t["sig"] = np.where(t.p_holm < A6, np.where(t["mean"] < 0, "better", "WORSE"), "")
say("\n== CONFIRMATORY round 6: rpt_auto_m - full (Holm over 5, alpha_6 = 0.00078125)")
say(t.round(5).to_string(index=False))
ok = (t.sig == "better").any() and not (t.sig == "WORSE").any() and (t["mean"] <= 0.3).all() and (t.pess <= 1.0).all()
say(f"CONFIRMED: {bool(ok)}")
# secondary: non-inferiority on hdd3, seed bootstrap
dif = (W.loc["hdd3"]["rpt_auto_m"] - W.loc["hdd3"]["full"]).dropna()
seeds = sorted(dif.index.get_level_values("seed").unique())
rng = np.random.default_rng(3535)
bs = [np.mean([dif.xs(s, level="seed").mean() for s in rng.choice(seeds, len(seeds))]) for _ in range(20000)]
lo, hi = np.percentile(bs, [5, 95])
say(f"\nhdd3 non-inferiority: mean {dif.mean():+.3f}, 90% CI [{lo:+.3f}, {hi:+.3f}] -> {'NON-INFERIOR' if hi < 0.3 else 'not shown'} at +0.3")
say("\n== descriptive: rpt_L - full, rpt_auto - full, rpt_auto_m vs directional estimators")
for a, b in [("rpt_L", "full"), ("rpt_auto", "full"), ("rpt_auto_m", "gauss_bias_aware"), ("rpt_auto_m", "err_reg")]:
    say(f"{a} - {b}: " + "  ".join(f"{ds} {(W.loc[ds][a] - W.loc[ds][b]).mean():+.2f}" for ds in DS))
say("\nswitch share (rpt_auto_m chose rpt_L): " + str(d.drop_duplicates(["dataset", "model", "seed", "fold"]).groupby("dataset").choice_m.apply(lambda s: round((s == 'rpt_L').mean(), 2)).to_dict()))
say("switch share (rpt_auto R0): " + str(d.drop_duplicates(["dataset", "model", "seed", "fold"]).groupby("dataset").choice2.apply(lambda s: round((s == 'rpt_L').mean(), 2)).to_dict()))
say("clean risk, rpt_auto_m - full: " + str((WC.groupby("dataset").mean()["rpt_auto_m"] - WC.groupby("dataset").mean()["full"]).reindex(DS).round(2).to_dict()))
t.to_csv(out / "ext35_confirm.csv", index=False)
(out / "ext35_output.txt").write_text("\n".join(lines) + "\n")
print("EXT35_SUMMARY_DONE")
