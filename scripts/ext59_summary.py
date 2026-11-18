"""EXT59 analysis (EXT59_PROTOCOL.md), confirmation round 10, seed level.
usage: ext59_summary.py ALL_EXT59_RAMP.csv.gz [ALL_EXT59_DEFAULT.csv.gz] OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

args = sys.argv[1:]
fr, out = args[0], Path(args[-1])
fd = args[1] if len(args) == 3 else None
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "battery_cv", "bearings"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
A10 = 0.000048828125
L = []


def say(s=""):
    print(s); L.append(str(s))


def load(f):
    d = pd.read_csv(f)
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    d = d[d.dataset.isin(DS)]
    nr = d.groupby(["dataset", "model", "seed"]).fold.nunique().unstack("model")
    ff = d.groupby("dataset").fold.nunique()
    ok = nr.apply(lambda r: (r == ff[r.name[0]]).all(), axis=1)
    return d.merge(ok[ok].reset_index()[["dataset", "seed"]], on=["dataset", "seed"])


def pool(d, scen, col, scores):
    g = d[d.scenario.isin(scen) & d.score.isin(scores)].groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0); g["v"] = g[col] * g.w
    s = g.groupby(["dataset", "score", "model", "seed"])[["v", "w"]].sum()
    return ((s.v / s.w).unstack("score") * 100).groupby(["dataset", "seed"]).mean()


R = load(fr)
say(f"complete seeds (RAMP): {R.drop_duplicates(['dataset','seed']).groupby('dataset').size().to_dict()}")
W = pool(R, FA, "risk_matched", ["full", "rpt_auto_d"]); P = pool(R, FA, "pess_matched", ["full", "rpt_auto_d"])
rows = []
for ds in DS:
    dif = (W.loc[ds].rpt_auto_d - W.loc[ds].full).dropna()
    rows.append(dict(dataset=ds, osc=W.loc[ds].full.mean(), elpis_g=W.loc[ds].rpt_auto_d.mean(), mean=dif.mean(), better=int((dif < 0).sum()),
                     n=len(dif), p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0, pess=(P.loc[ds].rpt_auto_d - P.loc[ds].full).mean()))
t = pd.DataFrame(rows)
p = t.p.to_numpy(); adj, run = np.empty(len(p)), 0.0
for r, i in enumerate(np.argsort(p)):
    run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
t["p_holm"] = adj
t["sig"] = np.where(t.p_holm < A10, np.where(t["mean"] < 0, "better", "WORSE"), "")
say(f"\n== CONFIRMATORY round 10: ELPIS-G - OSC on the RAMP-V13 predictor (seed level, Holm over 3, alpha_10={A10})")
say(t.round(6).to_string(index=False))
okc = (t.sig == "better").any() and not (t.sig == "WORSE").any() and (t["mean"] <= 0.3).all() and (t.pess <= 1.0).all()
say(f"CONFIRMED: {bool(okc)}")
t.to_csv(out / "ext59_confirm.csv", index=False)
u = R.drop_duplicates(["dataset", "model", "seed", "fold"])
say("switch share: " + str(u.groupby("dataset").choice_d.apply(lambda s: round((s == "rpt_L").mean(), 2)).to_dict()))
if fd:
    D = load(fd)
    WD = pool(D, FA, "risk_matched", ["full", "rpt_auto_d"])
    rm = lambda d: pool(d, ["clean"], "rmse_full", ["predictor"]).predictor / 100
    say("\n== secondary: RAMP-V13 system vs default system (same seeds), ELPIS-G risk under faults")
    for ds in DS:
        a, b = W.loc[ds].rpt_auto_d, WD.loc[ds].rpt_auto_d.reindex(W.loc[ds].index)
        dif = (a - b).dropna()
        say(f"  {ds:10s} RAMP {a.mean():.2f} default {b.mean():.2f} diff {dif.mean():+.3f} (RAMP better {int((dif<0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue:.2g}) | RMSE RAMP {rm(R).loc[ds].mean():.3f} default {rm(D).loc[ds].mean():.3f}")
(out / "ext59_output.txt").write_text("\n".join(L) + "\n")
print("EXT59_SUMMARY_DONE")
