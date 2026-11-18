"""EXT62 analysis (EXT62_PROTOCOL.md, exploratory): pooled default+RAMP-V13 ensemble vs default and RAMP-V13 (EXT59),
seeds 222-251, seed level. usage: ext62_summary.py HYBRID.csv.gz EXT59_DEFAULT.csv.gz EXT59_RAMP.csv.gz OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

fh, fd, fr, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "battery_cv", "bearings"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
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


def holm(p):
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


S = {k: load(f) for k, f in dict(hybrid=fh, default=fd, ramp=fr).items()}
say("complete seeds: " + str({k: v.drop_duplicates(["dataset", "seed"]).groupby("dataset").size().to_dict() for k, v in S.items()}))
W = {k: pool(v, FA, "risk_matched", ["full", "rpt_auto_d"]) for k, v in S.items()}
P = {k: pool(v, FA, "pess_matched", ["full", "rpt_auto_d"]) for k, v in S.items()}
RM = {k: pool(v, ["clean"], "rmse_full", ["predictor"]).predictor / 100 for k, v in S.items()}
rng = np.random.default_rng(0)
for ref in ("default", "ramp"):
    rows = []
    for ds in DS:
        a = W["hybrid"].loc[ds].rpt_auto_d
        dif = (a - W[ref].loc[ds].rpt_auto_d.reindex(a.index)).dropna()
        bs = rng.choice(dif.to_numpy(), (20000, len(dif))).mean(1)
        pd_ = (P["hybrid"].loc[ds].rpt_auto_d - P[ref].loc[ds].rpt_auto_d.reindex(a.index)).dropna()
        rows.append(dict(dataset=ds, hybrid=a.mean(), ref=W[ref].loc[ds].rpt_auto_d.mean(), diff=dif.mean(),
                         ci90_hi=np.quantile(bs, 0.95), better=int((dif < 0).sum()), n=len(dif),
                         p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0, pess_diff=pd_.mean(),
                         pess_better=int((pd_ < 0).sum()), rmse_h=RM["hybrid"].loc[ds].mean(), rmse_ref=RM[ref].loc[ds].mean()))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p.to_numpy())
    if ref == "default":
        t["noninf_0.3"] = t.ci90_hi <= 0.3
    say(f"\n== ELPIS-G risk under faults: hybrid vs {ref} (seed level)")
    say(t.round(4).to_string(index=False))
    t.to_csv(out / f"ext62_vs_{ref}.csv", index=False)
say("\n== ELPIS-G - OSC on the hybrid predictor")
for ds in DS:
    dif = (W["hybrid"].loc[ds].rpt_auto_d - W["hybrid"].loc[ds].full).dropna()
    say(f"  {ds:10s} {dif.mean():+.3f} (better {int((dif < 0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue:.2g})")
(out / "ext62_output.txt").write_text("\n".join(L) + "\n")
print("EXT62_SUMMARY_DONE")
