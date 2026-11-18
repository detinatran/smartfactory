"""EXT15 re-analyses requested by the simulated review (no new runs), on round-2 data (seeds 72-81).
  (a) seed-level tests: GRU and TCN share a unit split per seed, so average them per seed (n = 10) and report paired
      t-test 95% CIs and Wilcoxon p (sensitivity analysis; the pre-registered unit was the backbone-seed pair);
  (b) fault-induced excess risk: risk under faults minus clean risk, per family and per fault;
  (c) cost trade-off: c_opt * optimistic + c_pess * pessimistic rate at 90% coverage for cost ratios 1, 2, 5, 10.
usage: ext15_reanalysis.py all_ext12conf.csv.gz all_ext14sota.csv.gz all_ext14more.csv.gz all_ext14ablate.csv.gz OUTDIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fc, fs, fm, fa, out = map(Path, sys.argv[1:6])
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
M = ["risk_matched", "pess_matched"]
cols = ["dataset", "model", "seed", "fold", "scenario", "score"] + M
F = "rpt_L"
dc = pd.read_csv(fc)[cols]
da = pd.read_csv(fa)[cols]
d = pd.concat([dc, pd.read_csv(fs)[cols], pd.read_csv(fm)[cols],
               da[da.score.isin(["symmetric", "fault_det", "ours_white"])]], ignore_index=True)
d = d[~d.score.str.startswith("predictor")]
lines = []


def say(s):
    print(s)
    lines.append(str(s))


def pool(sub, cs, level=("dataset", "score", "model", "seed")):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cs].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    res = {}
    for col in cs:
        gg = g.assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(list(level) if "model" in level else ["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    r = pd.DataFrame(res)
    if "model" not in level:  # average backbones within seed
        r = r.groupby(["dataset", "score", "seed"]).mean()
    return r


# (a) seed level
xs = pool(d[d.scenario.isin(FA)], M, level=("dataset", "score", "seed"))
W = xs.risk_matched.unstack("score")
comps = ["full", "gauss_bias_aware", "err_reg", "ensemble_std", "cqr_lower", "deep_ens_nll", "selectivenet",
         "mahalanobis", "ar_residual", "symmetric"]
rows = []
for b in comps:
    for ds in DS:
        dif = 100 * (W.loc[ds][F] - W.loc[ds][b]).dropna()
        ci = stats.t.interval(0.95, len(dif) - 1, loc=dif.mean(), scale=stats.sem(dif))
        rows.append(dict(b=b, dataset=ds, n=len(dif), mean=dif.mean(), lo=ci[0], hi=ci[1],
                         wins=int((dif < 0).sum()), p_t=stats.ttest_1samp(dif, 0).pvalue,
                         p_w=stats.wilcoxon(dif).pvalue))
t = pd.DataFrame(rows)
adj, run = np.empty(len(t)), 0.0
for r, i in enumerate(np.argsort(t.p_t.to_numpy())):
    run = max(run, (len(t) - r) * t.p_t.iloc[i])
    adj[i] = min(1.0, run)
t["p_t_holm"] = adj
say("== (a) seed-level (n = 10 seeds, GRU/TCN averaged): rpt_L - baseline, points, 95% t-CI, wins/10, "
    "Holm-adjusted paired t p (over the table), Wilcoxon p (unadjusted)")
say(t.assign(cell=lambda r: r["mean"].map("{:+.2f}".format) + " [" + r.lo.map("{:+.2f}".format) + "," +
             r.hi.map("{:+.2f}".format) + "] " + r.wins.astype(str) + "/10 pH=" + r.p_t_holm.map("{:.1g}".format)
             + " pW=" + r.p_w.map("{:.1g}".format)).pivot(index="b", columns="dataset", values="cell")
    .reindex(index=comps, columns=DS).to_string())
t.to_csv(out / "seed_level.csv", index=False)

# (b) excess risk
x = pool(d[d.scenario.isin(FA)], ["risk_matched"])
xc = pool(d[d.scenario == "clean"], ["risk_matched"])
fault = x.groupby(["dataset", "score"]).mean().risk_matched * 100
clean = xc.groupby(["dataset", "score"]).mean().risk_matched * 100
ex = (fault - clean).unstack("dataset")
say("\n== (b) fault-induced excess risk = risk(6 faults) - risk(clean), points")
say(pd.concat({"fault": fault.unstack("dataset").loc[[F, "full", "gauss_bias_aware", "err_reg", "ensemble_std"]],
               "clean": clean.unstack("dataset").loc[[F, "full", "gauss_bias_aware", "err_reg", "ensemble_std"]],
               "excess": ex.loc[[F, "full", "gauss_bias_aware", "err_reg", "ensemble_std"]]}).reindex(columns=DS)
    .round(2).to_string())
# paired excess difference rpt_L vs full (per pair)
xe = (x.risk_matched - xc.risk_matched).unstack("score") * 100
rows = []
for ds in DS:
    dif = (xe.loc[ds][F] - xe.loc[ds]["full"]).dropna()
    rows.append(dict(dataset=ds, mean=dif.mean(), wins=int((dif < 0).sum()), n=len(dif), p=stats.wilcoxon(dif).pvalue))
say("excess(rpt_L) - excess(full), per pair: " + pd.DataFrame(rows).round(4).to_string(index=False))
pf = pool(d[d.scenario.isin(FA)].assign(dataset=lambda q: q.dataset + "|" + q.scenario), ["risk_matched"])
pfm = pf.groupby(["dataset", "score"]).mean().risk_matched.unstack("score") * 100
pfm.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in pfm.index])
cl = clean.unstack("score")
say("\n-- per-fault excess, rpt_L minus full (points)")
exf = pd.DataFrame({ds: (pfm.loc[ds][F] - cl.loc[ds, F]) - (pfm.loc[ds]["full"] - cl.loc[ds, "full"]) for ds in DS})
say(exf.reindex(FA).round(2).to_string())
say("-- clean-risk difference rpt_L - full: " + (cl[F] - cl["full"]).reindex(DS).round(2).to_dict().__str__())

# (c) cost trade-off
xp = pool(d[d.scenario.isin(FA)], M)
mo = xp.groupby(["dataset", "score"]).mean() * 100
say("\n== (c) cost = r * optimistic + pessimistic at 90% coverage (r = cost ratio optimistic:pessimistic); "
    "rpt_L minus best EXTERNAL baseline (own variants excluded), and which baseline")
for r in (1, 2, 5, 10):
    c = (r * mo.risk_matched + mo.pess_matched).unstack("dataset")
    own = [k for k in c.index if k in ("rptw", "full_rpt", "rptw_L", "err_reg_rpt", "gba_rpt", "full_recon_rpt",
                                        "recon_rpt_L", "ours_white", F)]
    best_other = c.drop(own).min()
    who = c.drop(own).idxmin()
    say(f"  r={r:>2}: " + "  ".join(f"{ds}: {c.loc[F, ds] - best_other[ds]:+.2f} (best other {who[ds]})" for ds in DS))
(out / "reanalysis_output.txt").write_text("\n".join(lines) + "\n")
print("EXT15_REANALYSIS_DONE")
