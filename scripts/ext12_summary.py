"""EXT12 analyses (EXT12_PROTOCOL.md). Pooling as conf_test.py.

  ext12_summary.py select  DEV.csv.gz [EXT10_DEV.csv.gz]                 stage D selection rule (seeds 42-61)
  ext12_summary.py confirm CONF.csv.gz FINAL                              stage C test (seeds 72-81, Holm 12, a=0.0125)
  ext12_summary.py stress  CONF.csv.gz FINAL                              stage S
  ext12_summary.py robust  CONF.csv.gz ROBUST.csv.gz FINAL                stage R (system level)
  ext12_summary.py backbone TRANSFORMER.csv.gz FINAL                      stage B
Writes <mode>_output.txt and CSVs next to the first input.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

mode, args = sys.argv[1], sys.argv[2:]
src = Path(args[0])
out = src.parent
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
HELD = ["clip", "lag", "quant", "invert"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
DIRECTIONAL = ["gauss_bias_aware", "err_reg"]
CANDS = ["rptw", "full_rpt", "rptw_L", "rpt_L", "err_reg_rpt", "gba_rpt", "full_recon_rpt", "recon_rpt_L"]  # simplicity
pd.set_option("display.width", 250)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def load(path):
    d = pd.read_csv(path)
    return d[~d.score.str.startswith("predictor")]


def pool(sub, cols):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def tests(W, WP, pairs, alpha, title, fname, datasets=DS):
    rows = []
    for a, b in pairs:
        for ds in datasets:
            if ds not in W.index.get_level_values(0) or a not in W or b not in W:
                continue
            dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
            if len(dif) == 0:
                continue
            rows.append(dict(a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                             p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                             pess_pts=100 * (WP.loc[ds][a] - WP.loc[ds][b]).mean() if WP is not None else np.nan))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t["sig"] = np.where(t.p_holm < alpha, np.where(t.mean_pts < 0, "a better", "a worse"), "")
    say(f"\n== {title} (Holm over {len(t)}, alpha = {alpha})")
    piv = t.assign(cell=lambda r: r.mean_pts.map("{:+.2f}".format) + " (" + r.wins_a.astype(str) + "/" +
                   r.n.astype(str) + ", p=" + r.p_holm.map("{:.1g}".format) + ")").pivot(
        index=["a", "b"], columns="dataset", values="cell")
    say(piv.reindex(columns=[d for d in datasets if d in piv.columns]).to_string())
    t.to_csv(out / fname, index=False)
    return t


def means(x, col, scores):
    s = (x.groupby(["dataset", "score"]).mean() * 100)[col].unstack("dataset")
    return s.reindex(index=[c for c in scores if c in s.index], columns=[d for d in DS if d in s.columns])


def runs_per(d):
    return d.groupby("dataset").apply(lambda g: g.groupby(["model", "seed", "fold"]).ngroups,
                                      include_groups=False).to_dict()


if mode == "select":
    d = load(src)
    assert d.seed.between(42, 61).all()
    say("runs per dataset:", runs_per(d))
    x = pool(d[d.scenario.isin(FA)], M)
    W, WP = x.risk_matched.unstack("score"), x.pess_matched.unstack("score")
    if len(args) > 1:
        d10 = load(args[1])
        x10 = pool(d10[d10.scenario.isin(FA)], ["risk_matched"]).risk_matched.unstack("score")
        say("consistency vs EXT10 (max |diff| points):")
        for sc in ("full", "gauss_bias_aware", "full_recon_rpt"):
            say(f"  {sc:18s}", (100 * (W[sc] - x10[sc])).abs().groupby("dataset").max().reindex(DS).round(4).to_dict())
    mu = W.groupby("dataset").mean() * 100
    mp = WP.groupby("dataset").mean() * 100
    best = mu[["full"] + DIRECTIONAL].min(axis=1)
    say("\n== dev means (risk %, 40 pairs)")
    say(mu[["ensemble_std", "full"] + DIRECTIONAL + CANDS].T.reindex(columns=DS).round(2).to_string())
    rows = []
    for c in CANDS:
        dfull = mu[c] - mu["full"]
        dbest = mu[c] - best
        pess = mp[c] - mp["full"]
        safe = bool((dfull <= 0.10).all() and (pess <= 1.0).all())
        rows.append(dict(cand=c, **{f"vsfull_{k}": v for k, v in dfull.items()},
                         **{f"vsbest_{k}": v for k, v in dbest.items()}, sum_vsbest=dbest.sum(),
                         max_pess=pess.max(), safe=safe))
    t = pd.DataFrame(rows)
    say("\n== selection table (points; vsbest = minus min(full, gauss_bias_aware, err_reg) per dataset)")
    say(t.round(3).to_string(index=False))
    s = t[t.safe].copy()
    if s.empty:
        say("\nNO SAFE CANDIDATE -> no round-2 confirmation")
    else:
        top = s.sum_vsbest.min()
        tied = s[s.sum_vsbest <= top + 0.02]
        final = min(tied.cand, key=CANDS.index)
        say(f"\nSELECTED FINAL: {final}  (sum_vsbest={s.set_index('cand').loc[final, 'sum_vsbest']:.3f}; "
            f"best={s.set_index('cand').sum_vsbest.idxmin()} {top:.3f}; tie band 0.02 -> simplest)")
    t.to_csv(out / "select_table.csv", index=False)

elif mode == "confirm":
    final, alpha = args[1], 0.0125
    d = load(src)
    assert d.seed.between(72, 81).all()
    say("runs per dataset:", runs_per(d))
    x = pool(d[d.scenario.isin(FA)], M)
    W, WP = x.risk_matched.unstack("score"), x.pess_matched.unstack("score")
    t = tests(W, WP, [(final, b) for b in ["full"] + DIRECTIONAL], alpha,
              f"CONFIRMATORY round 2: {final} vs full / directional", f"confirm12_{final}.csv")
    tf = t[t.b == "full"]
    better = ((tf.mean_pts < 0) & (tf.p_holm < alpha)).any()
    worse = ((tf.mean_pts > 0) & (tf.p_holm < alpha)).any()
    ok = bool(better and not worse and (tf.mean_pts <= 0.3).all() and (tf.pess_pts <= 1.0).all())
    say(f"\nCONFIRMED (vs full, EXT9 rule): {ok}  better={better} worse={worse} max_mean={tf.mean_pts.max():.3f} "
        f"max_pess={tf.pess_pts.max():.3f}")
    both = [ds for ds in DS if all(((t.a == final) & (t.b == b) & (t.dataset == ds) & (t.sig == "a better")).any()
                                   for b in DIRECTIONAL)]
    say(f"secondary: datasets where {final} is significantly better than BOTH directional estimators: {both}")
    allsc = ["ensemble_std", "full"] + DIRECTIONAL + CANDS
    for col in M:
        say(f"\n== {col} (%, six unseen faults, seeds 72-81, descriptive)")
        say(means(x, col, allsc).round(2).to_string())
    xc = pool(d[d.scenario == "clean"], ["risk_matched"])
    say("\n== clean risk_matched (%)")
    say(means(xc, "risk_matched", allsc).round(2).to_string())
    pf = pool(d[d.scenario.isin(FA)].assign(dataset=lambda q: q.dataset + "|" + q.scenario), ["risk_matched"])
    pm = pf.groupby(["dataset", "score"]).mean().risk_matched.unstack("score") * 100
    dd = pm[final] - pm["full"]
    dd.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in dd.index])
    say(f"\n== per fault: {final} - full (points)")
    say(dd.unstack(0).reindex(index=FA, columns=DS).round(2).to_string())
    (x.groupby(["dataset", "score"]).mean() * 100).reset_index().to_csv(out / "summary_conf12.csv", index=False)

elif mode == "stress":
    final = args[1]
    d = load(src)
    base = ["ensemble_std", "full"] + DIRECTIONAL + ["full_recon_rpt"]
    base = [b for b in base if b != final]
    xh = pool(d[d.scenario.isin(HELD)], ["risk_matched", "pess_matched"])
    say("== S1: four held-out faults (clip, lag, quant, invert), mean risk %")
    say(means(xh, "risk_matched", base + [final]).round(2).to_string())
    tests(xh.risk_matched.unstack("score"), xh.pess_matched.unstack("score"), [(final, b) for b in base], 0.05,
          f"S1 held-out faults: {final} - baseline", "stress_S1.csv")
    for h in HELD:
        xx = pool(d[d.scenario == h], ["risk_matched"])
        say(f"\n-- {h}: {final} - baseline (points)")
        mm = means(xx, "risk_matched", base + [final])
        say((mm.loc[final] - mm.drop(final)).round(2).to_string())
    s2 = [s for s in d.scenario.unique() if ("@" in s or "#" in s)]
    rows = []
    for scn in sorted(s2):
        xx = pool(d[d.scenario == scn], ["risk_matched"])
        W = xx.risk_matched.unstack("score")
        for b in base:
            for ds in DS:
                dif = (W.loc[ds][final] - W.loc[ds][b]).dropna()
                rows.append(dict(scenario=scn, b=b, dataset=ds, mean_pts=100 * dif.mean(), wins=int((dif < 0).sum()),
                                 n=len(dif), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== S2 severity / channel count: {final} - baseline (points; * Holm over {len(t)} < 0.05)")
    t["cell"] = t.mean_pts.map("{:+.2f}".format) + np.where(t.p_holm < 0.05, "*", " ")
    for b in base:
        say(f"-- vs {b}")
        say(t[t.b == b].pivot(index="scenario", columns="dataset", values="cell").reindex(columns=DS).to_string())
    t.to_csv(out / "stress_S2.csv", index=False)

elif mode == "robust":
    rob, final = Path(args[1]), args[2]
    d0, d1 = load(src), load(rob)
    d0 = d0[d0.score.isin([final, "full", "ensemble_std"])].assign(score=lambda q: "orig:" + q.score)
    d1 = d1[d1.score.isin([final, "full", "ensemble_std"])].assign(score=lambda q: "robust:" + q.score)
    d = pd.concat([d0, d1], ignore_index=True)
    x = pool(d[d.scenario.isin(FA)], M)
    sys_ = [f"orig:{final}", "orig:full", "orig:ensemble_std", "robust:ensemble_std", "robust:full", f"robust:{final}"]
    say("== system-level risk @90% matched (%, six unseen faults)")
    say(means(x, "risk_matched", sys_).round(2).to_string())
    say("\n== pessimistic (%)")
    say(means(x, "pess_matched", sys_).round(2).to_string())
    tests(x.risk_matched.unstack("score"), x.pess_matched.unstack("score"),
          [(f"orig:{final}", b) for b in ["robust:ensemble_std", "robust:full", f"robust:{final}"]], 0.05,
          "R: original predictor + FINAL vs robust-predictor systems", "robust_tests.csv")
    p0 = pd.concat([pd.read_csv(src).assign(sysn="orig"), pd.read_csv(rob).assign(sysn="robust")])
    p0 = p0[p0.score == "predictor"].assign(score=lambda q: q.sysn)
    for scs, nm in ((["clean"], "clean"), (FA, "six faults")):
        xx = pool(p0[p0.scenario.isin(scs)], ["rmse_full", "opt_full"])
        say(f"\n== predictor alone on {nm}: RMSE / optimistic rate at full coverage")
        say((xx.groupby(["dataset", "score"]).mean()).assign(opt_full=lambda q: 100 * q.opt_full).round(3)
            .unstack("score").reindex(DS).to_string())

elif mode == "backbone":
    final = args[1]
    d = load(src)
    say("runs per dataset:", runs_per(d))
    x = pool(d[d.scenario.isin(FA)], M)
    say(means(x, "risk_matched", ["ensemble_std", "full"] + DIRECTIONAL + ["full_recon_rpt", final]).round(2).to_string())
    tests(x.risk_matched.unstack("score"), x.pess_matched.unstack("score"),
          [(final, b) for b in ["full"] + DIRECTIONAL], 0.05, f"B: transformer backbone, {final} vs baselines",
          "backbone_tests.csv")

(out / f"{mode}_output.txt").write_text("\n".join(lines) + "\n")
print(f"EXT12_{mode.upper()}_DONE")
