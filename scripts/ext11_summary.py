"""EXT11 analysis (EXT11_PROTOCOL.md).
usage: ext11_summary.py EXT10_all_conf62.csv.gz EXT11_ablate_all.csv.gz EXT11_sota_all.csv.gz OUTDIR
Pooling as conf_test.py: risk @90% matched coverage, mean over six unseen faults; C-MAPSS subsets averaged within a pair;
bearings folds unit-weighted. Two-sided Wilcoxon over (backbone, seed) pairs; Holm within each family.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

f10, fa, fs, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
OURS = "full_recon_rpt"
pd.set_option("display.width", 250)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


cols = ["dataset", "model", "seed", "fold", "scenario", "score"] + M
d10 = pd.read_csv(f10)[cols]
da = pd.read_csv(fa)[cols]
ds_ = pd.read_csv(fs)[cols]
for nm, dd in (("ext10", d10), ("ablate", da), ("sota", ds_)):
    assert dd.seed.between(62, 71).all(), nm
    say(f"{nm}: runs per dataset", dd.groupby("dataset").apply(lambda g: g.groupby(["model", "seed", "fold"]).ngroups,
                                                               include_groups=False).to_dict())


def pool(sub, cols):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cols].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    res = {}
    for col in cols:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


# consistency: recomputed scores must reproduce EXT10
x10 = pool(d10[d10.scenario.isin(FA)], ["risk_matched"]).risk_matched.unstack("score")
xa = pool(da[da.scenario.isin(FA)], ["risk_matched"]).risk_matched.unstack("score")
say("\n== consistency (EXT11 recomputed - EXT10, max |diff| in points per dataset)")
bad = False
for sc in ("full", "full_recon", "full_rpt", OURS):
    dif = (100 * (xa[sc] - x10[sc])).abs().groupby("dataset").max()
    say(f"  {sc:16s}", dif.reindex(DS).round(4).to_dict())
    bad |= bool((dif > 0.01).any())
say("CONSISTENCY_OK" if not bad else "CONSISTENCY_FAILED (>0.01 points): investigate before reporting")

keep10 = d10[d10.score.isin(["ensemble_std", "gauss_bias_aware"])]
allr = pd.concat([keep10, da, ds_], ignore_index=True)
allr = allr[~allr.score.str.startswith("predictor")]
x = pool(allr[allr.scenario.isin(FA)], M)
xc = pool(allr[allr.scenario == "clean"], ["risk_matched"])
mean = x.groupby(["dataset", "score"]).mean() * 100
mean.reset_index().to_csv(out / "summary_ext11.csv", index=False)
W = x.risk_matched.unstack("score")
WP = x.pess_matched.unstack("score")


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(pairs, title, fname, alpha=0.05):
    """pairs: list of (a, b); tests a - b per dataset."""
    rows = []
    for a, b in pairs:
        for ds in DS:
            dif = (W.loc[ds][a] - W.loc[ds][b]).dropna()
            rows.append(dict(a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                             p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                             pess_pts=100 * (WP.loc[ds][a] - WP.loc[ds][b]).mean()))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t["sig"] = np.where(t.p_holm < alpha, np.where(t.mean_pts < 0, "a better", "a worse"), "")
    say(f"\n== {title} (Holm over {len(t)} tests, alpha = {alpha})")
    piv = t.assign(cell=lambda r: r.mean_pts.map("{:+.2f}".format) + " (" + r.wins_a.astype(str) + "/" +
                   r.n.astype(str) + ", p=" + r.p_holm.map("{:.1g}".format) + ")").pivot(index=["a", "b"],
                                                                                       columns="dataset", values="cell")
    say(piv.reindex(columns=DS).to_string())
    t.to_csv(out / fname, index=False)
    return t


def table(scores, col="risk_matched", title=""):
    s = mean[col].unstack("dataset").reindex(index=scores, columns=DS)
    say(f"\n== {title} {col} (%, unseen faults)")
    say(s.round(2).to_string())


SOTA = ["ensemble_std", "mc_dropout", "deep_ens_nll", "evidential", "selectivenet", "cqr_lower", "symmetric",
        "fault_det", "err_reg", "gauss_bias_aware", "ours_white"]
ABL = ["ours_copy", "ours_nobias", "ours_noonset", "ours_amp_small", "ours_amp_large", "ours_k1", "ours_khalf",
       "ours_x2", "ours_white", "ours_gauss_all", "ours_noseen"]
CLF = ["lr", "rf", "mlp", "hgbdef"]

table(SOTA + ["full", OURS], title="A. SOTA")
table(SOTA + ["full", OURS], "pess_matched", "A. SOTA")
tA = family([(OURS, b) for b in SOTA], "A. OURS - baseline (primary EXT11 family)", "familyA_sota.csv")

table(["full", "full_recon", "full_rpt", OURS] + ABL + ["ours_oracle"], title="B. ablation")
tB = family([(a, OURS) for a in ABL], "B. ablation - OURS (exploratory)", "familyB_ablation.csv")
say("\n  oracle (upper bound, not tested):",
    (mean.risk_matched.unstack("dataset").loc["ours_oracle"] - mean.risk_matched.unstack("dataset").loc[OURS])
    .reindex(DS).round(2).to_dict())
say("  H-B1 (ours_copy worse than OURS):", tB[tB.a == "ours_copy"][["dataset", "mean_pts", "p_holm", "sig"]]
    .round(4).to_dict("records"))

table(["full"] + [f"osc_{c}" for c in CLF] + [OURS] + [f"ours_{c}" for c in CLF], title="C. classifier")
tC = family([(f"osc_{c}", "full") for c in CLF] + [(f"ours_{c}", f"osc_{c}") for c in CLF],
            "C. classifier swap (exploratory)", "familyC_classifier.csv")

# per fault, H-B2 and frozen
pf = pool(allr[allr.scenario.isin(FA)].assign(dataset=lambda t: t.dataset + "|" + t.scenario), ["risk_matched"])
pfm = pf.groupby(["dataset", "score"]).mean().risk_matched.unstack("score") * 100
say("\n== per fault: score - full (points)")
for sc in (OURS, "full_rpt", "ours_nobias", "ours_copy", "ours_white", "ours_oracle"):
    dd = (pfm[sc] - pfm["full"]).rename(index=lambda s: s)
    dd.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in dd.index], names=["dataset", "fault"])
    say(f"-- {sc}")
    say(dd.unstack("dataset").reindex(index=FA, columns=DS).round(2).to_string())
nb = pfm.copy()
nb.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in nb.index], names=["dataset", "fault"])
nob = nb[nb.index.get_level_values("fault") != "bias"].groupby("dataset").mean()
say("\n== H-B2: mean over the five non-bias faults, minus full_recon (points)")
for sc in ("ours_nobias", OURS):
    say(f"  {sc}:", (nob[sc] - nob["full_recon"]).reindex(DS).round(2).to_dict())

say("\n== clean risk_matched (%)")
say((xc.groupby(["dataset", "score"]).mean() * 100).risk_matched.unstack("dataset")
    .reindex(index=["full", OURS, "full_rpt"] + SOTA, columns=DS).round(2).to_string())
say("\n== frozen thresholds: risk / coverage (%)")
fr = mean[["risk_frozen", "coverage_frozen"]]
for sc in ["full", OURS] + SOTA:
    say(f"  {sc:16s}", "  ".join(f"{ds}: {fr.loc[(ds, sc), 'risk_frozen']:.2f}/{fr.loc[(ds, sc), 'coverage_frozen']:.1f}"
                                 for ds in DS if (ds, sc) in fr.index))
(out / "summary_output.txt").write_text("\n".join(lines) + "\n")
print("EXT11_SUMMARY_DONE")
