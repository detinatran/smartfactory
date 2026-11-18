"""EXT16/EXT17 analyses. Pooling as conf_test.py; two-sided Wilcoxon over (backbone, seed) pairs; Holm within family.
  ext16_summary.py heads  EXT12_CONF EXT14_SOTA EXT16_HEADS OUTDIR
  ext16_summary.py errreg EXT16_ERRREG OUTDIR
  ext16_summary.py m5     EXT16_M5EVAL OUTDIR
  ext16_summary.py raw    EXT17_RAW OUTDIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

mode, args = sys.argv[1], sys.argv[2:]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
RAW = ["raw_gain", "raw_clip", "raw_detach", "raw_noise", "raw_spike"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
F = "rpt_L"
cols = ["dataset", "model", "seed", "fold", "scenario", "score", "risk_matched", "pess_matched"]
lines = []


def say(s):
    print(s)
    lines.append(str(s))


def load(p):
    d = pd.read_csv(p)
    return d[~d.score.str.startswith("predictor")][[c for c in cols if c in d.columns]]


def pool(sub, col="risk_matched"):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
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


def family(W, comps, title, fname, out, datasets=DS):
    rows = []
    for b in comps:
        for ds in datasets:
            if ds not in W.index.get_level_values(0) or b not in W:
                continue
            dif = (W.loc[ds][F] - W.loc[ds][b]).dropna()
            if len(dif) == 0:
                continue
            rows.append(dict(b=b, dataset=ds, mean=dif.mean(), wins=int((dif < 0).sum()), n=len(dif),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    say(f"\n== {title} ({F} - baseline, points; Holm over {len(t)})")
    say(t.assign(c=lambda r: r["mean"].map("{:+.2f}".format) + " (" + r.wins.astype(str) + "/" + r.n.astype(str) +
                 ", p=" + r.p_holm.map("{:.1g}".format) + ")").pivot(index="b", columns="dataset", values="c")
        .reindex(index=comps, columns=[d for d in datasets if d in t.dataset.unique()]).to_string())
    t.to_csv(out / fname, index=False)
    return t


if mode == "heads":
    fc, fs, fh, out = map(Path, args)
    d = pd.concat([load(fc), load(fs), load(fh)], ignore_index=True)
    W = pool(d[d.scenario.isin(FA)])
    sc = [F, "deep_ens_nll", "deep_ens_nll_aug", "evidential", "evidential_aug", "cqr_lower", "cqr_lower_aug",
          "selectivenet", "selectivenet_aug"]
    say("== risk (%), six faults, seeds 72-81")
    say(W.groupby("dataset").mean()[sc].T.reindex(columns=DS).round(2).to_string())
    family(W, ["deep_ens_nll_aug", "evidential_aug", "cqr_lower_aug", "selectivenet_aug"],
           "A. head baselines trained with perturbed data", "ext16_heads.csv", out)
elif mode == "errreg":
    fe, out = map(Path, args)
    d = load(fe)
    W = pool(d[d.scenario.isin(FA)])
    sc = [F, "full", "err_reg", "err_reg_l1", "err_reg_clip", "err_reg_rptL", "err_reg_l1_rptL", "err_reg_clip_rptL"]
    m = W.groupby("dataset").mean()
    say("== risk (%), six faults, seeds 72-81")
    say(m[sc].T.reindex(columns=DS).round(2).to_string())
    variants = ["err_reg", "err_reg_l1", "err_reg_clip"]
    best = {ds: m.loc[ds, variants].idxmin() for ds in DS}
    say(f"strongest error-regression variant per dataset (conservative rule): {best}")
    W2 = W.copy()
    W2["err_reg_best"] = np.nan
    for ds in DS:
        W2.loc[ds, "err_reg_best"] = W.loc[ds, best[ds]].to_numpy()
    family(W2, ["err_reg_best"], "C. RPT vs strongest error-regression variant", "ext16_errreg.csv", out)
    family(W, ["err_reg_l1_rptL", "err_reg_clip_rptL"], "C2. error-regression variants with perturbed data",
           "ext16_errreg_rpt.csv", out)
elif mode == "m5":
    fm, out = map(Path, args)
    d = load(fm)
    W = pool(d[d.scenario.isin(FA)])
    sc = ["ensemble_std", "full", "gauss_bias_aware", "err_reg", F]
    say("== M=5 ensembles: risk (%), six faults, seeds 72-81")
    say(W.groupby("dataset").mean()[sc].T.reindex(columns=DS).round(2).to_string())
    family(W, ["ensemble_std", "full", "gauss_bias_aware", "err_reg"], "B. M=5", "ext16_m5.csv", out)
elif mode == "raw":
    fr, out = map(Path, args)
    d = load(fr)
    W = pool(d[d.scenario.isin(RAW)])
    Wc = pool(d[d.scenario == "clean"])
    sc = ["ensemble_std", "full", "gauss_bias_aware", "err_reg", F]
    say("== EXT17 bearings, physical raw-signal faults: risk (%), mean of 5 kinds; clean; excess")
    m, mc = W.groupby("dataset").mean(), Wc.groupby("dataset").mean()
    say(pd.DataFrame({"raw": m.loc["bearings", sc], "clean": mc.loc["bearings", sc],
                      "excess": m.loc["bearings", sc] - mc.loc["bearings", sc]}).round(2).to_string())
    family(W, ["ensemble_std", "full", "gauss_bias_aware", "err_reg"], "EXT17 raw faults", "ext17_raw.csv", out,
           datasets=["bearings"])
    ex = (W - Wc)
    dif = (ex.loc["bearings"][F] - ex.loc["bearings"]["full"]).dropna()
    say(f"excess(rpt_L) - excess(full): {dif.mean():+.2f} ({(dif < 0).sum()}/{len(dif)}, p={stats.wilcoxon(dif).pvalue:.2g})")
    for k in RAW:
        w = pool(d[d.scenario == k]).groupby("dataset").mean()
        say(f"-- {k}: " + "  ".join(f"{s}={w.loc['bearings', s]:.2f}" for s in sc) +
            f"  | rpt_L-full={w.loc['bearings', F] - w.loc['bearings', 'full']:+.2f}")
(out / f"ext16_{mode}_output.txt").write_text("\n".join(lines) + "\n")
print("DONE")
