"""EXT14 analysis (EXT14_PROTOCOL.md): rpt_L vs 17 baselines on seeds 72-81, Holm over 68; writes a LaTeX table.
usage: ext14_summary.py EXT12_CONF.csv.gz EXT14_SOTA.csv.gz EXT14_MORE.csv.gz EXT14_ABLATE.csv.gz OUTDIR TABLE.tex"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

fc, fs, fm, fa, out, tex = map(Path, sys.argv[1:7])
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
M = ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]
cols = ["dataset", "model", "seed", "fold", "scenario", "score"] + M
FINAL = "rpt_L"
ROWS = [  # (group, label, score, cite)
    ("Dispersion", "Ensemble spread", "ensemble_std", "lakshminarayanan2017simple"),
    ("", "MC dropout", "mc_dropout", "gal2016dropout"),
    ("", "Deep ensemble, Gaussian NLL", "deep_ens_nll", "lakshminarayanan2017simple"),
    ("", "Deep evidential regression", "evidential", "amini2020deep"),
    ("", "Last-layer Laplace", "laplace", "daxberger2021laplace"),
    ("", "Test-time augmentation", "tta_std", "wang2019aleatoric"),
    ("OOD / anomaly", "Mahalanobis distance", "mahalanobis", "lee2018simple"),
    ("", "$k$-NN feature distance", "knn_feat", "sun2022out"),
    ("", "Isolation forest", "iforest", "liu2008isolation"),
    ("", "Analytic-redundancy residual", "ar_residual", "chow1984analytical"),
    ("Selective / interval", "SelectiveNet", "selectivenet", "geifman2019selectivenet"),
    ("", "CQR lower bound", "cqr_lower", "romano2019cqr"),
    ("Learned event / error", "Symmetric classifier, $|E|>\\delta$", "symmetric", "corbiere2019addressing,kamath2020selective"),
    ("", "Fault detector", "fault_det", "isermann2005model"),
    ("", "Signed-error regression", "err_reg", "yoo2019learning"),
    ("", "Gaussian with learned bias", "gauss_bias_aware", "nix1994estimating"),
    ("Augmentation", "Sparse white noise (round 1, recon. feat., ours)", "ours_white", "bishop1995training"),
]
lines = []


def say(s):
    print(s)
    lines.append(str(s))


dc = pd.read_csv(fc)[cols]
da = pd.read_csv(fa)[cols]
d = pd.concat([dc[dc.score.isin([FINAL, "full", "ensemble_std", "gauss_bias_aware", "err_reg"])],
               pd.read_csv(fs)[cols], pd.read_csv(fm)[cols],
               da[da.score.isin(["symmetric", "fault_det", "ours_white"])]], ignore_index=True)
assert d.seed.between(72, 81).all()


def pool(sub, cs):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[cs].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    res = {}
    for col in cs:
        gg = g[g[col].notna()].assign(_v=lambda t: t[col] * t._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


# consistency: ext11_ablate recomputation vs ext12_conf
xa = pool(da[da.scenario.isin(FA)], ["risk_matched"]).risk_matched.unstack("score")
xc = pool(dc[dc.scenario.isin(FA)], ["risk_matched"]).risk_matched.unstack("score")
say("consistency (ext14 ablate vs ext12 conf, max |diff| points):")
for sc in ("full", "err_reg", "full_recon_rpt"):
    say(f"  {sc:16s} {(100 * (xa[sc] - xc[sc])).abs().groupby('dataset').max().reindex(DS).round(4).to_dict()}")

x = pool(d[d.scenario.isin(FA)], M)
mean = x.groupby(["dataset", "score"]).mean() * 100
W, WP = x.risk_matched.unstack("score"), x.pess_matched.unstack("score")
rows = []
for _, _, b, _ in ROWS:
    for ds in DS:
        dif = (W.loc[ds][FINAL] - W.loc[ds][b]).dropna()
        rows.append(dict(b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins=int((dif < 0).sum()),
                         p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0,
                         pess_pts=100 * (WP.loc[ds][FINAL] - WP.loc[ds][b]).mean()))
t = pd.DataFrame(rows)
adj, run = np.empty(len(t)), 0.0
for r, i in enumerate(np.argsort(t.p.to_numpy())):
    run = max(run, (len(t) - r) * t.p.iloc[i])
    adj[i] = min(1.0, run)
t["p_holm"] = adj
t["sig"] = (t.p_holm < 0.05) & (t.mean_pts < 0)
t["worse"] = (t.p_holm < 0.05) & (t.mean_pts > 0)
t.to_csv(out / "familyEXT14.csv", index=False)
say(f"\n== {FINAL} - baseline, Holm over {len(t)}")
say(t.assign(cell=lambda r: r.mean_pts.map("{:+.2f}".format) + " (" + r.wins.astype(str) + ", p=" +
             r.p_holm.map("{:.1g}".format) + ")").pivot(index="b", columns="dataset", values="cell")
    .reindex(index=[r[2] for r in ROWS], columns=DS).to_string())
say(f"\nsignificantly better: {int(t.sig.sum())}/{len(t)}; significantly worse: {int(t.worse.sum())}; "
    f"wins range {t.wins.min()}-{t.wins.max()}; diff range {t.mean_pts.min():.2f} to {t.mean_pts.max():.2f}")
for col in M:
    say(f"\n== {col}")
    say(mean[col].unstack("dataset").reindex(index=[r[2] for r in ROWS] + ["full", FINAL], columns=DS).round(2)
        .to_string())
xcl = pool(d[d.scenario == "clean"], ["risk_matched"])
say("\n== clean risk_matched")
say((xcl.groupby(["dataset", "score"]).mean() * 100).risk_matched.unstack("dataset")
    .reindex(index=[r[2] for r in ROWS] + ["full", FINAL], columns=DS).round(2).to_string())

# LaTeX
R = mean.risk_matched.unstack("dataset")
best = {ds: R.loc[[r[2] for r in ROWS] + ["full", FINAL], ds].min() for ds in DS}
tl = ["\\begin{table*}[t]",
      "\\caption{Comparison with sixteen external baselines and one variant of our method on fresh seeds 72--81 (240 ensembles). Risk (\\%) at 90\\% matched "
      "coverage, averaged over six unseen faults; in parentheses, \\RPT{} minus the baseline in points (negative: \\RPT{} "
      "better). \\s: significant after Holm over all 68 tests ($\\alpha=0.05$). All selectors rank the same ensemble "
      "predictions. Bold: lowest risk in the column.}",
      "\\label{tab:sota}", "\\centering\\small", "\\begin{tabular}{llllll}", "\\toprule",
      "Group & Baseline & C-MAPSS & N-CMAPSS & Bearings & Batteries\\\\", "\\midrule"]


def cell(v, ds, diff=None, sig=False):
    txt = f"{v:.2f}"
    if abs(v - best[ds]) < 1e-9:
        txt = f"\\textbf{{{txt}}}"
    if diff is not None:
        txt += f" (${diff:+.2f}$" + ("\\s" if sig else "") + ")"
    return txt.replace("+-", "-")


for grp, lab, sc, cite in ROWS:
    cells = []
    for ds in DS:
        rr = t[(t.b == sc) & (t.dataset == ds)].iloc[0]
        cells.append(cell(R.loc[sc, ds], ds, rr.mean_pts, bool(rr.sig)))
    tl.append(f"{grp} & {lab}~\\cite{{{cite}}} & " + " & ".join(cells) + "\\\\")
tl.append(" & \\OSC{} (one-sided classifier, ours) & " + " & ".join(cell(R.loc["full", ds], ds) for ds in DS)
          + "\\\\")
tl += ["\\midrule", " & \\RPT{} (ours) & " + " & ".join(cell(R.loc[FINAL, ds], ds) for ds in DS) + "\\\\",
       "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
Path(tex).write_text("\n".join(tl) + "\n")
mean.reset_index().to_csv(out / "summary_ext14.csv", index=False)
(out / "summary_output.txt").write_text("\n".join(lines) + "\n")
print("EXT14_SUMMARY_DONE")
