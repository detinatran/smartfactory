"""EXT24b analysis (EXT24_PROTOCOL.md, EXT24b). usage: ext24b_summary.py ALL_EXT24B.csv.gz OUTDIR
Descriptive per family x pool x policy x pi; D4: joint90[rpt_auto] - offset_only (CP, pi=0.3) on risk and RMSE ratio,
Holm over 10; joint90[rpt_auto] - joint90[full] descriptive. Exploratory."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "ncmapss", "bearings", "battery", "hdd"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
POL = ["offset_only"] + [f"joint90[{m}]" for m in ("ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_auto")]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
x = d[d.kind == "deploy"].copy()
x["violation"] = x.violation.astype(str).str.lower().eq("true").astype(float)
x["w"] = np.where(x.dataset == "bearings", x.fold.map(BW), 1.0)


def wmean(g, col):
    v = g[col].to_numpy(float)
    m = ~np.isnan(v)
    return np.average(v[m], weights=g.w.to_numpy()[m]) if m.any() else np.nan


desc = x.groupby(["dataset", "pool", "policy", "pi"]).apply(lambda g: pd.Series({
    "cov": 100 * wmean(g, "coverage"), "risk": 100 * wmean(g, "risk"), "pess": 100 * wmean(g, "pess"),
    "rmse_ratio": wmean(g, "rmse_ratio"), "b": wmean(g, "b"), "viol": 100 * wmean(g, "violation"),
    "r_star": 100 * wmean(g, "r_star")}), include_groups=False).reset_index()
desc.to_csv(out / "ext24b_descriptive.csv", index=False)
for pk in ("C0", "CP"):
    for pi in (0.0, 0.3, 0.5):
        t = desc[(desc.pool == pk) & (desc.pi == pi)]
        cell = t.assign(c=t["cov"].map("{:.0f}".format) + "|" + t.risk.map("{:.2f}".format) + "|" +
                        t.rmse_ratio.map("{:.2f}".format) + "|b" + t.b.map("{:.2f}".format) + "|v" + t.viol.map("{:.0f}".format))
        say(f"\n== pool={pk}, pi={pi}: coverage% | risk% | RMSE ratio | b/delta | violation%")
        say(cell.pivot_table(index="policy", columns="dataset", values="c", aggfunc="first").reindex(index=POL, columns=DS).to_string())


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


s = x[(x.pool == "CP") & (x.pi == 0.3)]
P = {col: s.groupby(["dataset", "policy", "model", "seed"]).apply(lambda h: wmean(h, col), include_groups=False)
     for col in ("risk", "rmse_ratio", "coverage")}


def comp(a, b, cols, title, fname, do_holm=True):
    rows = []
    for col in cols:
        for ds in DS:
            try:
                dif = (P[col].loc[(ds, a)] - P[col].loc[(ds, b)]).dropna()
            except KeyError:
                continue
            k = 100 if col in ("risk", "coverage") else 1
            rows.append(dict(metric=col, dataset=ds, mean=k * dif.mean(), n=len(dif), neg=int((dif < 0).sum()),
                             p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p) if do_holm else np.nan
    say(f"\n== {title}")
    say(t.round(4).to_string(index=False))
    t.to_csv(out / fname, index=False)


comp("joint90[rpt_auto]", "offset_only", ["risk", "rmse_ratio"], "D4: joint90[rpt_auto] - offset_only (CP, pi=0.3); Holm over 10", "ext24b_D4.csv")
comp("joint90[rpt_auto]", "joint90[full]", ["risk", "rmse_ratio", "coverage"], "descriptive: joint90[rpt_auto] - joint90[full] (CP, pi=0.3)", "ext24b_vs_full.csv", False)
(out / "ext24b_output.txt").write_text("\n".join(lines) + "\n")
print("EXT24B_SUMMARY_DONE")
