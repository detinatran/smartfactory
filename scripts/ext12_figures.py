"""Paper figures from EXT12 (seeds 72-81): risk-coverage curves and fault-severity curves.
usage: ext12_figures.py all_ext12conf.csv.gz OUTDIR"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
TITLE = {"cmapss": "C-MAPSS", "ncmapss": "N-CMAPSS", "bearings": "Bearings", "battery": "Batteries"}
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
# fixed categorical order (validated palette, light mode); ours = slot 1
S = [("rpt_L", "ELPIS (ours)", "#2a78d6", "o"), ("full", "OSC", "#eb6834", "s"),
     ("gauss_bias_aware", "Gaussian w/ bias", "#1baf7a", "^"), ("err_reg", "Signed-error regr.", "#eda100", "D"),
     ("ensemble_std", "Ensemble spread", "#e87ba4", "v")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif"})
d = pd.read_csv(src)
d = d[d.score.isin([s[0] for s in S])]


def pairs(sub, col):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[col].mean().reset_index()
    g["_w"] = np.where(g.dataset.str.startswith("bearings"), g.fold.map(BW), 1.0)
    g["_v"] = g[col] * g._w
    s = g.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
    return (s._v / s._w) * 100


def pooled(sub, col):
    return pairs(sub, col).groupby(["dataset", "score"]).mean()


def se(sub, col):
    x = pairs(sub, col).groupby(["dataset", "score"])
    return x.std() / np.sqrt(x.count())


# Figure: risk-coverage
covs = [50, 60, 70, 80, 85, 90, 95]
cols = {c: (f"risk_c{c}" if c != 90 else "risk_matched") for c in covs}
fa = d[d.scenario.isin(FA)]
curves = {c: pooled(fa, col) for c, col in cols.items()}
ses = {c: se(fa, col) for c, col in cols.items()}
P = {c: pairs(fa, col) for c, col in cols.items()}
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3), sharex=True)
for j, ds in enumerate(DS):
    ax = axes[0][j]
    for key, lab, colr, mk in S:
        y = [curves[c].loc[(ds, key)] for c in covs]
        ax.plot(covs, y, color=colr, lw=2 if key == "rpt_L" else 1.2, marker=mk, ms=3.2, label=lab,
                zorder=3 if key == "rpt_L" else 2, markeredgecolor="#fcfcfb", markeredgewidth=0.6)
    ax.set_title(TITLE[ds], fontsize=7.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[1][j]
    ax.axhline(0, color=MUTED, lw=0.8)
    for key, lab, colr, mk in S:
        if key == "full":
            continue
        diffs = [(P[c].loc[ds].unstack("score")[key] - P[c].loc[ds].unstack("score")["full"]) for c in covs]
        m = np.array([x.mean() for x in diffs])
        e = np.array([1.96 * x.std() / np.sqrt(x.count()) for x in diffs])
        if key == "ensemble_std" and ds == "ncmapss":
            continue  # off scale (+6 to +8 points); see Table 3
        ax.fill_between(covs, m - e, m + e, color=colr, alpha=0.18, lw=0)
        ax.plot(covs, m, color=colr, lw=2 if key == "rpt_L" else 1.2, marker=mk, ms=3.2,
                zorder=3 if key == "rpt_L" else 2, markeredgecolor="#fcfcfb", markeredgewidth=0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlabel("Coverage (%)")
    ax.set_xticks([50, 70, 90])
axes[0][0].set_ylabel("Optimistic-error rate (%)")
axes[1][0].set_ylabel("Paired difference\nto OSC (points)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7)
fig.tight_layout(h_pad=0.5, w_pad=0.8)
fig.savefig(out / "risk_coverage.pdf", bbox_inches="tight")
fig.savefig(out / "risk_coverage.png", dpi=200, bbox_inches="tight")

# Figure: severity (bias and noise magnitude)
sev = {"Bias": [("bias@1.5", 1.5), ("bias", 3), ("bias@6", 6)], "Noise": [("noise@1", 1), ("noise", 2), ("noise@4", 4)]}
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.1), sharex="row")
for r, (fam, pts) in enumerate(sev.items()):
    vals = {scn: pooled(d[d.scenario == scn], "risk_matched") for scn, _ in pts}
    sev_se = {scn: se(d[d.scenario == scn], "risk_matched") for scn, _ in pts}
    for ax, ds in zip(axes[r], DS):
        for key, lab, colr, mk in S:
            yy = np.array([vals[scn].loc[(ds, key)] for scn, _ in pts])
            ee = np.array([1.96 * sev_se[scn].loc[(ds, key)] for scn, _ in pts])
            ax.fill_between([m for _, m in pts], yy - ee, yy + ee, color=colr, alpha=0.15, lw=0)
            ax.plot([m for _, m in pts], [vals[scn].loc[(ds, key)] for scn, _ in pts], color=colr,
                    lw=2 if key == "rpt_L" else 1.2, marker=mk, ms=3.2, label=lab,
                    zorder=3 if key == "rpt_L" else 2, markeredgecolor="#fcfcfb", markeredgewidth=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([m for _, m in pts])
        ax.set_xticklabels([f"{m:g}" for _, m in pts])
        ax.grid(axis="y", color=GRID, lw=0.6)
        if r == 0:
            ax.set_title(TITLE[ds], fontsize=7.5, color=INK)
        ax.set_xlabel(f"{fam} magnitude ($\\sigma$)")
    axes[r][0].set_ylabel("Risk @90% (%)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7)
fig.tight_layout(h_pad=0.6, w_pad=0.8)
fig.savefig(out / "severity.pdf", bbox_inches="tight")
fig.savefig(out / "severity.png", dpi=200, bbox_inches="tight")
print("FIGURES_DONE")
