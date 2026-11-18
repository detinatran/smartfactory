"""Figure: RPT - OSC (points) against the number of selector units, all families (EXT43 benchmarks, EXT37 hard drives).
usage: ext43_figure.py ext43_lc.csv ALL_EXT37.csv.gz OUTDIR"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lc43, f37, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
FAM = [("cmapss", "C-MAPSS", "#2a78d6", "o"), ("ncmapss_cv", "N-CMAPSS", "#eb6834", "s"),
       ("bearings", "Bearings", "#1baf7a", "^"), ("battery_cv", "Batteries", "#eda100", "D")]
t = pd.read_csv(lc43)
d = pd.read_csv(f37)
h = d[(d.kind == "lc") & d.dataset.str.startswith("hdd")].copy()
h["gain"] = (h.risk_rpt - h.risk_full) * 100
ALL = {"hdd": 296, "hdd2": 160, "hdd3": 96}
fig, ax = plt.subplots(figsize=(3.35, 2.2))
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for key, name, col, mk in FAM:
    x = t[t.dataset == key].sort_values("n_units")
    ax.plot(x.n_units, x["mean"], color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=0.8, zorder=3, label=name)
grid_n = [15, 30, 60, 120]
hm = h[h.n_sel.isin(grid_n)].groupby("n_sel").gain.mean()
ax.plot(hm.index, hm.values, color="#e87ba4", lw=2, marker="v", ms=4.5, mec="white", mew=0.8, zorder=3, label="Hard drives")
for ds, n in ALL.items():
    v = h[(h.dataset == ds) & (h.n_sel >= n - 1) & ~h.n_sel.isin(grid_n)].gain.mean()
    ax.plot([n], [v], color="#e87ba4", marker="v", ms=4.5, mec="white", mew=0.8, ls="none", zorder=3)
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50, 100, 300])
ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "300"])
ax.set_xlabel("Selector units per deployment (log scale)")
ax.set_ylabel("ELPIS $-$ OSC (points)")
ax.legend(frameon=False, fontsize=6, loc="lower right", ncol=1, handlelength=1.6)
fig.savefig(out / "learning_curve.pdf", bbox_inches="tight")
fig.savefig(out / "learning_curve.png", dpi=220, bbox_inches="tight")
print(hm.round(3).to_dict())
