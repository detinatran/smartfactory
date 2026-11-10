"""Fig. 1 -> outputs/figures/mechanism_example.pdf. Mechanism illustration on real data: N-CMAPSS test unit 7, GRU seed 42, clean vs 'bias' fault (same windows)."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4})
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
c = pd.read_csv(ROOT / "data/fig1/predictions_clean.csv")
f = pd.read_csv(ROOT / "data/fig1/predictions_bias.csv")
u = 7
mc, mf = c[c.unit == u].reset_index(drop=True), f[f.unit == u].reset_index(drop=True)
x = mc.cycle.to_numpy()
hit = np.abs(mf.prediction - mc.prediction).to_numpy() > 1e-6
fig, ax = plt.subplots(1, 2, figsize=(4.8, 1.6))
for a in ax:
    if hit.any():
        a.axvspan(x[hit].min(), x[hit].max(), color="0.9", lw=0)
    a.grid(alpha=0.3)
    a.set_xlabel("Flight cycle", labelpad=1)
a = ax[0]
a.plot(x, mc.rul_raw, "k-", lw=1.3, label="True RUL")
for d, col, lab in ((mc, "C0", "Clean pred."), (mf, "C3", "Faulty pred.")):
    a.plot(x, d.prediction, color=col, lw=1.3, label=lab)
    a.fill_between(x, d.prediction - d.score_ensemble_std, d.prediction + d.score_ensemble_std, color=col, alpha=0.25, lw=0)
opt = (mf.prediction - mf.rul_raw).to_numpy() > 0.16
a.fill_between(x, mf.rul_raw, mf.prediction, where=opt, color="C3", alpha=0.12, lw=0, hatch="////", label="Optimistic")
a.set_ylabel("Remaining life")
a.set_ylim(-0.05, 1.75)
a.set_yticks([0, 0.5, 1])
a.legend(fontsize=7, frameon=False, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)
a.set_title("(a) prediction (band: $\\pm$ spread)", fontsize=9, pad=2)
a = ax[1]
a.plot(x, mc.score_asymmetric_risk, color="C0", lw=1.3, label="clean")
a.plot(x, mf.score_asymmetric_risk, color="C3", lw=1.3, label="faulty")
a.set_ylim(-0.03, 1.05)
a.set_ylabel("$\\hat\\eta(x)$")
a.legend(fontsize=7.5, frameon=False, loc="upper right")
a.set_title("(b) one-sided score", fontsize=9, pad=2)
fig.tight_layout(w_pad=0.8, pad=0.2)
fig.savefig(ROOT / "paper/outputs/figures/mechanism_example.pdf", bbox_inches="tight", pad_inches=0.01)
s = mf.prediction - mc.prediction
print({"max_shift": round(float(s.max()), 3), "spread_clean_mean_hit": round(float(mc.score_ensemble_std[hit].mean()), 3),
       "spread_fault_mean_hit": round(float(mf.score_ensemble_std[hit].mean()), 3),
       "eta_clean_mean_hit": round(float(mc.score_asymmetric_risk[hit].mean()), 3),
       "eta_fault_mean_hit": round(float(mf.score_asymmetric_risk[hit].mean()), 3), "n_hit": int(hit.sum())})
