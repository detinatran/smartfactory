"""Supplementary S1: is Fig. 1 typical? Per-unit mechanism over all 39 N-CMAPSS test units (same run as Fig. 1:
GRU, seed 42, clean vs. bias fault, identical windows). Per unit, over windows the fault changes:
mean prediction shift, change of mean ensemble spread, change of mean one-sided score.
Outputs: outputs/stats/figS1_unit_mechanism.csv, outputs/figures/figS1_unit_distribution.{pdf,png}."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 8})
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
c = pd.read_csv(ROOT / "data/fig1/predictions_clean.csv")
f = pd.read_csv(ROOT / "data/fig1/predictions_bias.csv")
assert (c[["unit", "cycle"]].to_numpy() == f[["unit", "cycle"]].to_numpy()).all()
rows = []
for u in c.unit.unique():
    mc, mf = c[c.unit == u].reset_index(drop=True), f[f.unit == u].reset_index(drop=True)
    hit = (np.abs(mf.prediction - mc.prediction) > 1e-6).to_numpy()
    new_opt = ((mf.prediction - mf.rul_raw > 0.16) & ~(mc.prediction - mc.rul_raw > 0.16)).to_numpy() & hit
    rows.append(dict(unit=u, n_fault_windows=int(hit.sum()), new_optimistic=int(new_opt.sum()),
                     shift=(mf.prediction - mc.prediction)[hit].mean(),
                     spread_clean=mc.score_ensemble_std[hit].mean(), spread_fault=mf.score_ensemble_std[hit].mean(),
                     eta_clean=mc.score_asymmetric_risk[hit].mean(), eta_fault=mf.score_asymmetric_risk[hit].mean()))
r = pd.DataFrame(rows)
r["d_spread"], r["d_eta"] = r.spread_fault - r.spread_clean, r.eta_fault - r.eta_clean
r["shift_rank"] = r["shift"].rank(ascending=False).astype(int)
r.sort_values("shift_rank").to_csv(ROOT / "paper/outputs/stats/figS1_unit_mechanism.csv", index=False, float_format="%.4f")
up = r["shift"] > 0
summ = {"n_up": int(up.sum()), "n_down": int((~up).sum()),
        "med_d_eta_up": r.d_eta[up].median(), "med_d_eta_down": r.d_eta[~up].median(),
        "med_d_spread_up": r.d_spread[up].median(), "med_d_spread_down": r.d_spread[~up].median(),
        "rho_shift_eta": spearmanr(r["shift"], r.d_eta)[0], "rho_shift_spread": spearmanr(r["shift"], r.d_spread)[0],
        "fig1_rank": int(r.set_index("unit").shift_rank[7])}
print({k: round(float(v), 3) for k, v in summ.items()})

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
for a, col, lab in ((ax[0], "d_eta", "$\\Delta$ mean one-sided score $\\hat\\eta$"), (ax[1], "d_spread", "$\\Delta$ mean ensemble spread")):
    a.axvline(0, color="0.6", lw=0.8)
    a.axhline(0, color="0.6", lw=0.8)
    a.scatter(r["shift"], r[col], s=14, c=np.where(up, "C3", "C0"), alpha=0.8, lw=0)
    s7 = r[r.unit == 7]
    a.scatter(s7["shift"], s7[col], s=60, facecolors="none", edgecolors="k", lw=1)
    a.annotate("Fig. 1", (s7["shift"].iloc[0], s7[col].iloc[0]), xytext=(-28, 6), textcoords="offset points", fontsize=7)
    a.set_xlabel("Mean prediction shift (fraction of life)")
    a.set_ylabel(lab)
    a.grid(alpha=0.3)
ax[0].set_title(f"(a) $\\rho_S$ = {summ['rho_shift_eta']:.2f}", fontsize=8)
ax[1].set_title(f"(b) $\\rho_S$ = {summ['rho_shift_spread']:.2f}", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "paper/outputs/figures/figS1_unit_distribution.pdf")
fig.savefig(ROOT / "paper/outputs/figures/figS1_unit_distribution.png", dpi=160)
