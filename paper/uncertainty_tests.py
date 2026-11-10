"""Exploratory uncertainty decomposition (added after the main results; unadjusted). Input: results/uncertainty_per_fold.csv.gz,
written by scripts/uncertainty_v1.py (CPU replay of the same checkpoints and splits; MC dropout with T = 10 passes per member).

Scores compared (higher = abstain first):
  ensemble_std      epistemic, between-model variance (the "ensemble spread" of the paper)
  mc_dropout        epistemic, within-model (MC dropout, mean over members of the std over passes)
  mc_ensemble       epistemic, total: std over all M x T stochastic predictions
  resid_sigma       learned conditional error scale sqrt(v(x))
  gauss_unbiased    1 - Phi(delta / s),          s^2 = v(x) + mc_ensemble^2   (variance only)
  bias_only         learned conditional bias b(x) = E[yhat - y | x]
  gauss_bias_aware  1 - Phi((delta - b(x)) / s)                              (bias-aware predictive distribution)
  full              the one-sided classifier of the paper
  full_plus_unc     the same classifier with mc_dropout and mc_ensemble as extra features

-> outputs/stats/uncertainty_summary.csv      per dataset x score, mean over the six unseen faults
-> outputs/stats/uncertainty_tests.csv        paired comparisons: mean diff, seed bootstrap 95% interval (5 000), Wilcoxon
-> outputs/stats/uncertainty_differences.csv  the 20 paired differences behind each test
-> outputs/tables/uncertainty.tex             risk at 90% matched coverage per score (supplementary table)
"""
import numpy as np
import pandas as pd

from common import DATASETS, DNAME, FAULTS, OUT, per_seed, read, seed_bootstrap, wilcoxon_details, write_tex

ORDER = {"ensemble_std": "Ensemble spread (between-model)", "mc_dropout": "MC dropout (within-model)",
         "mc_ensemble": "MC-dropout ensemble (total epistemic)", "resid_sigma": "Learned error scale $\\sqrt{v(x)}$",
         "gauss_unbiased": "Gaussian, variance only", "bias_only": "Learned bias $b(x)$",
         "gauss_bias_aware": "Gaussian, bias-aware", "full": "One-sided classifier (ours)",
         "full_plus_unc": "Ours + MC-dropout features"}
METRICS = ["risk_matched", "risk_frozen", "coverage_frozen", "reject_new", "rho_shift"]
PAIRS = [("gauss_bias_aware", "gauss_unbiased"), ("gauss_bias_aware", "full"), ("gauss_bias_aware", "ensemble_std"),
         ("mc_ensemble", "ensemble_std"), ("mc_dropout", "ensemble_std"), ("full_plus_unc", "full"),
         ("full", "mc_ensemble")]

rng = np.random.default_rng(2)
d = read("uncertainty_per_fold.csv.gz")
fa = d[d.scenario.isin(FAULTS)]
x = per_seed(fa, METRICS).groupby(["dataset", "score", "model", "seed"]).mean()
summ = x.groupby(["dataset", "score"]).mean()
pct = ["risk_matched", "risk_frozen", "coverage_frozen", "reject_new"]
summ[pct] *= 100
clean = per_seed(d[d.scenario == "clean"], ["coverage_frozen"]).groupby(["dataset", "score"]).mean() * 100
summ["coverage_clean"] = clean.coverage_frozen
summ["coverage_loss"] = summ.coverage_clean - summ.coverage_frozen
summ.reset_index().to_csv(OUT / "stats/uncertainty_summary.csv", index=False, float_format="%.10g")

rows, diffs = [], []
for metric in ["risk_matched", "risk_frozen", "reject_new"]:
    w = x[metric].unstack("score")
    for a, b in PAIRS:
        for ds in DATASETS:
            dif = (w.loc[ds][a] - w.loc[ds][b]).dropna()
            lo, hi = seed_bootstrap(dif, rng, 5000)
            wins = int((dif < 0).sum()) if "risk" in metric else int((dif > 0).sum())
            rows.append(dict(metric=metric, a=a, b=b, dataset=ds, mean=100 * dif.mean(), lo=100 * lo, hi=100 * hi,
                             wins_a=wins, **wilcoxon_details(dif.to_numpy())))
            diffs += [dict(metric=metric, a=a, b=b, dataset=ds, model=k[0], seed=k[1], diff=100 * v) for k, v in dif.items()]
res = pd.DataFrame(rows)
res.to_csv(OUT / "stats/uncertainty_tests.csv", index=False, float_format="%.10g")
pd.DataFrame(diffs).to_csv(OUT / "stats/uncertainty_differences.csv", index=False, float_format="%.10g")

r = summ.risk_matched.unstack("dataset")[DATASETS]
lines = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Score & " + " & ".join(DNAME[k] for k in DATASETS) + " \\\\", "\\midrule"]
for k, name in ORDER.items():
    best = r.min()
    cells = [f"\\textbf{{{v:.2f}}}" if np.isclose(v, best[ds]) else f"{v:.2f}" for ds, v in r.loc[k].items()]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write_tex("uncertainty.tex", lines)

pd.set_option("display.width", 200)
for col in ["risk_matched", "risk_frozen", "coverage_loss", "reject_new", "rho_shift"]:
    print(f"\n== {col}")
    print(summ[col].unstack("dataset")[DATASETS].loc[list(ORDER)].round(2).to_string())
print("\n== paired tests (risk at 90% matched coverage, points)")
print(res[res.metric == "risk_matched"][["dataset", "a", "b", "mean", "lo", "hi", "wins_a", "p"]].round(4).to_string(index=False))
