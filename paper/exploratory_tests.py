"""Exploratory comparisons (unadjusted; added after the main results) on the CPU-replay scores of Table 2.

-> outputs/stats/exploratory.csv              every comparison x metric: mean diff, seed-bootstrap 95% interval (5 000),
                                              wins, Wilcoxon details (W, zeros, ties, exact/normal p)
-> outputs/stats/exploratory_differences.csv  the 20 paired differences behind each test
-> outputs/stats/selectivity.csv              acceptance of fault-affected non-optimistic windows (G1) and rejection of
                                              severe fault errors with below-median spread (G3), quoted in the text
Pair = (backbone, seed), 20 per dataset; value = mean over the six unseen faults (bearing folds pooled by unit count).
"""
import numpy as np
import pandas as pd

from common import FAULTS, OUT, per_seed, read, seed_bootstrap, wilcoxon_details

rng = np.random.default_rng(1)
rp = read("replay_scores_per_fold.csv.gz")
rp = rp[rp.scenario.isin(FAULTS)]
METRICS = ["risk_matched", "risk_frozen", "coverage_frozen", "reject_new"]
PAIRS = [("full", "fault_detector"), ("full", "fault_detector_full"), ("full", "ensemble_std"),
         ("pred_cycle_only", "full"), ("one_sided_plus_detector", "full"),
         ("fault_detector_full", "fault_detector"), ("fault_detector_full", "ensemble_std")]
agg = per_seed(rp, METRICS + ["G1_accept_fault_ok", "G3_reject_fault_lowspread_severe"])
rows, diffs = [], []
for metric in METRICS:
    x = agg[metric].groupby(["dataset", "score", "model", "seed"]).mean().unstack("score")
    for a, b in PAIRS:
        for ds, g in x.groupby(level="dataset"):
            dif = (g[a] - g[b]).dropna().droplevel("dataset")
            lo, hi = seed_bootstrap(dif, rng, 5000)
            wins = int((dif < 0).sum()) if "risk" in metric else int((dif > 0).sum())
            rows.append(dict(metric=metric, a=a, b=b, dataset=ds, mean=100 * dif.mean(), lo=100 * lo, hi=100 * hi,
                             wins_a=wins, **wilcoxon_details(dif.to_numpy())))
            diffs += [dict(metric=metric, a=a, b=b, dataset=ds, model=k[0], seed=k[1], diff=100 * v) for k, v in dif.items()]
res = pd.DataFrame(rows)
res.to_csv(OUT / "stats/exploratory.csv", index=False, float_format="%.10g")
pd.DataFrame(diffs).to_csv(OUT / "stats/exploratory_differences.csv", index=False, float_format="%.10g")
sel = agg[["G1_accept_fault_ok", "G3_reject_fault_lowspread_severe"]].groupby(["dataset", "score"]).mean() * 100
sel.loc[(slice(None), ["full", "ensemble_std", "fault_detector", "fault_detector_full"]), :].round(1).to_csv(OUT / "stats/selectivity.csv")
print(res[res.metric == "risk_matched"][["dataset", "a", "b", "mean", "wins_a", "W", "p"]].round(5).to_string(index=False))
