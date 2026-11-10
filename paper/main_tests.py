"""Main test family (Holm over 100 tests) -> outputs/stats/main_family.csv, main_family_differences.csv.

Pair = (backbone, seed): GRU/TCN x 10 seeds = 20 per dataset; value = mean over the five standard scenarios or the six
unseen faults. Metrics: risk at 90% matched coverage and AURC over coverage 0.5-1.0 (trapezoid). Two-sided Wilcoxon
signed-rank; 95% interval from a seed bootstrap (10 000 replicates). These tests were added after the first results
(the pre-registered primary protocol is the frozen threshold, Table 3).
"""
import numpy as np
import pandas as pd

from common import COMPARATORS, DATASETS, FAULTS, OUT, STANDARD, holm, read, seed_bootstrap, wilcoxon_details

rng = np.random.default_rng(20261109)
m = read("matched_per_seed.csv.gz")


def aurc(u):
    u = u.sort_values("cov_level")
    return np.trapezoid(u.optimistic_error_rate, u.cov_level) / (u.cov_level.max() - u.cov_level.min())


rows, diffs = [], []
for kind, setting, scen in (("base", "standard", STANDARD), ("fault", "faults", FAULTS)):
    d = m[(m.setting == setting) & m.scenario.isin(scen)]
    u = d.groupby(["dataset", "method", "model", "seed", "cov_level"]).optimistic_error_rate.mean().reset_index()
    at90 = u[u.cov_level == 0.9].rename(columns={"optimistic_error_rate": "risk90"})
    au = u.groupby(["dataset", "method", "model", "seed"]).apply(aurc, include_groups=False).rename("aurc").reset_index()
    tests = []
    for ds in DATASETS:
        for b in COMPARATORS:
            tests += [(ds, "risk90", "asymmetric_risk", b), (ds, "aurc", "asymmetric_risk", b)]
        if kind == "fault":
            tests.append((ds, "risk90", "asym_aug", "asymmetric_risk"))
    for ds, metric, a, b in tests:
        t = at90 if metric == "risk90" else au
        x = t[(t.dataset == ds) & (t.method == a)].set_index(["model", "seed"])[metric]
        y = t[(t.dataset == ds) & (t.method == b)].set_index(["model", "seed"])[metric]
        dif = (x - y).dropna()
        lo, hi = seed_bootstrap(dif, rng, 10000)
        comp = "aug_vs_asym" if a == "asym_aug" else b
        rows.append(dict(setting=kind, dataset=ds, metric=metric, comparator=comp, a=a, b=b, mean=100 * dif.mean(),
                         ci_lo=100 * lo, ci_hi=100 * hi, wins=int((dif < 0).sum()), **wilcoxon_details(dif.to_numpy())))
        diffs += [dict(setting=kind, dataset=ds, metric=metric, a=a, b=b, model=k[0], seed=k[1], diff_pts=100 * v)
                  for k, v in dif.items()]
res = pd.DataFrame(rows)
res["p_holm"] = holm(res.p.to_numpy())
assert len(res) == 100
res.to_csv(OUT / "stats/main_family.csv", index=False, float_format="%.10g")
pd.DataFrame(diffs).to_csv(OUT / "stats/main_family_differences.csv", index=False, float_format="%.10g")
print(res[res.metric == "risk90"].pivot_table(index=["setting", "comparator"], columns="dataset", values="p_holm").round(4))
