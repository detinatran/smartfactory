"""Confirmatory test of EXT6C_PROTOCOL.md. usage: ext6c_test.py all_ext6c_v1.csv.gz

H: full_recon has lower risk than full on C-MAPSS (risk at 90% matched coverage, mean over six unseen faults, subsets
averaged within each (backbone, seed)); 10 pairs; one-sided Wilcoxon signed-rank, alpha = 0.05.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

src = Path(sys.argv[1])
out = src.parent
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
SC = ["ensemble_std", "impact", "gauss_bias_aware", "full", "full_recon", "repair_full"]
d = pd.read_csv(src)
d = d[d.score.isin(SC)]
pd.set_option("display.width", 250)
fa = d[d.scenario.isin(FA)].groupby(["model", "seed", "fold", "score"])[
    ["risk_matched", "pess_matched", "risk_frozen", "coverage_frozen"]].mean()
cl = d[d.scenario == "clean"].groupby(["model", "seed", "fold", "score"])[["risk_matched"]].mean()
pair = fa.groupby(["model", "seed", "score"]).mean()
print(f"runs: {d.groupby(['model', 'seed', 'fold']).ngroups}, pairs: {pair.groupby(['model', 'seed']).ngroups}\n")
print("== unseen faults, mean over pairs (%)")
print((pair.groupby("score").mean() * 100).reindex(SC).round(2).to_string(), "\n")
print("== clean risk_matched (%)")
print((cl.groupby(["model", "seed", "score"]).mean().groupby("score").mean() * 100).reindex(SC).round(2).to_string(), "\n")

r = pair.risk_matched.unstack("score")
dif = (r.full_recon - r.full).dropna()
res = wilcoxon(dif, alternative="less")
print("== CONFIRMATORY TEST: full_recon - full, C-MAPSS, one-sided Wilcoxon")
print((dif * 100).round(3).to_string())
confirmed = bool(res.pvalue < 0.05 and dif.mean() < 0)
print(f"n={len(dif)} mean={100 * dif.mean():.3f} pts wins={(dif < 0).sum()}/{len(dif)} W={res.statistic} "
      f"p_one_sided={res.pvalue:.4g} -> CONFIRMED: {confirmed}\n")
sub = fa.risk_matched.unstack("score")
print("== by subset: full_recon - full (points, mean over pairs)")
print(((sub.full_recon - sub.full).groupby("fold").agg(["mean", lambda v: (v < 0).sum(), "count"]) *
       [100, 1, 1]).round(3).to_string())
pd.DataFrame([dict(n=len(dif), mean_pts=100 * dif.mean(), wins=int((dif < 0).sum()), W=res.statistic,
                   p_one_sided=res.pvalue, confirmed=confirmed)]).to_csv(out / "confirm_ext6c.csv", index=False)
(pair.groupby("score").mean() * 100).to_csv(out / "summary_ext6c_v1.csv")
