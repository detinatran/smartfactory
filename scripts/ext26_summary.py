"""EXT26 analysis (EXT26_PROTOCOL.md). usage: ext26_summary.py ALL_EXT26.csv.gz OUTDIR
Primary: rpt_auto vs {full, gauss_bias_aware, err_reg}, risk @90% matched, mean of six faults, Holm over 3, alpha 0.05.
Success: rpt_auto not significantly worse than full and mean difference <= +0.3 points."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "full", "rpt_L", "rpt_auto", "auto3"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d = d[~d.score.str.startswith("predictor")]
say(f"runs: {d.groupby(['model', 'seed']).ngroups}; datasets: {d.dataset.unique().tolist()}")
ch = d.drop_duplicates(["model", "seed"]).choice2.value_counts().to_dict()
say(f"rpt_auto choice: {ch}")
W = d[d.scenario.isin(FA)].groupby(["score", "model", "seed"]).risk_matched.mean().unstack("score") * 100
Wc = d[d.scenario == "clean"].groupby(["score", "model", "seed"]).risk_matched.mean().unstack("score") * 100
Wp = d[d.scenario.isin(FA)].groupby(["score", "model", "seed"]).pess_matched.mean().unstack("score") * 100
say("\n== risk (%), six faults / clean / excess / pessimistic (faults)")
say(pd.DataFrame({"faults": W.mean(), "clean": Wc.mean(), "excess": (W - Wc).mean(), "pess": Wp.mean()})
    .reindex([s for s in SC if s in W]).round(2).to_string())


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


rows = []
for b in ("full", "gauss_bias_aware", "err_reg"):
    dif = (W["rpt_auto"] - W[b]).dropna()
    rows.append(dict(b=b, mean=dif.mean(), better=int((dif < 0).sum()), worse=int((dif > 0).sum()), n=len(dif),
                     p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0))
t = pd.DataFrame(rows)
t["p_holm"] = holm(t.p)
say("\n== PRIMARY: rpt_auto - baseline (points), Holm over 3")
say(t.round(4).to_string(index=False))
r = t.set_index("b").loc["full"]
ok = not (r.p_holm < 0.05 and r["mean"] > 0) and r["mean"] <= 0.3
say(f"SUCCESS (not significantly worse than full, mean <= +0.3): {ok}")
say("\n== descriptive")
for a, b in (("rpt_L", "full"), ("auto3", "full"), ("rpt_auto", "rpt_L")):
    if a in W and b in W:
        dif = (W[a] - W[b]).dropna()
        say(f"{a} - {b}: {dif.mean():+.2f} ({(dif < 0).sum()}/{len(dif)} better, p={stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1:.3g})")
    dif = ((W - Wc)[a] - (W - Wc)[b]).dropna() if a in W and b in W else None
    if dif is not None:
        say(f"   excess {a} - {b}: {dif.mean():+.2f}")
t.to_csv(out / "ext26_primary.csv", index=False)
(out / "ext26_output.txt").write_text("\n".join(lines) + "\n")
print("EXT26_SUMMARY_DONE")
