"""EXT39a post hoc (exploratory): share of the gap to a hindsight-oracle selector closed by each score.
usage: ext39_oracle_gap.py ALL.csv.gz OUTFILE   (risk at 90% matched coverage, six faults)"""
import sys
import numpy as np
import pandas as pd
fe, out = sys.argv[1:3]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
d = d[d.scenario.isin(FA)]
key = ["dataset", "model", "seed", "fold", "scenario"]
acc = d[d.score == "predictor"].set_index(key).opt_full.rename("all")
r = d[~d.score.str.startswith("predictor")].pivot_table(index=key, columns="score", values="risk_matched")
r = r.join(acc)
r["oracle"] = np.clip((r["all"] - 0.10) / 0.90, 0, None)
r = r.groupby(["dataset", "model", "seed", "fold"]).mean().reset_index()
r["w"] = np.where(r.dataset == "bearings", r.fold.map(BW), 1.0)
cols = [c for c in r.columns if c not in ("dataset", "model", "seed", "fold", "w")]
for c in cols:
    r[c] = r[c] * r.w
p = r.groupby(["dataset", "model", "seed"])[cols + ["w"]].sum()
for c in cols:
    p[c] = 100 * p[c] / p.w
m = p.groupby("dataset")[cols].mean()
lines = ["risk (%) at 90% matched coverage, six faults; all = accept-all; oracle = hindsight selector", m.round(2).to_string(), ""]
lines.append("share of the accept-all -> oracle gap closed (%) = (all - score)/(all - oracle)")
g = m.drop(columns=["all", "oracle"]).apply(lambda s: 100 * (m["all"] - s) / (m["all"] - m["oracle"]))
lines.append(g.round(1).to_string())
best = [c for c in ("rpt_auto_d", "rpt_L") if c in m]
for c in best:
    lines.append(f"\nremaining gap OSC->oracle closed by {c} (%): " + str((100 * (m["full"] - m[c]) / (m["full"] - m["oracle"])).round(1).to_dict()))
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
