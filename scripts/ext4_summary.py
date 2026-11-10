"""Summarise EXT4 (EXT4_PROTOCOL.md). usage: ext4_summary.py all_ext4_v1.csv.gz

Per run the selected score is chosen from the scenario="cal_pool" rows (calibration units x seen shifts); the
evaluation rows of that score are then used. Aggregation as in the paper; pairs = (backbone, seed), 20 per dataset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

src = Path(sys.argv[1])
out = src.parent
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
DS = ["cmapss", "ncmapss", "bearings", "battery"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
RUN = ["dataset", "model", "seed", "fold"]
d = pd.read_csv(src)
cal = d[d.scenario == "cal_pool"].set_index(RUN + ["score"])


def choose(cands, crit):
    c = cal[crit].unstack("score")[cands]
    best = c.idxmin(axis=1)  # first minimum in candidate order
    full_is_min = c["full"].eq(c.min(axis=1))
    best[full_is_min] = "full"  # ties involving the paper's classifier keep it
    return best


rules = {"select_cal": choose(["full", "gauss_bias_aware"], "cal_aurc"),
         "select_cal_90": choose(["full", "gauss_bias_aware"], "risk_matched"),
         "select_cal3": choose(["full", "gauss_bias_aware", "rank_avg"], "cal_aurc")}
ev = d[d.scenario != "cal_pool"]
add = []
for rule, pick in rules.items():
    m = ev.merge(pick.rename("pick").reset_index(), on=RUN)
    add.append(m[m.score == m.pick].drop(columns="pick").assign(score=rule))
    print(f"{rule}: share choosing each candidate by dataset")
    print(pick.reset_index().groupby("dataset")[0].value_counts(normalize=True).round(2).unstack().to_string(), "\n")
ev = pd.concat([ev] + add, ignore_index=True)

M = ["risk_matched", "risk_frozen", "coverage_frozen"]


def pool(sub):
    g = sub.groupby(["dataset", "score", "model", "seed", "fold"])[M].mean().reset_index()
    g["_w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0)
    res = {}
    for col in M:
        gg = g.assign(_v=g[col] * g._w)
        s = gg.groupby(["dataset", "score", "model", "seed"])[["_v", "_w"]].sum()
        res[col] = s._v / s._w
    return pd.DataFrame(res)


x = pool(ev[ev.scenario.isin(FA)])
summ = x.groupby(["dataset", "score"]).mean() * 100
SC = ["ensemble_std", "full", "gauss_bias_aware", "rank_avg", "conformal", "select_cal", "select_cal_90", "select_cal3"]
pd.set_option("display.width", 250)
for col in M:
    print(f"== {col} (unseen faults, %)")
    print(summ[col].unstack("dataset").reindex(columns=DS).reindex(SC).round(2).to_string(), "\n")

w = x.risk_matched.unstack("score")


def test(a, b, ds):
    dif = (w.loc[ds][a] - w.loc[ds][b]).dropna()
    return dict(a=a, b=b, dataset=ds, n=len(dif), mean_pts=100 * dif.mean(), wins_a=int((dif < 0).sum()),
                ties=int((dif == 0).sum()), p=wilcoxon(dif).pvalue if (dif != 0).any() else 1.0)


main = pd.DataFrame([test("select_cal", "full", ds) for ds in DS])
order = np.argsort(main.p.to_numpy())
adj, run = np.empty(len(main)), 0.0
for r, i in enumerate(order):
    run = max(run, (len(main) - r) * main.p.iloc[i])
    adj[i] = min(1.0, run)
main["p_holm"] = adj
print("== MAIN FAMILY: select_cal - full (risk at 90% matched coverage, points), Holm over 4")
print(main.round(4).to_string(index=False))
worse = ((main.mean_pts > 0) & (main.p_holm < 0.05)).any()
better = ((main.mean_pts < 0) & (main.p_holm < 0.05)).any()
print(f"\ncriterion 'better than ESANN' (no significant loss, >=1 significant win): {bool(better and not worse)}")
sec = pd.DataFrame([test(a, "full", ds) for a in ["select_cal_90", "select_cal3"] for ds in DS])
print("\n== secondary (unadjusted)")
print(sec.round(4).to_string(index=False))
summ.reset_index().to_csv(out / "summary_ext4_v1.csv", index=False)
main.to_csv(out / "main_family_ext4_v1.csv", index=False)
sec.to_csv(out / "secondary_ext4_v1.csv", index=False)
