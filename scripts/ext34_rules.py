"""EXT34 offline rule evaluation (EXT34_PROTOCOL.md). usage: ext34_rules.py ALL_EXT34.csv.gz OUTDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fe, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
SEEN = ["clean", "dropout", "drift"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
FAM = ["cmapss", "ncmapss", "bearings", "battery", "hdd", "hdd2"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
key = ["dataset", "model", "seed", "fold"]
test = d[(d.part == "test") & d.scenario.isin(FA)].groupby(key + ["score"]).risk_matched.mean().unstack("score")
cal = d[d.part == "cal"]
rng = np.random.default_rng(3434)
rows = []
for k, g in cal.groupby(key):
    r = {"key": k}
    piv = {}
    for sc in SEEN:
        h = g[g.scenario == sc]
        for nm in ("full", "rpt_L"):
            x = h[h.score == nm].set_index("unit")
            piv[(sc, nm)] = x
            r[f"{sc}_{nm}"] = x.o.sum() / max(x.a.sum(), 1e-12)
    # paired bootstrap over calibration units (same units across shifts and scores)
    units = sorted(set.intersection(*[set(v.index) for v in piv.values()]))
    A = {kk: v.reindex(units)[["a", "o"]].to_numpy() for kk, v in piv.items()}
    wins = 0
    for _ in range(500):
        idx = rng.integers(0, len(units), len(units))
        rr = {}
        for nm in ("full", "rpt_L"):
            rr[nm] = np.mean([A[(sc, nm)][idx, 1].sum() / max(A[(sc, nm)][idx, 0].sum(), 1e-12) for sc in SEEN])
        wins += rr["rpt_L"] < rr["full"]
    r["p_boot"] = wins / 500
    rows.append(r)
c = pd.DataFrame(rows)
c[key] = pd.DataFrame(c.key.tolist(), index=c.index)
c = c.drop(columns="key").set_index(key)
c["m_full"] = c[[f"{s}_full" for s in SEEN]].mean(1)
c["m_rpt"] = c[[f"{s}_rpt_L" for s in SEEN]].mean(1)
c = c.join(test, how="inner")
RULES = {"R0": lambda c: c.m_rpt < c.m_full}
for m in (0.001, 0.0025, 0.005, 0.01):
    RULES[f"M{m}"] = (lambda mm: (lambda c: c.m_rpt < c.m_full - mm))(m)
RULES["U"] = lambda c: np.all([c[f"{s}_rpt_L"] < c[f"{s}_full"] for s in SEEN], axis=0)
for q in (0.8, 0.9):
    RULES[f"B{q}"] = (lambda qq: (lambda c: c.p_boot >= qq))(q)
res = []
for name, rule in RULES.items():
    choose = np.asarray(rule(c), bool)
    risk = np.where(choose, c.rpt_L, c.full)
    t = c.assign(rule=risk, chose=choose).reset_index()
    t["w"] = np.where(t.dataset == "bearings", t.fold.map(BW), 1.0)
    t["dv"] = (t.rule - t.full) * 100 * t.w
    t["dr"] = (t.rpt_L - t.full) * 100 * t.w
    pp = t.groupby(["dataset", "model", "seed"])[["dv", "dr", "w"]].sum()
    pp["diff"], pp["diff_rpt"] = pp.dv / pp.w, pp.dr / pp.w
    fam = pp.groupby("dataset")["diff"].mean().reindex(FAM)
    ch = t.groupby("dataset").chose.mean().reindex(FAM)
    res.append(dict(rule=name, **{f"d_{f}": fam[f] for f in FAM}, **{f"ch_{f}": ch[f] for f in FAM},
                    total=fam.sum(), safe=bool((fam <= 0.05).all())))
r = pd.DataFrame(res)
pd.set_option("display.width", 250)
say("== rule - full (points, mean over pairs) and share choosing rpt_L")
say(r.round(3).to_string(index=False))
say("\nrpt_L - full (always switch): " + str(pp.groupby("dataset").diff_rpt.mean().reindex(FAM).round(3).to_dict()))
order = ["R0", "M0.001", "M0.0025", "M0.005", "M0.01", "U", "B0.8", "B0.9"]
s = r[r.safe].copy()
if s.empty:
    say("\nNO SAFE RULE")
else:
    top = s.total.min()
    tied = s[s.total <= top + 0.05]
    final = min(tied.rule, key=order.index)
    say(f"\nSELECTED: {final} (total={s.set_index('rule').loc[final, 'total']:.3f}; best={s.set_index('rule').total.idxmin()} {top:.3f})")
r.to_csv(out / "ext34_rules.csv", index=False)
(out / "ext34_output.txt").write_text("\n".join(lines) + "\n")
print("EXT34_RULES_DONE")
