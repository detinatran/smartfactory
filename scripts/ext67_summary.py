"""EXT67 analysis (EXT67_PROTOCOL.md, exploratory): per-deployment calibration gate between ELPIS-J V13 (EMA teacher,
EXT59) and V20 (SIGReg, EXT66 script) on seeds 222-251, seed level.
usage: ext67_summary.py ALL_EXT59_V13.csv.gz ALL_EXT67_V20.csv.gz OUTDIR [SEED_LO SEED_HI]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

f13, f20, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (222, 251)
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "battery_cv", "bearings"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
K = ["dataset", "model", "seed", "fold"]
L = []
rng = np.random.default_rng(0)


def say(s=""):
    print(s); L.append(str(s))


def per_run(f):
    d = pd.read_csv(f)
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    d = d[d.dataset.isin(DS) & d.seed.between(lo, hi)]
    r = d.drop_duplicates(K)[K + ["choice_d", "calrisk_full", "calrisk_rpt_L"] + [c for c in ("emb_effrank",) if c in d]].copy()
    r["cal"] = np.where(r.choice_d == "rpt_L", r.calrisk_rpt_L, r.calrisk_full)
    g = d[d.scenario.isin(FA)]
    for sc, nm in (("rpt_auto_d", "g"), ("full", "osc")):
        x = g[g.score == sc].groupby(K)[["risk_matched", "pess_matched"]].mean().add_prefix(nm + "_").reset_index()
        r = r.merge(x, on=K)
    c = d[(d.scenario == "clean") & (d.score == "predictor")].groupby(K).rmse_full.mean().reset_index()
    return r.merge(c, on=K, how="left")


a, b = per_run(f13), per_run(f20)
m = a.merge(b, on=K, suffixes=("_13", "_20"))
pick = m.cal_20 < m.cal_13
for col in ("g_risk_matched", "g_pess_matched", "rmse_full"):
    m[col + "_gate"] = np.where(pick, m[col + "_20"], m[col + "_13"])
m["pick20"] = pick.astype(float)
# keep only (dataset, seed) with every fold of both backbones present in both systems
nr = m.groupby(["dataset", "model", "seed"]).fold.nunique().unstack("model")
ff = m.groupby("dataset").fold.nunique()
ok = nr.apply(lambda r: (r == ff[r.name[0]]).all(), axis=1)
m = m.merge(ok[ok].reset_index()[["dataset", "seed"]], on=["dataset", "seed"])
say(f"runs matched: {len(m)}; complete seeds: {m.drop_duplicates(['dataset', 'seed']).groupby('dataset').size().to_dict()}")


def seed(col, scale=100.0):
    g = m.copy(); g["w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0); g["v"] = g[col] * g.w
    s = g.groupby(["dataset", "model", "seed"])[["v", "w"]].sum()
    return (s.v / s.w * scale).groupby(["dataset", "seed"]).mean()


def boot_hi(x, q=0.95, n=10000):
    x = np.asarray(x); return float(np.quantile(rng.choice(x, (n, len(x))).mean(1), q))


def cmp(c1, c0, scale=100.0):
    rows = []
    A, B = seed(c1, scale), seed(c0, scale)
    for ds in DS:
        if ds not in A.index.get_level_values(0):
            continue
        d = (A.loc[ds] - B.loc[ds].reindex(A.loc[ds].index)).dropna()
        rows.append(dict(dataset=ds, a=A.loc[ds].mean(), b=B.loc[ds].mean(), diff=d.mean(), ci90_hi=boot_hi(d), lower=int((d < 0).sum()),
                         n=len(d), p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0))
    return pd.DataFrame(rows)


say("\n== PRIMARY: ELPIS-G risk, gate - V13 (seed level, Wilcoxon two-sided, 90% bootstrap CI upper bound)")
P = cmp("g_risk_matched_gate", "g_risk_matched_13")
P["better"] = (P["diff"] < 0) & (P.p < 0.05); P["noninferior"] = P.ci90_hi <= 0.3
say(P.round(4).to_string(index=False))
P.to_csv(out / "ext67_primary.csv", index=False)
improved = bool(P.better.sum() >= 2 and P.noninferior.all() and len(P) == 3)
say(f"GATE IMPROVES ELPIS-J (better on >= 2 families, non-inferior on all 3): {improved}")

say("\n== secondary")
for nm, c1, c0, sc in (("V20 - V13, ELPIS-G risk", "g_risk_matched_20", "g_risk_matched_13", 100.0),
                       ("gate - V13, pessimistic", "g_pess_matched_gate", "g_pess_matched_13", 100.0),
                       ("gate - V13, clean RMSE", "rmse_full_gate", "rmse_full_13", 1.0),
                       ("V20: ELPIS-G - OSC", "g_risk_matched_20", "osc_risk_matched_20", 100.0)):
    say(f"-- {nm}"); say(cmp(c1, c0, sc).round(4).to_string(index=False))
say("-- share of deployments using V20: " + str(m.groupby("dataset").pick20.mean().round(3).to_dict()))
if "emb_effrank_20" in m:
    say("-- V20 effective rank: " + str(m.groupby("dataset").emb_effrank_20.mean().round(2).to_dict()))
(out / "ext67_summary.txt").write_text("\n".join(L) + "\n")
