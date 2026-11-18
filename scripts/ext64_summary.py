"""EXT64 analysis (EXT64_PROTOCOL.md, exploratory, fresh seeds 252-281): predictor gate (default vs pooled
default+RAMP-V13 ensemble, chosen per deployment by the calibration risk of the selector ELPIS-G picks) vs ELPIS-G on the
default predictor. usage: ext64_summary.py DEFAULT.csv.gz HYBRID.csv.gz OUTDIR [SEED_LO SEED_HI]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

fd, fh, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (252, 281)
out.mkdir(parents=True, exist_ok=True)
DS = ["bearings", "cmapss", "battery_cv"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
K = ["dataset", "model", "seed", "fold"]
L = []


def say(s=""):
    print(s); L.append(str(s))


def prep(f):
    d = pd.read_csv(f)
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    d = d[d.dataset.isin(DS) & d.seed.between(lo, hi)]
    run = d.drop_duplicates(K)[K + ["choice_d", "calrisk_full", "calrisk_rpt_L"]].copy()
    run["cal"] = np.where(run.choice_d == "rpt_L", run.calrisk_rpt_L, run.calrisk_full)
    r = d[d.scenario.isin(FA) & (d.score == "rpt_auto_d")].groupby(K)[["risk_matched", "pess_matched"]].mean().reset_index()
    c = d[(d.scenario == "clean") & (d.score == "predictor")].groupby(K).rmse_full.mean().reset_index()
    return run.merge(r, on=K).merge(c, on=K, how="left")


def agg(t, col):
    t = t.copy(); t["w"] = np.where(t.dataset == "bearings", t.fold.map(BW), 1.0); t["v"] = t[col] * t.w
    s = t.groupby(["dataset", "model", "seed"])[["v", "w"]].sum()
    return (s.v / s.w).groupby(["dataset", "seed"]).mean()


m = prep(fd).merge(prep(fh), on=K, suffixes=("_d", "_h"))
# keep only seeds complete in both systems (all folds, both backbones)
nf = m.groupby(["dataset", "model", "seed"]).fold.nunique().unstack("model")
ff = m.groupby("dataset").fold.nunique()
ok = nf.apply(lambda r: (r == ff[r.name[0]]).all(), axis=1)
m = m.merge(ok[ok].reset_index()[["dataset", "seed"]], on=["dataset", "seed"])
use = m.cal_h < m.cal_d
sysd = {"gate": m[K].assign(**{c: np.where(use, m[c + "_h"], m[c + "_d"]) for c in ("risk_matched", "pess_matched", "rmse_full")}),
        "default": m[K].assign(**{c: m[c + "_d"] for c in ("risk_matched", "pess_matched", "rmse_full")}),
        "pooled": m[K].assign(**{c: m[c + "_h"] for c in ("risk_matched", "pess_matched", "rmse_full")})}
say(f"seeds {lo}-{hi}; complete seeds per family: {m.drop_duplicates(['dataset','seed']).groupby('dataset').size().to_dict()}")
rng = np.random.default_rng(0)
rows = []
for ds in DS:
    if ds not in m.dataset.values:
        continue
    R = {k: 100 * agg(v[v.dataset == ds], "risk_matched").loc[ds] for k, v in sysd.items()}
    P = {k: 100 * agg(v[v.dataset == ds], "pess_matched").loc[ds] for k, v in sysd.items()}
    E = {k: agg(v[v.dataset == ds], "rmse_full").loc[ds] for k, v in sysd.items()}
    dif = R["gate"] - R["default"]
    bs = rng.choice(dif.to_numpy(), (20000, len(dif))).mean(1)
    dp = R["gate"] - R["pooled"]
    rows.append(dict(dataset=ds, n=len(dif), gate=R["gate"].mean(), default=R["default"].mean(), diff=dif.mean(),
                     ci90_lo=np.quantile(bs, 0.05), ci90_hi=np.quantile(bs, 0.95), better=int((dif < 0).sum()), worse=int((dif > 0).sum()),
                     p=stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0, pess_diff=(P["gate"] - P["default"]).mean(),
                     share_pooled=float(use[m.dataset == ds].mean()), gate_vs_pooled=dp.mean(),
                     rmse_gate=E["gate"].mean(), rmse_default=E["default"].mean()))
t = pd.DataFrame(rows)
p = t.p.to_numpy(); adj, run = np.empty(len(p)), 0.0
for r, i in enumerate(np.argsort(p)):
    run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
t["p_holm"] = adj
t["tie_(ci90hi<=0.3)"] = t.ci90_hi <= 0.3
t["win_(mean<0,p<.05)"] = (t["diff"] < 0) & (t.p < 0.05)
say("\n== predictor gate vs ELPIS-G on the default predictor (risk under faults @90%, seed level)")
say(t.round(4).to_string(index=False))
t.to_csv(out / "ext64_gate.csv", index=False)
(out / "ext64_output.txt").write_text("\n".join(L) + "\n")
print("EXT64_SUMMARY_DONE")
