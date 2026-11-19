"""EXT66 analysis (EXT66_PROTOCOL.md, exploratory): ELPIS-J pretrained LeWM-style (V20: end-to-end, MSE + 0.05 SIGReg)
vs the EMA-teacher V13 rerun with the same script (V13r), seeds 162-171, seed level.
usage: ext66_summary.py ALL_EXT66_V20.csv.gz ALL_EXT66_V13R.csv.gz [EXT56_V13.csv.gz EXT56_V13_OTHER.csv.gz] OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

args = sys.argv[1:]
f20, f13, out = args[0], args[1], Path(args[-1])
old = args[2:-1]
out.mkdir(parents=True, exist_ok=True)
DS = ["cmapss", "battery_cv", "bearings"]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
L = []


def say(s=""):
    print(s); L.append(str(s))


def load(fs):
    d = pd.concat([pd.read_csv(f) for f in ([fs] if isinstance(fs, str) else fs)], ignore_index=True)
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    d = d[d.dataset.isin(DS) & d.seed.between(162, 171)]
    nr = d.groupby(["dataset", "model", "seed"]).fold.nunique().unstack("model")
    ff = d.groupby("dataset").fold.nunique()
    ok = nr.apply(lambda r: (r == ff[r.name[0]]).all(), axis=1)
    return d.merge(ok[ok].reset_index()[["dataset", "seed"]], on=["dataset", "seed"])


def pool(d, scen, col, score, scale=100.0):
    g = d[d.scenario.isin(scen) & (d.score == score)].groupby(["dataset", "model", "seed", "fold"])[col].mean().reset_index()
    g["w"] = np.where(g.dataset == "bearings", g.fold.map(BW), 1.0); g["v"] = g[col] * g.w
    s = g.groupby(["dataset", "model", "seed"])[["v", "w"]].sum()
    return (s.v / s.w * scale).groupby(["dataset", "seed"]).mean()


def diag(d, col):
    g = d.drop_duplicates(["dataset", "model", "seed", "fold"]).groupby(["dataset", "model", "seed"])[col].mean()
    return g.groupby(["dataset", "seed"]).mean()


V = {"V20": load(f20), "V13r": load(f13)}
if old:
    V["V13(EXT56)"] = load(old)
for k, d in V.items():
    say(f"complete seeds {k}: {d.drop_duplicates(['dataset', 'seed']).groupby('dataset').size().to_dict()}")
M = {k: dict(elpis_g=pool(d, FA, "risk_matched", "rpt_auto_d"), osc=pool(d, FA, "risk_matched", "full"),
             pess_g=pool(d, FA, "pess_matched", "rpt_auto_d"), opt_all=pool(d, FA, "opt_full", "predictor"),
             rmse=pool(d, ["clean"], "rmse_full", "predictor", 1.0)) for k, d in V.items()}
for k in ("V20", "V13r"):
    for c in ("emb_effrank", "emb_std", "emb_sigreg"):
        if c in V[k]:
            M[k][c] = diag(V[k], c)

say("\n== means per family")
tab = {(k, m): s.groupby("dataset").mean() for k, r in M.items() for m, s in r.items()}
say(pd.DataFrame(tab).T.round(3).to_string())

say("\n== PRIMARY: ELPIS-G risk under six faults, 90% matched coverage, V20 - V13r (seed level, Wilcoxon, 10 seeds)")
rows = []
for ds in DS:
    for m in ("elpis_g", "osc", "pess_g", "opt_all", "rmse", "emb_effrank", "emb_std", "emb_sigreg"):
        if m not in M["V20"] or m not in M["V13r"] or ds not in M["V20"][m].index.get_level_values(0) or ds not in M["V13r"][m].index.get_level_values(0):
            continue
        a, b = M["V20"][m].loc[ds], M["V13r"][m].loc[ds]
        dif = (a - b.reindex(a.index)).dropna()
        rows.append(dict(dataset=ds, metric=m, v20=a.mean(), v13r=b.mean(), diff=dif.mean(), lower=int((dif < 0).sum()), n=len(dif),
                         p=stats.wilcoxon(dif).pvalue if (dif != 0).any() and len(dif) > 1 else 1.0))
t = pd.DataFrame(rows)
say(t.round(4).to_string(index=False))
t.to_csv(out / "ext66_compare.csv", index=False)

say("\n== inside V20: ELPIS-G - OSC (seed level)")
for ds in [x for x in DS if x in M["V20"]["elpis_g"].index.get_level_values(0)]:
    dif = (M["V20"]["elpis_g"].loc[ds] - M["V20"]["osc"].loc[ds]).dropna()
    say(f"  {ds:10s} {dif.mean():+.3f} (better {int((dif < 0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0:.2g})")
if "V13(EXT56)" in M:
    say("\n== cross-check: V13r - V13 of EXT56, ELPIS-G risk (should be near 0)")
    for ds in [x for x in DS if x in M["V13r"]["elpis_g"].index.get_level_values(0)]:
        a, b = M["V13r"]["elpis_g"].loc[ds], M["V13(EXT56)"]["elpis_g"].loc[ds]
        say(f"  {ds:10s} {(a - b.reindex(a.index)).mean():+.3f}")

p = t[t.metric == "elpis_g"].set_index("dataset")["diff"]
promising = bool((p < 0).sum() >= 2 and (p <= 0.3).all())
say(f"\nPROMISING (mean ELPIS-G diff < 0 on >= 2 families and <= +0.3 on all): {promising}")
(out / "ext66_summary.txt").write_text("\n".join(L) + "\n")
