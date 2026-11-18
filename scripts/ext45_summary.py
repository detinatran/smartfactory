"""EXT45 analysis (EXT45_PROTOCOL.md, exploratory). usage: ext45_summary.py OUT.txt ALL_o1.csv.gz [ALL_o2.csv.gz ...]"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

out, files = sys.argv[1], sys.argv[2:]
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
k = ["dataset", "model", "seed", "fold"]
lines = []
for fe in files:
    d = pd.read_csv(fe)
    d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
    off = d.offset.iloc[0]
    f = d[d.scenario.isin(FA) & d.score.isin(["full", "rpt_L", "rpt_auto_d"])].groupby(k + ["score"]).risk_matched.mean().unstack() * 100
    u = d.drop_duplicates(k).set_index(k)[["D"]].join(f).reset_index()
    u["dr"], u["dd"] = u.rpt_L - u.full, u.rpt_auto_d - u.full
    u["w"] = np.where(u.dataset == "bearings", u.fold.map(BW), 1.0)
    lines.append(f"\n######## offset = {off} x delta  (runs {len(u)})")
    rows = []
    for ds, g in u.groupby("dataset"):
        sd = g.assign(a=g.dr * g.w, b=g.dd * g.w).groupby("seed")[["a", "b", "w"]].sum()
        right = np.where(g.D > 0, g.dr < 0, g.dr >= 0).mean()
        pos, neg = g[g.D > 0], g[g.D <= 0]
        rows.append(dict(dataset=ds, D=100 * g.D.mean(), share_Dpos=(g.D > 0).mean(), risk_osc=g.full.mean(),
                         rpt_osc=(sd.a / sd.w).mean(), rptd_osc=(sd.b / sd.w).mean(), acc=right,
                         gain_if_Dpos=pos.dr.mean() if len(pos) else np.nan, n_pos=len(pos),
                         gain_if_Dneg=neg.dr.mean() if len(neg) else np.nan, n_neg=len(neg)))
    lines.append(pd.DataFrame(rows).round(3).to_string(index=False))
    pos, neg = u[u.D > 0], u[u.D <= 0]
    lines.append(f"pooled over families: RPT-OSC when D>0: {pos.dr.mean():+.3f} (n={len(pos)}, share helped {(pos.dr < 0).mean():.2f}); "
                 f"when D<=0: {neg.dr.mean():+.3f} (n={len(neg)}, share helped {(neg.dr < 0).mean():.2f}); "
                 f"Mann-Whitney p={stats.mannwhitneyu(pos.dr, neg.dr).pvalue if len(pos) and len(neg) else float('nan'):.2g}")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
