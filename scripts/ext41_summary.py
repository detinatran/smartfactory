"""EXT41 analysis (EXT41_PROTOCOL.md, exploratory). usage: ext41_summary.py ALL_EXT41.csv.gz GLITCH.csv OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

fe, fg, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
W = 30
SC = ["ensemble_std", "gauss_bias_aware", "err_reg", "err_reg_rptL", "full", "rpt_L", "rpt_auto_d"]
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(fe)
g = pd.read_csv(fg)
# window = cycles (end-29 .. end) of the unit: flag if any glitch cycle inside
win = {}
for col in ("g_main", "g_ct", "g_irt"):
    gg = g[g[col]][["unit", "cycle"]]
    s = set()
    for u, c in zip(gg.unit.astype(int), gg.cycle.astype(int)):
        for e in range(c, c + W):
            s.add((u, e))
    d[col] = [(int(u), int(c)) in s for u, c in zip(d.unit, d.cycle)]
d["opt"] = (d.pred - d.y) > d.delta
say(f"runs: {d.groupby(['model', 'seed', 'fold']).ngroups}; windows per run ~{len(d) / d.groupby(['model', 'seed', 'fold']).ngroups:.0f}")
for col in ("g_main", "g_ct", "g_irt"):
    say(f"\n######## glitch definition: {col}")
    say(f"share of test windows flagged: {d[col].mean():.4f}")

    def wrate(x, w):
        return float((x * w).sum() / max(w.sum(), 1e-12))

    rows = []
    for (m, sd, f), r in d.groupby(["model", "seed", "fold"]):
        gm, ng = r[col].to_numpy(), ~r[col].to_numpy()
        if gm.sum() == 0:
            continue
        # life-matched controls: non-glitch windows of the same unit within 50 cycles of a glitch window
        near = np.zeros(len(r), bool)
        uu, cc = r.unit.to_numpy(), r.cycle.to_numpy()
        for u, c in zip(uu[gm], cc[gm]):
            near |= (uu == u) & (np.abs(cc - c) <= 50)
        near &= ng
        row = dict(model=m, seed=sd, fold=f, n_g=int(gm.sum()), opt_g=wrate(r.opt[gm], r.uw[gm]), opt_ng=wrate(r.opt[ng], r.uw[ng]),
                   opt_near=wrate(r.opt[near], r.uw[near]) if near.any() else np.nan)
        for s_ in SC:
            a = r[f"a_{s_}"].to_numpy().astype(bool)
            row[f"risk_g_{s_}"] = wrate(r.opt[gm & a], r.uw[gm & a]) if (gm & a).any() else np.nan
            row[f"rej_g_{s_}"] = 1 - wrate(a[gm], r.uw[gm])
            row[f"risk_ng_{s_}"] = wrate(r.opt[ng & a], r.uw[ng & a])
            row[f"caught_{s_}"] = wrate(~a[gm & r.opt.to_numpy()], r.uw[gm & r.opt.to_numpy()]) if (gm & r.opt.to_numpy()).any() else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    s = t.groupby("seed").mean(numeric_only=True) * 100
    say(f"runs with >=1 glitch window: {len(t)}; glitch windows per such run: {t.n_g.mean():.1f}")
    say(f"(1) predictor optimistic rate (%): glitch {s.opt_g.mean():.2f} vs non-glitch {s.opt_ng.mean():.2f}; diff {(s.opt_g - s.opt_ng).mean():+.2f} "
        f"(seeds higher on glitch {int(((s.opt_g - s.opt_ng) > 0).sum())}/{len(s)}, p={stats.wilcoxon(s.opt_g - s.opt_ng).pvalue:.2g})")
    dm = (s.opt_g - s.opt_near).dropna()
    say(f"(1b) life-matched (same unit, within 50 cycles): glitch {s.opt_g.mean():.2f} vs near non-glitch {s.opt_near.mean():.2f}; diff {dm.mean():+.2f} (p={stats.wilcoxon(dm).pvalue if (dm != 0).any() else 1:.2g})")
    say("(2/3) per selector, mean over seeds (%): risk on accepted glitch windows | rejected share of glitch windows | share of optimistic glitch windows rejected | risk on accepted non-glitch")
    for s_ in SC:
        say(f"  {s_:17s} {s[f'risk_g_{s_}'].mean():6.2f} | {s[f'rej_g_{s_}'].mean():6.2f} | {s[f'caught_{s_}'].mean():6.2f} | {s[f'risk_ng_{s_}'].mean():6.2f}")
    for a, b_ in [("rpt_auto_d", "full"), ("rpt_auto_d", "gauss_bias_aware"), ("rpt_auto_d", "err_reg"), ("rpt_auto_d", "ensemble_std"), ("err_reg_rptL", "full")]:
        dif = (s[f"risk_g_{a}"] - s[f"risk_g_{b_}"]).dropna()
        say(f"  risk_g {a} - {b_}: {dif.mean():+.2f} points, better seeds {int((dif < 0).sum())}/{len(dif)}, p={stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1:.2g}")
(out / "ext41_output.txt").write_text("\n".join(lines) + "\n")
print("EXT41_SUMMARY_DONE")
