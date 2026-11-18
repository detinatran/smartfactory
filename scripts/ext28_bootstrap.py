"""EXT28 hierarchical bootstrap (EXT28_PROTOCOL.md).
usage: ext28_bootstrap.py UNITS.csv.gz A B LABEL OUTDIR   (difference A - B, e.g. rpt_L full)
Resample seeds (both backbones together), then test units within each selected run (same units for all scores and
scenarios of that run). Risk = sum o / sum a; pooled as the primary metric. B = 4000, seed 2828."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fu, A, B_, label, out = sys.argv[1:6]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
FA = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
BW = {"femto_c1": 7, "femto_c2": 7, "femto_c3": 3, "xjtu_sy_c1": 5, "xjtu_sy_c2": 5, "xjtu_sy_c3": 5}
NB = 4000
d = pd.read_csv(fu)
d = d[d.scenario.isin(FA) & d.score.isin([A, B_])]
d["dataset"] = np.where(d.dataset.str.startswith("bearings"), "bearings", d.dataset)
rng = np.random.default_rng(2828)
rows = []
for ds, g in d.groupby("dataset"):
    # per run: arrays [unit, scenario, score] of a and o
    runs = {}
    for key, r in g.groupby(["model", "seed", "fold"]):
        units = np.sort(r.unit.unique())
        ui = {u: i for i, u in enumerate(units)}
        a = np.zeros((len(units), len(FA), 2))
        o = np.zeros_like(a)
        si = {s: i for i, s in enumerate(FA)}
        ci = {A: 0, B_: 1}
        idx = (r.unit.map(ui).to_numpy(), r.scenario.map(si).to_numpy(), r.score.map(ci).to_numpy())
        np.add.at(a, idx, r.a.to_numpy())
        np.add.at(o, idx, r.o.to_numpy())
        runs[key] = (a, o)
    seeds = sorted({k[1] for k in runs})
    models = sorted({k[0] for k in runs})

    def pair_diff(model, seed, resample):
        num = den = 0.0
        for (m, s, f), (a, o) in runs.items():
            if m != model or s != seed:
                continue
            n = a.shape[0]
            pick = rng.integers(0, n, n) if resample else np.arange(n)
            aa, oo = a[pick].sum(0), o[pick].sum(0)
            risk = np.where(aa > 0, oo / np.maximum(aa, 1e-12), np.nan)  # [scenario, score]
            dr = np.nanmean(risk[:, 0] - risk[:, 1])
            w = BW.get(f, 1.0) if ds == "bearings" else 1.0
            num += w * dr
            den += w
        return num / den if den else np.nan

    point = np.nanmean([pair_diff(m, s, False) for s in seeds for m in models])
    boot = np.empty(NB)
    for b in range(NB):
        ss = rng.choice(seeds, len(seeds), replace=True)
        boot[b] = np.nanmean([pair_diff(m, s, True) for s in ss for m in models])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append(dict(dataset=ds, comparison=f"{A} - {B_}", label=label, point=100 * point, lo=100 * lo, hi=100 * hi,
                     frac_ge0=float((boot >= 0).mean())))
    print(rows[-1], flush=True)
t = pd.DataFrame(rows)
t.to_csv(out / f"ext28_{label}.csv", index=False)
print(t.round(3).to_string(index=False))
print("EXT28_BOOT_DONE")
