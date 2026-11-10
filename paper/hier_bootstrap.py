"""Hierarchical (seed x physical-unit) bootstrap for unseen-fault risk differences -> outputs/stats/hierarchical_bootstrap.csv.

Input: results/unit_masses.csv.gz (per run, scenario, method, unit: accepted mass a and optimistic mass b at matched 90%).
Replicate: resample seeds (both backbones of a seed together), then units:
  N-CMAPSS, battery, bearings: one shared draw over the dataset's units (paired across runs);
  C-MAPSS: an independent draw per (subset, seed) development-unit set (shared by backbones and methods).
Statistic: mean over (seed, backbone, scenario[, subset]) of risk(one-sided) - risk(comparator), risk = sum b / sum a.
"""
import numpy as np
import pandas as pd

from common import FAULTS, OUT, read

COMP = ["ensemble_std", "mc_dropout", "deep_ens_nll", "selectivenet", "symmetric_risk", "cqr_lower"]
B = 2000
rng = np.random.default_rng(20261109)

d = read("unit_masses.csv.gz")
d = d[d.scenario.isin(FAULTS)]
d["block"] = np.where(d.dataset == "cmapss", d.fold.astype(str) + "|" + d.seed.astype(str), "all")
out = []
for ds, g in d.groupby("dataset"):
    cols = g[["block", "unit"]].drop_duplicates().reset_index(drop=True)
    cols["c"] = np.arange(len(cols))
    g = g.merge(cols, on=["block", "unit"])
    g["sub"] = g.fold.astype(str) if ds == "cmapss" else pd.Series("all", index=g.index)
    keys = ["seed", "model", "scenario", "sub", "method"]
    rows = g[keys].drop_duplicates().reset_index(drop=True)
    rows["r"] = np.arange(len(rows))
    g = g.merge(rows, on=keys)
    A = np.zeros((len(rows), len(cols)))
    Bm = np.zeros_like(A)
    A[g.r, g.c] = g.a
    Bm[g.r, g.c] = g.b
    seeds = np.sort(rows.seed.unique())
    blocks = cols.groupby("block").c.apply(np.array).to_dict()

    def stat(wcol, seed_counts):
        risk = (Bm @ wcol) / np.maximum(A @ wcol, 1e-12)
        piv = rows.assign(risk=risk).pivot_table(index=["seed", "model", "scenario", "sub"], columns="method", values="risk")
        w = piv.index.get_level_values("seed").map(seed_counts).to_numpy(dtype=float)
        res = {}
        for m in COMP + ["aug_vs_asym"]:
            diff = piv["asym_aug"] - piv["asymmetric_risk"] if m == "aug_vs_asym" else piv["asymmetric_risk"] - piv[m]
            ok = diff.notna().to_numpy() & (w > 0)
            res[m] = float(np.sum(diff.to_numpy()[ok] * w[ok]) / np.sum(w[ok]))
        return res

    point = stat(np.ones(len(cols)), {s: 1 for s in seeds})
    boots = {m: [] for m in point}
    for _ in range(B):
        sc = pd.Series(rng.choice(seeds, len(seeds), replace=True)).value_counts().to_dict()
        wcol = np.zeros(len(cols))
        for idx in blocks.values():
            np.add.at(wcol, rng.choice(idx, len(idx), replace=True), 1.0)
        for m, v in stat(wcol, {s: sc.get(s, 0) for s in seeds}).items():
            boots[m].append(v)
    for m in point:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        out.append(dict(dataset=ds, comparator=m, mean=100 * point[m], lo=100 * lo, hi=100 * hi))
    print(ds, "done", flush=True)
pd.DataFrame(out).to_csv(OUT / "stats/hierarchical_bootstrap.csv", index=False)
