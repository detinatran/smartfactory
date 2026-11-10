"""Shared paths, constants and helpers for the paper scripts. Everything is read from results/ and data/ in this repo."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"
OUT = ROOT / "paper" / "outputs"
for sub in ("tables", "figures", "stats"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

DATASETS = ["cmapss", "ncmapss", "bearings", "battery"]
DNAME = {"cmapss": "C-MAPSS", "ncmapss": "N-CMAPSS", "bearings": "Bearings", "battery": "Battery"}
STANDARD = ["clean", "dropout", "drift", "dropout_strong", "drift_strong"]
FAULTS = ["stuck", "bias", "noise", "intermittent", "spike", "gain"]
COMPARATORS = {"ensemble_std": "Ensemble spread", "mc_dropout": "MC dropout", "deep_ens_nll": "Deep ens.\\ (NLL)",
               "selectivenet": "SelectiveNet", "symmetric_risk": "Symmetric classifier", "cqr_lower": "CQR lower bound"}


def read(name, **kw):
    return pd.read_csv(RESULTS / name, **kw)


def write_tex(name, lines):
    (OUT / "tables" / name).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ aggregation over folds
def per_seed(d, metrics, keys=("dataset", "score", "model", "seed", "scenario")):
    """Per-fold replay rows -> one value per (dataset, score, model, seed, scenario), as in the main pipeline:
    C-MAPSS subsets macro-averaged; the six bearing folds pooled with weight = number of evaluation units (the 32
    bearings form one unit-balanced population); NaNs skipped with renormalised weights."""
    um = pd.read_csv(SPLITS / "unit_map.csv")
    w_b = um[um.dataset == "bearings"].groupby("fold").size().to_dict()
    d = d.copy()
    d["_w"] = np.where(d.dataset == "bearings", d.fold.map(w_b), 1.0)
    out = {}
    for m in metrics:
        v = d[m].to_numpy(float)
        ok = np.isfinite(v)
        d["_wv"], d["_wo"] = np.where(ok, v, 0.0) * d._w, np.where(ok, d._w, 0.0)
        g = d.groupby(list(keys))[["_wv", "_wo"]].sum()
        out[m] = g._wv / g._wo.replace(0, np.nan)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ tests
def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_details(dif):
    """Two-sided Wilcoxon signed-rank with scipy defaults, plus zero/tie counts and exact/normal p-values."""
    dif = np.asarray(dif, float)
    nz = dif[dif != 0]
    _, counts = np.unique(np.round(np.abs(nz), 12), return_counts=True)
    if not len(nz):
        return dict(n_pairs=len(dif), n_zero=len(dif), n_tied_abs=0, W=np.nan, p=1.0, p_exact=1.0, p_normal=1.0, method="-")
    res = wilcoxon(dif)
    return dict(n_pairs=len(dif), n_zero=int((dif == 0).sum()), n_tied_abs=int(counts[counts > 1].sum()),
                W=float(res.statistic), p=float(res.pvalue), p_exact=float(wilcoxon(nz, method="exact").pvalue),
                p_normal=float(wilcoxon(nz, method="approx", correction=False).pvalue),
                method="exact" if (len(dif) <= 50 and not (dif == 0).any()) else "normal approx.")


def seed_bootstrap(dif, rng, reps):
    """95% interval of the mean paired difference, resampling seeds (both backbones of a seed together)."""
    seeds = np.array(sorted(dif.index.get_level_values("seed").unique()))
    by_seed = dif.groupby(level="seed").apply(list)
    boots = [np.mean(np.concatenate([by_seed[s] for s in rng.choice(seeds, len(seeds), replace=True)])) for _ in range(reps)]
    return np.percentile(boots, [2.5, 97.5])
