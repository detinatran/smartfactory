"""Controlled simulation of common-mode vs member-specific fault effects (REVIEW_V4_PROTOCOL.md, item 7).

Three ensemble members predict y = y0(x) + z (z: aleatoric noise the members cannot see):
  yhat_m = y0(x) + b(x) + eps_m,  eps_m ~ N(0, s(x)^2),  y0 = 3*x3 (an 'age' feature), b = 0.5 sin(2 pi x1)
A fault (on half of the samples) adds a common shift d and member-specific terms u_m ~ N(0, (kappa |d|)^2).
Training faults: d ~ N(0, 0.5^2), kappa_train = 0.5. Test faults: d ~ N(0, 1.5^2), kappa in {0, .25, .5, 1, 2}.
Scores: ensemble spread; one-sided HGB classifier for E > delta on [mean prediction, spread, x1, x2, x3].
Metric: share of fault-induced optimistic errors (optimistic with the fault, not without) rejected at
90% coverage, and overall risk at 90% coverage. 20 seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DELTA = 1.0
HGB = dict(max_iter=100, max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)


def draw(n, rng, d_sd, kappa, fault_share=0.5):
    x = rng.uniform(size=(n, 3))
    y0, b, s = 3 * x[:, 2], 0.5 * np.sin(2 * np.pi * x[:, 0]), 0.3 + 0.5 * x[:, 1]
    y = y0 + rng.normal(0, 0.5, n)
    eps = rng.normal(0, 1, (n, 3)) * s[:, None]
    clean = y0[:, None] + b[:, None] + eps
    faulty = rng.random(n) < fault_share
    d = rng.normal(0, d_sd, n) * faulty
    u = rng.normal(0, 1, (n, 3)) * (kappa * np.abs(d))[:, None]
    shifted = clean + d[:, None] + u
    return x, y, clean, shifted, faulty


def feats(x, members):
    return np.column_stack([members.mean(1), members.std(1), x])


def accept90(score):
    order = np.argsort(score, kind="mergesort")
    acc = np.zeros(len(score), bool)
    acc[order[: int(0.9 * len(score))]] = True
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 777]))
        x, y, clean, shifted, _ = draw(6000, rng, 0.5, 0.5)
        f_tr = feats(x, shifted)
        clf = HistGradientBoostingClassifier(random_state=seed, **HGB).fit(f_tr, shifted.mean(1) - y > DELTA)
        for kappa in (0.0, 0.25, 0.5, 1.0, 2.0):
            x, y, clean, shifted, faulty = draw(50000, rng, 1.5, kappa)
            e_clean, e_fault = clean.mean(1) - y, shifted.mean(1) - y
            new = (e_fault > DELTA) & ~(e_clean > DELTA)
            for name, score in {"spread": shifted.std(1), "one_sided": clf.predict_proba(feats(x, shifted))[:, 1],
                                "random": rng.random(len(y))}.items():
                acc = accept90(score)
                rows.append(dict(seed=seed, kappa=kappa, score=name, reject_new=float((~acc[new]).mean()),
                                 risk90=float((e_fault[acc] > DELTA).mean()),
                                 spread_ratio_fault_vs_clean=float(shifted.std(1)[faulty].mean() / clean.std(1)[faulty].mean())))
    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "sim_common_shift.csv", index=False)
    summ = df.groupby(["kappa", "score"])[["reject_new", "risk90", "spread_ratio_fault_vs_clean"]].mean()
    summ.to_csv(a.out / "sim_common_shift_summary.csv")
    print(summ.round(3).to_string())
    print(json.dumps({"rows": len(df)}))


if __name__ == "__main__":
    main()
