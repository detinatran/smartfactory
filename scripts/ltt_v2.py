"""LTT v2: weaker targets, pooled units, betting (WSR) bound. Offline on saved scores.

Pre-registered in LTT_V2_PROTOCOL.md (2026-11-09) before any result was computed.

Unit-level loss at coverage level k: L_u = b_u - r* a_u (a_u: accepted fraction of unit u's
windows, b_u: accepted-and-optimistic fraction). Risk = sum b / sum a over units (equal unit
weight). Fixed-sequence test over levels 0.05..1.00 (FWER delta = 0.1); abstain if level 0.05
is not certified. Thresholds at level k = unit-weighted k-quantile of the SOURCE scores
(per subset when pooling, so every subset keeps its own threshold for the same level).

Label sources
  cal       : clean calibration units of the run (or of the 4 C-MAPSS subsets, pooled)
  tgt       : labeled target batch A (30% of target units, stratified per subset), applied to B
Bounds: hoeffding, bernstein (Maurer-Pontil), wsr (one-sided betting, Ville), plus plugin.
Targets: r* = f x accept-all risk of the clean CALIBRATION units (pooled when pooling),
f in {1.0, 0.85, 0.7}; the same r* is used for cal and tgt rules.
Scopes: per_run (every run) and cmapss_pooled (4 subsets of one model/seed).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_study import describe  # noqa: E402

METHODS = ("asymmetric_risk", "ensemble_std", "signed_error_q90")
LEVELS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
FACTORS = (1.0, 0.85, 0.7)
DELTA = 0.1


# ------------------------------------------------------------------ bounds (x = unit losses)
def hoeffding_ok(x, r_star):
    return len(x) > 0 and x.mean() + math.sqrt(math.log(1 / DELTA) / (2 * len(x))) < 0


def bernstein_ok(x, r_star):
    n = len(x)
    if n < 2:
        return False
    t = math.log(2 / DELTA)
    return x.mean() + math.sqrt(2 * x.var(ddof=1) * t / n) + 7 * t / (3 * (n - 1)) < 0


def wsr_ok(x, r_star):
    """H0: E[y] >= m with y = x + r* in [0, 1], m = r*. Predictable WSR-style bets."""
    y, m, n = x + r_star, r_star, len(x)
    if n == 0 or m >= 1:
        return False
    capital, mu_sum, var_sum = 1.0, 0.5, 0.25
    for t, value in enumerate(y, start=1):
        mu_hat, sigma2 = mu_sum / t, var_sum / t
        lam = min(math.sqrt(2 * math.log(1 / DELTA) / (n * sigma2)), 0.75 / (1 - m))
        capital *= 1 + lam * (m - value)
        if capital >= 1 / DELTA:
            return True
        mu_sum += value
        var_sum += (value - mu_hat) ** 2
    return False


def plugin_ok(x, r_star):
    return len(x) > 0 and x.mean() <= 0


TESTS = {"hoeffding": hoeffding_ok, "bernstein": bernstein_ok, "wsr": wsr_ok,
         "plugin": plugin_ok}


# ------------------------------------------------------------------ vectorised helpers
class Part:
    """Windows of one subset/run restricted to some units, with per-level unit fractions."""

    def __init__(self, frame, col, delta):
        self.units, codes = np.unique(frame.unit.to_numpy(), return_inverse=True)
        self.codes, self.n_units = codes, len(self.units)
        self.counts = np.bincount(codes, minlength=self.n_units).astype(float)
        self.scores = frame[col].to_numpy(float)
        self.ties = frame.tie_breaker.to_numpy(float)
        self.bad = ((frame.prediction - frame.rul_raw).to_numpy() > delta).astype(float)
        self.w = (1.0 / self.counts)[codes]

    def thresholds(self):
        order = np.lexsort((self.ties, self.scores))
        cum = np.cumsum(self.w[order]) / self.w.sum()
        rows = order[np.minimum(np.searchsorted(cum, LEVELS), len(order) - 1)]
        return self.scores[rows], self.ties[rows]

    def fractions(self, thresholds):
        s, t = thresholds
        acc = (self.scores[:, None] < s) | ((self.scores[:, None] == s) & (self.ties[:, None] <= t))
        a = np.stack([np.bincount(self.codes, acc[:, k], self.n_units) for k in range(len(s))], 1)
        b = np.stack([np.bincount(self.codes, acc[:, k] * self.bad, self.n_units)
                      for k in range(len(s))], 1)
        return a / self.counts[:, None], b / self.counts[:, None]

    def subset(self, mask_units, frame, col, delta):
        return Part(frame[frame.unit.isin(self.units[mask_units])], col, delta)


def certify(sources, r_star, test):
    """sources: list of (a, b) unit x level matrices (thresholds from that source). -> level idx."""
    a = np.vstack([s[0] for s in sources])
    b = np.vstack([s[1] for s in sources])
    chosen = None
    for k in range(len(LEVELS)):
        if not test(b[:, k] - r_star * a[:, k], r_star):
            break
        chosen = k
    return chosen


def outcome(evals, k):
    """evals: list of (a, b) on evaluation units (thresholds from the matching source)."""
    if k is None:
        return 0.0, np.nan
    a = np.concatenate([e[0][:, k] for e in evals])
    b = np.concatenate([e[1][:, k] for e in evals])
    return a.mean(), (b.sum() / a.sum() if a.sum() > 0 else np.nan)


# ------------------------------------------------------------------ one analysis group
def analyse(group, scope, repeats, adapt_fraction, seed):
    """group: list of run infos (1 for per_run, 4 subsets for cmapss_pooled)."""
    delta = group[0]["delta"]
    cals = [pd.read_csv(info["calib"]) for info in group]
    pooled_bad = np.concatenate([((c.prediction - c.rul_raw).to_numpy() > delta) for c in cals])
    pooled_units = np.concatenate([c.unit.to_numpy() + 10_000 * i for i, c in enumerate(cals)])
    per_unit = pd.Series(pooled_bad.astype(float)).groupby(pooled_units).mean()
    base = float(per_unit.mean())  # equal unit weight, as in the risk definition
    scenarios = list(group[0]["evals"])
    targets = {s: [pd.read_csv(info["evals"][s]) for info in group] for s in scenarios}
    rows = []
    for method in METHODS:
        col = f"score_{method}"
        cal_parts = [Part(c, col, delta) for c in cals]
        cal_thr = [p.thresholds() for p in cal_parts]
        cal_src = [p.fractions(t) for p, t in zip(cal_parts, cal_thr)]
        for scenario in scenarios:
            tparts = [Part(t, col, delta) for t in targets[scenario]]
            rng = np.random.default_rng(np.random.SeedSequence([seed, len(group), 6262]))
            for rep in range(repeats):
                masks = []
                for p in tparts:
                    n_a = max(1, round(adapt_fraction * p.n_units))
                    if p.n_units - n_a < 1:
                        masks = None
                        break
                    m = np.zeros(p.n_units, bool)
                    m[rng.choice(p.n_units, n_a, replace=False)] = True
                    masks.append(m)
                if masks is None:
                    break
                a_parts = [p.subset(m, t, col, delta)
                           for p, m, t in zip(tparts, masks, targets[scenario])]
                b_parts = [p.subset(~m, t, col, delta)
                           for p, m, t in zip(tparts, masks, targets[scenario])]
                a_thr = [p.thresholds() for p in a_parts]
                tgt_src = [p.fractions(t) for p, t in zip(a_parts, a_thr)]
                eval_cal = [bp.fractions(t) for bp, t in zip(b_parts, cal_thr)]
                eval_tgt = [bp.fractions(t) for bp, t in zip(b_parts, a_thr)]
                n_b = sum(bp.n_units for bp in b_parts)
                for factor in FACTORS:
                    r_star = factor * base
                    for source, src, ev in (("cal", cal_src, eval_cal), ("tgt", tgt_src, eval_tgt)):
                        n_lab = sum(s[0].shape[0] for s in src)
                        for bound, test in TESTS.items():
                            k = certify(src, r_star, test)
                            cov, risk = outcome(ev, k)
                            rows.append(dict(scope=scope, scenario=scenario, method=method,
                                             factor=factor, source=source, bound=bound,
                                             repeat=rep, r_star=r_star, level=(
                                                 LEVELS[k] if k is not None else 0.0),
                                             abstain_all=k is None, coverage=cov, risk=risk,
                                             n_labeled_units=n_lab, n_eval_units=n_b))
    out = pd.DataFrame(rows)
    first = group[0]
    for key in ("dataset", "model", "seed"):
        out.insert(0, key, first[key])
    out.insert(3, "fold", "pooled" if scope == "cmapss_pooled" else first["fold"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=["per_run", "cmapss_pooled"], required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--adapt-fraction", type=float, default=0.3)
    parser.add_argument("--shard", default="0/1", help="i/N: process every N-th group")
    args = parser.parse_args()
    shard, n_shards = map(int, args.shard.split("/"))
    infos = []
    for root in args.runs:
        for done in sorted(root.rglob("complete.json")):
            info = describe(done.parent)
            if info["variant"] != "raw_softplus":
                infos.append(info)
    if args.scope == "per_run":
        groups = [[i] for i in infos]
    else:
        keyed = {}
        for i in infos:
            if i["dataset"] == "cmapss":
                keyed.setdefault((i["model"], i["seed"]), []).append(i)
        groups = [sorted(g, key=lambda i: i["fold"]) for g in keyed.values() if len(g) == 4]
    groups = groups[shard::n_shards]
    frames = []
    for n, group in enumerate(groups, 1):
        frames.append(analyse(group, args.scope, args.repeats, args.adapt_fraction,
                              group[0]["seed"]))
        print(f"{n}/{len(groups)} {group[0]['dataset']} {group[0]['model']} "
              f"{group[0]['seed']} {group[0]['fold']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    per = pd.concat(frames, ignore_index=True)
    per.to_csv(args.out / "per_repeat.csv.gz", index=False)
    print(json.dumps({"rows": len(per), "groups": len(groups)}))


if __name__ == "__main__":
    main()
