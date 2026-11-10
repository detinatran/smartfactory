"""EXT4 (protocol: EXT4_PROTOCOL.md): per-run selection between the paper's classifier and the bias-aware Gaussian,
by risk on labelled calibration units under the SEEN shifts only. Built on ext3_conformal.py (same fits); the
weighted-conformal, Mondrian and Laplace scores are dropped. Extra rows scenario="cal_pool" give the selection criterion.

EXT3 description, for reference:

Same checkpoints, splits, windows and faults as the paper (all GRU/TCN runs, CPU replay). Per run, fitted on selector
units x {clean, dropout, drift} (classifier, bias b, variance v) and calibration units (conformal quantiles):

  full              one-sided HGB classifier (the paper's method)
  gauss_bias_aware  1 - Phi((delta - b) / s),  s^2 = v + mc_ensemble^2
  conformal         b + q s, q = unit-weighted 90% quantile of r = (e - b) / s on clean calibration windows
  conformal_w       weighted split conformal under covariate shift (Tibshirani et al., 2019): calibration pool =
                    calibration units x {clean, dropout, drift}; density ratio from a domain classifier (pool vs. the
                    unlabeled evaluation windows of the scenario); per-window quantile q(x) with the test point's mass
                    at +inf; score b + q(x) s
  mondrian          b + q_k s with q_k the 90% quantile of r within the calibration-pool quartile bin k of the
                    predicted RUL (group-conditional conformal)
  rank_avg          mean of the clean-calibration CDF transforms of `full` and `gauss_bias_aware`
  laplace_ev        last-layer Laplace std with the prior precision chosen by the Laplace evidence (grid 1e-4..1e4)
  ensemble_std      ensemble spread (reference)

Per score and scenario: risk at matched 90% coverage, frozen-threshold risk/coverage, and, for the conformal scores,
the "certify" rule (accept iff the one-sided bound b + q s <= delta): its risk and coverage. Bound coverage
P(e <= b + q s) is reported on all windows and on fault-affected windows only.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext2_pilot import last_layer, wquantile  # noqa: E402
from extra_baselines import cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import mc_draws, unit_risk  # noqa: E402

from selective_rul.data import engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

SEEN = ("clean", "dropout", "drift")
EVAL = ("clean", "stuck", "bias", "noise", "intermittent", "spike", "gain")
HGB = dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)
ALPHA = 0.9
CONFORMAL = ("conformal", "conformal_w", "mondrian")


# ------------------------------------------------------------------ Laplace with evidence-chosen prior
def fit_laplace_ev(models, train, device, grid=10.0 ** np.arange(-4, 5)):
    post = []
    for model in models:
        phi, g = last_layer(model, train.x, device)
        pred = predict_details([model], train.x, device)[0][:, 0]
        noise = max(float(np.mean((pred - train.y) ** 2)), 1e-12)
        jac = phi * g[:, None]
        ggn = jac.T @ jac / noise
        theta = np.concatenate([model.head.weight.detach().cpu().double().numpy().ravel(),
                                model.head.bias.detach().cpu().double().numpy().ravel()])
        p = len(theta)
        ev = [0.5 * p * np.log(lam) - 0.5 * lam * theta @ theta - 0.5 * np.linalg.slogdet(ggn + lam * np.eye(p))[1]
              for lam in grid]
        lam = grid[int(np.argmax(ev))]
        post.append((np.linalg.inv(ggn + lam * np.eye(p)), lam))
    return post


def laplace_std(models, post, x, device):
    var = []
    for model, (cov, _) in zip(models, post):
        phi, g = last_layer(model, x, device)
        jac = phi * g[:, None]
        var.append(np.einsum("ij,jk,ik->i", jac, cov, jac))
    return np.sqrt(np.mean(var, 0))


# ------------------------------------------------------------------ conformal helpers
def weighted_q(r_sorted, cw_sorted, w_test, alpha=ALPHA):
    """Per-test-point weighted conformal quantile; the test point carries mass w_test at +inf."""
    cum = np.cumsum(cw_sorted)
    need = alpha * (cum[-1] + w_test)
    idx = np.searchsorted(cum, need)
    q = np.full(len(w_test), np.inf)
    ok = idx < len(r_sorted)
    q[ok] = r_sorted[idx[ok]]
    return q


def ecdf(ref):
    ref = np.sort(ref)
    return lambda v: np.searchsorted(ref, v, side="right") / len(ref)


def process(run, raw, ext_dir, device, passes):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import external_features, load_dataset

        n_feat = len(load_dataset(cfg["dataset"], ext_dir)[1])

        def feats_of(w, ens):
            return external_features(w, ens, n_feat)
        scale = 1.0
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)

        def feats_of(w, ens):
            return selector_features(w, ens, cfg["rul_cap"], None)
        scale = float(cfg["rul_cap"])
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())

    def prep(w, k):
        ens, _ = predict_details(models, w.x, device)
        dr = mc_draws(models, w.x, device, passes, seed + 30000 + k)
        return dict(ens=ens, f=feats_of(w, ens), mce=dr.reshape(-1, dr.shape[-1]).std(0),
                    ties=tie_breakers(w.meta, seed + 20000),
                    y=w.meta.rul_raw.to_numpy(), units=w.meta.unit.to_numpy(), yc=w.y)

    S = [prep(windows("selector", sc), k) for k, sc in enumerate(SEEN)]
    F = np.concatenate([s["f"] for s in S])
    err = np.concatenate([s["ens"].mean(1) - s["y"] for s in S])
    U = np.concatenate([s["units"] for s in S])
    wts = engine_weights(U)
    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)
    clf = HistGradientBoostingClassifier(**mk).fit(F, (err > delta).astype(int), sample_weight=wts)
    bias = HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=wts)
    uu = np.unique(U)
    fold = np.isin(U, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])
    b_oof = np.empty_like(err)
    for msk in (fold, ~fold):
        b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(F[~msk], err[~msk], sample_weight=wts[~msk]).predict(F[msk])
    var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(F, (err - b_oof) ** 2, sample_weight=wts)

    def core(p):
        b = bias.predict(p["f"])
        s = np.sqrt(np.maximum(var.predict(p["f"]), 1e-12) + p["mce"] ** 2)
        return b, s, clf.predict_proba(p["f"])[:, 1], norm.sf((delta - b) / s)

    # calibration: clean (plain conformal, frozen thresholds, CDF transforms) and the pool (weighted / Mondrian)
    C = [prep(windows("calibration", sc), 10 + k) for k, sc in enumerate(SEEN)]
    cal_core = [core(c) for c in C]
    b0, s0, full0, gba0 = cal_core[0]
    e0 = C[0]["ens"].mean(1) - C[0]["y"]
    q_plain = wquantile((e0 - b0) / s0, engine_weights(C[0]["units"]), ALPHA)
    pool_f = np.concatenate([c["f"] for c in C])
    pool_r = np.concatenate([(c["ens"].mean(1) - c["y"] - cc[0]) / cc[1] for c, cc in zip(C, cal_core)])
    pool_w = np.concatenate([engine_weights(c["units"]) for c in C])
    pool_pred = np.concatenate([c["ens"].mean(1) for c in C])
    edges = np.quantile(pool_pred, [0.25, 0.5, 0.75])
    pool_bin = np.digitize(pool_pred, edges)
    q_bin = np.array([wquantile(pool_r[pool_bin == k], pool_w[pool_bin == k], ALPHA) for k in range(4)])
    F_full, F_gba = ecdf(full0), ecdf(gba0)

    def scores_for(p, b, s, fu, gba, q_w):
        return {"ensemble_std": p["ens"].std(1), "full": fu, "gauss_bias_aware": gba,
                "conformal": b + q_plain * s, "rank_avg": 0.5 * (F_full(fu) + F_gba(gba))}

    sc_cal = scores_for(C[0], b0, s0, full0, gba0, np.full(len(b0), q_plain))
    frozen = {k: fit_threshold(v, C[0]["ties"], C[0]["units"], ALPHA) for k, v in sc_cal.items()}

    rows, clean = [], None
    # selection criterion: risk at matched 90% coverage on calibration units x SEEN shifts (labels of calibration units)
    pool_scores = {}
    for c, cc in zip(C, cal_core):
        sc_c = scores_for(c, cc[0], cc[1], cc[2], cc[3], np.full(len(cc[0]), q_plain))
        for k2, v in sc_c.items():
            pool_scores.setdefault(k2, []).append(v)
    pp = np.concatenate([c["ens"].mean(1) for c in C])
    py = np.concatenate([c["y"] for c in C])
    pu = np.concatenate([c["units"] for c in C])
    pt = np.concatenate([c["ties"] for c in C])
    for name, parts in pool_scores.items():
        v = np.concatenate(parts)
        risks = {}
        for cov in (0.5, 0.6, 0.7, 0.8, 0.9):
            acc = apply_threshold(v, pt, fit_threshold(v, pt, pu, cov))
            risks[cov] = unit_risk(pp, py, pu, acc, delta)[0]
        rows.append(dict(scenario="cal_pool", score=name, risk_matched=risks[0.9],
                         cal_aurc=float(np.mean(list(risks.values())))))
    for k, scn in enumerate(EVAL):
        p = prep(windows(eval_part, scn), 20 + k)
        b, s, fu, gba = core(p)
        q_w = np.full(len(b), q_plain)
        sc = scores_for(p, b, s, fu, gba, q_w)
        pred, y, units = p["ens"].mean(1), p["y"], p["units"]
        e, uw = pred - y, engine_weights(units)
        if scn == "clean":
            clean = dict(pred=pred)
            aff = np.zeros(len(y), bool)
        else:
            aff = np.abs(pred - clean["pred"]) > 1e-6 * max(np.abs(y).max(), 1e-9)
        bounds = {"conformal": b + q_plain * s}
        for name, v in sc.items():
            acc_m = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], units, ALPHA))
            acc_f = apply_threshold(v, p["ties"], frozen[name])
            r_m, _ = unit_risk(pred, y, units, acc_m, delta)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            row = dict(scenario=scn, score=name, risk_matched=r_m, risk_frozen=r_f, coverage_frozen=c_f,
                       frac_affected=float(aff.mean()))
            if name in bounds:
                ok = e <= bounds[name]
                row["bound_coverage"] = float(np.average(ok, weights=uw))
                row["bound_coverage_aff"] = float(np.average(ok[aff], weights=uw[aff])) if aff.any() else np.nan
                acc_c = bounds[name] <= delta
                row["risk_certify"], row["coverage_certify"] = unit_risk(pred, y, units, acc_c, delta)
            rows.append(row)
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed, fold=cfg.get("fold", cfg.get("subset")),
                scale=scale, delta=delta, mc_passes=passes)
    for k, v in reversed(list(meta.items())):
        out.insert(0, k, v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/CMAPSSData"))
    ap.add_argument("--ext-data-dir", type=Path, default=Path("data/external/processed"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=["gru", "tcn"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("model") in a.models and cfg.get("point_target", "capped") != "raw" and \
                    cfg.get("head", "point") == "point" and (a.seeds is None or cfg["seed"] in a.seeds):
                runs.append(done.parent)
    runs = runs[i0::n][: a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs in shard {a.shard}", flush=True)
    for k, run in enumerate(runs, 1):
        dest = a.out / f"{run.parent.name}__{run.name}.csv".replace(":", "_")
        if dest.exists():
            continue
        t = time.time()
        try:
            process(run, a.data_dir, a.ext_data_dir, a.device, a.passes).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT4_DONE", flush=True)


if __name__ == "__main__":
    main()
