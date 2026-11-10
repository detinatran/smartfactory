"""UNCERTAINTY_V1 (exploratory pilot): uncertainty decomposition and a bias-aware predictive distribution.

Same checkpoints, splits, windows and fault injection as ablation_v5 (CPU replay). Per run, every score is fitted
on selector units x {clean, dropout, drift} only, with fixed HGB settings, and evaluated on clean + six unseen faults.

Uncertainty components (per window):
  ensemble_std  epistemic, between-model: std of the M = 3 member predictions
  mc_dropout    epistemic, within-model: MC dropout (T passes per member, dropout active), mean over members of the std
  mc_ensemble   epistemic, total: std over all M x T stochastic predictions (MC-dropout ensemble)
  b(x)          conditional bias E[yhat - y | x]   (HGB regression; out-of-fold residuals by unit for the next step)
  v(x)          conditional error variance E[(yhat - y - b)^2 | x]  (HGB, Poisson loss)  -> resid_sigma = sqrt(v)
Scores (higher = abstain first):
  ensemble_std, mc_dropout, mc_ensemble, resid_sigma        variance-only uncertainty
  gauss_unbiased   = 1 - Phi(delta / s),          s^2 = v(x) + mc_ensemble^2      (ranks like s: variance-only)
  gauss_bias_aware = 1 - Phi((delta - b(x)) / s)                                    (bias-aware predictive distribution)
  bias_only        = b(x)
  full             one-sided HGB classifier of 1[yhat - y > delta] (as in the paper)
  full_plus_unc    the same classifier with mc_dropout and mc_ensemble as extra features
Per score and scenario: risk at matched 90% coverage, frozen threshold (clean calibration) risk/coverage, share of
fault-induced optimistic errors rejected, and on fault-affected windows the Spearman correlation between the
prediction shift and the score change, and the mean score change on upward- / downward-shifted windows.
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
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_baselines import cmapss_windows, external_windows  # noqa: E402

from selective_rul.data import engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

SEEN = ("clean", "dropout", "drift")
EVAL = ("clean", "stuck", "bias", "noise", "intermittent", "spike", "gain")
HGB = dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)


@torch.no_grad()
def mc_draws(models, x, device, passes, seed, batch=2048):
    """Array (M, T, n) of predictions with dropout active."""
    torch.manual_seed(seed)
    out = []
    for model in models:
        model.train()
        draws = []
        for _ in range(passes):
            draws.append(np.concatenate([model(torch.from_numpy(x[i:i + batch]).to(device)).cpu().numpy().reshape(-1)
                                         for i in range(0, len(x), batch)]))
        model.eval()
        out.append(np.stack(draws))
    return np.stack(out)


def unit_risk(pred, y, units, acc, delta):
    w = engine_weights(units) * acc
    return float((w * ((pred - y) > delta)).sum() / max(w.sum(), 1e-12)), float(w.sum() / engine_weights(units).sum())


def process(run, raw, ext_dir, device, passes):
    cfg = json.loads((run / "config.json").read_text())
    external = "dataset" in cfg
    if external:
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

    def prep(w, tag):
        ens, _ = predict_details(models, w.x, device)
        dr = mc_draws(models, w.x, device, passes, seed + 30000 + tag)
        unc = dict(ensemble_std=ens.std(1), mc_dropout=dr.std(1).mean(0), mc_ensemble=dr.reshape(-1, dr.shape[-1]).std(0))
        f = feats_of(w, ens)
        f_unc = np.column_stack([f, unc["mc_dropout"] / scale, unc["mc_ensemble"] / scale])
        return ens.mean(1), f, f_unc, unc, tie_breakers(w.meta, seed + 20000)

    F, FU, P, Y, U = [], [], [], [], []
    for k, sc in enumerate(SEEN):
        w = windows("selector", sc)
        pred, f, fu, _, _ = prep(w, k)
        F.append(f), FU.append(fu), P.append(pred), Y.append(w.meta.rul_raw.to_numpy()), U.append(w.meta.unit.to_numpy())
    F, FU, P, Y, U = map(np.concatenate, (F, FU, P, Y, U))
    err = P - Y
    bad = (err > delta).astype(int)
    wts = engine_weights(U)
    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)
    clf = HistGradientBoostingClassifier(**mk).fit(F, bad, sample_weight=wts)
    clf_u = HistGradientBoostingClassifier(**mk).fit(FU, bad, sample_weight=wts)
    bias = HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=wts)
    # out-of-fold bias (2 folds by unit) so that the variance model sees honest residuals
    uu = np.unique(U)
    fold = np.isin(U, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])
    b_oof = np.empty_like(err)
    for msk in (fold, ~fold):
        b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(F[~msk], err[~msk], sample_weight=wts[~msk]).predict(F[msk])
    var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(F, (err - b_oof) ** 2, sample_weight=wts)

    def scores(f, fu, unc):
        b = bias.predict(f)
        v = np.maximum(var.predict(f), 1e-12)
        s = np.sqrt(v + unc["mc_ensemble"] ** 2)
        return {**unc, "resid_sigma": np.sqrt(v), "gauss_unbiased": norm.sf(delta / s),
                "gauss_bias_aware": norm.sf((delta - b) / s), "bias_only": b,
                "full": clf.predict_proba(f)[:, 1], "full_plus_unc": clf_u.predict_proba(fu)[:, 1]}

    cal = windows("calibration", "clean")
    _, f_c, fu_c, unc_c, t_c = prep(cal, 10)
    s_c = scores(f_c, fu_c, unc_c)
    frozen = {k: fit_threshold(v, t_c, cal.meta.unit.to_numpy(), 0.9) for k, v in s_c.items()}

    rows, clean = [], None
    for k, sc in enumerate(EVAL):
        w = windows(eval_part, sc)
        pred, f, fu, unc, ties = prep(w, 20 + k)
        y, units = w.meta.rul_raw.to_numpy(), w.meta.unit.to_numpy()
        sc_scores = scores(f, fu, unc)
        if sc == "clean":
            clean = dict(pred=pred, scores=sc_scores)
        e = pred - y
        for name, s in sc_scores.items():
            acc_m = apply_threshold(s, ties, fit_threshold(s, ties, units, 0.9))
            acc_f = apply_threshold(s, ties, frozen[name])
            r_m, _ = unit_risk(pred, y, units, acc_m, delta)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            row = dict(scenario=sc, score=name, risk_matched=r_m, risk_frozen=r_f, coverage_frozen=c_f,
                       mean_score=float(np.mean(s)))
            if sc != "clean":
                shift = pred - clean["pred"]
                aff = np.abs(shift) > 1e-6 * max(np.abs(y).max(), 1e-9)
                new = (e > delta) & ~((clean["pred"] - y) > delta)
                ds = s - clean["scores"][name]
                up, dn = aff & (shift > 0), aff & (shift < 0)
                row.update(frac_affected=float(aff.mean()),
                           reject_new=float((~acc_m[new]).mean()) if new.sum() >= 5 else np.nan,
                           rho_shift=float(spearmanr(shift[aff], ds[aff])[0]) if aff.sum() >= 10 else np.nan,
                           dscore_up=float(ds[up].mean()) if up.any() else np.nan,
                           dscore_down=float(ds[dn].mean()) if dn.any() else np.nan,
                           shift_up=float(shift[up].mean()) if up.any() else np.nan)
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
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--passes", type=int, default=10)
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
            if cfg.get("model") in ("gru", "tcn") and cfg.get("point_target", "capped") != "raw" and \
                    (a.seeds is None or cfg["seed"] in a.seeds):
                runs.append(done.parent)
    runs = runs[i0::n][: a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
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
    print("UNC1_DONE", flush=True)


if __name__ == "__main__":
    main()
