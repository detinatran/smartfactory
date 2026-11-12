"""EXT10 (protocol: EXT10_PROTOCOL.md): ext8_select.py unchanged plus randomized-perturbation training (RPT):
full_rpt / full_recon_rpt = the full / full_recon HGB trained on selector x {clean, dropout, drift, rand}, where rand
is the clean selector set perturbed by random smooth functions (see randomize()).

EXT8 description:
safeguarded stacking candidates. Identical to ext7_stack.py (same reconstruction,
features, HGB fits and EXT7 scores) plus four candidates that try to keep EXT7's C-MAPSS gain without its bearings loss:
  stack_plus  logistic stacker on [logit full, logit gauss_bias_aware, logit full_recon, logit gba_recon, impact], C=0.1
  blend       mean of clean-calibration ECDF ranks of full and stack (fixed 50/50 hedge, nothing fitted)
  stack_cv    stack if its 2-fold out-of-fold calibration risk (matched 90%, mean over SEEN shifts) < that of full,
              else full
  select_cv   per run, the lowest calibration risk among {full, full_recon, gauss_bias_aware, gba_recon, stack (OOF)};
              ties -> full
Any stacker that sees a single class falls back to full.

EXT6 description, for reference:

Per run: per-channel Ridge reconstruction of each sensor from the other sensors (+ settings on C-MAPSS) at the same time
step, fitted on clean fit-unit time steps without missing values. z = (x - xhat) / sigma; window suspicion s_c = mean_t|z|;
tau_c = 99.5% quantile of s_c on clean fit windows; a channel is suspicious if s_c > tau_c or it has missing values.
Suspicious channels are replaced by their reconstruction (3 sweeps, missing flags set to 0) and the ensemble is re-run.

Scores on the original prediction: ensemble_std, full, gauss_bias_aware, impact (= yhat - yhat_rep), full_recon
(full + impact, n_susp, max_ratio). On the repaired prediction: repair_full (classifier retrained on repaired errors).
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_baselines import cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import mc_draws, unit_risk  # noqa: E402

from selective_rul.data import SENSORS, SETTINGS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

SEEN = ("clean", "dropout", "drift")
EVAL = ("clean", "stuck", "bias", "noise", "intermittent", "spike", "gain")
HGB = dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)
ALPHA = 0.9


class Reconstructor:
    def __init__(self, n_ch, flag_off, n_set, seed, max_rows=200_000):
        self.n_ch, self.flag_off, self.n_set, self.seed, self.max_rows = n_ch, flag_off, n_set, seed, max_rows

    def _inputs(self, v, s, c):
        return np.column_stack([np.delete(v, c, axis=1), s]) if s is not None else np.delete(v, c, axis=1)

    def _split(self, x2):
        v = x2[:, : self.n_ch]
        s = x2[:, self.n_ch: self.n_ch + self.n_set] if self.n_set else None
        return v, s

    def fit(self, x):
        n, t, _ = x.shape
        flat = x.reshape(-1, x.shape[2])
        ok = flat[:, self.flag_off: self.flag_off + self.n_ch].sum(1) == 0
        rows = np.flatnonzero(ok)
        rng = np.random.default_rng(self.seed)
        if len(rows) > self.max_rows:
            rows = rng.choice(rows, self.max_rows, replace=False)
        v, s = self._split(flat[rows])
        self.models, self.sigma = [], np.empty(self.n_ch)
        for c in range(self.n_ch):
            m = Ridge(alpha=1.0).fit(self._inputs(v, s if s is not None else None, c), v[:, c])
            self.models.append(m)
            self.sigma[c] = max(float(np.std(v[:, c] - m.predict(self._inputs(v, s, c)))), 1e-6)
        clean_s = self.suspicion(x)[0]
        self.tau = np.quantile(clean_s, 0.995, axis=0)
        return self

    def reconstruct(self, x):
        n, t, d = x.shape
        flat = x.reshape(-1, d)
        v, s = self._split(flat)
        xh = np.column_stack([self.models[c].predict(self._inputs(v, s, c)) for c in range(self.n_ch)])
        return xh.reshape(n, t, self.n_ch)

    def suspicion(self, x):
        xh = self.reconstruct(x)
        z = np.abs(x[:, :, : self.n_ch] - xh) / self.sigma
        miss = x[:, :, self.flag_off: self.flag_off + self.n_ch] > 0.5
        z = np.where(miss, 0.0, z)
        return z.mean(1), miss.any(1)

    def repair(self, x):
        s, miss = self.suspicion(x)
        susp = (s > self.tau) | miss
        xr = x.copy()
        if susp.any():
            xr[:, :, self.flag_off: self.flag_off + self.n_ch] = np.where(
                susp[:, None, :], 0.0, xr[:, :, self.flag_off: self.flag_off + self.n_ch])
            for _ in range(3):
                xh = self.reconstruct(xr)
                xr[:, :, : self.n_ch] = np.where(susp[:, None, :], xh, xr[:, :, : self.n_ch])
        return xr.astype(np.float32), susp, s / self.tau


def randomize(x, n_ch, seed):
    """EXT10 generator: per window, k ~ U{1..max(1, round(0.2 n_ch))} value channels get A * phi(t), phi a random
    3-term cosine series normalised to max|phi| = 1, A ~ LogUniform(0.5, 4) (normalised units), and with prob. 0.5
    phi = 0 before a random onset. Missing flags untouched."""
    rng = np.random.default_rng(seed)
    n, T, _ = x.shape
    out = x.copy()
    kmax = max(1, int(round(0.2 * n_ch)))
    t = np.arange(T) / max(T - 1, 1)
    m = np.arange(3)
    for i in range(n):
        k = int(rng.integers(1, kmax + 1))
        for c in rng.choice(n_ch, k, replace=False):
            a, b = rng.normal(size=3), rng.uniform(0, 2 * np.pi, 3)
            phi = (a[:, None] * np.cos(np.pi * m[:, None] * t[None, :] + b[:, None])).sum(0)
            phi /= max(np.abs(phi).max(), 1e-6)
            amp = float(np.exp(rng.uniform(np.log(0.5), np.log(4.0))))
            if rng.random() < 0.5:
                phi[: int(rng.integers(0, T))] = 0.0
            out[i, :, c] += amp * phi
    return out.astype(x.dtype)


def process(run, raw, ext_dir, device, passes):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import external_features, load_dataset

        n_ch = len(load_dataset(cfg["dataset"], ext_dir)[1])
        flag_off, n_set = n_ch, 0

        def feats_of(w_x, meta, ens):
            return external_features(type("W", (), {"x": w_x, "meta": meta})(), ens, n_ch)
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)
        n_ch, n_set = len(SENSORS), len(SETTINGS)
        flag_off = n_ch + n_set

        def feats_of(w_x, meta, ens):
            return selector_features(type("W", (), {"x": w_x, "meta": meta})(), ens, cfg["rul_cap"], None)
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())

    fitw = windows("fit", "clean")
    assert fitw.x.shape[2] == flag_off + n_ch, (fitw.x.shape, flag_off, n_ch)
    rec = Reconstructor(n_ch, flag_off, n_set, seed).fit(fitw.x)
    del fitw

    def prep(w, k):
        ens, _ = predict_details(models, w.x, device)
        dr = mc_draws(models, w.x, device, passes, seed + 30000 + k)
        xr, susp, ratio = rec.repair(w.x)
        ens_r, _ = predict_details(models, xr, device)
        extra = np.column_stack([ens.mean(1) - ens_r.mean(1), susp.sum(1), ratio.max(1)])
        f = feats_of(w.x, w.meta, ens)
        fr = feats_of(xr, w.meta, ens_r)
        return dict(x=w.x, ens=ens, ens_r=ens_r, susp=susp, f=f, fx=np.column_stack([f, extra]),
                    frx=np.column_stack([fr, extra]), mce=dr.reshape(-1, dr.shape[-1]).std(0),
                    ties=tie_breakers(w.meta, seed + 20000), y=w.meta.rul_raw.to_numpy(), yc=np.asarray(w.y),
                    units=w.meta.unit.to_numpy())

    S = [prep(windows("selector", sc), k) for k, sc in enumerate(SEEN)]
    _b = windows("selector", "clean")
    SA = prep(type("W", (), {"x": randomize(_b.x, n_ch, seed + 50000), "meta": _b.meta, "y": _b.y})(), 9)
    del _b
    cat = lambda key: np.concatenate([s[key] for s in S])  # noqa: E731
    y_s = cat("y")
    err = cat("ens").mean(1) - y_s
    err_r = cat("ens_r").mean(1) - y_s
    U = cat("units")
    wts = engine_weights(U)
    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)
    clf = HistGradientBoostingClassifier(**mk).fit(cat("f"), (err > delta).astype(int), sample_weight=wts)
    clf_x = HistGradientBoostingClassifier(**mk).fit(cat("fx"), (err > delta).astype(int), sample_weight=wts)
    clf_r = HistGradientBoostingClassifier(**mk).fit(cat("frx"), (err_r > delta).astype(int), sample_weight=wts)
    catA = lambda key: np.concatenate([s[key] for s in S + [SA]])  # noqa: E731
    errA = catA("ens").mean(1) - catA("y")
    wA = engine_weights(catA("units"))
    clf_rpt = HistGradientBoostingClassifier(**mk).fit(catA("f"), (errA > delta).astype(int), sample_weight=wA)
    clf_x_rpt = HistGradientBoostingClassifier(**mk).fit(catA("fx"), (errA > delta).astype(int), sample_weight=wA)
    F = cat("f")
    bias = HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=wts)
    uu = np.unique(U)
    fold = np.isin(U, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])
    b_oof = np.empty_like(err)
    for msk in (fold, ~fold):
        b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(F[~msk], err[~msk], sample_weight=wts[~msk]).predict(F[msk])
    var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(F, (err - b_oof) ** 2, sample_weight=wts)
    FX = cat("fx")
    bias_x = HistGradientBoostingRegressor(**mk).fit(FX, err, sample_weight=wts)
    bx_oof = np.empty_like(err)
    for msk in (fold, ~fold):
        bx_oof[msk] = HistGradientBoostingRegressor(**mk).fit(FX[~msk], err[~msk],
                                                              sample_weight=wts[~msk]).predict(FX[msk])
    var_x = HistGradientBoostingRegressor(loss="poisson", **mk).fit(FX, (err - bx_oof) ** 2, sample_weight=wts)
    eps = 1e-6

    def logit(q):
        q = np.clip(q, eps, 1 - eps)
        return np.log(q / (1 - q))

    def base_scores(p):
        b = bias.predict(p["f"])
        s = np.sqrt(np.maximum(var.predict(p["f"]), 1e-12) + p["mce"] ** 2)
        bx = bias_x.predict(p["fx"])
        sx = np.sqrt(np.maximum(var_x.predict(p["fx"]), 1e-12) + p["mce"] ** 2)
        return {"ensemble_std": ("orig", p["ens"].std(1)), "full": ("orig", clf.predict_proba(p["f"])[:, 1]),
                "gauss_bias_aware": ("orig", norm.sf((delta - b) / s)),
                "impact": ("orig", p["ens"].mean(1) - p["ens_r"].mean(1)),
                "full_recon": ("orig", clf_x.predict_proba(p["fx"])[:, 1]),
                "repair_full": ("rep", clf_r.predict_proba(p["frx"])[:, 1]),
                "gba_recon": ("orig", norm.sf((delta - bx) / sx)),
                "full_rpt": ("orig", clf_rpt.predict_proba(p["f"])[:, 1]),
                "full_recon_rpt": ("orig", clf_x_rpt.predict_proba(p["fx"])[:, 1])}

    def stack_inputs(sc):
        return np.column_stack([logit(sc["full_recon"][1]), logit(sc["gba_recon"][1]), sc["impact"][1]])

    def stack_inputs_plus(sc):
        return np.column_stack([logit(sc["full"][1]), logit(sc["gauss_bias_aware"][1]), logit(sc["full_recon"][1]),
                                logit(sc["gba_recon"][1]), sc["impact"][1]])

    def fit_lr(Z, y, w, c):
        if len(np.unique(y)) < 2:
            return None
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000)).fit(
            Z, y, logisticregression__sample_weight=w)

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    # calibration units x SEEN shifts: labelled, disjoint from selector and evaluation units
    CS = [prep(windows("calibration", sc), 10 + k) for k, sc in enumerate(SEEN)]
    cal_sc = [base_scores(c) for c in CS]
    Zl = [stack_inputs(sc) for sc in cal_sc]
    Z, ZP = np.concatenate(Zl), np.concatenate([stack_inputs_plus(sc) for sc in cal_sc])
    yl = [(c["ens"].mean(1) - c["y"] > delta).astype(int) for c in CS]
    yz = np.concatenate(yl)
    wz = np.concatenate([engine_weights(c["units"]) for c in CS])
    stacker = fit_lr(Z, yz, wz, 1.0)
    stacker_p = fit_lr(ZP, yz, wz, 0.1)
    coef = stacker[-1].coef_.ravel() if stacker is not None else np.full(3, np.nan)

    # 2-fold (by calibration unit) out-of-fold stack scores, for the CV guards
    cu = np.unique(np.concatenate([c["units"] for c in CS]))
    half = np.random.default_rng(seed + 40000).permutation(cu)[: len(cu) // 2]
    oof = [np.empty(len(c["y"])) for c in CS]
    for side in (True, False):
        tr = [np.isin(c["units"], half) != side for c in CS]
        m = fit_lr(np.concatenate([z[t] for z, t in zip(Zl, tr)]), np.concatenate([y[t] for y, t in zip(yl, tr)]),
                   np.concatenate([engine_weights(c["units"][t]) if t.any() else np.zeros(0) for c, t in zip(CS, tr)]),
                   1.0) if all(t.any() for t in tr) else None
        for k, c in enumerate(CS):
            te = ~tr[k]
            if te.any():
                oof[k][te] = m.predict_proba(Zl[k][te])[:, 1] if m is not None else cal_sc[k]["full"][1][te]

    def cal_risk(vals):
        out = []
        for c, v in zip(CS, vals):
            acc = apply_threshold(v, c["ties"], fit_threshold(v, c["ties"], c["units"], ALPHA))
            out.append(rates(c["ens"].mean(1), c["y"], c["yc"], c["units"], acc)[0])
        return float(np.mean(out))

    cand = {"full": cal_risk([sc["full"][1] for sc in cal_sc])}
    for nm in ("full_recon", "gauss_bias_aware", "gba_recon"):
        cand[nm] = cal_risk([sc[nm][1] for sc in cal_sc])
    cand["stack"] = cal_risk(oof) if stacker is not None else np.inf
    use_stack = bool(cand["stack"] < cand["full"])
    choice = min(cand, key=lambda k: (cand[k], k != "full"))

    C = CS[0]
    c0 = cal_sc[0]
    ref_fr, ref_gb = np.sort(c0["full_recon"][1]), np.sort(c0["gba_recon"][1])
    ref_full = np.sort(c0["full"][1])

    def ecdf(ref, v):
        return np.searchsorted(ref, v, side="right") / len(ref)

    def stack_of(sc):
        return stacker.predict_proba(stack_inputs(sc))[:, 1] if stacker is not None else sc["full"][1]

    ref_st = np.sort(stack_of(c0))

    def scores(p, sc=None):
        sc = dict(sc or base_scores(p))
        sc["stack"] = ("orig", stack_of(sc))
        sc["rank_recon"] = ("orig", 0.5 * (ecdf(ref_fr, sc["full_recon"][1]) + ecdf(ref_gb, sc["gba_recon"][1])))
        sc["stack_plus"] = ("orig", stacker_p.predict_proba(stack_inputs_plus(sc))[:, 1] if stacker_p is not None
                            else sc["full"][1])
        sc["blend"] = ("orig", 0.5 * (ecdf(ref_full, sc["full"][1]) + ecdf(ref_st, sc["stack"][1])))
        sc["stack_cv"] = sc["stack"] if use_stack else sc["full"]
        sc["select_cv"] = sc[choice]
        return sc

    frozen = {k: fit_threshold(v, C["ties"], C["units"], ALPHA) for k, (_, v) in scores(C, c0).items()}

    rows = []
    clean_x = None
    for k, scn in enumerate(EVAL):
        p = prep(windows(eval_part, scn), 20 + k)
        y, yc, units = p["y"], p["yc"], p["units"]
        preds = {"orig": p["ens"].mean(1), "rep": p["ens_r"].mean(1)}
        if scn == "clean":
            clean_x = p["x"]
            false_alarm, sens = float(p["susp"].mean()), np.nan
        else:
            changed = np.abs(p["x"][:, :, :n_ch] - clean_x[:, :, :n_ch]).max(1) > 1e-6
            changed |= (p["x"][:, :, flag_off:] != clean_x[:, :, flag_off:]).any(1)
            false_alarm = float(p["susp"][~changed].mean())
            sens = float(p["susp"][changed].mean()) if changed.any() else np.nan
        allacc = np.ones(len(y), bool)
        for which, pred in preds.items():
            o, pe, r = rates(pred, y, yc, units, allacc)
            rows.append(dict(scenario=scn, score=f"predictor_{which}", opt_full=o, pess_full=pe, rmse_full=r,
                             false_alarm=false_alarm, sensitivity=sens))
        for name, (which, v) in scores(p).items():
            pred = preds[which]
            acc_m = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], units, ALPHA))
            acc_f = apply_threshold(v, p["ties"], frozen[name])
            r_m, pe_m, rmse_m = rates(pred, y, yc, units, acc_m)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            rows.append(dict(scenario=scn, score=name, risk_matched=r_m, pess_matched=pe_m, rmse_matched=rmse_m,
                             risk_frozen=r_f, coverage_frozen=c_f))
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, n_channels=n_ch,
                coef_full_recon=coef[0], coef_gba_recon=coef[1], coef_impact=coef[2],
                stack_fallback=stacker is None, cv_use_stack=use_stack, select_choice=choice,
                **{f"calrisk_{k}": v for k, v in cand.items()})
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
    ap.add_argument("--datasets", nargs="*", default=None)
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
                    cfg.get("head", "point") == "point" and (a.seeds is None or cfg["seed"] in a.seeds) and \
                    (a.datasets is None or cfg.get("dataset", "cmapss") in a.datasets):
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
    print("EXT10_DONE", flush=True)


if __name__ == "__main__":
    main()
