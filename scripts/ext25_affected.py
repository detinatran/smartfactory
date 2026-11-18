"""EXT25 (protocol: EXT25_PROTOCOL.md): risk on windows the fault actually changes (affected windows) and new baselines
eq90 (HGB quantile regression of E), pca_spe / pca_t2 (PCA sensor validation). Predictor, selectors and rpt_auto as ext22.py.\nOriginal EXT12 doc: EXT12 (protocol: EXT12_PROTOCOL.md): candidates for the final method, directional baselines, and optional stress
scenarios. Shared by stages D (seeds 42-61), C and S (72-81), R (robust-predictor mirror runs) and B (transformer).

`full`, `gauss_bias_aware` and `full_recon_rpt` follow ext10_rpt.py exactly (same features, seeds, MC draws), so they
reproduce EXT10 on the same runs.
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
from ext10_rpt import ALPHA, EVAL, HGB, SEEN, Reconstructor, cmapss_windows, external_windows  # noqa: E402
from ext11_ablate import perturb  # noqa: E402
from uncertainty_v1 import mc_draws, unit_risk  # noqa: E402

from selective_rul.data import SENSORS, SETTINGS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.95)  # risk-coverage curve (0.9 is risk_matched)
STRESS = ("clip", "lag", "quant", "invert", "bias@1.5", "bias@6", "noise@1", "noise@4", "spike@3", "spike@12",
          "gain@1.25", "gain@2", "stuck#1", "stuck#6", "bias#1", "bias#6", "noise#6")


def hist_feats(x, meta, n_ch, ref_x=None):
    """EXT18 unit-history features: deviation of the window from the unit's earliest window (ref from ref_x if given)."""
    src = x if ref_x is None else ref_x
    units, cyc = meta.unit.to_numpy(), meta.cycle.to_numpy()
    m, sd = x[:, :, :n_ch].mean(1), x[:, :, :n_ch].std(1)
    ms, sds = src[:, :, :n_ch].mean(1), src[:, :, :n_ch].std(1)
    ref_m, ref_s = np.empty_like(m), np.empty_like(sd)
    for u in np.unique(units):
        idx = np.flatnonzero(units == u)
        j = idx[np.argmin(cyc[idx])]
        ref_m[idx], ref_s[idx] = ms[j], sds[j]
    d = m - ref_m
    r = np.log((sd + 1e-3) / (ref_s + 1e-3))
    return np.column_stack([d, r, np.abs(d).max(1), np.abs(d).mean(1)])


def process(run, raw, ext_dir, device, passes, extra_scenarios):
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
    globals_box = {}
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())


    def prep(x, meta, y, k=None, ref_x=None):
        ens, _ = predict_details(models, x, device)
        f = feats_of(x, meta, ens)
        mce = None
        if k is not None:  # MC draws only where a Gaussian score is evaluated (same seeds as ext10_rpt.py)
            mce = mc_draws(models, x, device, passes, seed + 30000 + k)
            mce = mce.reshape(-1, mce.shape[-1]).std(0)
        return dict(ens=ens, f=f, mce=mce, ties=tie_breakers(meta, seed + 20000), x=x,
                    y=meta.rul_raw.to_numpy(), yc=np.asarray(y), units=meta.unit.to_numpy())

    S = [prep(w.x, w.meta, w.y) for w in (windows("selector", sc) for sc in SEEN)]
    b = windows("selector", "clean")
    P = lambda x: prep(x, b.meta, b.y, ref_x=b.x)  # noqa: E731
    base = seed + 50000
    L = dict(amp=(1.0, 8.0), kfrac=0.5)
    aug = {
        "smooth_L1": P(perturb(b.x, n_ch, base, **L)),
        "smooth_L2": P(perturb(b.x, n_ch, seed + 51000, **L)),
    }
    del b
    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)

    def data(parts, key):
        X = np.concatenate([p[key] for p in parts])
        err = np.concatenate([p["ens"].mean(1) - p["y"] for p in parts])
        return X, err, engine_weights(np.concatenate([p["units"] for p in parts])), \
            np.concatenate([p["units"] for p in parts])

    def clf(parts, key):
        X, err, w, _ = data(parts, key)
        return key, HistGradientBoostingClassifier(**mk).fit(X, (err > delta).astype(int), sample_weight=w)

    def gauss(parts):
        F, err, w, U = data(parts, "f")
        bias = HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=w)
        uu = np.unique(U)
        fold = np.isin(U, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])
        b_oof = np.empty_like(err)
        for msk in (fold, ~fold):
            b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(F[~msk], err[~msk],
                                                                 sample_weight=w[~msk]).predict(F[msk])
        var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(F, (err - b_oof) ** 2, sample_weight=w)
        return bias, var

    def ereg(parts):
        F, err, w, _ = data(parts, "f")
        return HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=w)

    C = {"full": clf(S, "f"),
         "rpt_L": clf(S + [aug["smooth_L1"], aug["smooth_L2"]], "f"),
         }
    G = {"gauss_bias_aware": gauss(S)}
    R = {"err_reg": ereg(S)}
    Fq, eq_, wq, _ = data(S, "f")
    R["eq90"] = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **mk).fit(Fq, eq_, sample_weight=wq)
    del Fq, eq_, wq
    fitw = windows("fit", "clean")
    rec_fit = np.nan_to_num(fitw.x[:, :, :n_ch].reshape(-1, n_ch))
    del fitw
    pca_mu = rec_fit.mean(0)
    Zc = rec_fit - pca_mu
    U_, sv, Vt = np.linalg.svd(Zc[np.random.default_rng(seed).permutation(len(Zc))[:200000]], full_matrices=False)
    ev = sv ** 2 / max((sv ** 2).sum(), 1e-12)
    kp = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
    P_ = Vt[:kp].T
    lam = np.maximum(sv[:kp] ** 2 / max(len(U_) - 1, 1), 1e-12)
    del rec_fit, Zc, U_

    def pca_scores(x):
        z = np.nan_to_num(x[:, :, :n_ch]) - pca_mu
        t = z @ P_
        spe = ((z - t @ P_.T) ** 2).sum(-1).max(1)
        t2 = ((t ** 2) / lam).sum(-1).max(1)
        return spe, t2
    del aug, S

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    def scores(p):
        out = {"ensemble_std": p["ens"].std(1)}
        out.update({nm: m.predict_proba(p[key])[:, 1] for nm, (key, m) in C.items()})
        for nm, (bias, var) in G.items():
            s = np.sqrt(np.maximum(var.predict(p["f"]), 1e-12) + p["mce"] ** 2)
            out[nm] = norm.sf((delta - bias.predict(p["f"])) / s)
        out.update({nm: m.predict(p["f"]) for nm, m in R.items()})
        if "pca" in p:
            out["pca_spe"], out["pca_t2"] = p["pca"]
        if "choice2" in globals_box:
            out["rpt_auto"] = out[globals_box["choice2"]]
        return out

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    cw = windows("calibration", "clean")
    calp = [prep(cw.x, cw.meta, cw.y)] + [prep(w.x, w.meta, w.y) for w in (windows("calibration", sc) for sc in SEEN[1:])]
    del cw

    def cal_risk(name):
        out = []
        for p in calp:
            v = C[name][1].predict_proba(p[C[name][0]])[:, 1]
            acc = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], p["units"], ALPHA))
            out.append(rates(p["ens"].mean(1), p["y"], p["yc"], p["units"], acc)[0])
        return float(np.mean(out))

    cr = {k: cal_risk(k) for k in ("full", "rpt_L")}
    globals_box.update(choice2="rpt_L" if cr["rpt_L"] < cr["full"] else "full", choice3="full")
    del calp

    def wrate(w, acc, opt):
        a = (w * acc).sum()
        return (float((w * acc * opt).sum() / a) if a > 0 else np.nan), float(a / max(w.sum(), 1e-12))

    rows, clean = [], None
    for k, scn in enumerate(EVAL):
        wd = windows(eval_part, scn)
        p = prep(wd.x, wd.meta, wd.y, 20 + k)
        p["pca"] = pca_scores(wd.x)
        key = np.column_stack([wd.meta.unit.to_numpy(), wd.meta[[c for c in ("cycle", "position") if c in wd.meta][0]].to_numpy()])
        del wd
        sc = scores(p)
        w = engine_weights(p["units"]).astype(float)
        opt = (p["ens"].mean(1) - p["y"]) > delta
        acc = {nm: apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], p["units"], ALPHA)) for nm, v in sc.items()}
        if scn == "clean":
            clean = dict(x=p["x"], key=key, opt=opt, acc=acc)
            for nm in sc:
                rows.append(dict(scenario=scn, score=nm, risk_matched=wrate(w, acc[nm], opt)[0]))
            del p
            continue
        assert np.array_equal(key, clean["key"]), "window order differs from clean"
        xa, xc = p["x"], clean["x"]
        diff = ~((xa == xc) | (np.isnan(xa) & np.isnan(xc)))
        A = diff.reshape(len(xa), -1).any(1)
        wa = w * A
        po_f = wrate(wa, np.ones(len(A), bool), opt)[0]
        po_c = wrate(wa, np.ones(len(A), bool), clean["opt"])[0]
        for nm in sc:
            r_all = wrate(w, acc[nm], opt)[0]
            rA, cA = wrate(wa, acc[nm], opt)
            rAc, cAc = wrate(wa, clean["acc"][nm], clean["opt"])
            rows.append(dict(scenario=scn, score=nm, risk_matched=r_all, risk_A=rA, cov_A=cA, risk_A_clean=rAc,
                             cov_A_clean=cAc, excess_A=rA - rAc if np.isfinite(rA) and np.isfinite(rAc) else np.nan,
                             share_A=float(wa.sum() / w.sum()), pred_opt_A=po_f, pred_opt_A_clean=po_c))
        del p
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, choice2=globals_box["choice2"], pca_k=kp)
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
    ap.add_argument("--stress", action="store_true", help="also evaluate the EXT12 stress scenarios")
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
            process(run, a.data_dir, a.ext_data_dir, a.device, a.passes,
                    ()).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT25_DONE", flush=True)


if __name__ == "__main__":
    main()
