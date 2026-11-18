"""EXT15 (protocol: EXT15_PROTOCOL.md): novelty controls (outlier-exposure labels, perturbation detector, learned
baselines on RPT data), all-channel faults, generator-fault overlap (AUC) and the mu-vs-sigma premise of Prop. 1.
`full` and `rpt_L` reproduce ext12_main.py exactly (same features, generator seeds and HGB settings).
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext10_rpt import ALPHA, EVAL, HGB, SEEN, cmapss_windows, external_windows  # noqa: E402
from ext11_ablate import perturb  # noqa: E402
from uncertainty_v1 import mc_draws, unit_risk  # noqa: E402

from selective_rul.data import SENSORS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

FAULTS6 = ("stuck", "bias", "noise", "intermittent", "spike", "gain")
HELD = ("clip", "lag", "quant", "invert")
ALLCH = ("bias@1#99", "stuck#99", "gain@1.25#99", "noise@1#99")


def process(run, raw, ext_dir, device, passes):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import external_features, load_dataset

        n_ch = len(load_dataset(cfg["dataset"], ext_dir)[1])
        flag_off, cap = n_ch, 1.0

        def feats_of(w_x, meta, ens):
            return external_features(type("W", (), {"x": w_x, "meta": meta})(), ens, n_ch)
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)
        n_ch, flag_off, cap = len(SENSORS), None, float(cfg["rul_cap"])

        def feats_of(w_x, meta, ens):
            return selector_features(type("W", (), {"x": w_x, "meta": meta})(), ens, cfg["rul_cap"], None)
    if flag_off is None:
        from selective_rul.data import SETTINGS
        flag_off = n_ch + len(SETTINGS)
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())

    def prep(x, meta, y, k=None):
        ens, _ = predict_details(models, x, device)
        mce = None
        if k is not None:
            mce = mc_draws(models, x, device, passes, seed + 30000 + k)
            mce = mce.reshape(-1, mce.shape[-1]).std(0)
        return dict(ens=ens, f=feats_of(x, meta, ens), mce=mce, ties=tie_breakers(meta, seed + 20000),
                    y=meta.rul_raw.to_numpy(), yc=np.asarray(y), units=meta.unit.to_numpy())

    S = [prep(w.x, w.meta, w.y) for w in (windows("selector", sc) for sc in SEEN)]
    b = windows("selector", "clean")
    L = dict(amp=(1.0, 8.0), kfrac=0.5)
    A = [prep(perturb(b.x, n_ch, seed + 50000, **L), b.meta, b.y),
         prep(perturb(b.x, n_ch, seed + 51000, **L), b.meta, b.y)]

    # ---- overlap: generator (copy 1) vs truly changed windows of each fault, selector units, 2-fold-by-unit AUC
    aux = []
    fa = A[0]["f"]
    su = b.meta.unit.to_numpy()
    half = np.isin(su, np.random.default_rng(seed + 80000).permutation(np.unique(su))[: len(np.unique(su)) // 2])
    for scn in FAULTS6 + HELD + ALLCH:
        w = windows("selector", scn)
        ch = (np.abs(np.nan_to_num(w.x[:, :, :n_ch]) - np.nan_to_num(b.x[:, :, :n_ch])).max(1) > 1e-6).any(1)
        ch |= (w.x[:, :, flag_off:] != b.x[:, :, flag_off:]).any((1, 2))
        if ch.sum() < 20:
            aux.append(dict(kind="overlap", scenario=scn, auc=np.nan, n_changed=int(ch.sum())))
            continue
        ens, _ = predict_details(models, w.x[ch], device)
        ff = feats_of(w.x[ch], w.meta[ch].reset_index(drop=True), ens)
        X = np.vstack([fa, ff])
        yb = np.r_[np.zeros(len(fa)), np.ones(len(ff))]
        hu = np.r_[half, half[ch]]
        sc = np.empty(len(yb))
        for msk in (hu, ~hu):
            m_ = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(X[~msk], yb[~msk])
            sc[msk] = m_.predict_proba(X[msk])[:, 1]
        aux.append(dict(kind="overlap", scenario=scn, auc=float(roc_auc_score(yb, sc)), n_changed=int(ch.sum())))
    del b

    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)

    def data(parts):
        X = np.concatenate([p["f"] for p in parts])
        err = np.concatenate([p["ens"].mean(1) - p["y"] for p in parts])
        U = np.concatenate([p["units"] for p in parts])
        return X, err, engine_weights(U), U

    def clf_fit(X, yb, w):
        return HistGradientBoostingClassifier(**mk).fit(X, yb.astype(int), sample_weight=w)

    X, err, w, _ = data(S)
    XA, errA, wA, UA = data(S + A)
    nS = len(X)
    is_aug = np.r_[np.zeros(nS), np.ones(len(XA) - nS)]
    C = {"full": clf_fit(X, err > delta, w),
         "rpt_L": clf_fit(XA, errA > delta, wA),
         "oe_label": clf_fit(XA, np.where(is_aug == 1, 1, errA > delta), wA),
         "det_rpt": clf_fit(XA, is_aug, wA),
         "symmetric": clf_fit(X, np.abs(err) > delta, w),
         "symmetric_rpt": clf_fit(XA, np.abs(errA) > delta, wA)}

    def gauss(Xg, eg, wg, Ug):
        bias = HistGradientBoostingRegressor(**mk).fit(Xg, eg, sample_weight=wg)
        uu = np.unique(Ug)
        fold = np.isin(Ug, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])
        b_oof = np.empty_like(eg)
        for msk in (fold, ~fold):
            b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(Xg[~msk], eg[~msk],
                                                                 sample_weight=wg[~msk]).predict(Xg[msk])
        var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(Xg, (eg - b_oof) ** 2, sample_weight=wg)
        return bias, var

    _, _, _, U = data(S)
    G = {"gauss_bias_aware": gauss(X, err, w, U), "gba_rptL": gauss(XA, errA, wA, UA)}
    R = {"err_reg": HistGradientBoostingRegressor(**mk).fit(X, err, sample_weight=w),
         "err_reg_rptL": HistGradientBoostingRegressor(**mk).fit(XA, errA, sample_weight=wA)}
    del S, A

    def scores(p):
        out = {"ensemble_std": p["ens"].std(1)}
        out.update({nm: m.predict_proba(p["f"])[:, 1] for nm, m in C.items()})
        for nm, (bias, var) in G.items():
            s = np.sqrt(np.maximum(var.predict(p["f"]), 1e-12) + p["mce"] ** 2)
            out[nm] = norm.sf((delta - bias.predict(p["f"])) / s)
        out.update({nm: m.predict(p["f"]) for nm, m in R.items()})
        return out

    def rates(pred, y, yc, units, acc):
        ww = engine_weights(units) * acc
        tot = max(ww.sum(), 1e-12)
        e = pred - y
        return (float((ww * (e > delta)).sum() / tot), float((ww * (e < -delta)).sum() / tot),
                float(np.sqrt((ww * (pred - yc) ** 2).sum() / tot)))

    cw = windows("calibration", "clean")
    Cp = prep(cw.x, cw.meta, cw.y, 10)
    frozen = {k: fit_threshold(v, Cp["ties"], Cp["units"], ALPHA) for k, v in scores(Cp).items()}
    rows = []
    clean = None
    for k, scn in enumerate(list(EVAL) + list(HELD) + list(ALLCH)):
        wv = windows(eval_part, scn)
        p = prep(wv.x, wv.meta, wv.y, 20 + k)
        y, yc, units, pred = p["y"], p["yc"], p["units"], p["ens"].mean(1)
        if scn == "clean":
            clean = dict(x=wv.x, ens=p["ens"], y=y)
        else:  # mu-vs-sigma premise on truly changed windows
            ch = (np.abs(np.nan_to_num(wv.x[:, :, :n_ch]) - np.nan_to_num(clean["x"][:, :, :n_ch])).max(1) > 1e-6).any(1)
            ch |= (wv.x[:, :, flag_off:] != clean["x"][:, :, flag_off:]).any((1, 2))
            if ch.sum() >= 20:
                dmu = (p["ens"].mean(1) - clean["ens"].mean(1))[ch] / cap
                dsd = (p["ens"].std(1) - clean["ens"].std(1))[ch] / cap
                new_opt = ((pred - y) > delta) & ((clean["ens"].mean(1) - clean["y"]) <= delta) & ch
                med_sd = np.median(clean["ens"].std(1))
                aux.append(dict(kind="musigma", scenario=scn, n_changed=int(ch.sum()),
                                med_abs_dmu=float(np.median(np.abs(dmu))), med_abs_dsd=float(np.median(np.abs(dsd))),
                                n_new_opt=int(new_opt.sum()),
                                frac_new_opt_low_sd=float((p["ens"].std(1)[new_opt] < med_sd).mean())
                                if new_opt.any() else np.nan))
        for name, v in scores(p).items():
            acc_m = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], units, ALPHA))
            acc_f = apply_threshold(v, p["ties"], frozen[name])
            r_m, pe_m, rmse_m = rates(pred, y, yc, units, acc_m)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            rows.append(dict(scenario=scn, score=name, risk_matched=r_m, pess_matched=pe_m, rmse_matched=rmse_m,
                             risk_frozen=r_f, coverage_frozen=c_f))
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta)
    out, ax = pd.DataFrame(rows), pd.DataFrame(aux)
    for df in (out, ax):
        for k, v in reversed(list(meta.items())):
            df.insert(0, k, v)
    return out, ax


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
    (a.out / "aux").mkdir(exist_ok=True)
    print(f"{len(runs)} runs in shard {a.shard}", flush=True)
    for k, run in enumerate(runs, 1):
        name = f"{run.parent.name}__{run.name}.csv".replace(":", "_")
        if (a.out / name).exists():
            continue
        t = time.time()
        try:
            main_rows, aux = process(run, a.data_dir, a.ext_data_dir, a.device, a.passes)
            aux.to_csv(a.out / "aux" / name, index=False)
            main_rows.to_csv(a.out / name, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT15_DONE", flush=True)


if __name__ == "__main__":
    main()
