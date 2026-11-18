"""EXT41 (protocol: EXT41_PROTOCOL.md): per-window scores on the CLEAN evaluation windows (natural-glitch analysis).
Derived from EXT38 (protocol: EXT38_PROTOCOL.md): adds rpt_auto_d (use rpt_L iff the calibration diagnostic D > 0). Derived from
EXT35 (protocol: EXT35_PROTOCOL.md): adds rpt_auto_m (switch to rpt_L only if its calibration risk is lower by > 0.01).
Derived from EXT22 (protocol: EXT22_PROTOCOL.md): rpt_auto = per-run choice between full and rpt_L by calibration risk under
the seen shifts (clean, dropout, drift); auto3 adds err_reg_rptL. Lean ext12_main v2 derivative.\nOriginal EXT12 doc: EXT12 (protocol: EXT12_PROTOCOL.md): candidates for the final method, directional baselines, and optional stress
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

KEEP = ("ensemble_std", "gauss_bias_aware", "err_reg", "err_reg_rptL", "full", "rpt_L", "rpt_auto_d")
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
        return dict(ens=ens, f=f, mce=mce, ties=tie_breakers(meta, seed + 20000),
                    fh=np.column_stack([f, hist_feats(x, meta, n_ch, ref_x)]),
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
    R = {"err_reg": ereg(S), "err_reg_rptL": ereg(S + [aug["smooth_L1"], aug["smooth_L2"]])}
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
        if "choice2" in globals_box:
            out["rpt_auto"] = out[globals_box["choice2"]]
            out["auto3"] = out[globals_box["choice3"]]
            out["rpt_auto_m"] = out[globals_box["choice_m"]]
            out["rpt_auto_d"] = out[globals_box["choice_d"]]
        return out

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    cw = windows("calibration", "clean")
    Cp = prep(cw.x, cw.meta, cw.y, 10)
    calp = [Cp] + [prep(w.x, w.meta, w.y) for w in (windows("calibration", sc) for sc in SEEN[1:])]

    def cal_risk(name):
        out = []
        for p in calp:
            v = C[name][1].predict_proba(p[C[name][0]])[:, 1] if name in C else R[name].predict(p["f"])
            acc = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], p["units"], ALPHA))
            out.append(rates(p["ens"].mean(1), p["y"], p["yc"], p["units"], acc)[0])
        return float(np.mean(out))

    cr = {k: cal_risk(k) for k in ("full", "rpt_L", "err_reg_rptL")}
    choice2 = "rpt_L" if cr["rpt_L"] < cr["full"] else "full"
    choice3 = min(cr, key=lambda k: (cr[k], k != "full"))
    choice_m = "rpt_L" if cr["rpt_L"] < cr["full"] - 0.01 else "full"
    wc_ = engine_weights(Cp["units"]).astype(float)
    opt_c = float((wc_ * ((Cp["ens"].mean(1) - Cp["y"]) > delta)).sum() / wc_.sum())
    opt_p = []
    for sd in (seed + 53000, seed + 54000):
        pp = prep(perturb(cw.x, n_ch, sd, **L), cw.meta, cw.y)
        opt_p.append(float((wc_ * ((pp["ens"].mean(1) - pp["y"]) > delta)).sum() / wc_.sum()))
        del pp
    D = float(np.mean(opt_p) - opt_c)
    choice_d = "rpt_L" if D > 0 else "full"
    globals_box["D"] = D
    globals_box.update(choice2=choice2, choice3=choice3, choice_m=choice_m, choice_d=choice_d)
    frozen = {k: fit_threshold(v, Cp["ties"], Cp["units"], ALPHA) for k, v in scores(Cp).items()}
    w = windows(eval_part, "clean")
    p = prep(w.x, w.meta, w.y, 20)
    pred = p["ens"].mean(1)
    out = pd.DataFrame({"unit": w.meta.unit.to_numpy(), "cycle": w.meta.cycle.to_numpy(), "y": p["y"], "pred": pred,
                        "uw": engine_weights(p["units"])})
    for name, v in scores(p).items():
        if name in KEEP:
            out[f"s_{name}"] = v
            out[f"a_{name}"] = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], p["units"], ALPHA)).astype(np.int8)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, n_channels=n_ch,
                robust=bool(cfg.get("robust_predictor", False)), choice2=globals_box.get("choice2"),
                choice3=globals_box.get("choice3"), choice_m=globals_box.get("choice_m"), choice_d=globals_box.get("choice_d"), D=globals_box.get("D"), **{f"calrisk_{k}": v for k, v in cr.items()})
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
                    STRESS if a.stress else ()).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT41_DONE", flush=True)


if __name__ == "__main__":
    main()
