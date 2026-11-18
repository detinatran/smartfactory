"""EXT12 (protocol: EXT12_PROTOCOL.md): candidates for the final method, directional baselines, and optional stress
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
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())

    fitw = windows("fit", "clean")
    rec = Reconstructor(n_ch, flag_off, n_set, seed).fit(fitw.x)
    del fitw

    def prep(x, meta, y, k=None):
        ens, _ = predict_details(models, x, device)
        xr, susp, ratio = rec.repair(x)
        ens_r, _ = predict_details(models, xr, device)
        extra = np.column_stack([ens.mean(1) - ens_r.mean(1), susp.sum(1), ratio.max(1)])
        f = feats_of(x, meta, ens)
        mce = None
        if k is not None:  # MC draws only where a Gaussian score is evaluated (same seeds as ext10_rpt.py)
            mce = mc_draws(models, x, device, passes, seed + 30000 + k)
            mce = mce.reshape(-1, mce.shape[-1]).std(0)
        return dict(ens=ens, f=f, fx=np.column_stack([f, extra]), mce=mce, ties=tie_breakers(meta, seed + 20000),
                    y=meta.rul_raw.to_numpy(), yc=np.asarray(y), units=meta.unit.to_numpy())

    S = [prep(w.x, w.meta, w.y) for w in (windows("selector", sc) for sc in SEEN)]
    b = windows("selector", "clean")
    P = lambda x: prep(x, b.meta, b.y)  # noqa: E731
    base = seed + 50000
    L = dict(amp=(1.0, 8.0), kfrac=0.5)
    aug = {
        "rand": P(perturb(b.x, n_ch, base)),
        "white": P(perturb(b.x, n_ch, base, kind="white")),
        "white_L1": P(perturb(b.x, n_ch, base, kind="white", **L)),
        "white_L2": P(perturb(b.x, n_ch, seed + 51000, kind="white", **L)),
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
         "full_rpt": clf(S + [aug["rand"]], "f"),
         "full_recon_rpt": clf(S + [aug["rand"]], "fx"),
         "rptw": clf(S + [aug["white"]], "f"),
         "rptw_L": clf(S + [aug["white_L1"], aug["white_L2"]], "f"),
         "rpt_L": clf(S + [aug["smooth_L1"], aug["smooth_L2"]], "f"),
         "recon_rpt_L": clf(S + [aug["smooth_L1"], aug["smooth_L2"]], "fx")}
    G = {"gauss_bias_aware": gauss(S), "gba_rpt": gauss(S + [aug["rand"]])}
    R = {"err_reg": ereg(S), "err_reg_rpt": ereg(S + [aug["rand"]])}
    del aug, S

    def scores(p):
        out = {"ensemble_std": p["ens"].std(1)}
        out.update({nm: m.predict_proba(p[key])[:, 1] for nm, (key, m) in C.items()})
        for nm, (bias, var) in G.items():
            s = np.sqrt(np.maximum(var.predict(p["f"]), 1e-12) + p["mce"] ** 2)
            out[nm] = norm.sf((delta - bias.predict(p["f"])) / s)
        out.update({nm: m.predict(p["f"]) for nm, m in R.items()})
        return out

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    cw = windows("calibration", "clean")
    Cp = prep(cw.x, cw.meta, cw.y, 10)
    frozen = {k: fit_threshold(v, Cp["ties"], Cp["units"], ALPHA) for k, v in scores(Cp).items()}
    rows = []
    for k, scn in enumerate(list(EVAL) + list(extra_scenarios)):
        w = windows(eval_part, scn)
        p = prep(w.x, w.meta, w.y, 20 + k)
        y, yc, units, pred = p["y"], p["yc"], p["units"], p["ens"].mean(1)
        o, pe, r = rates(pred, y, yc, units, np.ones(len(y), bool))
        rows.append(dict(scenario=scn, score="predictor", opt_full=o, pess_full=pe, rmse_full=r))
        for name, v in scores(p).items():
            acc_m = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], units, ALPHA))
            acc_f = apply_threshold(v, p["ties"], frozen[name])
            r_m, pe_m, rmse_m = rates(pred, y, yc, units, acc_m)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            curve = {f"risk_c{int(round(100 * c))}": rates(pred, y, yc, units, apply_threshold(
                v, p["ties"], fit_threshold(v, p["ties"], units, c)))[0] for c in COVERAGES}
            rows.append(dict(scenario=scn, score=name, risk_matched=r_m, pess_matched=pe_m, rmse_matched=rmse_m,
                             risk_frozen=r_f, coverage_frozen=c_f, **curve))
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, n_channels=n_ch,
                robust=bool(cfg.get("robust_predictor", False)))
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
    print("EXT12_DONE", flush=True)


if __name__ == "__main__":
    main()
