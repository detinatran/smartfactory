"""EXT5 (protocol: EXT5_PROTOCOL.md): leave-one-channel-out (LOCO) sensitivity as selector features.

Same checkpoints, splits, windows and faults as the paper (all GRU/TCN runs, CPU replay). For every window and sensor
channel c, d_c = yhat - yhat_{-c}, where yhat_{-c} is the ensemble-mean prediction with channel c masked exactly as missing
data is encoded at input (value 0, missingness flag 1). Features: loco_up = max_c d_c, loco_down = min_c d_c,
loco_abs = max_c |d_c|, loco_std = std_c d_c.

Scores (fitted on selector units x {clean, dropout, drift}):
  ensemble_std, full (the paper's classifier), full_loco (same classifier + LOCO features), loco_up (raw),
  gauss_bias_aware (as EXT3), gba_loco (bias/variance regressors on features + LOCO).
Per score and scenario: risk at matched 90% coverage, frozen-threshold risk/coverage, rejection of fault-induced
optimistic errors at frozen thresholds; per faulty scenario the LOCO localisation rate on affected windows.
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
from extra_baselines import cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import mc_draws, unit_risk  # noqa: E402

from selective_rul.data import SENSORS, SETTINGS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

SEEN = ("clean", "dropout", "drift")
EVAL = ("clean", "stuck", "bias", "noise", "intermittent", "spike", "gain")
HGB = dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)
ALPHA = 0.9


def loco(models, x, device, n_ch, flag_off):
    """d[:, c] = yhat - yhat with channel c masked (value 0, missing flag 1)."""
    base = predict_details(models, x, device)[0].mean(1)
    d = np.empty((len(x), n_ch))
    for c in range(n_ch):
        xm = x.copy()
        xm[:, :, c] = 0.0
        xm[:, :, flag_off + c] = 1.0
        d[:, c] = base - predict_details(models, xm, device)[0].mean(1)
    return d


def loco_feats(d):
    return np.column_stack([d.max(1), d.min(1), np.abs(d).max(1), d.std(1)])


def process(run, raw, ext_dir, device, passes):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import external_features, load_dataset

        n_ch = len(load_dataset(cfg["dataset"], ext_dir)[1])
        flag_off = n_ch

        def feats_of(w, ens):
            return external_features(w, ens, n_ch)
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)
        n_ch, flag_off = len(SENSORS), len(SENSORS) + len(SETTINGS)

        def feats_of(w, ens):
            return selector_features(w, ens, cfg["rul_cap"], None)
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())

    def prep(w, k):
        assert w.x.shape[2] == flag_off + n_ch, (w.x.shape, flag_off, n_ch)
        ens, _ = predict_details(models, w.x, device)
        dr = mc_draws(models, w.x, device, passes, seed + 30000 + k)
        d = loco(models, w.x, device, n_ch, flag_off)
        f = feats_of(w, ens)
        return dict(x=w.x, ens=ens, d=d, f=f, fl=np.column_stack([f, loco_feats(d)]),
                    mce=dr.reshape(-1, dr.shape[-1]).std(0), ties=tie_breakers(w.meta, seed + 20000),
                    y=w.meta.rul_raw.to_numpy(), units=w.meta.unit.to_numpy())

    S = [prep(windows("selector", sc), k) for k, sc in enumerate(SEEN)]
    err = np.concatenate([s["ens"].mean(1) - s["y"] for s in S])
    U = np.concatenate([s["units"] for s in S])
    wts = engine_weights(U)
    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)
    uu = np.unique(U)
    fold = np.isin(U, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])

    def fit_set(key):
        F = np.concatenate([s[key] for s in S])
        clf = HistGradientBoostingClassifier(**mk).fit(F, (err > delta).astype(int), sample_weight=wts)
        bias = HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=wts)
        b_oof = np.empty_like(err)
        for msk in (fold, ~fold):
            b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(F[~msk], err[~msk],
                                                                 sample_weight=wts[~msk]).predict(F[msk])
        var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(F, (err - b_oof) ** 2, sample_weight=wts)
        return clf, bias, var

    base_fit, loco_fit = fit_set("f"), fit_set("fl")

    def scores(p):
        out = {"ensemble_std": p["ens"].std(1), "loco_up": p["d"].max(1)}
        for (clf, bias, var), key, cname, gname in ((base_fit, "f", "full", "gauss_bias_aware"),
                                                     (loco_fit, "fl", "full_loco", "gba_loco")):
            b = bias.predict(p[key])
            s = np.sqrt(np.maximum(var.predict(p[key]), 1e-12) + p["mce"] ** 2)
            out[cname] = clf.predict_proba(p[key])[:, 1]
            out[gname] = norm.sf((delta - b) / s)
        return out

    C = prep(windows("calibration", "clean"), 10)
    frozen = {k: fit_threshold(v, C["ties"], C["units"], ALPHA) for k, v in scores(C).items()}

    rows, clean = [], None
    for k, scn in enumerate(EVAL):
        p = prep(windows(eval_part, scn), 20 + k)
        sc = scores(p)
        pred, y, units = p["ens"].mean(1), p["y"], p["units"]
        uw = engine_weights(units)
        bad = pred - y > delta
        if scn == "clean":
            clean = dict(pred=pred, bad=bad, x=p["x"])
            aff = np.zeros(len(y), bool)
            new_bad = np.zeros(len(y), bool)
            loc_rate, n_hit = np.nan, np.nan
        else:
            aff = np.abs(pred - clean["pred"]) > 1e-6 * max(np.abs(y).max(), 1e-9)
            new_bad = bad & ~clean["bad"]
            changed = np.abs(p["x"][:, :, :n_ch] - clean["x"][:, :, :n_ch]).max(1) > 1e-6  # [n, n_ch]
            changed |= (p["x"][:, :, flag_off:] != clean["x"][:, :, flag_off:]).any(1)
            top = np.abs(p["d"]).argmax(1)
            m = aff & changed.any(1)
            loc_rate = float(np.average(changed[np.arange(len(top)), top][m], weights=uw[m])) if m.any() else np.nan
            n_hit = float(changed[m].sum(1).mean()) if m.any() else np.nan
        for name, v in sc.items():
            acc_m = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], units, ALPHA))
            acc_f = apply_threshold(v, p["ties"], frozen[name])
            r_m, _ = unit_risk(pred, y, units, acc_m, delta)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            rej = float(np.average(~acc_f[new_bad], weights=uw[new_bad])) if new_bad.any() else np.nan
            rows.append(dict(scenario=scn, score=name, risk_matched=r_m, risk_frozen=r_f, coverage_frozen=c_f,
                             reject_new_frozen=rej, frac_affected=float(aff.mean()), frac_new_bad=float(new_bad.mean()),
                             loco_localisation=loc_rate, channels_hit=n_hit, n_channels=n_ch))
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, mc_passes=passes)
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
    print("EXT5_DONE", flush=True)


if __name__ == "__main__":
    main()
