"""EXT11 (protocol: EXT11_PROTOCOL.md) parts A (HGB-based baselines), B (RPT ablations) and C (classifier swap).

Reuses ext10_rpt.py (reconstruction, features, weights, matched/frozen evaluation). `full` and `full_recon_rpt` are
recomputed and must reproduce EXT10 on the same runs.
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext10_rpt as E  # noqa: E402
from ext10_rpt import ALPHA, EVAL, HGB, SEEN, Reconstructor, cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import unit_risk  # noqa: E402

from selective_rul.data import SENSORS, SETTINGS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

FAULTS = ("stuck", "bias", "noise", "intermittent", "spike", "gain")


def perturb(x, n_ch, seed, kind="smooth", const=True, onset_p=0.5, amp=(0.5, 4.0), kfrac=0.2, k1=False):
    """Generalised EXT10 generator; with defaults it reproduces ext10_rpt.randomize draw for draw."""
    rng = np.random.default_rng(seed)
    n, T, _ = x.shape
    out = x.copy()
    kmax = max(1, int(round(kfrac * n_ch)))
    t = np.arange(T) / max(T - 1, 1)
    m = np.arange(3)
    for i in range(n):
        k = int(rng.integers(1, kmax + 1))
        if k1:
            k = 1
        for c in rng.choice(n_ch, k, replace=False):
            a, b = rng.normal(size=3), rng.uniform(0, 2 * np.pi, 3)
            if kind == "smooth":
                if not const:
                    a[0] = 0.0
                phi = (a[:, None] * np.cos(np.pi * m[:, None] * t[None, :] + b[:, None])).sum(0)
                phi /= max(np.abs(phi).max(), 1e-6)
            else:  # white
                phi = rng.normal(size=T)
            A = float(np.exp(rng.uniform(np.log(amp[0]), np.log(amp[1]))))
            if rng.random() < onset_p:
                phi[: int(rng.integers(0, T))] = 0.0
            out[i, :, c] += A * phi
    return out.astype(x.dtype)


def gauss_all(x, n_ch, seed):
    rng = np.random.default_rng(seed)
    out = x.copy()
    sig = np.exp(rng.uniform(np.log(0.1), np.log(1.0), size=len(x)))
    out[:, :, :n_ch] += sig[:, None, None] * rng.normal(size=out[:, :, :n_ch].shape)
    return out.astype(x.dtype)


def process(run, raw, ext_dir, device, threads):
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

    def prep(x, meta, y, keep_x=False):
        ens, _ = predict_details(models, x, device)
        xr, susp, ratio = rec.repair(x)
        ens_r, _ = predict_details(models, xr, device)
        extra = np.column_stack([ens.mean(1) - ens_r.mean(1), susp.sum(1), ratio.max(1)])
        f = feats_of(x, meta, ens)
        return dict(x=x if keep_x else None, ens=ens, f=f, fx=np.column_stack([f, extra]), ties=tie_breakers(meta, seed + 20000),
                    y=meta.rul_raw.to_numpy(), yc=np.asarray(y), units=meta.unit.to_numpy())

    def prep_w(w, keep_x=False):
        return prep(w.x, w.meta, w.y, keep_x)

    S = [prep_w(windows("selector", sc), True) for sc in SEEN]
    b = windows("selector", "clean")
    P = lambda x: prep(x, b.meta, b.y)  # noqa: E731
    base = seed + 50000
    xr = perturb(b.x, n_ch, base)
    assert np.array_equal(xr, E.randomize(b.x, n_ch, base)), "generator does not reproduce EXT10"
    aug = {
        "rand": P(xr),
        "copy": S[0],
        "nobias": P(perturb(b.x, n_ch, base, const=False)),
        "noonset": P(perturb(b.x, n_ch, base, onset_p=0.0)),
        "amp_small": P(perturb(b.x, n_ch, base, amp=(0.25, 1.0))),
        "amp_large": P(perturb(b.x, n_ch, base, amp=(2.0, 8.0))),
        "k1": P(perturb(b.x, n_ch, base, k1=True)),
        "khalf": P(perturb(b.x, n_ch, base, kfrac=0.5)),
        "rand2": P(perturb(b.x, n_ch, seed + 51000)),
        "white": P(perturb(b.x, n_ch, base, kind="white")),
        "gauss_all": P(gauss_all(b.x, n_ch, base)),
    }
    oracle = [prep_w(windows("selector", sc)) for sc in FAULTS]
    del b, xr

    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)

    def data(parts, key):
        X = np.concatenate([p[key] for p in parts])
        err = np.concatenate([p["ens"].mean(1) - p["y"] for p in parts])
        return X, err, engine_weights(np.concatenate([p["units"] for p in parts]))

    def hgb(parts, key, **kw):
        X, err, w = data(parts, key)
        return HistGradientBoostingClassifier(**dict(mk, **kw)).fit(X, (err > delta).astype(int), sample_weight=w)

    fitted = {}  # name -> (feature key, model with predict_proba)
    fitted["full"] = ("f", hgb(S, "f"))
    fitted["full_recon"] = ("fx", hgb(S, "fx"))
    fitted["full_rpt"] = ("f", hgb(S + [aug["rand"]], "f"))
    fitted["full_recon_rpt"] = ("fx", hgb(S + [aug["rand"]], "fx"))
    # B: ablations on OURS
    for nm in ("copy", "nobias", "noonset", "amp_small", "amp_large", "k1", "khalf", "white", "gauss_all"):
        fitted[f"ours_{nm}"] = ("fx", hgb(S + [aug[nm]], "fx"))
    fitted["ours_x2"] = ("fx", hgb(S + [aug["rand"], aug["rand2"]], "fx"))
    fitted["ours_noseen"] = ("fx", hgb([S[0], aug["rand"]], "fx"))
    fitted["ours_oracle"] = ("fx", hgb(S + oracle, "fx"))
    # A: HGB-based baselines (OSC features and data)
    X, err, w = data(S, "f")
    fitted["symmetric"] = ("f", HistGradientBoostingClassifier(**mk).fit(X, (np.abs(err) > delta).astype(int),
                                                                         sample_weight=w))
    xs = [p["x"] for p in S]
    changed = [np.zeros(len(xs[0]), bool)] + [
        (np.abs(x[:, :, :n_ch] - xs[0][:, :, :n_ch]).max(1) > 1e-6).any(1) | (x[:, :, flag_off:] != xs[0][:, :, flag_off:]).any((1, 2))
        for x in xs[1:]]
    fitted["fault_det"] = ("f", HistGradientBoostingClassifier(**mk).fit(X, np.concatenate(changed).astype(int),
                                                                         sample_weight=w))
    for p in S:
        p["x"] = None
    err_reg = HistGradientBoostingRegressor(**mk).fit(X, err, sample_weight=w)
    # C: classifier swap
    clfs = {
        "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "rf": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=25, max_features="sqrt",
                                             n_jobs=threads, random_state=seed),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3,
                                                                     early_stopping=True, max_iter=200,
                                                                     random_state=seed)),
        "hgbdef": lambda: HistGradientBoostingClassifier(max_iter=cfg["selector_iterations"], random_state=seed,
                                                         early_stopping=False),
    }
    for tag, parts, key in (("osc", S, "f"), ("ours", S + [aug["rand"]], "fx")):
        X, err, w = data(parts, key)
        yb = (err > delta).astype(int)
        for c, make in clfs.items():
            mdl = make()
            if isinstance(mdl, HistGradientBoostingClassifier | RandomForestClassifier):
                mdl.fit(X, yb, sample_weight=w)
            else:
                mdl.fit(X, yb, **{f"{mdl.steps[-1][0]}__sample_weight": w})
            fitted[f"{tag}_{c}"] = (key, mdl)

    def scores(p):
        out = {nm: mdl.predict_proba(p[key])[:, 1] for nm, (key, mdl) in fitted.items()}
        out["err_reg"] = err_reg.predict(p["f"])
        return out

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    C = prep_w(windows("calibration", "clean"))
    frozen = {k: fit_threshold(v, C["ties"], C["units"], ALPHA) for k, v in scores(C).items()}
    rows = []
    for scn in EVAL:
        p = prep_w(windows(eval_part, scn))
        y, yc, units, pred = p["y"], p["yc"], p["units"], p["ens"].mean(1)
        for name, v in scores(p).items():
            acc_m = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], units, ALPHA))
            acc_f = apply_threshold(v, p["ties"], frozen[name])
            r_m, pe_m, rmse_m = rates(pred, y, yc, units, acc_m)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            rows.append(dict(scenario=scn, score=name, risk_matched=r_m, pess_matched=pe_m, rmse_matched=rmse_m,
                             risk_frozen=r_f, coverage_frozen=c_f))
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, n_channels=n_ch)
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
            process(run, a.data_dir, a.ext_data_dir, a.device, a.threads).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT11A_DONE", flush=True)


if __name__ == "__main__":
    main()
