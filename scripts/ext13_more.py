"""EXT13 (protocol: EXT13_PROTOCOL.md): six more baseline scores on the same runs and shared ensemble predictions:
laplace, mahalanobis, knn_feat, iforest, ar_residual, tta_std.
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext10_rpt import ALPHA, EVAL, Reconstructor, cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import unit_risk  # noqa: E402

from selective_rul.data import SENSORS, SETTINGS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

GRID = 10.0 ** np.arange(-4, 5)


@torch.no_grad()
def penult(model, x, device, batch=2048):
    model.eval()
    return np.concatenate([model.features(torch.from_numpy(x[i:i + batch]).to(device)).cpu().double().numpy()
                           for i in range(0, len(x), batch)])


@torch.no_grad()
def last_layer(model, x, device):
    """Penultimate features (with bias column) and d yhat / d logit for the sigmoid point readout (as EXT2/EXT3)."""
    phi = penult(model, x, device)
    z = phi @ model.head.weight.detach().cpu().double().numpy().ravel() + float(model.head.bias.detach().cpu())
    s = 1.0 / (1.0 + np.exp(-z))
    return np.column_stack([phi, np.ones(len(phi))]), model.cap * s * (1 - s)


def fit_laplace_ev(models, x, y, device):
    post = []
    for model in models:
        phi, g = last_layer(model, x, device)
        pred = predict_details([model], x, device)[0][:, 0]
        noise = max(float(np.mean((pred - y) ** 2)), 1e-12)
        jac = phi * g[:, None]
        ggn = jac.T @ jac / noise
        theta = np.concatenate([model.head.weight.detach().cpu().double().numpy().ravel(),
                                model.head.bias.detach().cpu().double().numpy().ravel()])
        p = len(theta)
        ev = [0.5 * p * np.log(lam) - 0.5 * lam * theta @ theta - 0.5 * np.linalg.slogdet(ggn + lam * np.eye(p))[1]
              for lam in GRID]
        post.append(np.linalg.inv(ggn + GRID[int(np.argmax(ev))] * np.eye(p)))
    return post


def process(run, raw, ext_dir, device):
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
    rng = np.random.default_rng(seed + 70000)

    fit = windows("fit", "clean")
    rec = Reconstructor(n_ch, flag_off, n_set, seed).fit(fit.x)
    post = fit_laplace_ev(models, fit.x, np.asarray(fit.y, float), device)
    maha, knn = [], []
    sub = rng.choice(len(fit.x), min(len(fit.x), 20000), replace=False)
    for model in models:
        h = penult(model, fit.x, device)
        mu = h.mean(0)
        prec = np.linalg.pinv(np.cov(h - mu, rowvar=False) + 1e-6 * np.eye(h.shape[1]))
        maha.append((mu, prec))
        hn = h[sub] / np.maximum(np.linalg.norm(h[sub], axis=1, keepdims=True), 1e-12)
        knn.append(NearestNeighbors(n_neighbors=50).fit(hn))
    del fit
    sel = windows("selector", "clean")
    ens_s, _ = predict_details(models, sel.x, device)
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(feats_of(sel.x, sel.meta, ens_s))
    del sel

    def scores(w):
        x = w.x
        ens, _ = predict_details(models, x, device)
        out = {}
        var = []
        for model, cov in zip(models, post):
            phi, g = last_layer(model, x, device)
            jac = phi * g[:, None]
            var.append(np.einsum("ij,jk,ik->i", jac, cov, jac))
        out["laplace"] = np.sqrt(np.mean(var, 0))
        dm, dk = [], []
        for model, (mu, prec), nn_ in zip(models, maha, knn):
            h = penult(model, x, device)
            c = h - mu
            dm.append(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", c, prec, c), 0)))
            hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
            dk.append(nn_.kneighbors(hn)[0][:, -1])
        out["mahalanobis"], out["knn_feat"] = np.mean(dm, 0), np.mean(dk, 0)
        out["iforest"] = -iso.score_samples(feats_of(x, w.meta, ens))
        _, _, ratio = rec.repair(x)
        out["ar_residual"] = ratio.max(1)
        g = np.random.default_rng(seed + 71000)
        draws = []
        for _ in range(10):
            xa = x.copy()
            xa[:, :, :n_ch] += g.normal(0.0, 0.1, xa[:, :, :n_ch].shape).astype(x.dtype)
            draws.append(predict_details(models, xa, device)[0].mean(1))
        out["tta_std"] = np.std(draws, axis=0)
        return ens.mean(1), out

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    cal = windows("calibration", "clean")
    _, csc = scores(cal)
    cties, cunits = tie_breakers(cal.meta, seed + 20000), cal.meta.unit.to_numpy()
    frozen = {k: fit_threshold(v, cties, cunits, ALPHA) for k, v in csc.items()}
    rows = []
    for scn in EVAL:
        w = windows(eval_part, scn)
        pred, sc = scores(w)
        y, yc, units = w.meta.rul_raw.to_numpy(), np.asarray(w.y), w.meta.unit.to_numpy()
        ties = tie_breakers(w.meta, seed + 20000)
        for name, v in sc.items():
            acc_m = apply_threshold(v, ties, fit_threshold(v, ties, units, ALPHA))
            acc_f = apply_threshold(v, ties, frozen[name])
            r_m, pe_m, rmse_m = rates(pred, y, yc, units, acc_m)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            rows.append(dict(scenario=scn, score=name, risk_matched=r_m, pess_matched=pe_m, rmse_matched=rmse_m,
                             risk_frozen=r_f, coverage_frozen=c_f))
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta)
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
            process(run, a.data_dir, a.ext_data_dir, a.device).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT13_DONE", flush=True)


if __name__ == "__main__":
    main()
