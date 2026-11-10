"""EXT2 pilot (protocol: EXT2_PILOT_PROTOCOL.md): backbones x uncertainty scores under unseen sensor faults.

For every existing GRU run (same data, splits, normaliser, fault seeds) three backbones are compared:
  gru        the run's own 3-member GRU ensemble (checkpoints reloaded)
  film_gru   FiLM-Ensemble GRU, M = 8 implicit members, trained here on the run's fit/validation units
  lipschitz  3-member Lipschitz RNN ensemble, trained here (member seeds = run seed + 100 m)
For each backbone the scores below are fitted on selector units x {clean, dropout, drift} (calibration units for the
conformal quantile) and evaluated on the evaluation units, clean and under six unseen faults:
  ensemble_std, mc_dropout, mc_ensemble   variance-type epistemic estimates (MC dropout: T passes)
  laplace          last-layer Laplace approximation (GGN, prior precision 1, noise variance = fit MSE), per-member
                   predictive std, averaged over members
  gauss_unbiased   1 - Phi(delta / s),          s^2 = v(x) + mc_ensemble^2
  gauss_bias_aware 1 - Phi((delta - b(x)) / s)
  conformal        b(x) + q s(x), q = unit-weighted 90% quantile of (e - b) / s on clean calibration windows
                   (split-conformal one-sided bound e <= b + q s; its empirical coverage is reported)
  full             one-sided HGB classifier of 1[yhat - y > delta]
"""

import argparse
import copy
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_baselines import cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import mc_draws, unit_risk  # noqa: E402

from selective_rul.data import engine_weights  # noqa: E402
from selective_rul.ext_backbones import FiLMEnsembleGRU  # noqa: E402
from selective_rul.model import build_model, predict_details, seed_everything, train_member  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, selector_features, tie_breakers  # noqa: E402

SEEN = ("clean", "dropout", "drift")
EVAL = ("clean", "stuck", "bias", "noise", "intermittent", "spike", "gain")
HGB = dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)
FILM = dict(members=8, init_std=0.5)
LIP = dict(lip_beta=0.75, lip_gamma=0.001, lip_eps=0.1)


# ------------------------------------------------------------------ training of the new backbones
def train_film(train, validation, cfg, device, seed):
    seed_everything(seed)
    model = FiLMEnsembleGRU(train.x.shape[-1], cfg["hidden"], cfg["dropout"], cfg["rul_cap"], **FILM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    w = engine_weights(train.meta.unit.to_numpy()).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y), torch.from_numpy(w)),
                        batch_size=cfg["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    vw = engine_weights(validation.meta.unit.to_numpy())
    best, stale, best_state, hist = np.inf, 0, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y, wt in loader:
            x, y, wt = x.to(device), y.to(device), wt.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model.forward_all(x)  # [B, M]; every member gets its own loss on the same batch
            loss = (wt[:, None] * ((pred - y[:, None]) / cfg["rul_cap"]).square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model.forward_all(torch.from_numpy(validation.x[i:i + 2048]).to(device)).cpu().numpy()
                                 for i in range(0, len(validation.x), 2048)]).mean(1)
        rmse = float(np.sqrt(np.average((pv - validation.y) ** 2, weights=vw)))
        hist.append(rmse)
        if rmse < best:
            best, stale, best_state = rmse, 0, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval(), hist


def get_backbones(run, cfg, model_cfg, windows, device, ckpt_dir, which=("gru", "film_gru", "lipschitz")):
    out = {}
    gru = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        gru.append(net.to(device).eval())
    out["gru"] = gru
    train, validation = windows("fit", "clean"), windows("validation", "clean")
    seed = cfg["seed"]
    if "film_gru" not in which and "lipschitz" not in which:
        return out, train
    f = ckpt_dir / "film_gru.pt"
    film = FiLMEnsembleGRU(train.x.shape[-1], model_cfg["hidden"], model_cfg["dropout"], model_cfg["rul_cap"], **FILM)
    if f.exists():
        film.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        film = film.to(device).eval()
    else:
        film, _ = train_film(train, validation, model_cfg, device, seed + 900)
        torch.save({k: v.cpu() for k, v in film.state_dict().items()}, f)
    out["film_gru"] = film.member_views()
    if "lipschitz" not in which:
        return out, train
    lcfg = dict(model_cfg, model="lipschitz_rnn", **LIP)
    lip = []
    for m in range(cfg["ensemble_size"]):
        f = ckpt_dir / f"lipschitz_{m}.pt"
        if f.exists():
            net = build_model(train.x.shape[-1], lcfg)
            net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
            net = net.to(device).eval()
        else:
            net, _ = train_member(train, validation, lcfg, device, seed + 100 * m)
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, f)
        lip.append(net.eval())
    out["lipschitz"] = lip
    return out, train


# ------------------------------------------------------------------ last-layer Laplace
@torch.no_grad()
def last_layer(model, x, device, batch=2048):
    """Penultimate features phi (with a bias column) and d yhat / d logit for a sigmoid point readout."""
    model.eval()
    phis, grads = [], []
    for i in range(0, len(x), batch):
        phi = model.features(torch.from_numpy(x[i:i + batch]).to(device))
        z = model.head(phi).squeeze(-1)
        s = torch.sigmoid(z)
        phis.append(torch.cat([phi, torch.ones_like(phi[:, :1])], 1).cpu().double().numpy())
        grads.append((model.cap * s * (1 - s)).cpu().double().numpy())
    return np.concatenate(phis), np.concatenate(grads)


def fit_laplace(models, train, device, prior=1.0):
    post = []
    for model in models:
        phi, g = last_layer(model, train.x, device)
        pred = predict_details([model], train.x, device)[0][:, 0]
        noise = float(np.mean((pred - train.y) ** 2))
        jac = phi * g[:, None]
        h = jac.T @ jac / max(noise, 1e-12) + prior * np.eye(jac.shape[1])
        post.append(np.linalg.inv(h))
    return post


def laplace_std(models, post, x, device):
    var = []
    for model, cov in zip(models, post):
        phi, g = last_layer(model, x, device)
        jac = phi * g[:, None]
        var.append(np.einsum("ij,jk,ik->i", jac, cov, jac))
    return np.sqrt(np.mean(var, 0))


def wquantile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


# ------------------------------------------------------------------ evaluation of one backbone
def evaluate_backbone(models, train, windows, eval_part, feats_of, cfg, scale, device, passes, tag):
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    post = fit_laplace(models, train, device)

    def prep(w, k):
        ens, _ = predict_details(models, w.x, device)
        dr = mc_draws(models, w.x, device, passes, seed + 30000 + 100 * tag + k)
        unc = dict(ensemble_std=ens.std(1), mc_dropout=dr.std(1).mean(0),
                   mc_ensemble=dr.reshape(-1, dr.shape[-1]).std(0), laplace=laplace_std(models, post, w.x, device))
        return ens, feats_of(w, ens), unc, tie_breakers(w.meta, seed + 20000)

    F, P, Y, U = [], [], [], []
    for k, sc in enumerate(SEEN):
        w = windows("selector", sc)
        ens, f, _, _ = prep(w, k)
        F.append(f), P.append(ens.mean(1)), Y.append(w.meta.rul_raw.to_numpy()), U.append(w.meta.unit.to_numpy())
    F, P, Y, U = map(np.concatenate, (F, P, Y, U))
    err, wts = P - Y, engine_weights(U)
    mk = dict(max_iter=cfg["selector_iterations"], random_state=seed, **HGB)
    clf = HistGradientBoostingClassifier(**mk).fit(F, (err > delta).astype(int), sample_weight=wts)
    bias = HistGradientBoostingRegressor(**mk).fit(F, err, sample_weight=wts)
    uu = np.unique(U)
    fold = np.isin(U, np.random.default_rng(seed).permutation(uu)[: len(uu) // 2])
    b_oof = np.empty_like(err)
    for msk in (fold, ~fold):
        b_oof[msk] = HistGradientBoostingRegressor(**mk).fit(F[~msk], err[~msk], sample_weight=wts[~msk]).predict(F[msk])
    var = HistGradientBoostingRegressor(loss="poisson", **mk).fit(F, (err - b_oof) ** 2, sample_weight=wts)

    def base_scores(f, unc):
        b = bias.predict(f)
        s = np.sqrt(np.maximum(var.predict(f), 1e-12) + unc["mc_ensemble"] ** 2)
        return b, s, {**unc, "gauss_unbiased": norm.sf(delta / s), "gauss_bias_aware": norm.sf((delta - b) / s),
                      "full": clf.predict_proba(f)[:, 1]}

    cal = windows("calibration", "clean")
    ens_c, f_c, unc_c, t_c = prep(cal, 10)
    b_c, s_c, sc_c = base_scores(f_c, unc_c)
    e_c = ens_c.mean(1) - cal.meta.rul_raw.to_numpy()
    q = wquantile((e_c - b_c) / s_c, engine_weights(cal.meta.unit.to_numpy()), 0.9)
    sc_c["conformal"] = b_c + q * s_c
    frozen = {k: fit_threshold(v, t_c, cal.meta.unit.to_numpy(), 0.9) for k, v in sc_c.items()}

    rows, clean = [], None
    for k, sc in enumerate(EVAL):
        w = windows(eval_part, sc)
        ens, f, unc, ties = prep(w, 20 + k)
        pred, y, units = ens.mean(1), w.meta.rul_raw.to_numpy(), w.meta.unit.to_numpy()
        b, s, scores = base_scores(f, unc)
        scores["conformal"] = b + q * s
        e = pred - y
        uw = engine_weights(units)
        common = dict(rmse=float(np.sqrt(np.average((pred - w.y) ** 2, weights=uw))) / scale,
                      conf_coverage=float(np.average(e <= b + q * s, weights=uw)), conf_q=q)
        if sc == "clean":
            clean = dict(pred=pred, spread=ens.std(1), scores=scores)
        for name, sv in scores.items():
            acc_m = apply_threshold(sv, ties, fit_threshold(sv, ties, units, 0.9))
            acc_f = apply_threshold(sv, ties, frozen[name])
            r_m, _ = unit_risk(pred, y, units, acc_m, delta)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            row = dict(scenario=sc, score=name, risk_matched=r_m, risk_frozen=r_f, coverage_frozen=c_f, **common)
            if sc != "clean":
                shift = pred - clean["pred"]
                aff = np.abs(shift) > 1e-6 * max(np.abs(y).max(), 1e-9)
                new = (e > delta) & ~((clean["pred"] - y) > delta)
                ds = sv - clean["scores"][name]
                row.update(frac_affected=float(aff.mean()),
                           reject_new=float((~acc_m[new]).mean()) if new.sum() >= 5 else np.nan,
                           rho_shift=float(spearmanr(shift[aff], ds[aff])[0]) if aff.sum() >= 10 else np.nan,
                           abs_shift=float(np.abs(shift[aff]).mean()) / scale if aff.any() else np.nan,
                           shift_over_spread=float(np.abs(shift[aff]).mean() / ens.std(1)[aff].mean())
                           if aff.any() else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def process(run, raw, ext_dir, device, passes, ckpt_root, which):
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
    ckpt = ckpt_root / f"{run.parent.name}__{run.name}".replace(":", "_")
    ckpt.mkdir(parents=True, exist_ok=True)
    t = time.time()
    backbones, train = get_backbones(run, cfg, model_cfg, windows, device, ckpt, which)
    t_train = time.time() - t
    outs = []
    for tag, (name, models) in enumerate(backbones.items()):
        d = evaluate_backbone(models, train, windows, eval_part, feats_of, cfg, scale, device, passes, tag)
        d.insert(0, "backbone", name)
        outs.append(d)
    out = pd.concat(outs, ignore_index=True)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=cfg["seed"],
                fold=cfg.get("fold", cfg.get("subset")), scale=scale, delta=float(cfg["risk_delta"]), mc_passes=passes,
                train_seconds=round(t_train, 1))
    for k, v in reversed(list(meta.items())):
        out.insert(0, k, v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/CMAPSSData"))
    ap.add_argument("--ext-data-dir", type=Path, default=Path("data/external/processed"))
    ap.add_argument("--datasets", nargs="*", default=["cmapss", "ncmapss"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--base-model", default="gru")
    ap.add_argument("--backbones", nargs="+", default=["gru", "film_gru", "lipschitz"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("model") == a.base_model and cfg.get("point_target", "capped") != "raw" and cfg["seed"] in a.seeds \
                    and cfg.get("dataset", "cmapss") in a.datasets and cfg.get("head", "point") == "point":
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
            process(run, a.data_dir, a.ext_data_dir, a.device, a.passes, a.out / "checkpoints", a.backbones).to_csv(dest, index=False)
            print(f"[{k}/{len(runs)}] {run.name} {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{len(runs)}] FAILED {run}: {e}\n{traceback.format_exc()}", flush=True)
    print("EXT2_DONE", flush=True)


if __name__ == "__main__":
    main()
