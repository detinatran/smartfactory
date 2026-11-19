"""EXT50 (EXT50_PROTOCOL.md, revised before running): JEPA-style predictor (horizon-aligned masked JEPA + gated symbolic branch). Stage I: an online encoder sees
corrupted windows (RPT perturbations or contiguous channel masks) and predicts, through a small predictor, the latent of the
clean window produced by an EMA target encoder (cosine energy). Stage II: the encoder and a RUL head are fine-tuned on clean
windows (unit-weighted MSE, early stopping). (Original jd feature dropped in the design revision.)
Everything else as EXT38. Derived from EXT38 (protocol: EXT38_PROTOCOL.md): adds rpt_auto_d (use rpt_L iff the calibration diagnostic D > 0). Derived from
EXT35 (protocol: EXT35_PROTOCOL.md): adds rpt_auto_m (switch to rpt_L only if its calibration risk is lower by > 0.01).
Derived from EXT22 (protocol: EXT22_PROTOCOL.md): rpt_auto = per-run choice between full and rpt_L by calibration risk under
the seen shifts (clean, dropout, drift); auto3 adds err_reg_rptL. Lean ext12_main v2 derivative.\nOriginal EXT12 doc: EXT12 (protocol: EXT12_PROTOCOL.md): candidates for the final method, directional baselines, and optional stress
scenarios. Shared by stages D (seeds 42-61), C and S (72-81), R (robust-predictor mirror runs) and B (transformer).

`full`, `gauss_bias_aware` and `full_recon_rpt` follow ext10_rpt.py exactly (same features, seeds, MC draws), so they
reproduce EXT10 on the same runs.
"""

import argparse
import copy
import os
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



def _corrupt(x, rng, n_ch, flag_off, L):
    """JEPA-style structured corruption: half the windows get the RPT perturbation, half a contiguous channel-subset
    mask (values 0, missing flag 1)"""
    n, T, _ = x.shape
    out = x.copy()
    pert = rng.random(n) < (0.0 if os.environ.get("ELPISJ_CORRUPT", "mix") == "mask" else 0.5)
    if pert.any():
        out[pert] = perturb(x[pert], n_ch, int(rng.integers(1 << 30)), **L)
    for i in np.flatnonzero(~pert):
        k = int(rng.integers(1, max(1, n_ch // 2) + 1))
        ch = rng.choice(n_ch, k, replace=False)
        ln = int(rng.integers(max(1, T // 3), T + 1))
        st = int(rng.integers(0, T - ln + 1))
        out[i, st:st + ln, ch] = 0.0
        if flag_off + n_ch <= x.shape[2]:
            out[i, st:st + ln, flag_off + ch] = 1.0
    return out.astype(np.float32)


class JepaRUL(torch.nn.Module):
    """JEPA-style RUL model: causal encoder z; symbolic trend features s (per-channel slope, last-first, std) and
    observation quality q (per-channel missing share); future-latent predictor r = P([z, Es(s), Eq(q)]);
    v = F(z) + sigmoid(gamma) r; output = cap * sigmoid(H(v) + alpha * C(s)), alpha = sigmoid(G([z, Es(s), Eq(q)]))."""

    def __init__(self, base, n_ch, flag_off, d, cap, dropout=0.1):
        super().__init__()
        self.base, self.n_ch, self.flag_off, self.cap = base, n_ch, flag_off, cap
        ns = 3 * n_ch
        self.Es, self.Eq = torch.nn.Linear(ns, d), torch.nn.Linear(n_ch, d)
        self.P = torch.nn.Sequential(torch.nn.Linear(3 * d, d), torch.nn.GELU(), torch.nn.Linear(d, d))
        self.F, self.H = torch.nn.Linear(d, d), torch.nn.Linear(d, 1)
        self.G, self.C = torch.nn.Linear(3 * d, 1), torch.nn.Linear(ns, 1)
        self.gamma = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("offset", torch.zeros(()))
        self.drop = torch.nn.Dropout(dropout)

    def sym(self, x):
        if os.environ.get("ELPISJ_NOSYM", "0") == "1":
            return torch.zeros(x.shape[0], 3 * self.n_ch, device=x.device)
        v = x[:, :, : self.n_ch]
        T = v.shape[1]
        t = torch.linspace(-1, 1, T, device=x.device)[None, :, None]
        slope = (v * t).mean(1) / (t ** 2).mean()
        return torch.cat([slope, v[:, -1] - v[:, 0], v.std(1)], -1)

    def qual(self, x):
        if self.flag_off + self.n_ch <= x.shape[2]:
            return x[:, :, self.flag_off: self.flag_off + self.n_ch].mean(1)
        return torch.zeros(x.shape[0], self.n_ch, device=x.device)

    def parts(self, x):
        z = self.base.features(x)
        s = self.sym(x)
        cond = torch.cat([z, self.Es(s), self.Eq(self.qual(x))], -1)
        return z, s, cond, self.P(cond)

    def forward(self, x):
        z, s, cond, r = self.parts(x)
        v = self.F(z) + torch.sigmoid(self.gamma) * r
        alpha = torch.sigmoid(self.G(cond)).squeeze(-1)
        logit = self.H(self.drop(v)).squeeze(-1) + alpha * self.C(s).squeeze(-1)
        return torch.sigmoid(logit) * self.cap - self.offset

    def forward_outputs(self, x):
        return {"point": self(x), "quantiles": None}


def _future_index(meta, k=3):
    """index of the window k positions later on the same unit (last window of the unit if fewer remain)"""
    order = np.lexsort((meta.cycle.to_numpy(), meta.unit.to_numpy()))
    fut = np.empty(len(meta), int)
    u = meta.unit.to_numpy()[order]
    for st in np.flatnonzero(np.r_[True, u[1:] != u[:-1]]):
        en = st
        while en + 1 < len(u) and u[en + 1] == u[st]:
            en += 1
        for p in range(st, en + 1):
            fut[order[p]] = order[min(p + k, en)]
    return fut


def sigreg(z, gen, n_slices=256, t_max=3.0, n_points=17):
    """SIGReg (LeJEPA): sliced Epps-Pulley statistic of embeddings z [N, d] against N(0, 1), mean over random unit directions."""
    A = torch.randn(z.shape[1], n_slices, device=z.device, generator=gen)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(0, t_max, n_points, device=z.device)
    phi = torch.exp(-t ** 2 / 2)
    dt = t_max / (n_points - 1)
    w = torch.full((n_points,), 2 * dt, device=z.device)
    w[[0, -1]] = dt
    xt = (z @ A).unsqueeze(-1) * t
    err = (xt.cos().mean(0) - phi) ** 2 + xt.sin().mean(0) ** 2
    return ((err @ (w * phi)) * z.shape[0]).mean()


def _emb_diag(net, x, device):
    """collapse diagnostics of the encoder on clean validation windows: effective rank, mean per-dim std, SIGReg"""
    with torch.no_grad():
        z = torch.cat([net.base.features(torch.from_numpy(x[i:i + 1024]).to(device)) for i in range(0, len(x), 1024)])
        zc = z - z.mean(0)
        s = torch.linalg.svdvals(zc)
        p = s / s.sum().clamp_min(1e-12)
        er = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
        g = torch.Generator(device=z.device).manual_seed(0)
        return dict(emb_effrank=er, emb_std=float(z.std(0).mean()), emb_sigreg=float(sigreg(z, g)))


def jepa_train(windows, model_cfg, device, seed, n_ch, flag_off, cfg, e1=20, e2=40, patience=8, ema=0.99):
    tr, va = windows("fit", "clean"), windows("validation", "clean")
    cap = float(model_cfg["rul_cap"])
    L = dict(amp=(1.0, 8.0), kfrac=0.5)
    pre_parts = os.environ.get("ELPISJ_PRE_PARTS", "fit").split(",")
    if pre_parts == ["fit"]:
        pre_x, pre_meta = tr.x, tr.meta
    else:  # unlabeled windows of several non-test parts for the self-supervised stage (no labels used)
        ws = [windows(pp, "clean") for pp in pre_parts]
        pre_x = np.concatenate([w_.x for w_ in ws])
        pre_meta = pd.concat([w_.meta for w_ in ws], ignore_index=True)
    fut = torch.from_numpy(_future_index(pre_meta, k=int(os.environ.get("ELPISJ_K", "3")))).to(device)
    wtr = torch.from_numpy(engine_weights(tr.meta.unit.to_numpy()).astype(np.float32)).to(device)
    ytr = torch.from_numpy(np.asarray(tr.y, np.float32)).to(device)
    wva = engine_weights(va.meta.unit.to_numpy())
    models = []
    for m in range(cfg["ensemble_size"]):
        ms = seed * 100 + 7000 + m
        torch.manual_seed(ms)
        rng = np.random.default_rng(ms)
        base = build_model(tr.x.shape[-1], model_cfg).to(device)
        with torch.no_grad():
            d = base.features(torch.from_numpy(tr.x[:2]).to(device)).shape[-1]
        net = JepaRUL(base, n_ch, flag_off, d, cap).to(device)
        tgt = copy.deepcopy(base)
        for p_ in tgt.parameters():
            p_.requires_grad_(False)
        Xc = torch.from_numpy(tr.x).to(device)
        Xp = torch.from_numpy(pre_x).to(device)
        # Stage I: horizon-aligned masked JEPA (context: corrupted window; target: EMA latent of a later clean window)
        opt = torch.optim.AdamW([p_ for n_, p_ in net.named_parameters() if not n_.startswith(("F.", "H.", "G.", "C.", "gamma"))],
                                lr=1e-3, weight_decay=1e-4)
        ssl, lam = os.environ.get("ELPISJ_SSL", "ema"), float(os.environ.get("ELPISJ_SIGREG_LAMBDA", "0.05"))
        gen, step = torch.Generator(device=device), 0
        for ep in range(e1):
            net.train()
            xcor = torch.from_numpy(_corrupt(pre_x, rng, n_ch, flag_off, L)).to(device)
            for idx in torch.from_numpy(rng.permutation(len(pre_x))).split(128):
                idx = idx.to(device)
                if ssl == "sigreg":  # EXT66 (LeWM-style): end-to-end, no EMA/stop-grad; MSE prediction + lambda * SIGReg
                    z_, _, _, r = net.parts(xcor[idx])
                    t = net.base.features(Xp[fut[idx]])
                    step += 1
                    loss = (1 - lam) * ((r - t) ** 2).mean() + lam * sigreg(torch.cat([z_, t]), gen.manual_seed(ms * 100000 + step))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    continue
                r = net.parts(xcor[idx])[3]
                with torch.no_grad():
                    t = tgt.features(Xp[fut[idx]])
                loss = (2 - 2 * (torch.nn.functional.normalize(r, dim=-1) * torch.nn.functional.normalize(t, dim=-1)).sum(-1)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for pt, po in zip(tgt.parameters(), net.base.parameters()):
                        pt.mul_(ema).add_(po.detach(), alpha=1 - ema)
        # Stage II: transfer all modules, supervised RUL on clean windows (target encoder discarded)
        p2 = [p_ for n_, p_ in net.named_parameters() if not (os.environ.get("ELPISJ_FREEZE", "0") == "1" and n_.startswith("base."))]
        opt2 = torch.optim.AdamW(p2, lr=float(os.environ.get("ELPISJ_LR2", "1e-3")), weight_decay=1e-4)
        best, stale, best_state = float("inf"), 0, None
        for ep in range(e2):
            net.train()
            Xs = torch.from_numpy(_corrupt(tr.x, rng, n_ch, flag_off, L)).to(device) if os.environ.get("ELPISJ_CORR2", "0") == "1" else None
            for idx in torch.from_numpy(rng.permutation(len(tr.x))).split(128):
                idx = idx.to(device)
                loss = (wtr[idx] * ((net(Xc[idx]) - ytr[idx]) / cap) ** 2).mean()
                if Xs is not None:
                    loss = 0.5 * loss + 0.5 * (wtr[idx] * ((net(Xs[idx]) - ytr[idx]) / cap) ** 2).mean()
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                opt2.step()
            net.eval()
            with torch.no_grad():
                pv = np.concatenate([net(torch.from_numpy(va.x[i:i + 1024]).to(device)).cpu().numpy() for i in range(0, len(va.x), 1024)])
            rm = float(np.sqrt((wva * (pv - np.asarray(va.y)) ** 2).sum() / wva.sum()))
            if rm < best - 1e-6:
                best, stale, best_state = rm, 0, copy.deepcopy(net.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        bc = os.environ.get("ELPISJ_BIASCAL", "0")
        if bc in ("1", "late"):  # post-hoc bias correction on validation units (no test data)
            with torch.no_grad():
                pv = np.concatenate([net(torch.from_numpy(va.x[i:i + 1024]).to(device)).cpu().numpy() for i in range(0, len(va.x), 1024)])
            yv = np.asarray(va.y)
            m = np.ones(len(yv), bool) if bc == "1" else (yv <= 0.3 * cap)  # "late": end-of-life windows only
            net.offset.fill_(float((wva[m] * (pv[m] - yv[m])).sum() / max(wva[m].sum(), 1e-12)))
        net.diag = _emb_diag(net, va.x, device)
        models.append(net)
    return models


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
    models = jepa_train(windows, model_cfg, device, seed, n_ch, flag_off, cfg, e1=int(os.environ.get("EXT50_E1", "20")))
    # EXT61 (exploratory): validation-only alignment with / pooling of the run's default ensemble
    pmatch = os.environ.get("ELPISJ_PMATCH", "0") == "1"
    hybrid = os.environ.get("ELPISJ_HYBRID", "0") == "1"
    if pmatch or hybrid:
        dmods = []
        for m in range(cfg["ensemble_size"]):
            ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
            dn = build_model(ck["input_dim"], model_cfg)
            dn.load_state_dict(ck["state_dict"])
            dmods.append(dn.to(device).eval())
        if pmatch:  # shift ELPIS-J so its validation optimistic share equals the default ensemble's (no test data)
            va = windows("validation", "clean")
            yv, wv = va.meta.rul_raw.to_numpy(), engine_weights(va.meta.unit.to_numpy())
            pdv = predict_details(dmods, va.x, device)[0].mean(1)
            prv = predict_details(models, va.x, device)[0].mean(1)
            share = lambda p: float((wv * ((p - yv) > delta)).sum() / wv.sum())  # noqa: E731
            target, span = share(pdv), float(np.abs(prv - yv).max()) + 1.0
            lo, hi = -span, span  # share(prv - s) is non-increasing in s
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if share(prv - mid) > target else (lo, mid)
            for net in models:
                net.offset.add_(hi)
            print(f"pmatch target={target:.4f} shift={hi:+.4f} got={share(prv - hi):.4f}", flush=True)
        if hybrid:  # pooled ensemble: default members + ELPIS-J members
            models = dmods + models

    pass

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
                robust=bool(cfg.get("robust_predictor", False)), choice2=globals_box.get("choice2"),
                choice3=globals_box.get("choice3"), choice_m=globals_box.get("choice_m"), choice_d=globals_box.get("choice_d"), D=globals_box.get("D"),
                ssl=os.environ.get("ELPISJ_SSL", "ema"), **{k: float(np.mean([m_.diag[k] for m_ in models if hasattr(m_, "diag")]))
                                                           for k in ("emb_effrank", "emb_std", "emb_sigreg")}, **{f"calrisk_{k}": v for k, v in cr.items()})
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
    print("EXT50_DONE", flush=True)


if __name__ == "__main__":
    main()
