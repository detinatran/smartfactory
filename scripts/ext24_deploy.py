"""EXT24 (protocol: EXT24_PROTOCOL.md): deployment with thresholds fixed on calibration units (C0 clean, CS seen shifts,
CP clean + perturbed copies at assumed prevalence 0.3) applied to test mixtures with fault prevalence pi.
Predictor, selectors and the rpt_auto choice are identical to ext22.py (consistency rows reproduce EXT22).\nOriginal EXT12 doc: EXT12 (protocol: EXT12_PROTOCOL.md): candidates for the final method, directional baselines, and optional stress
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
    R = {"err_reg": ereg(S)}
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
        return out

    def rates(pred, y, yc, units, acc):
        w = engine_weights(units) * acc
        tot = max(w.sum(), 1e-12)
        e = pred - y
        return (float((w * (e > delta)).sum() / tot), float((w * (e < -delta)).sum() / tot),
                float(np.sqrt((w * (pred - yc) ** 2).sum() / tot)))

    cw = windows("calibration", "clean")
    Cp = prep(cw.x, cw.meta, cw.y, 10)
    calp = [Cp] + [prep(w.x, w.meta, w.y, 40 + j) for j, w in enumerate(windows("calibration", sc) for sc in SEEN[1:])]

    def cal_risk(name):
        out = []
        for p in calp:
            v = C[name][1].predict_proba(p[C[name][0]])[:, 1] if name in C else R[name].predict(p["f"])
            acc = apply_threshold(v, p["ties"], fit_threshold(v, p["ties"], p["units"], ALPHA))
            out.append(rates(p["ens"].mean(1), p["y"], p["yc"], p["units"], acc)[0])
        return float(np.mean(out))

    cr = {k: cal_risk(k) for k in ("full", "rpt_L")}
    globals_box.update(choice2="rpt_L" if cr["rpt_L"] < cr["full"] else "full", choice3="full")
    Pp = prep(perturb(cw.x, n_ch, seed + 52000, **L), cw.meta, cw.y, 43)
    del cw

    def part(p, wt):
        """one mixture component: scores, ties, normalised engine weights times component weight, errors"""
        w = engine_weights(p["units"]).astype(float)
        e = p["ens"].mean(1) - p["y"]
        return dict(s=scores(p), t=p["ties"], w=wt * w / w.sum(), opt=e > delta, pes=e < -delta)

    PIA = 0.3
    pools = {"C0": [part(Cp, 1.0)],
             "CS": [part(p, 1.0 / len(calp)) for p in calp],
             "CP": [part(Cp, 1 - PIA), part(Pp, PIA)]}
    del calp, Pp
    r_star = 0.5 * float((pools["C0"][0]["w"] * pools["C0"][0]["opt"]).sum())

    def thr_at(parts, name, cov):
        if cov >= 1:
            return {"score": None, "tie": None}
        s = np.concatenate([q["s"][name] for q in parts])
        t = np.concatenate([q["t"] for q in parts])
        w = np.concatenate([q["w"] for q in parts])
        o = np.lexsort((t, s))
        i = min(np.searchsorted(np.cumsum(w[o]) / w.sum(), cov), len(o) - 1)
        return {"score": float(s[o[i]]), "tie": float(t[o[i]])}

    def evaluate(parts, name, th):
        a = o = pe = 0.0
        for q in parts:
            acc = apply_threshold(q["s"][name], q["t"], th)
            a += (q["w"] * acc).sum()
            o += (q["w"] * acc * q["opt"]).sum()
            pe += (q["w"] * acc * q["pes"]).sum()
        tot = sum(q["w"].sum() for q in parts)
        return a / tot, (o / a if a > 0 else np.nan), (pe / a if a > 0 else np.nan)

    LEVELS = np.round(np.arange(0.30, 1.0001, 0.05), 2)
    names = list(pools["C0"][0]["s"].keys())
    rule = {}
    for pk, parts in pools.items():
        for name in names:
            rule[(pk, name, "cov90")] = (0.9, thr_at(parts, name, 0.9))
            ths = {lv: thr_at(parts, name, lv) for lv in LEVELS}
            ok = [lv for lv in LEVELS if evaluate(parts, name, ths[lv])[1] <= r_star]
            lv = max(ok) if ok else 0.0
            rule[(pk, name, "risk")] = (lv, ths[lv] if ok else None)
    rows, tparts = [], {}
    for k, scn in enumerate(EVAL):
        w = windows(eval_part, scn)
        p = prep(w.x, w.meta, w.y, 20 + k)
        del w
        if scn != "clean":  # consistency rows: matched 90% risk per fault, as in ext22.py
            sc = scores(p)
            for name in ("full", "rpt_L", "rpt_auto"):
                acc = apply_threshold(sc[name], p["ties"], fit_threshold(sc[name], p["ties"], p["units"], ALPHA))
                rows.append(dict(kind="consistency", scenario=scn, score=name,
                                 risk_matched=rates(p["ens"].mean(1), p["y"], p["yc"], p["units"], acc)[0]))
        tparts[scn] = part(p, 1.0)
        del p
    faults = EVAL[1:]
    for pi in (0.0, 0.1, 0.3, 0.5):
        mix = [dict(tparts["clean"], w=tparts["clean"]["w"] * (1 - pi))] + \
              [dict(tparts[f], w=tparts[f]["w"] * pi / len(faults)) for f in faults if pi > 0]
        for (pk, name, target), (lv, th) in rule.items():
            if th is None:
                cov, risk, pes = 0.0, np.nan, np.nan
            else:
                cov, risk, pes = evaluate(mix, name, th)
            viol = bool(target == "risk" and th is not None and risk > r_star)
            rows.append(dict(kind="deploy", pi=pi, pool=pk, score=name, target=target, level=lv, coverage=cov,
                             risk=risk, pess=pes, r_star=r_star, violation=viol,
                             excess=max(0.0, risk - r_star) if th is not None else 0.0))
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed,
                fold=cfg.get("fold", cfg.get("subset")), delta=delta, choice2=globals_box["choice2"])
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
    print("EXT24_DONE", flush=True)


if __name__ == "__main__":
    main()
