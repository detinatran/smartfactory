"""REVIEW_V5 (exploratory): ablation_v4 plus a fault detector on the SAME features as the one-sided classifier.

fault_detector_full: HGB, same settings/data/weights as fault_detector, but trained on all selector features
(prediction, spread, elapsed time, window statistics) to predict "window corrupted" (clean/dropout/drift).

Original v4 docstring follows. Fault-detector baseline, extra ablations and case groups (REVIEW_V4_PROTOCOL.md, exploratory).

Derived from ablation_severity.py (REVIEW_V3).

Per existing run (C-MAPSS ext_v1 capped gru/tcn incl. s2 seeds; external_v1; battery_v1):
 * Feature ablation of the one-sided classifier (same data, labels, HGB settings and seed as v1;
   trained on clean/dropout/drift selector windows): full | no_spread | no_sensor_stats |
   sensor_stats_only | pred_cycle_only.
 * Evaluation on clean, the six default unseen faults and a severity sweep:
   bias@1, bias@6, noise@1, noise@4, gain@1.25, gain@2, stuck#1, stuck#6.
 * Per scenario and score (ensemble spread + every ablation variant): risk at matched 90% coverage,
   frozen 90% threshold (clean calibration) -> coverage and risk, and on windows whose prediction the
   fault changes: decision flip rate (frozen threshold) and share of fault-induced optimistic errors
   rejected (matched 90%).
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_baselines import cmapss_windows, external_windows  # noqa: E402

from selective_rul.data import engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import (  # noqa: E402
    apply_threshold,
    fit_threshold,
    selector_features,
    tie_breakers,
)

SEEN = ("clean", "dropout", "drift")
EVAL = ("clean", "stuck", "bias", "noise", "intermittent", "spike", "gain")
HGB = dict(max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0, early_stopping=False)


def subsets(n_cols):
    allc = np.arange(n_cols)
    return {"full": allc, "pred_only": np.array([0]), "pred_cycle_only": np.array([0, 2])}


def unit_risk(pred, y, units, acc, delta):
    w = engine_weights(units) * acc
    return float((w * ((pred - y) > delta)).sum() / max(w.sum(), 1e-12)), float(w.sum() / engine_weights(units).sum())


def process(run, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    external = "dataset" in cfg
    if external:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import external_features, load_dataset

        n_feat = len(load_dataset(cfg["dataset"], ext_dir)[1])

        def feats_of(w, ens):
            return external_features(w, ens, n_feat)
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)

        def feats_of(w, ens):
            return selector_features(w, ens, cfg["rul_cap"], None)
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    models = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        models.append(net.to(device).eval())

    def prep(w):
        ens, _ = predict_details(models, w.x, device)
        return ens, feats_of(w, ens), tie_breakers(w.meta, seed + 20000)

    F, P, Y, U, C = [], [], [], [], []
    x_clean = None
    for sc in SEEN:
        w = windows("selector", sc)
        if sc == "clean":
            x_clean = w.x
        C.append(np.any(np.abs(w.x - x_clean) > 1e-7, axis=(1, 2)).astype(int))
        ens, feats, _ = prep(w)
        F.append(feats)
        P.append(ens.mean(1))
        Y.append(w.meta.rul_raw.to_numpy())
        U.append(w.meta.unit.to_numpy())
    F, P, Y, U, C = map(np.concatenate, (F, P, Y, U, C))
    bad = (P - Y > delta).astype(int)
    wts = engine_weights(U)
    detector = HistGradientBoostingClassifier(max_iter=cfg["selector_iterations"], random_state=seed, **HGB).fit(
        F[:, 3:], C, sample_weight=wts)
    det_tr = detector.predict_proba(F[:, 3:])[:, 1]
    detector_full = HistGradientBoostingClassifier(max_iter=cfg["selector_iterations"], random_state=seed, **HGB).fit(
        F, C, sample_weight=wts)
    combo = HistGradientBoostingClassifier(max_iter=cfg["selector_iterations"], random_state=seed, **HGB).fit(
        np.column_stack([F, det_tr]), bad, sample_weight=wts)
    clfs = {name: HistGradientBoostingClassifier(max_iter=cfg["selector_iterations"], random_state=seed, **HGB).fit(F[:, cols], bad, sample_weight=engine_weights(U))
            for name, cols in subsets(F.shape[1]).items()}

    def scores(ens, feats):
        out = {"ensemble_std": ens.std(1)}
        for name, cols in subsets(feats.shape[1]).items():
            out[name] = clfs[name].predict_proba(feats[:, cols])[:, 1]
        det = detector.predict_proba(feats[:, 3:])[:, 1]
        out["fault_detector"] = det
        out["fault_detector_full"] = detector_full.predict_proba(feats)[:, 1]
        out["one_sided_plus_detector"] = combo.predict_proba(np.column_stack([feats, det]))[:, 1]
        return out

    cal = windows("calibration", "clean")
    ens_c, f_c, t_c = prep(cal)
    s_c = scores(ens_c, f_c)
    frozen = {k: fit_threshold(v, t_c, cal.meta.unit.to_numpy(), 0.9) for k, v in s_c.items()}
    v1 = joblib.load(run / "selectors.joblib")
    check = float(np.max(np.abs(v1.scores(f_c, ens_c, t_c)["asymmetric_risk"] - s_c["full"])))

    rows, clean = [], None
    for sc in EVAL:
        w = windows(eval_part, sc)
        ens, feats, ties = prep(w)
        pred, y, units = ens.mean(1), w.meta.rul_raw.to_numpy(), w.meta.unit.to_numpy()
        sc_scores = scores(ens, feats)
        if sc == "clean":
            clean = dict(pred=pred, scores=sc_scores, ties=ties)
        std_now = ens.std(1)
        err = pred - y
        for name, s in sc_scores.items():
            acc_m = apply_threshold(s, ties, fit_threshold(s, ties, units, 0.9))
            acc_f = apply_threshold(s, ties, frozen[name])
            r_m, _ = unit_risk(pred, y, units, acc_m, delta)
            r_f, c_f = unit_risk(pred, y, units, acc_f, delta)
            row = dict(scenario=sc, score=name, risk_matched=r_m, risk_frozen=r_f, coverage_frozen=c_f)
            if sc == "clean":
                g2 = err > delta
                row["G2_reject_clean_optimistic"] = float((~acc_m[g2]).mean()) if g2.sum() >= 5 else np.nan
            if sc != "clean":
                aff = np.abs(pred - clean["pred"]) > 1e-6 * max(np.abs(y).max(), 1e-9)
                acc_clean = apply_threshold(clean["scores"][name], clean["ties"], frozen[name])
                new = ((pred - y) > delta) & ~((clean["pred"] - y) > delta)
                g1 = aff & ~(err > delta)
                g3 = aff & (err > 2 * delta) & (std_now <= np.median(std_now))
                row.update(G1_accept_fault_ok=float(acc_m[g1].mean()) if g1.sum() >= 5 else np.nan,
                           G3_reject_fault_lowspread_severe=float((~acc_m[g3]).mean()) if g3.sum() >= 5 else np.nan,
                           frac_affected=float(aff.mean()),
                           flip_rate=float((acc_f[aff] != acc_clean[aff]).mean()) if aff.any() else np.nan,
                           reject_new=float((~acc_m[new]).mean()) if new.sum() >= 5 else np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    meta = dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=seed, fold=cfg.get("fold", cfg.get("subset")),
                v1_reproduction_max_abs_diff=check, detector_train_positive_rate=float(C.mean()))
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
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("model") in ("gru", "tcn") and cfg.get("point_target", "capped") != "raw":
                runs.append(done.parent)
    runs = runs[i0::n]
    a.out.mkdir(parents=True, exist_ok=True)
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
    print("ABL4_DONE", flush=True)


if __name__ == "__main__":
    main()
