"""Unseen-fault evaluation + fault-augmented selectors + one-sided SelectiveNet (FAULT_PROTOCOL.md).

For each existing run (C-MAPSS ext_v1 capped gru/tcn; external_v1) -> mirrored dir under --out:
  * predictions_{scenario}.csv (C-MAPSS: development/...) for clean + 6 unseen faults, with
    score_* columns for: all v1 selectors (frozen, trained on clean/dropout/drift), mc_dropout,
    selectivenet, deep_ens_nll, cqr_lower, and the leave-one-fault-out augmented selectors
    asym_aug__{f}, asym_aug_dro__{f}, q90_aug__{f} (present only in the CSV of fault f)
  * thresholds.json (frozen thresholds on clean calibration windows for every score)
  * own_metrics.csv: SelectiveNet / one-sided SelectiveNet evaluated with THEIR OWN predictions
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_baselines import cmapss_windows, external_windows  # noqa: E402

from selective_rul.baselines import (  # noqa: E402
    HeadNet,
    baseline_scores,
    head_outputs,
    train_head_model,
)
from selective_rul.data import engine_weights  # noqa: E402
from selective_rul.faults import FAULTS  # noqa: E402
from selective_rul.metrics import selective_metrics  # noqa: E402
from selective_rul.model import build_model, predict_details, seed_everything  # noqa: E402
from selective_rul.selection import (  # noqa: E402
    apply_threshold,
    fit_threshold,
    selector_features,
    tie_breakers,
)

SEEN = ("clean", "dropout", "drift")
HGB = dict(max_iter=100, max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=2.0,
           early_stopping=False)


# ------------------------------------------------------------------ one-sided SelectiveNet
def train_os_selectivenet(train, validation, cfg, device, seed, delta):
    """SelectiveNet whose selective loss is squared error + smooth one-sided event term."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    model = HeadNet(build_model(train.x.shape[-1], dict(cfg, head="point")), 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    cap = float(cfg["rul_cap"])
    tau = delta / 4.0

    def loss_fn(point, ext, y, w):
        err = (point - y) / cap
        event = torch.sigmoid((point - y - delta) / tau)
        ell = err.square() + event
        g = torch.sigmoid(ext[:, 0])
        sel = (g * w * ell).sum() / (g * w).sum().clamp_min(1e-6)
        cov = (g * w).sum() / w.sum()
        pen = 32.0 * torch.clamp(0.9 - cov, min=0).square()
        aux = (w * (torch.sigmoid(ext[:, 1]) - y / cap).square()).mean()
        return 0.5 * (sel + pen) + 0.5 * aux

    w_tr = engine_weights(train.meta.unit.to_numpy()).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y),
                                      torch.from_numpy(w_tr)), batch_size=cfg["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    xv, yv = torch.from_numpy(validation.x).to(device), torch.from_numpy(validation.y).to(device)
    wv = torch.from_numpy(engine_weights(validation.meta.unit.to_numpy()).astype(np.float32)
                          ).to(device)
    best, stale, state = float("inf"), 0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            point, ext = model(x)
            loss_fn(point, ext, y, w).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            parts = [model(xv[i:i + 1024]) for i in range(0, len(xv), 1024)]
            value = float(loss_fn(torch.cat([p[0] for p in parts]),
                                  torch.cat([p[1] for p in parts]), yv, wv))
        if value < best:
            best, stale = value, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(state)
    return model.eval()


# ------------------------------------------------------------------ augmented selectors
def fit_aug(features, pred, y, units, groups, delta, seed, dro):
    bad = (pred - y > delta).astype(int)
    base_w = engine_weights(units)
    names = np.unique(groups)
    gw = {g: 1.0 for g in names}
    rounds = 3 if dro else 1
    clf = None
    for _ in range(rounds):
        w = base_w.copy()
        for g in names:  # group-balanced (each corruption group has equal total weight)
            m = groups == g
            w[m] *= gw[g] / base_w[m].sum()
        w *= len(w) / w.sum()
        clf = HistGradientBoostingClassifier(random_state=seed, **HGB).fit(features, bad,
                                                                           sample_weight=w)
        if dro:
            p = np.clip(clf.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
            nll = -(bad * np.log(p) + (1 - bad) * np.log(1 - p))
            losses = {g: np.average(nll[groups == g], weights=base_w[groups == g]) for g in names}
            mean = np.mean(list(losses.values()))
            gw = {g: gw[g] * np.exp(2.0 * losses[g] / mean) for g in names}
            total = sum(gw.values())
            gw = {g: v * len(names) / total for g, v in gw.items()}
    return clf


def fit_q90_aug(features, pred, y, units, groups, seed):
    base_w = engine_weights(units)
    w = base_w.copy()
    for g in np.unique(groups):
        m = groups == g
        w[m] /= base_w[m].sum()
    w *= len(w) / w.sum()
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=seed,
                                         **HGB).fit(features, pred - y, sample_weight=w)


# ------------------------------------------------------------------ per run
def process(run, out, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    external = "dataset" in cfg
    if external:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import external_features, load_dataset

        n_feat = len(load_dataset(cfg["dataset"], ext_dir)[1])

        def features_of(w, ens):
            return external_features(w, ens, n_feat)
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)

        def features_of(w, ens):
            return selector_features(w, ens, cfg["rul_cap"], None)
    delta, cap, seed = float(cfg["risk_delta"]), float(model_cfg["rul_cap"]), cfg["seed"]
    selectors = joblib.load(run / "selectors.joblib")
    ensemble = []
    for member in range(cfg["ensemble_size"]):
        ckpt = torch.load(run / f"model_{member}.pt", map_location="cpu", weights_only=True)
        model = build_model(ckpt["input_dim"], model_cfg)
        model.load_state_dict(ckpt["state_dict"])
        ensemble.append(model.to(device).eval())
    train, validation = windows("fit", "clean"), windows("validation", "clean")
    fitted = {
        "selectivenet": [train_head_model("selectivenet", train, validation, model_cfg, device,
                                          seed + 700)],
        "deep_ens_nll": [train_head_model("deep_ens_nll", train, validation, model_cfg, device,
                                          seed + 800 + 100 * m) for m in range(3)],
        "cqr_lower": [train_head_model("cqr_lower", train, validation, model_cfg, device,
                                       seed + 900 + 100 * m) for m in range(3)],
    }
    os_net = train_os_selectivenet(train, validation, model_cfg, device, seed + 1200, delta)

    def all_scores(w):
        ens, _ = predict_details(ensemble, w.x, device)
        pred = ens.mean(1)
        ties = tie_breakers(w.meta, seed + 20000)
        feats = features_of(w, ens)
        scores = selectors.scores(feats, ens, ties)
        scores.update(baseline_scores(fitted, ensemble, w.x, pred, cap, device, seed + 30000))
        return pred, ties, feats, scores

    # selector-split windows for every scenario (augmented selectors, leave-one-fault-out)
    sel = {}
    for scenario in SEEN + FAULTS:
        w = windows("selector", scenario)
        pred, _, feats, _ = all_scores(w)
        sel[scenario] = (feats, pred, w.y if external else w.meta.rul_raw.to_numpy(),
                         w.meta.unit.to_numpy())
    calib = windows("calibration", "clean")
    c_pred, c_ties, c_feats, c_scores = all_scores(calib)
    c_units = calib.meta.unit.to_numpy()
    aug = {}
    for held in FAULTS:
        keep = [s for s in SEEN + FAULTS if s != held]
        F = np.concatenate([sel[s][0] for s in keep])
        P = np.concatenate([sel[s][1] for s in keep])
        Y = np.concatenate([sel[s][2] for s in keep])
        U = np.concatenate([sel[s][3] for s in keep])
        G = np.concatenate([np.full(len(sel[s][1]), s) for s in keep])
        aug[held] = {
            "asym_aug": fit_aug(F, P, Y, U, G, delta, seed, dro=False),
            "asym_aug_dro": fit_aug(F, P, Y, U, G, delta, seed, dro=True),
            "q90_aug": fit_q90_aug(F, P, Y, U, G, seed),
        }

    def aug_scores(held, feats):
        return {
            f"asym_aug__{held}": aug[held]["asym_aug"].predict_proba(feats)[:, 1],
            f"asym_aug_dro__{held}": aug[held]["asym_aug_dro"].predict_proba(feats)[:, 1],
            f"q90_aug__{held}": aug[held]["q90_aug"].predict(feats),
        }

    thresholds = {}
    coverages = cfg["coverages"]
    for m, s in c_scores.items():
        thresholds[m] = {str(c): fit_threshold(s, c_ties, c_units, c) for c in coverages}
    for held in FAULTS:
        for m, s in aug_scores(held, c_feats).items():
            thresholds[m] = {str(c): fit_threshold(s, c_ties, c_units, c) for c in coverages}
    # own-prediction SelectiveNets: thresholds on their g over calibration
    own_nets = {"selectivenet_own": fitted["selectivenet"][0], "os_selectivenet_own": os_net}
    own_thr = {}
    for name, net in own_nets.items():
        _, ext = head_outputs(net, calib.x, device)
        own_thr[name] = fit_threshold(-ext[:, 0], c_ties, c_units, 0.9)

    out.mkdir(parents=True, exist_ok=True)
    own_rows = []
    scenarios = ["clean", *FAULTS]
    for scenario in scenarios:
        w = windows(eval_part, scenario)
        pred, ties, feats, scores = all_scores(w)
        if scenario in FAULTS:
            scores.update(aug_scores(scenario, feats))
        meta = w.meta.copy()
        meta["rul_capped"] = w.y
        meta["prediction"], meta["tie_breaker"] = pred, ties
        for m, s in scores.items():
            meta[f"score_{m}"] = s
        path = (out / f"predictions_{scenario}.csv") if external else (
            out / "development" / f"predictions_{scenario}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(path, index=False)
        raw_t = meta.rul_raw.to_numpy()
        units = meta.unit.to_numpy()
        for name, net in own_nets.items():
            own_pred, ext = head_outputs(net, w.x, device)
            score = -ext[:, 0]
            for protocol, thr in (("frozen", own_thr[name]),
                                  ("matched", fit_threshold(score, ties, units, 0.9))):
                acc = apply_threshold(score, ties, thr)
                mtr = selective_metrics(own_pred, raw_t, w.y, units, acc, delta,
                                        cfg["critical_rul"])
                own_rows.append(dict(scenario=scenario, method=name, protocol=protocol,
                                     **{k: mtr[k] for k in ("coverage", "optimistic_error_rate",
                                                            "critical_optimistic_error_rate",
                                                            "rmse_raw", "rmse_capped")}))
            full = selective_metrics(own_pred, raw_t, w.y, units, np.ones(len(own_pred), bool),
                                     delta, cfg["critical_rul"])
            own_rows.append(dict(scenario=scenario, method=name, protocol="always",
                                 **{k: full[k] for k in ("coverage", "optimistic_error_rate",
                                                         "critical_optimistic_error_rate",
                                                         "rmse_raw", "rmse_capped")}))
    pd.DataFrame(own_rows).to_csv(out / "own_metrics.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    new_cfg = dict(cfg, eval_scenarios=scenarios)
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    shutil.copy2(run / "splits.json", out / "splits.json")
    (out / "complete.json").write_text(json.dumps({"source_run": str(run)}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gru", "tcn"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/CMAPSSData"))
    parser.add_argument("--ext-data-dir", type=Path,
                        default=Path("data/external/processed"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    shard, n_shards = map(int, args.shard.split("/"))
    runs = []
    for root in args.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("point_target") == "raw" or cfg["model"] not in args.models:
                continue
            runs.append((root, done.parent))
    runs = runs[shard::n_shards]
    for i, (root, run) in enumerate(runs, 1):
        out = args.out / root.name / run.relative_to(root)
        if (out / "complete.json").exists():
            continue
        start = time.time()
        try:
            process(run, out, args.data_dir, args.ext_data_dir, args.device)
            print(f"[{i}/{len(runs)}] {run.name} {time.time() - start:.0f}s", flush=True)
        except Exception as error:
            print(f"[{i}/{len(runs)}] FAILED {run}: {error}\n{traceback.format_exc()}", flush=True)
    print("FAULT_DONE", flush=True)


if __name__ == "__main__":
    main()
