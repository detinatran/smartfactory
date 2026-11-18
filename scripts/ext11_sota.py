"""EXT11 part A (protocol: EXT11_PROTOCOL.md): baselines that need extra trained models, on the same runs and the same
shared point-ensemble predictions as EXT10.

  selectivenet, deep_ens_nll, cqr_lower, mc_dropout : exactly as the ESANN study (selective_rul.baselines)
  evidential : Deep Evidential Regression (Amini et al., NeurIPS 2020), 1 model, NIG head, lambda = 0.01,
               score = predictive (Student-t) std sqrt(beta (1 + nu) / (nu (alpha - 1)))
Head models are trained on clean fit units with early stopping on clean validation units (ESANN configuration).
"""

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext10_rpt import ALPHA, EVAL, cmapss_windows, external_windows  # noqa: E402
from uncertainty_v1 import unit_risk  # noqa: E402

from selective_rul import baselines as B  # noqa: E402
from selective_rul.data import engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details, seed_everything  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, tie_breakers  # noqa: E402

LAMBDA_EVID = 0.01


def nig(extra):
    sp = nn.functional.softplus
    return sp(extra[:, 0]) + 1e-6, sp(extra[:, 1]) + 1.0 + 1e-6, sp(extra[:, 2]) + 1e-6  # nu, alpha, beta


def loss_fn(kind, point, extra, y, w, cap):
    if kind != "evidential":
        return B._loss(kind, point, extra, y, w, cap)
    nu, al, be = nig(extra)
    g, yy = point / cap, y / cap
    om = 2.0 * be * (1.0 + nu)
    nll = (0.5 * torch.log(math.pi / nu) - al * torch.log(om) + (al + 0.5) * torch.log(nu * (yy - g) ** 2 + om)
           + torch.lgamma(al) - torch.lgamma(al + 0.5))
    reg = (yy - g).abs() * (2.0 * nu + al)
    return (w * (nll + LAMBDA_EVID * reg)).mean()


def train_head(kind, train, validation, cfg, device, seed):
    """Copy of selective_rul.baselines.train_head_model with an extra 'evidential' kind."""
    seed_everything(seed)
    extra = {"selectivenet": 2, "deep_ens_nll": 1, "cqr_lower": 0, "evidential": 3}[kind]
    model = B.HeadNet(build_model(train.x.shape[-1], dict(cfg, head="point")), extra).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    cap = float(cfg["rul_cap"])
    w_train = engine_weights(train.meta.unit.to_numpy()).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y), torch.from_numpy(w_train)),
                        batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(validation.x).to(device)
    yv = torch.from_numpy(validation.y).to(device)
    wv = torch.from_numpy(engine_weights(validation.meta.unit.to_numpy()).astype(np.float32)).to(device)
    best, stale, best_state = float("inf"), 0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            point, ext = model(x)
            loss_fn(kind, point, ext, y, w, cap).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            parts = [model(xv[i:i + 1024]) for i in range(0, len(xv), 1024)]
            point = torch.cat([p[0] for p in parts])
            ext = torch.cat([p[1] for p in parts]) if parts[0][1] is not None else None
            value = float(loss_fn(kind, point, ext, yv, wv, cap))
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite validation loss for {kind}")
        if value < best:
            best, stale = value, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval()


def process(run, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    cap = float(model_cfg["rul_cap"])
    train, validation = windows("fit", "clean"), windows("validation", "clean")
    fitted = {
        "selectivenet": [train_head("selectivenet", train, validation, model_cfg, device, seed + 700)],
        "deep_ens_nll": [train_head("deep_ens_nll", train, validation, model_cfg, device, seed + 800 + 100 * m)
                         for m in range(3)],
        "cqr_lower": [train_head("cqr_lower", train, validation, model_cfg, device, seed + 900 + 100 * m)
                      for m in range(3)],
    }
    evid = train_head("evidential", train, validation, model_cfg, device, seed + 1300)
    del train, validation
    ensemble = []
    for m in range(cfg["ensemble_size"]):
        ck = torch.load(run / f"model_{m}.pt", map_location="cpu", weights_only=True)
        net = build_model(ck["input_dim"], model_cfg)
        net.load_state_dict(ck["state_dict"])
        ensemble.append(net.to(device).eval())

    def scores(w):
        ens, _ = predict_details(ensemble, w.x, device)
        pred = ens.mean(1)
        sc = B.baseline_scores(fitted, ensemble, w.x, pred, cap, device, seed + 30000)
        _, ext = B.head_outputs(evid, w.x, device)
        nu, al, be = (t.numpy() for t in nig(torch.from_numpy(ext)))
        sc["evidential"] = np.sqrt(be * (1.0 + nu) / (nu * (al - 1.0)))
        return pred, sc

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
    print("EXT11S_DONE", flush=True)


if __name__ == "__main__":
    main()
