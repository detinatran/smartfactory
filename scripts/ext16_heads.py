"""EXT16 A (protocol: EXT16_PROTOCOL.md): head-based baselines trained on clean fit windows PLUS one perturbed copy
(L configuration, true RUL labels): deep_ens_nll_aug, evidential_aug, cqr_lower_aug, selectivenet_aug, mc_dropout (on the
shared ensemble, unchanged; for reference). Same configuration as ext11_sota.py otherwise.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext10_rpt import ALPHA, EVAL, cmapss_windows, external_windows  # noqa: E402
from ext11_ablate import perturb  # noqa: E402
from ext11_sota import nig, train_head  # noqa: E402
from uncertainty_v1 import unit_risk  # noqa: E402

from selective_rul import baselines as B  # noqa: E402
from selective_rul.data import SENSORS, engine_weights  # noqa: E402
from selective_rul.model import build_model, predict_details  # noqa: E402
from selective_rul.selection import apply_threshold, fit_threshold, tie_breakers  # noqa: E402


def process(run, raw, ext_dir, device):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, eval_part, model_cfg = external_windows(run, cfg, ext_dir)
        from selective_rul.external import load_dataset
        n_ch = len(load_dataset(cfg["dataset"], ext_dir)[1])
    else:
        windows, eval_part, model_cfg = cmapss_windows(run, cfg, raw)
        n_ch = len(SENSORS)
    delta, seed = float(cfg["risk_delta"]), cfg["seed"]
    cap = float(model_cfg["rul_cap"])
    tr, va = windows("fit", "clean"), windows("validation", "clean")
    xa = perturb(tr.x, n_ch, seed + 52000, amp=(1.0, 8.0), kfrac=0.5)
    train = type("W", (), {"x": np.concatenate([tr.x, xa]), "y": np.concatenate([np.asarray(tr.y)] * 2),
                           "meta": pd.concat([tr.meta, tr.meta], ignore_index=True)})()
    fitted = {
        "selectivenet": [train_head("selectivenet", train, va, model_cfg, device, seed + 700)],
        "deep_ens_nll": [train_head("deep_ens_nll", train, va, model_cfg, device, seed + 800 + 100 * m)
                         for m in range(3)],
        "cqr_lower": [train_head("cqr_lower", train, va, model_cfg, device, seed + 900 + 100 * m) for m in range(3)],
    }
    evid = train_head("evidential", train, va, model_cfg, device, seed + 1300)
    del tr, va, train, xa
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
        sc.pop("mc_dropout", None)
        _, ext = B.head_outputs(evid, w.x, device)
        nu, al, be = (t.numpy() for t in nig(torch.from_numpy(ext)))
        sc["evidential"] = np.sqrt(be * (1.0 + nu) / (nu * (al - 1.0)))
        return pred, {f"{k}_aug": v for k, v in sc.items()}

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
    print("EXT16H_DONE", flush=True)


if __name__ == "__main__":
    main()
