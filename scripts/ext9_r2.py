"""EXT9: per-run analytic redundancy rho = median over channels of R^2_c = 1 - sigma_c^2 / Var(x_c) of the EXT6 Ridge
reconstruction, on clean fit-unit time steps (training data only). usage: ext9_r2.py --runs DIR... --out rho.csv"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext8_select import Reconstructor  # noqa: E402
from extra_baselines import cmapss_windows, external_windows  # noqa: E402

from selective_rul.data import SENSORS, SETTINGS  # noqa: E402


def channel_r2(rec, x):
    flat = x.reshape(-1, x.shape[2])
    ok = flat[:, rec.flag_off: rec.flag_off + rec.n_ch].sum(1) == 0
    v = flat[ok, : rec.n_ch]
    var = np.maximum(v.var(0), 1e-12)
    return 1.0 - rec.sigma ** 2 / var


def rho_of(run, raw, ext_dir):
    cfg = json.loads((run / "config.json").read_text())
    if "dataset" in cfg:
        windows, _, _ = external_windows(run, cfg, ext_dir)
        from selective_rul.external import load_dataset
        n_ch = len(load_dataset(cfg["dataset"], ext_dir)[1])
        flag_off, n_set = n_ch, 0
    else:
        windows, _, _ = cmapss_windows(run, cfg, raw)
        n_ch, n_set = len(SENSORS), len(SETTINGS)
        flag_off = n_ch + n_set
    fitw = windows("fit", "clean")
    rec = Reconstructor(n_ch, flag_off, n_set, cfg["seed"]).fit(fitw.x)
    r2 = channel_r2(rec, fitw.x)
    return dict(dataset=cfg.get("dataset", "cmapss"), model=cfg["model"], seed=cfg["seed"],
                fold=cfg.get("fold", cfg.get("subset")), n_channels=n_ch, rho=float(np.median(r2)),
                r2_min=float(r2.min()), r2_max=float(r2.max()), r2_all=" ".join(f"{v:.3f}" for v in r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/CMAPSSData"))
    ap.add_argument("--ext-data-dir", type=Path, default=Path("data/external/processed"))
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i0, n = map(int, a.shard.split("/"))
    runs = []
    for root in a.runs:
        for done in sorted(root.rglob("complete.json")):
            cfg = json.loads((done.parent / "config.json").read_text())
            if cfg.get("model") in ("gru", "tcn") and cfg.get("point_target", "capped") != "raw" and \
                    cfg.get("head", "point") == "point":
                runs.append(done.parent)
    rows = []
    for k, run in enumerate(runs[i0::n], 1):
        try:
            rows.append(rho_of(run, a.data_dir, a.ext_data_dir))
            print(f"[{k}] {run.name} rho={rows[-1]['rho']:.3f}", flush=True)
        except Exception as e:
            print(f"[{k}] FAILED {run}: {e}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("EXT9_R2_DONE", flush=True)


if __name__ == "__main__":
    main()
