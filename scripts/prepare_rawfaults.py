"""EXT17 (EXT17_PROTOCOL.md): physical sensor faults on RAW bearing acceleration (FEMTO, XJTU-SY). For each fault kind,
writes OUT/<kind>/{femto,xjtu_sy}.csv: identical to the processed tables except that, inside each unit's fault interval,
the features of the faulted axis are recomputed from the faulted raw signal with prepare_external.signal_features."""
import argparse
import io
import shutil
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_external import SEVEN_ZIP, signal_features  # noqa: E402

KINDS = ("raw_gain", "raw_clip", "raw_detach", "raw_noise", "raw_spike")


def plan(frame):
    """unit -> (axis, set of faulted steps, r_ref) with unit-name seeds."""
    out = {}
    for unit, g in frame.groupby("unit"):
        rng = np.random.default_rng(np.random.SeedSequence([17, zlib.crc32(str(unit).encode())]))
        axis = ["h", "v"][int(rng.integers(0, 2))]
        onset = rng.uniform(0.2, 0.4)
        g = g.sort_values("step")
        pos = np.arange(len(g)) / max(len(g) - 1, 1)
        steps = set(g.step.to_numpy()[(pos >= onset) & (pos < onset + 0.3)])
        early = g[f"{axis}_rms"].to_numpy()[: max(1, len(g) // 10)]
        out[unit] = (axis, steps, float(np.median(early)), zlib.crc32(str(unit).encode()))
    return out


def fault(x, kind, r, rng):
    x = np.asarray(x, dtype=np.float64)
    if kind == "raw_gain":
        return 0.5 * x
    if kind == "raw_clip":
        return np.clip(x, -2 * r, 2 * r)
    if kind == "raw_detach":
        return 0.6 * np.convolve(x, np.ones(8) / 8, mode="same")
    if kind == "raw_noise":
        return x + rng.normal(0.0, r, len(x))
    if kind == "raw_spike":
        y = x.copy()
        idx = rng.random(len(x)) < 0.002
        y[idx] += rng.choice([-1, 1], idx.sum()) * 10 * r
        return y
    raise ValueError(kind)


def apply(frame, unit, step, sig_h, sig_v, P, tables):
    axis, steps, r, salt = P[unit]
    if step not in steps:
        return
    sig = sig_h if axis == "h" else sig_v
    i = frame.index[(frame.unit == unit) & (frame.step == step)]
    for k, kind in enumerate(KINDS):
        rng = np.random.default_rng(np.random.SeedSequence([17, salt, int(step), k]))
        feats = signal_features(fault(sig, kind, r, rng), axis)
        for col, v in feats.items():
            tables[kind].loc[i, col] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, required=True)
    a = ap.parse_args()
    # FEMTO
    fe = pd.read_csv(a.processed / "femto.csv")
    P = plan(fe)
    tables = {k: fe.copy() for k in KINDS}
    outer = zipfile.ZipFile(a.root / "femto/FEMTO_Bearing.zip")
    inner = zipfile.ZipFile(outer.open("10. FEMTO Bearing/FEMTOBearingDataSet.zip"))
    n = 0
    for part in ("Training_set.zip", "Validation_Set.zip"):
        z = zipfile.ZipFile(io.BytesIO(inner.read(part)))
        for name in sorted(x for x in z.namelist() if x.split("/")[-1].startswith("acc_")):
            _, bearing, file = name.split("/")
            step = int(file[4:9])
            if bearing not in P or step not in P[bearing][1]:
                continue
            raw = z.read(name).decode()
            data = pd.read_csv(io.StringIO(raw), sep=";" if ";" in raw[:200] else ",", header=None).to_numpy()
            apply(fe, bearing, step, data[:, 4], data[:, 5], P, tables)
            n += 1
    print(f"femto: {n} faulted snapshots", flush=True)
    for k in KINDS:
        (a.out / k).mkdir(parents=True, exist_ok=True)
        tables[k].to_csv(a.out / k / "femto.csv", index=False)
    # XJTU-SY
    xj = pd.read_csv(a.processed / "xjtu_sy.csv")
    P = plan(xj)
    tables = {k: xj.copy() for k in KINDS}
    first = a.root / "xjtu_sy" / "XJTU-SY_Bearing_Datasets.part01.rar"
    n = 0
    for condition, code in (("35Hz12kN", 1), ("37.5Hz11kN", 2), ("40Hz10kN", 3)):
        target = a.tmp / f"rawf_xjtu_{code}"
        if target.exists():
            shutil.rmtree(target)
        subprocess.run([SEVEN_ZIP, "x", "-y", f"-o{target}", str(first), f"XJTU-SY_Bearing_Datasets\\{condition}\\*"],
                       check=True, stdout=subprocess.DEVNULL)
        for bdir in sorted((target / "XJTU-SY_Bearing_Datasets" / condition).iterdir()):
            unit = bdir.name
            if unit not in P:
                continue
            for path in bdir.glob("*.csv"):
                step = int(path.stem)
                if step not in P[unit][1]:
                    continue
                data = pd.read_csv(path).to_numpy()
                apply(xj, unit, step, data[:, 0], data[:, 1], P, tables)
                n += 1
        shutil.rmtree(target)
        print(f"xjtu {condition}: cumulative {n} faulted snapshots", flush=True)
    for k in KINDS:
        tables[k].to_csv(a.out / k / "xjtu_sy.csv", index=False)
    print("RAWFAULTS_DONE", flush=True)


if __name__ == "__main__":
    main()
