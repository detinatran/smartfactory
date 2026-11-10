"""Convert downloaded external run-to-failure datasets into compact per-step feature tables.

Archives are read in place (or extracted piecewise to a temp dir that is deleted again);
raw archives are never modified. Outputs one CSV per dataset plus a JSON manifest.

  ncmapss : N-CMAPSS (NASA DS01-DS08d). One row per (file, split, unit, flight cycle):
            mean of operating conditions W and physical sensors X_s over the flight,
            flight length, flight class Fc, RUL label Y. Virtual sensors X_v and health
            parameters T are latent/simulator-only and are NOT exported.
  femto   : FEMTO/PRONOSTIA (IEEE PHM 2012). Learning_set + Full_Test_Set (17 complete
            run-to-failure bearings; the truncated Test_set is skipped). One row per
            10-s snapshot (2560 samples, 25.6 kHz, horizontal+vertical acceleration).
  xjtu_sy : XJTU-SY. 15 bearings, 3 conditions; one row per 1-min snapshot (32768 samples).

Labels: rul_steps = number of later snapshots/cycles until the last recorded one
(last = 0); rul_time in minutes (bearings) or flight cycles (N-CMAPSS: dataset Y).
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"
N_BANDS = 8


def signal_features(x, prefix):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    absx = np.abs(x)
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    std = x.std() + 1e-12
    peak = absx.max()
    mean_abs = absx.mean() + 1e-12
    z = x / std
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.linspace(0.0, 1.0, len(power))
    bands = np.array_split(power[1:], N_BANDS)
    total = power[1:].sum() + 1e-12
    out = {
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_p2p": x.max() - x.min(),
        f"{prefix}_kurtosis": float(np.mean(z**4)),
        f"{prefix}_skewness": float(np.mean(z**3)),
        f"{prefix}_crest": peak / rms,
        f"{prefix}_shape": rms / mean_abs,
        f"{prefix}_impulse": peak / mean_abs,
        f"{prefix}_spectral_centroid": float((freqs * power).sum() / (power.sum() + 1e-12)),
    }
    for i, band in enumerate(bands):
        out[f"{prefix}_band{i}_rel"] = float(band.sum() / total)
    return out


def label_steps(frame, keys):
    frame = frame.sort_values(keys + ["step"]).reset_index(drop=True)
    last = frame.groupby(keys).step.transform("max")
    frame["rul_steps"] = last - frame.step
    return frame


# ---------------------------------------------------------------- FEMTO / PRONOSTIA
def femto(archive):
    rows = []
    outer = zipfile.ZipFile(archive)
    inner = zipfile.ZipFile(outer.open("10. FEMTO Bearing/FEMTOBearingDataSet.zip"))
    for part, top in (("Training_set.zip", "Learning_set"), ("Validation_Set.zip", "Full_Test_Set")):
        z = zipfile.ZipFile(io.BytesIO(inner.read(part)))
        names = sorted(n for n in z.namelist() if n.split("/")[-1].startswith("acc_"))
        for name in names:
            _, bearing, file = name.split("/")
            raw = z.read(name).decode()
            sep = ";" if ";" in raw[:200] else ","
            data = pd.read_csv(io.StringIO(raw), sep=sep, header=None).to_numpy()
            row = dict(
                dataset="femto",
                source_set=top,
                condition=int(bearing[7]),
                unit=bearing,
                step=int(file[4:9]),
                clock_s=float(data[0, 0] * 3600 + data[0, 1] * 60 + data[0, 2] + data[0, 3] * 1e-6),
            )
            row.update(signal_features(data[:, 4], "h"))
            row.update(signal_features(data[:, 5], "v"))
            rows.append(row)
        print(f"femto {top}: {len(names)} snapshots", flush=True)
    frame = label_steps(pd.DataFrame(rows), ["unit"])
    frame["rul_time"] = frame.rul_steps * 10.0 / 60.0  # minutes (10-s snapshot spacing)
    return frame


# ---------------------------------------------------------------- XJTU-SY
def xjtu(folder, tmp):
    first = folder / "XJTU-SY_Bearing_Datasets.part01.rar"
    rows = []
    for condition, code in (("35Hz12kN", 1), ("37.5Hz11kN", 2), ("40Hz10kN", 3)):
        target = tmp / f"xjtu_{code}"
        if target.exists():
            shutil.rmtree(target)
        pattern = f"XJTU-SY_Bearing_Datasets\\{condition}\\*"
        subprocess.run([SEVEN_ZIP, "x", "-y", f"-o{target}", str(first), pattern],
                       check=True, stdout=subprocess.DEVNULL)
        for bearing_dir in sorted((target / "XJTU-SY_Bearing_Datasets" / condition).iterdir()):
            files = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(p.stem))
            for path in files:
                data = pd.read_csv(path).to_numpy()
                row = dict(dataset="xjtu_sy", source_set=condition, condition=code,
                           unit=bearing_dir.name, step=int(path.stem))
                row.update(signal_features(data[:, 0], "h"))
                row.update(signal_features(data[:, 1], "v"))
                rows.append(row)
            print(f"xjtu {condition}/{bearing_dir.name}: {len(files)} snapshots", flush=True)
        shutil.rmtree(target)
    frame = label_steps(pd.DataFrame(rows), ["unit"])
    frame["rul_time"] = frame.rul_steps.astype(float)  # minutes (1-min spacing)
    return frame


# ---------------------------------------------------------------- N-CMAPSS
def open_h5(path, notes, name):
    """Open an HDF5 file; if it is truncated (seen for DS08d-010 in the NASA archive), pad the
    TEMP copy with zero bytes up to the EOF stored in its superblock and record that it was padded."""
    import re

    import h5py

    try:
        return h5py.File(path, "r")
    except OSError as error:
        match = re.search(r"eof = (\d+).*stored_eof = (\d+)", str(error))
        if not match:
            raise
        eof, stored = map(int, match.groups())
        if not 0 < stored - eof <= 4096:
            raise
        with path.open("ab") as handle:
            handle.write(b"\0" * (stored - eof))
        notes[name] = f"truncated in source archive by {stored - eof} bytes; temp copy zero-padded"
        print(f"WARNING {name}: {notes[name]}", flush=True)
        return h5py.File(path, "r")


def read_ncmapss_file(path, notes, name):
    frames = []
    with open_h5(path, notes, name) as h5:
        def names(key):
            return [v.decode().strip() if isinstance(v, bytes) else str(v).strip()
                    for v in np.array(h5[key]).ravel()]
        w_names, xs_names, a_names = names("W_var"), names("X_s_var"), names("A_var")
        for split in ("dev", "test"):
            w = pd.DataFrame(np.array(h5[f"W_{split}"]), columns=w_names)
            xs = pd.DataFrame(np.array(h5[f"X_s_{split}"]), columns=xs_names)
            a = pd.DataFrame(np.array(h5[f"A_{split}"]), columns=a_names)
            y = np.array(h5[f"Y_{split}"]).ravel()
            df = pd.concat([a[["unit", "cycle", "Fc", "hs"]], w, xs], axis=1)
            df["Y"] = y
            g = df.groupby(["unit", "cycle"], sort=True)
            agg = g[w_names + xs_names].mean()
            agg["flight_seconds"] = g.size()
            agg["Fc"] = g.Fc.first()
            agg["hs_min"] = g.hs.min()
            agg["rul_time"] = g.Y.first()
            agg = agg.reset_index()
            agg.insert(0, "split", split)
            agg.insert(0, "file", name)
            frames.append(agg)
            print(f"ncmapss {name}/{split}: {agg.unit.nunique()} units, {len(agg)} cycles",
                  flush=True)
    return frames


def ncmapss(archive, tmp, only=None, notes=None):
    notes = {} if notes is None else notes
    frames = []
    outer = zipfile.ZipFile(archive)
    with outer.open("17. Turbofan Engine Degradation Simulation Data Set 2/data_set.zip") as fo:
        inner = zipfile.ZipFile(fo)
        members = sorted(n for n in inner.namelist() if n.endswith(".h5"))
        for member in members:
            name = Path(member).stem.replace("N-CMAPSS_", "")
            if only and name not in only:
                continue
            path = tmp / f"{name}.h5"
            with inner.open(member) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            try:
                file_frames = read_ncmapss_file(path, notes, name)
            except (OSError, KeyError) as error:
                notes[name] = f"SKIPPED, unreadable HDF5 in source archive: {error}"
                print(f"WARNING {name}: {notes[name]}", flush=True)
                file_frames = []
            frames += file_frames
            path.unlink()
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "dataset", "ncmapss")
    frame["unit_id"] = frame.file + "_" + frame.split + "_u" + frame.unit.astype(int).astype(str)
    frame = frame.rename(columns={"cycle": "step"})
    frame = label_steps(frame, ["unit_id"])
    return frame


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    parser.add_argument("--datasets", nargs="+", default=["femto", "xjtu_sy", "ncmapss"])
    parser.add_argument("--ncmapss-files", nargs="*")
    args = parser.parse_args()
    out, tmp = args.root / "processed", args.root / "tmp"
    out.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    sources = {
        "femto": [args.root / "femto/FEMTO_Bearing.zip"],
        "xjtu_sy": sorted((args.root / "xjtu_sy").glob("*.rar")),
        "ncmapss": [args.root / "ncmapss/NCMAPSS.zip"],
    }
    for name in args.datasets:
        notes = {}
        if name == "femto":
            frame = femto(sources[name][0])
        elif name == "xjtu_sy":
            frame = xjtu(args.root / "xjtu_sy", tmp)
        else:
            frame = ncmapss(sources[name][0], tmp, args.ncmapss_files, notes)
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        unit_col = "unit_id" if "unit_id" in frame else "unit"
        manifest = dict(
            dataset=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            rows=len(frame),
            units=int(frame[unit_col].nunique()),
            steps_per_unit=frame.groupby(unit_col).size().describe().round(1).to_dict(),
            columns=list(frame.columns),
            output_sha256=sha256(path),
            sources={str(p): sha256(p) for p in sources[name]},
            script=__file__,
            notes=notes,
        )
        (out / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {path}: {len(frame)} rows, {manifest['units']} units", flush=True)


if __name__ == "__main__":
    main()
