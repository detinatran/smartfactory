"""MIT-Stanford-TRI battery cycle-life data (Severson et al., Nature Energy 2019) -> per-cycle table.

Downloads each official .mat batch from data.matr.io one at a time, records SHA256, extracts the
per-cycle SUMMARY data (QDischarge, QCharge, IR, Tavg, Tmin, Tmax, chargetime) plus cycle life
and charging policy, then DELETES the .mat (user-approved, 2026-11-09) before the next download.
The merge step follows the authors' public loading code (rdbraatz/data-driven-prediction-...):
drop b1c8,b1c10,b1c12,b1c13,b1c22 (did not reach 80%); append the batch-2 continuations
b2c7,b2c8,b2c9,b2c15,b2c16 to b1c0..b1c4 (cycle_life += 662,981,1060,208,482); drop the noisy
b3c37,b3c2,b3c23,b3c32,b3c42,b3c43  -> 124 cells.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    1: ("5c86c0b5fa2ede00015ddf66", "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
    2: ("5c86bf13fa2ede00015ddd82", "2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
    3: ("5c86bd64fa2ede00015ddbb2", "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
}
FIELDS = ("cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmin", "Tmax", "chargetime")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(mat, batch_no):
    import h5py

    rows, cells = [], []
    with h5py.File(mat, "r") as f:
        batch = f["batch"]
        for i in range(batch["summary"].shape[0]):
            life = float(np.array(f[batch["cycle_life"][i, 0]]).ravel()[0])
            policy = np.array(f[batch["policy_readable"][i, 0]]).tobytes()[::2].decode()
            summary = f[batch["summary"][i, 0]]
            data = {k: np.array(summary[k]).ravel() for k in FIELDS}
            cell = f"b{batch_no}c{i}"
            cells.append(dict(cell=cell, cycle_life=life, policy=policy, n_summary=len(data["cycle"])))
            frame = pd.DataFrame(data)
            frame.insert(0, "cell", cell)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cells)


def merge(parts, cells):
    drop = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22", "b3c37", "b3c2", "b3c23", "b3c32",
            "b3c42", "b3c43"}
    carry = dict(zip(["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"],
                     ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]))
    add_len = dict(zip(["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"], [662, 981, 1060, 208, 482]))
    data = pd.concat(parts, ignore_index=True)
    out = []
    for cell, g in data.groupby("cell", sort=False):
        if cell in drop or cell in carry:
            continue
        g = g.copy()
        if cell in add_len:
            cont = data[data.cell == {v: k for k, v in carry.items()}[cell]].copy()
            cont["cycle"] = cont["cycle"] + len(g)
            cont["cell"] = cell
            g = pd.concat([g, cont], ignore_index=True)
        out.append(g)
    table = pd.concat(out, ignore_index=True)
    info = cells.set_index("cell")
    life = {c: info.loc[c, "cycle_life"] + add_len.get(c, 0) for c in table.cell.unique()}
    table["cycle_life"] = table.cell.map(life)
    table["policy"] = table.cell.map(info.policy)
    table["batch"] = table.cell.str[1].astype(int)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external/battery"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    parts, cells, sources = [], [], {}
    for batch_no, (file_id, name) in FILES.items():
        part_csv = args.root / f"batch{batch_no}_summary.csv"
        cell_csv = args.root / f"batch{batch_no}_cells.csv"
        if not part_csv.exists():
            mat = args.root / name
            url = f"https://data.matr.io/1/api/v1/file/{file_id}/download"
            print(f"download {name}", flush=True)
            subprocess.run(["curl.exe", "-L", "--fail", "--retry", "5", "-C", "-", "-s", "-o",
                            str(mat), url], check=True)
            sources[name] = {"url": url, "bytes": mat.stat().st_size, "sha256": sha256(mat)}
            (args.root / f"batch{batch_no}_source.json").write_text(json.dumps(sources[name], indent=2))
            part, cell = extract(mat, batch_no)
            part.to_csv(part_csv, index=False)
            cell.to_csv(cell_csv, index=False)
            mat.unlink()
            print(f"batch {batch_no}: {cell.shape[0]} cells, {len(part)} cycles; .mat deleted",
                  flush=True)
        else:
            sources[name] = json.loads((args.root / f"batch{batch_no}_source.json").read_text())
        parts.append(pd.read_csv(part_csv))
        cells.append(pd.read_csv(cell_csv))
    table = merge(parts, pd.concat(cells, ignore_index=True))
    out = args.root.parent / "processed" / "battery.csv"
    table.to_csv(out, index=False)
    manifest = dict(
        dataset="battery_severson2019", created_at=datetime.now(timezone.utc).isoformat(),
        cells=int(table.cell.nunique()), rows=len(table),
        cells_per_batch=table.groupby("batch").cell.nunique().to_dict(),
        cycle_life=table.groupby("cell").cycle_life.first().describe().round(1).to_dict(),
        sources=sources, output_sha256=sha256(out), notes=__doc__,
    )
    (out.parent / "battery.manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("cells", "rows", "cells_per_batch")}, default=str))
    print("BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
