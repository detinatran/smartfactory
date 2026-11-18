"""EXT21 (EXT21_PROTOCOL.md, revised): build a run-to-failure table from Backblaze daily SMART snapshots.
Each drive keeps its observed history (up to 365 days before failure; >= 120 days required); rul_steps = min(days to
failure, 90), so window position no longer reveals the target.
Two passes over the quarterly zips (read in place): (1) failures, (2) SMART rows of the selected failed drives in the
90 days before failure. Writes OUT/hdd.csv with columns unit, model, day, rul_steps, fold and the SMART channels."""
import argparse
import io
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 90  # RUL cap (days)
HIST = 365  # maximum history kept per drive
MIN_SPAN = 120
RAW = [5, 9, 12, 187, 188, 192, 193, 194, 197, 198, 199, 240, 241, 242]
NORM = [1, 3, 7]
COLS = [f"smart_{a}_raw" for a in RAW] + [f"smart_{a}_normalized" for a in NORM]


def daily(zips):
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".csv") and "__MACOSX" not in n):
                yield name, zf, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ranks", default="1-4", help="model ranks by eligible failures, e.g. 1-4 (EXT21) or 5-8 (EXT26)")
    ap.add_argument("--name", default="hdd", help="output file stem")
    a = ap.parse_args()
    # pass 1: failures and first-seen dates
    fails, first = [], {}
    for name, zf, _ in daily(a.zips):
        df = pd.read_csv(io.BytesIO(zf.read(name)), usecols=["date", "serial_number", "model", "failure"])
        d0 = df.date.iloc[0]
        new = df.loc[~df.serial_number.isin(first.keys()), "serial_number"]
        first.update(dict.fromkeys(new, d0))
        fails.append(df[df.failure == 1])
    fails = pd.concat(fails).drop_duplicates("serial_number")
    fails["first"] = fails.serial_number.map(first)
    fails["span"] = (pd.to_datetime(fails.date) - pd.to_datetime(fails["first"])).dt.days
    elig = fails[fails.span >= MIN_SPAN]
    r0, r1 = map(int, a.ranks.split("-"))
    vc = elig.model.value_counts()
    print("eligible failures by model (top 12):", vc.head(12).to_dict(), flush=True)
    top = vc.iloc[r0 - 1:r1]
    print(f"selected ranks {a.ranks}:", top.to_dict(), flush=True)
    sel = elig[elig.model.isin(top.index)].set_index("serial_number")
    fdate = pd.to_datetime(sel.date)
    # pass 2: SMART rows of the selected drives
    rows = []
    for name, zf, _ in daily(a.zips):
        hdr = pd.read_csv(io.BytesIO(zf.read(name)), nrows=0).columns
        use = ["date", "serial_number"] + [c for c in COLS if c in hdr]
        df = pd.read_csv(io.BytesIO(zf.read(name)), usecols=use)
        df = df[df.serial_number.isin(sel.index)]
        if df.empty:
            continue
        dt = pd.to_datetime(df.date)
        left = (fdate.reindex(df.serial_number).to_numpy() - dt.to_numpy()).astype("timedelta64[D]").astype(int)
        keep_rows = (left >= 0) & (left < HIST)
        df = df[keep_rows].assign(days_left=left[keep_rows])
        rows.append(df)
    t = pd.concat(rows, ignore_index=True)
    for c in COLS:
        if c not in t:
            t[c] = np.nan
    keep = [c for c in COLS if t[c].isna().mean() <= 0.5] if a.name == "hdd" else list(COLS)  # hdd2: same 17 channels
    print("channels kept:", keep, flush=True)
    t = t.sort_values(["serial_number", "days_left"], ascending=[True, False])
    # one row per day from the first kept day to failure, forward-filled per drive
    full = []
    for s, g in t.groupby("serial_number"):
        top = int(g.days_left.max())
        g = g.drop_duplicates("days_left").set_index("days_left").reindex(range(top, -1, -1))
        g[keep] = g[keep].ffill().bfill()
        g["serial_number"] = s
        g.index.name = "days_left"
        full.append(g.reset_index())
    t = pd.concat(full, ignore_index=True)
    t = t.dropna(subset=keep, how="all")
    for c in keep:
        if c.endswith("_raw"):
            t[c] = np.log1p(t[c].clip(lower=0))
        t[c] = t[c].fillna(t[c].median()).fillna(0.0)
    t["model"] = t.serial_number.map(sel.model)
    t["day"] = t.groupby("serial_number").days_left.transform("max") - t.days_left
    t["rul_steps"] = t.days_left.clip(upper=HORIZON)
    t["fold"] = np.where([zlib.crc32(str(s).encode()) % 10 < 3 for s in t.serial_number], "test", "dev")
    a.out.mkdir(parents=True, exist_ok=True)
    t[["serial_number", "model", "day", "rul_steps", "fold"] + keep].to_csv(a.out / f"{a.name}.csv", index=False)
    print(f"drives={t.serial_number.nunique()} rows={len(t)} test drives={t[t.fold == 'test'].serial_number.nunique()}",
          flush=True)
    print("HDD_DONE", flush=True)


if __name__ == "__main__":
    main()
