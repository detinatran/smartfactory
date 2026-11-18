"""EXT41: per-(unit, cycle) natural-glitch flags for battery.csv (EXT41_PROTOCOL.md). usage: ext41_glitch.py battery.csv OUT.csv"""
import sys
import numpy as np
import pandas as pd

b = pd.read_csv(sys.argv[1])
b = b[b.cycle <= b.cycle_life].copy()  # same rows as the loader
codes = {u: i + 1 for i, u in enumerate(sorted(b.cell.unique()))}
b["unit"] = b.cell.map(codes).astype(int)
b = b.sort_values(["cell", "cycle"])
MAIN = {"QDischarge": ("rel", 0.02), "QCharge": ("rel", 0.02), "IR": ("rel", 0.10), "Tavg": ("abs", 2.0), "Tmin": ("abs", 2.0), "Tmax": ("abs", 2.0)}
CT = {"chargetime": ("rel", 0.05)}


def flags(spec):
    f = np.zeros(len(b), bool)
    for c, (k, t) in spec.items():
        med = b.groupby("cell")[c].transform(lambda s: s.rolling(11, center=True, min_periods=3).median())
        dev = (b[c] - med).abs()
        f |= (dev > (t * med.abs() if k == "rel" else t)).to_numpy()
    return f & (b.cycle > 1).to_numpy()


b["g_main"] = flags(MAIN)
b["g_ct"] = flags(CT) | b.g_main
b["g_irt"] = flags({k: v for k, v in MAIN.items() if not k.startswith("Q")})  # sensitivity: no capacity channels (rest-period rebounds)
b[["unit", "cycle", "g_main", "g_ct", "g_irt"]].to_csv(sys.argv[2], index=False)
print("glitch share main/ct:", b.g_main.mean(), b.g_ct.mean())
