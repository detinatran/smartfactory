"""Run on the training host: map integer unit codes (as stored in splits.json) to original unit names for the
external datasets (-> supplementary/splits/unit_map.csv). Codes are 1-based ranks of the sorted unit names."""
import argparse
import pandas as pd
from selective_rul.external import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/external/processed")
ap.add_argument("--out", default="unit_map.csv")
a = ap.parse_args()
rows = []
for name in ("ncmapss", "bearings", "battery"):
    f, _, _ = load_dataset(name, a.data_dir)
    g = f.groupby("unit").agg(unit_name=("unit_name", "first"), fold=("fold", "first"), regime=("regime", "first"),
                              n_records=("step", "size"), first_step=("step", "min"), last_step=("step", "max"),
                              life_steps=("life_steps", "first")).reset_index()
    g.insert(0, "dataset", name)
    rows.append(g)
pd.concat(rows).to_csv(a.out, index=False)
