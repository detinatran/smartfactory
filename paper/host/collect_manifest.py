"""Run on the training host: collect config.json, splits.json, complete.json and checkpoint SHA-256 prefixes of every
completed run under the paper's run roots into one gzipped JSON (-> supplementary/config/run_manifest.json.gz)."""
import argparse, gzip, hashlib, json
from pathlib import Path

ROOTS = ["ext_v1_capped_gru_tcn", "ext_v1_capped_gru_tcn_s2_gru", "ext_v1_capped_gru_tcn_s2_tcn", "external_v1",
         "battery_v1", "fault_v1", "battery_fault_v1", "extra_v1", "battery_extra_v1"]
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=Path("runs"))
ap.add_argument("--out", type=Path, default=Path("supp_manifest.json.gz"))
a = ap.parse_args()
out = []
for r in ROOTS:
    for done in sorted((a.runs / r).rglob("complete.json")):
        d = done.parent
        rec = {"root": r, "run": str(d.relative_to(a.runs / r)).replace("\\", "/")}
        for name in ("config.json", "splits.json", "complete.json"):
            if (d / name).exists():
                rec[name[:-5]] = json.loads((d / name).read_text(encoding="utf-8"))
        rec["checkpoints"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(d.glob("model_*.pt"))}
        out.append(rec)
with gzip.open(a.out, "wt", encoding="utf-8") as f:
    json.dump(out, f)
print(len(out))
