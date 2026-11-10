"""Download NASA's public C-MAPSS benchmark, with an explicit pinned mirror fallback."""

import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NASA = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
LANDING = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
COMMIT = "dcb137fe50277307a954f4e530e3f68bc7016070"
MIRROR = (
    "https://raw.githubusercontent.com/jiaxiang-cheng/"
    f"PyTorch-Transformer-for-RUL-Prediction/{COMMIT}/CMAPSSData"
)
FILES = [f"{kind}_FD00{i}.txt" for i in range(1, 5) for kind in ("train", "test", "RUL")]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "selective-rul-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-mirror", action="store_true")
    args = parser.parse_args()
    raw = ROOT / "data/raw/CMAPSSData"
    downloads = ROOT / "data/downloads"
    raw.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(
            (raw / n).exists() and sha256(raw / n) == manifest["files"][n]["sha256"] for n in FILES
        ):
            print("All 12 files already present; SHA-256 checks passed.")
            return
        raise SystemExit(
            "Existing data differs from manifest; inspect it before downloading again."
        )

    errors = []
    archive = downloads / "CMAPSSData.zip"
    try:
        print(f"Downloading {NASA}", flush=True)
        download(NASA, archive)
        with zipfile.ZipFile(archive) as zf:
            names = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
            missing = set(FILES) - names.keys()
            if missing:
                raise ValueError(f"NASA archive missing files: {missing}")
            # Extract only whitelisted basenames, never arbitrary archive paths.
            for name in FILES + ["readme.txt"]:
                if name in names:
                    (raw / name).write_bytes(zf.read(names[name]))
        source = {"kind": "NASA archive", "url": NASA, "archive_sha256": sha256(archive)}
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, ValueError) as exc:
        errors.append(str(exc))
        if not args.allow_mirror:
            raise SystemExit(f"NASA download failed: {exc}. Retry with --allow-mirror.") from exc
        print(f"NASA unavailable ({exc}); using pinned public GitHub mirror.", flush=True)
        for name in FILES + ["readme.txt"]:
            print(f"Downloading {name}", flush=True)
            download(f"{MIRROR}/{name}", raw / name)
        source = {"kind": "pinned GitHub mirror", "url": MIRROR, "commit": COMMIT}

    manifest = {
        "dataset": "NASA C-MAPSS FD001-FD004 (2008), not N-CMAPSS",
        "landing_page": LANDING,
        "citation": "A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "download_errors": errors,
        "files": {n: {"bytes": (raw / n).stat().st_size, "sha256": sha256(raw / n)} for n in FILES},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved 12 benchmark files and provenance to {manifest_path}")


if __name__ == "__main__":
    main()
