"""Freeze selected existing runs, evaluate official test once, and resume safely."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import selective_rul
from selective_rul.data import EXPECTED
from selective_rul.experiment import evaluate_official

ARTIFACTS = (
    "config.json",
    "complete.json",
    "splits.json",
    "environment.json",
    "data_manifest.json",
    "normalizer.joblib",
    "selectors.joblib",
    "thresholds.json",
)
METRICS = (
    "coverage",
    "window_coverage",
    "optimistic_error_rate",
    "critical_coverage",
    "critical_optimistic_error_rate",
    "rmse_capped",
    "rmse_raw",
    "mae_raw",
    "mean_overestimation",
    "nasa_score_raw_sum",
    "nasa_score_raw_engine_mean",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    files = sorted(Path(selective_rul.__file__).parent.glob("*.py"))
    return (
        hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
        {p.name: sha(p) for p in files},
    )


def inventory(batch, spec, raw):
    """Read training artifacts only; no official labels or predictions are opened here."""
    if sha(batch / "plan.json") != spec["training_plan_sha256"]:
        raise ValueError("Training plan differs from the selected development benchmark")
    source_digest, source_files = source_hashes()
    if source_digest != spec["training_source_sha256"]:
        raise ValueError("Source differs from the code used for the frozen benchmark")
    data_files = read_json(raw.parent.parent / "manifest.json")["files"]
    expected = {
        (variant["model"], variant["head"], subset, seed)
        for variant in (spec["primary"], spec["control"])
        for subset in spec["subsets"]
        for seed in spec["seeds"]
    }
    records, seen = [], set()
    for job in read_json(batch / "plan.json")["jobs"]:
        cfg = job["config"]
        identity = tuple(cfg[k] for k in ("model", "head", "subset", "seed"))
        if identity not in expected:
            continue
        if identity in seen or Path(job["name"]).name != job["name"]:
            raise ValueError("Duplicate or invalid selected job")
        seen.add(identity)
        completed = sorted((batch / job["name"]).glob("attempt_*/complete.json"))
        if len(completed) != 1:
            raise ValueError(f"Need exactly one completed training attempt: {job['name']}")
        run = completed[0].parent
        if read_json(run / "config.json") != cfg or cfg.get("smoke", False):
            raise ValueError(f"Unexpected/smoke config: {job['name']}")
        if (run / "official_test").exists() or (run / "official_test_used.json").exists():
            raise ValueError(f"Official test was already opened in source run: {job['name']}")
        if (
            cfg["coverages"] != spec["report_coverages"]
            or cfg["eval_scenarios"] != spec["scenarios"]
        ):
            raise ValueError("Coverage/scenario list differs from frozen selection")
        for key, spec_key in (
            ("risk_delta", "risk_delta_cycles"),
            ("critical_rul", "critical_rul_cycles"),
            ("rul_cap", "rul_cap_cycles"),
        ):
            if cfg[key] != spec[spec_key]:
                raise ValueError(f"Changed evaluation definition: {key}")
        env = read_json(run / "environment.json")
        if env.get("official_test_used") or env["source_sha256"] != source_files:
            raise ValueError(f"Source provenance/test status differs: {job['name']}")
        if read_json(run / "data_manifest.json")["files"] != data_files:
            raise ValueError("Data manifest differs from training")
        filenames = list(ARTIFACTS) + [f"model_{i}.pt" for i in range(cfg["ensemble_size"])]
        records.append(
            {
                "name": job["name"],
                "source_run": str(run.relative_to(batch)),
                "config": cfg,
                "files": {name: sha(run / name) for name in filenames},
                "training_packages": env["packages"],
            }
        )
    if seen != expected or len(records) != spec["expected_jobs"]:
        raise ValueError("Selected training runs are incomplete")
    return sorted(records, key=lambda r: r["name"])


def verify_files(root, hashes):
    for name, expected in hashes.items():
        if not (root / name).is_file() or sha(root / name) != expected:
            raise ValueError(f"Missing or changed frozen artifact: {root / name}")


def prepare(batch, out, spec, raw):
    records = inventory(batch, spec, raw)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frozen_manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["selection"] != spec or manifest["runs"] != records:
            raise ValueError("Selection or source artifacts changed after freezing")
    else:
        if any(out.iterdir()):
            raise ValueError("Use an empty output folder for a new frozen evaluation")
        manifest = {"frozen_at_utc": utc_now(), "selection": spec, "runs": records}
        write_json(path, manifest)
    for record in records:
        destination = out / "frozen_runs" / record["name"]
        destination.mkdir(parents=True, exist_ok=True)
        for name, expected in record["files"].items():
            target = destination / name
            if not target.exists():
                temporary = target.with_name(target.name + ".copying")
                shutil.copy2(batch / record["source_run"] / name, temporary)
                if sha(temporary) != expected:
                    raise ValueError(f"Incomplete frozen copy: {temporary}")
                temporary.replace(target)
            if sha(target) != expected:
                raise ValueError(f"Frozen copy changed: {target}")
    print(f"Frozen {len(records)} existing runs; no training or calibration performed.", flush=True)
    return manifest


def check_joblib_environment(records):
    for record in records:
        for package in ("numpy", "scipy", "scikit-learn"):
            expected = record["training_packages"][package]
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise ValueError(
                    f"For saved joblib objects, install {package}=={expected}; current={actual}. "
                    "Use the notebook's environment cell before evaluation."
                )


def evaluate_all(out, manifest, raw, device, evaluator=evaluate_official):
    check_joblib_environment(manifest["runs"])
    freeze_hash = sha(out / "frozen_manifest.json")
    for index, record in enumerate(manifest["runs"], 1):
        run = out / "frozen_runs" / record["name"]
        verify_files(run, record["files"])
        marker = run / "official_eval_complete.json"
        if marker.exists():
            done = read_json(marker)
            if done["frozen_manifest_sha256"] != freeze_hash:
                raise ValueError("Evaluation marker refers to another frozen protocol")
            verify_files(run, done["output_files"])
            print(
                f"Already evaluated {index}/{len(manifest['runs'])}: {record['name']}", flush=True
            )
            continue
        # Preserve incomplete output from an interrupted execution before retrying.
        if (run / "official_test").exists():
            attempt = 1
            while (run / f"official_test_interrupted_{attempt:03d}").exists():
                attempt += 1
            (run / "official_test").rename(run / f"official_test_interrupted_{attempt:03d}")
            if (run / "official_test_used.json").exists():
                (run / "official_test_used.json").rename(
                    run / f"test_used_interrupted_{attempt:03d}.json"
                )
        started = time.monotonic()
        print(f"Official test {index}/{len(manifest['runs'])}: {record['name']}", flush=True)
        evaluator(run, raw, device)
        verify_files(run, record["files"])
        if (
            not (run / "official_test/metrics.csv").is_file()
            or not (run / "official_test_used.json").is_file()
        ):
            raise ValueError("Evaluation did not produce its required outputs")
        output_files = sorted(p for p in (run / "official_test").rglob("*") if p.is_file())
        output_files.append(run / "official_test_used.json")
        write_json(
            marker,
            {
                "completed_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "device_requested": device,
                "frozen_manifest_sha256": freeze_hash,
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "numpy", "pandas", "scipy", "scikit-learn")
                },
                "output_files": {str(p.relative_to(run)): sha(p) for p in output_files},
            },
        )


def summarize(out, manifest):
    frames = []
    freeze_hash = sha(out / "frozen_manifest.json")
    for record in manifest["runs"]:
        run = out / "frozen_runs" / record["name"]
        verify_files(run, record["files"])
        done = read_json(run / "official_eval_complete.json")
        if done["frozen_manifest_sha256"] != freeze_hash:
            raise ValueError("Summary input belongs to another frozen protocol")
        verify_files(run, done["output_files"])
        cfg = record["config"]
        frame = pd.read_csv(run / "official_test/metrics.csv")
        expected_engines = EXPECTED[cfg["subset"]][1]
        if not (
            (frame.n_windows == expected_engines) & (frame.n_engines == expected_engines)
        ).all():
            raise ValueError("Official test must have exactly one window per engine")
        methods = ["random", "ensemble_std", "absolute_error", "asymmetric_risk"]
        if cfg["head"] == "quantile":
            methods += ["quantile_width", "lower_tail_gap"]
        expected_rows = {
            (scenario, method, protocol, coverage)
            for scenario in cfg["eval_scenarios"]
            for method in methods
            for protocol in ("frozen_calibration", "matched_coverage_diagnostic")
            for coverage in cfg["coverages"]
        } | {(scenario, "always", "always", 1.0) for scenario in cfg["eval_scenarios"]}
        actual_rows = set(
            frame[["scenario", "method", "protocol", "target_coverage"]].itertuples(
                index=False, name=None
            )
        )
        if actual_rows != expected_rows or len(frame) != len(expected_rows):
            raise ValueError("Official-test metrics contain missing/duplicate/extra rows")
        for key in ("model", "head", "subset", "seed"):
            frame[key] = cfg[key]
        frame["role"] = (
            "primary" if cfg["head"] == manifest["selection"]["primary"]["head"] else "control"
        )
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    keys = ["role", "model", "head", "subset", "scenario", "method", "protocol", "target_coverage"]
    if all_rows.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate seed in official-test summary")
    grouped = all_rows.groupby(keys, dropna=False)
    summary = grouped[list(METRICS)].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.seed.nunique()
    summary = summary.reset_index()
    if not (summary.n_seeds == len(manifest["selection"]["seeds"])).all():
        raise ValueError("Incomplete official-test seed group")
    target = out / "summary"
    target.mkdir(exist_ok=True)
    all_rows.to_csv(target / "per_seed.csv", index=False)
    summary.to_csv(target / "summary.csv", index=False)
    summary[summary.protocol == "always"].to_csv(target / "always_all_predictions.csv", index=False)
    summary[summary.protocol == "frozen_calibration"].to_csv(
        target / "frozen_thresholds.csv", index=False
    )
    summary[
        (summary.protocol == "frozen_calibration")
        & (summary.target_coverage == manifest["selection"]["primary_target_coverage"])
    ].to_csv(target / "primary_frozen90.csv", index=False)
    summary[summary.protocol == "matched_coverage_diagnostic"].to_csv(
        target / "diagnostic_matched_coverage.csv", index=False
    )
    write_json(
        target / "evaluation_audit.json",
        {
            "complete_jobs": len(frames),
            "expected_jobs": manifest["selection"]["expected_jobs"],
            "frozen_manifest_sha256": freeze_hash,
            "summary_rows": len(summary),
            "primary_protocol": "frozen_calibration",
            "primary_target_coverage": 0.9,
            "notes": [
                "Primary/control were selected on development before official-test evaluation.",
                "Frozen calibration thresholds are unchanged; report actual coverage, not only its target.",
                "Matched coverage is a secondary diagnostic, not a deployment threshold.",
                "Mean/std are over training seeds on the same test engines, not independent fleets.",
                "Metric counts disclose undefined risk when no predictions/critical samples are accepted.",
            ],
        },
    )
    print(f"Official-test summary complete: {len(frames)} jobs. {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--official-test", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.batch, args.out, read_json(args.selection), args.data_dir)
    if args.prepare_only:
        return
    if args.official_test:
        evaluate_all(args.out, manifest, args.data_dir, args.device)
    summarize(args.out, manifest)


if __name__ == "__main__":
    main()
