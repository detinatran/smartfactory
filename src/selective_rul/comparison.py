"""Summarize all development seeds without looking at official-test metrics."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROTOCOL_FIELDS = (
    "window",
    "stride",
    "rul_cap",
    "risk_delta",
    "critical_rul",
    "coverages",
    "epochs",
    "patience",
    "ensemble_size",
    "selector_scenarios",
    "eval_scenarios",
    "selector_iterations",
)
METRICS = (
    "coverage",
    "optimistic_error_rate",
    "critical_coverage",
    "critical_optimistic_error_rate",
    "rmse_capped",
    "rmse_raw",
    "mean_overestimation",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def collect_runs(root, include_smoke=False):
    frames, seen = [], set()
    for completed in sorted(Path(root).rglob("complete.json")):
        run = completed.parent
        if not (run / "development/metrics.csv").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())
        if cfg.get("smoke", False) and not include_smoke:
            continue
        model, head = cfg.get("model", "tcn"), cfg.get("head", "point")
        cfg = dict(
            cfg,
            model=model,
            head=head,
            layers=cfg.get("layers", 2),
            quantile_weight=cfg.get("quantile_weight", 0.2),
        )
        manifest = json.loads((run / "data_manifest.json").read_text())
        protocol = {key: cfg[key] for key in PROTOCOL_FIELDS}
        protocol.update(data_files=manifest["files"], smoke=cfg.get("smoke", False))
        protocol_id = digest(protocol)
        variant_id = digest(
            {
                key: value
                for key, value in cfg.items()
                if key not in ("seed", "subset", "device", "threads")
            }
        )
        identity = (variant_id, cfg["subset"], cfg["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate variant/subset/seed at {run}; select one batch")
        seen.add(identity)
        frame = pd.read_csv(run / "development/metrics.csv")
        environment = json.loads((run / "environment.json").read_text())
        elapsed = (
            datetime.fromisoformat(json.loads(completed.read_text())["completed_at"])
            - datetime.fromisoformat(environment["created_at"])
        ).total_seconds()
        metadata = dict(
            model=model,
            head=head,
            subset=cfg["subset"],
            seed=cfg["seed"],
            window=cfg["window"],
            hidden=cfg["hidden"],
            run=str(run.resolve()),
            variant_id=variant_id,
            protocol_id=protocol_id,
            smoke=cfg.get("smoke", False),
            elapsed_seconds=elapsed,
            parameters=environment.get("parameters_per_member"),
        )
        for key, value in metadata.items():
            frame[key] = value
        frame["splits_hash"] = digest(json.loads((run / "splits.json").read_text()))
        frames.append(frame)
    if not frames:
        raise ValueError("No completed development runs (smoke excluded unless --include-smoke)")
    results = pd.concat(frames, ignore_index=True)
    for _, group in results.groupby(["protocol_id", "subset", "seed"]):
        if group.splits_hash.nunique() != 1:
            raise ValueError("Inconsistent engine splits in a proposed comparison")
    return results


def summarize(results):
    keys = [
        "protocol_id",
        "variant_id",
        "model",
        "head",
        "subset",
        "window",
        "hidden",
        "scenario",
        "method",
        "protocol",
        "target_coverage",
        "smoke",
    ]
    grouped = results.groupby(keys, dropna=False)
    summary = grouped[list(METRICS)].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.seed.nunique()
    summary["seeds"] = grouped.seed.apply(lambda s: ",".join(map(str, sorted(s.unique()))))
    summary["mean_seconds"] = grouped.elapsed_seconds.mean()
    summary["parameters"] = grouped.parameters.first()
    return summary.reset_index()


def compare_runs(root, out, include_smoke=False):
    results = collect_runs(root, include_smoke)
    summary = summarize(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    overview = summary[
        (summary.scenario == "clean")
        & (summary.protocol == "matched_coverage_diagnostic")
        & (summary.target_coverage == 0.9)
    ]
    overview.to_csv(out / "clean_90_percent.csv", index=False)
    lines = [
        "# Model comparison — development only",
        "",
        "All completed variants and seeds are retained. No automatic winner or paper inclusion decision.",
        "Compare only matching protocol_id and seed sets. Windows/epoch budgets remain separate.",
        "Mean/std are across seeds; one-seed standard deviations are undefined, not zero.",
        "Split seeds reuse the same engine dataset; they are not independent new fleets.",
        "Complete missing/failed seeds before selection; inspect critical coverage as well as risk.",
        "summary.csv includes all scenarios and frozen thresholds; per_seed.csv retains every result.",
        "",
    ]
    if results.smoke.any():
        lines += ["**SMOKE RESULTS INCLUDED: execution checks only, not model rankings.**", ""]
    plan_path = Path(root) / "plan.json"
    if plan_path.exists():
        jobs = json.loads(plan_path.read_text())["jobs"]
        missing = [
            j["name"]
            for j in jobs
            if not list((Path(root) / j["name"]).glob("attempt_*/complete.json"))
        ]
        (out / "missing_jobs.json").write_text(json.dumps(missing, indent=2) + "\n")
        lines += [f"Planned jobs: {len(jobs)}. Missing completed jobs: {len(missing)}.", ""]
    lines += [
        "| Protocol | Model/head | Subset | Selector | Seeds | Risk mean | Near-failure acceptance |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for _, row in overview.iterrows():
        lines.append(
            f"| {row.protocol_id} | {row.model}/{row['head']} ({row.variant_id}) | {row.subset} | {row.method} | {row.seeds} | {row.optimistic_error_rate_mean:.4f} | {row.critical_coverage_mean:.4f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved comparison: {out.resolve()}")
