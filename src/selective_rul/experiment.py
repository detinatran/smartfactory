"""Reproducible experiment orchestration and explicit final-test evaluation."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .data import (
    EXPECTED,
    Normalizer,
    corrupt,
    engine_weights,
    make_windows,
    read_frame,
    split_engines,
    validate_data,
)
from .metrics import bootstrap_risk, selective_metrics
from .model import HEAD_NAMES, MODEL_NAMES, build_model, predict_details, train_member
from .selection import Selectors, apply_threshold, fit_threshold, selector_features, tie_breakers

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve_device(name):
    if name == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or mps")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device: {name}")
    return name


def check_config(cfg):
    if cfg.get("model", "tcn") not in MODEL_NAMES or cfg.get("head", "point") not in HEAD_NAMES:
        raise ValueError("Unknown model or prediction head")
    if cfg["subset"] not in EXPECTED:
        raise ValueError("Unknown C-MAPSS subset")
    for key in (
        "window",
        "stride",
        "epochs",
        "ensemble_size",
        "batch_size",
        "hidden",
        "patience",
        "threads",
    ):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["ensemble_size"] < 2:
        raise ValueError("At least two ensemble members are needed for the uncertainty baseline")
    if cfg.get("layers", 2) < 1 or cfg.get("state_size", 8) < 1:
        raise ValueError("layers and state_size must be positive")
    if cfg.get("model") == "patch_transformer":
        length, stride, heads = (
            cfg.get("patch_length", 10),
            cfg.get("patch_stride", 5),
            cfg.get("attention_heads", 4),
        )
        if min(length, stride, heads) < 1 or cfg["hidden"] % heads:
            raise ValueError(
                "Positive patch settings and hidden divisible by attention_heads are required"
            )
        if cfg["window"] < length + stride or (cfg["window"] - length) % stride:
            raise ValueError("Need >=2 patches, tiling exactly through the current cycle")
    if cfg.get("head") == "quantile" and cfg.get("quantile_weight", 0.2) <= 0:
        raise ValueError("Quantile head needs a positive auxiliary-loss weight")
    if cfg.get("point_readout", "sigmoid") not in ("sigmoid", "softplus"):
        raise ValueError("point_readout must be sigmoid or softplus")
    if cfg.get("point_target", "capped") not in ("capped", "raw"):
        raise ValueError("point_target must be capped or raw")
    if cfg.get("point_target", "capped") == "raw" and cfg.get("point_readout", "sigmoid") != "softplus":
        raise ValueError("Raw (uncapped) targets need the unbounded softplus readout")
    if cfg.get("critical_weight", 3.0) <= 0:
        raise ValueError("critical_weight must be positive")
    if cfg["risk_delta"] <= 0 or cfg["rul_cap"] <= 0:
        raise ValueError("risk_delta and rul_cap must be positive")
    if any(not 0 < c <= 1 for c in cfg["coverages"]):
        raise ValueError("Invalid coverage")


def windows_for(frame, normalizer, cfg, scenario, endpoint_rul=None):
    altered = corrupt(frame, normalizer, scenario, cfg["seed"] + 10000)
    return make_windows(
        altered, normalizer, cfg["window"], cfg["stride"], cfg["rul_cap"], endpoint_rul
    )


def score_windows(windows, models, selectors, cfg, device):
    ensemble, quantiles = predict_details(models, windows.x, device)
    features = selector_features(windows, ensemble, cfg["rul_cap"], quantiles)
    ties = tie_breakers(windows.meta, cfg["seed"] + 20000)
    scores = selectors.scores(features, ensemble, ties, quantiles)
    return ensemble, ties, scores, quantiles


def evaluate_frame(
    frame, endpoint_rul, normalizer, models, selectors, thresholds, cfg, device, out, label
):
    out.mkdir(parents=True, exist_ok=False)
    rows, quantile_rows = [], []
    for scenario in cfg["eval_scenarios"]:
        windows = windows_for(frame, normalizer, cfg, scenario, endpoint_rul)
        ensemble, ties, scores, quantiles = score_windows(windows, models, selectors, cfg, device)
        mean = ensemble.mean(1)
        meta = windows.meta.copy()
        meta["rul_capped"] = windows.y
        meta["prediction"] = mean
        meta["ensemble_std"] = ensemble.std(1)
        meta["tie_breaker"] = ties
        if quantiles is not None:
            weights = engine_weights(meta.unit.to_numpy())
            raw = meta.rul_raw.to_numpy()
            record = {
                "scenario": scenario,
                "raw_q10_q90_coverage": float(
                    np.average((raw >= quantiles[:, 0]) & (raw <= quantiles[:, 2]), weights=weights)
                ),
                "mean_raw_interval_width": float(
                    np.average(quantiles[:, 2] - quantiles[:, 0], weights=weights)
                ),
            }
            for index, level in enumerate((0.1, 0.5, 0.9)):
                meta[f"q{int(level * 100)}_member_mean"] = quantiles[:, index]
                record[f"raw_fraction_below_q{int(level * 100)}"] = float(
                    np.average(raw <= quantiles[:, index], weights=weights)
                )
            quantile_rows.append(record)
        for method, score in scores.items():
            meta[f"score_{method}"] = score
        meta.to_csv(out / f"predictions_{scenario}.csv", index=False)
        units = meta.unit.to_numpy()
        raw_targets = meta.rul_raw.to_numpy()

        def record(method, protocol, coverage, accepted):
            result = dict(
                scenario=scenario, method=method, protocol=protocol, target_coverage=coverage
            )
            result.update(
                selective_metrics(
                    mean,
                    raw_targets,
                    windows.y,
                    units,
                    accepted,
                    cfg["risk_delta"],
                    cfg["critical_rul"],
                )
            )
            result.update(
                bootstrap_risk(
                    mean,
                    raw_targets,
                    units,
                    accepted,
                    cfg["risk_delta"],
                    cfg["seed"],
                    cfg["bootstrap_repeats"],
                )
            )
            rows.append(result)

        record("always", "always", 1.0, np.ones(len(mean), dtype=bool))
        for method, score in scores.items():
            for coverage in cfg["coverages"]:
                frozen = thresholds[method][str(coverage)]
                record(method, "frozen_calibration", coverage, apply_threshold(score, ties, frozen))
                # Diagnostic ranking only: target-domain scores but NEVER target labels
                # determine this threshold. It is not a deployable fixed threshold.
                matched = fit_threshold(score, ties, units, coverage)
                record(
                    method,
                    "matched_coverage_diagnostic",
                    coverage,
                    apply_threshold(score, ties, matched),
                )
        print(
            f"evaluated {label}/{scenario}: {len(mean)} windows, {len(np.unique(units))} engines",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    if quantile_rows:
        pd.DataFrame(quantile_rows).to_csv(out / "quantile_metrics.csv", index=False)
    plot_metrics(metrics, cfg, out, label)
    write_report(metrics, cfg, out, label)
    return metrics


def plot_metrics(metrics, cfg, out, label):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = cfg["eval_scenarios"]
    fig, axes = plt.subplots(2, len(scenarios), figsize=(3.6 * len(scenarios), 6.4), squeeze=False)
    matched = metrics[metrics.protocol == "matched_coverage_diagnostic"]
    for col, scenario in enumerate(scenarios):
        for method, group in matched[matched.scenario == scenario].groupby("method"):
            group = group.sort_values("coverage")
            axes[0, col].plot(group.coverage, group.optimistic_error_rate, marker=".", label=method)
            axes[1, col].plot(group.coverage, group.critical_coverage, marker=".", label=method)
        axes[0, col].set_title(scenario)
        for row in range(2):
            axes[row, col].set_xlabel("Engine-balanced acceptance rate")
            axes[row, col].grid(alpha=0.25)
            axes[row, col].set_xlim(0.45, 1.02)
            axes[row, col].set_ylim(-0.02, 1.02)
        max_risk = matched[matched.scenario == scenario].optimistic_error_rate.max()
        if pd.notna(max_risk):
            axes[0, col].set_ylim(0, min(1.02, max(0.05, max_risk * 1.15)))
    axes[0, 0].set_ylabel(f"P(overestimate > {cfg['risk_delta']:g} cycles | accepted)")
    axes[1, 0].set_ylabel(f"Acceptance when true RUL <= {cfg['critical_rul']:g}")
    axes[0, -1].legend(fontsize=7)
    suffix = "SMOKE TEST, not a research result" if cfg.get("smoke") else "single-seed experiment"
    fig.suptitle(
        f"{cfg.get('model', 'tcn')} / {cfg.get('head', 'point')} / {cfg['subset']} / {label} / {suffix}"
    )
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=160)
    fig.savefig(out / "risk_coverage.pdf")
    plt.close(fig)


def write_report(metrics, cfg, out, label):
    selected = metrics[
        (metrics.scenario == "clean")
        & (metrics.protocol == "matched_coverage_diagnostic")
        & np.isclose(metrics.target_coverage, 0.9)
    ]
    lines = [
        f"# {cfg.get('model', 'tcn')} / {cfg.get('head', 'point')} / {cfg['subset']} — {label}",
        "",
        "SMOKE TEST ONLY: verifies execution; does not establish research performance."
        if cfg.get("smoke")
        else "Single-seed experiment; multi-seed validation is still required.",
        "",
        f"Risk event: prediction − raw RUL > {cfg['risk_delta']:g} cycles.",
        "Engine-balanced metrics; bootstrap resamples whole engines, conditional on the fitted model.",
        "Capped regression targets and raw safety targets are stored separately.",
        "Quantile heads, when enabled, learn raw RUL with an auxiliary pinball loss. Point predictions still target capped RUL.",
        "",
        "## Clean data, approximately 90% matched acceptance",
        "",
        "| Selector | Actual acceptance | Optimistic error rate | Acceptance near failure |",
        "|---|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row.method} | {row.coverage:.3f} | {row.optimistic_error_rate:.3f} | {row.critical_coverage:.3f} |"
        )
    lines += [
        "",
        "See metrics.csv for both frozen calibration thresholds and matched-coverage diagnostics.",
        "Matched-coverage thresholds use evaluation scores, not labels; they are ranking diagnostics only.",
        "Frozen thresholds are selected on clean calibration engines and reused unchanged under corruption.",
        "Known dropout/drift enter selector training; strong variants are stress tests only.",
        "No conformal coverage guarantee, calibrated probability claim, or certified risk bound is made.",
        "Official-test results, if requested separately, use one final window per engine.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def train_experiment(cfg, raw, out):
    cfg = dict(cfg)
    cfg.setdefault("model", "tcn")
    cfg.setdefault("head", "point")
    cfg.setdefault("layers", 2)
    cfg.setdefault("quantile_weight", 0.2)
    check_config(cfg)
    validate_data(raw)
    device = resolve_device(cfg["device"])
    torch.set_num_threads(cfg["threads"])
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cfg["threads"]))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    frame = read_frame(raw / f"train_{cfg['subset']}.txt")
    splits = split_engines(frame.unit.to_numpy(), cfg["seed"])
    write_json(out / "splits.json", splits)
    parts = {name: frame[frame.unit.isin(units)].copy() for name, units in splits.items()}
    normalizer = Normalizer(6 if cfg["subset"] in ("FD002", "FD004") else 1, cfg["seed"])
    with threadpool_limits(limits=cfg["threads"]):
        normalizer.fit(parts["fit"])
    joblib.dump(normalizer, out / "normalizer.joblib")
    train = windows_for(parts["fit"], normalizer, cfg, "clean")
    validation = windows_for(parts["validation"], normalizer, cfg, "clean")
    environment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib")
        },
        "raw_data": str(raw.resolve()),
        "split_engines": {name: len(ids) for name, ids in splits.items()},
        "training_windows": len(train.y),
        "validation_windows": len(validation.y),
        "official_test_used": False,
        "model": cfg["model"],
        "head": cfg["head"],
    }
    source_dir = Path(__file__).parent
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in sorted(source_dir.glob("*.py")):
        shutil.copy2(path, snapshot / path.name)
    environment["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.glob("*.py")
    }
    for filename in ("uv.lock", "pyproject.toml"):
        shutil.copy2(ROOT / filename, out / filename)
    shutil.copy2(raw.parent.parent / "manifest.json", out / "data_manifest.json")
    write_json(out / "environment.json", environment)
    print(
        f"{cfg['model']}/{cfg['head']} / {cfg['subset']} / {device}: engines {environment['split_engines']}",
        flush=True,
    )
    models, histories = [], []
    for member in range(cfg["ensemble_size"]):
        model, history = train_member(train, validation, cfg, device, cfg["seed"] + member * 100)
        if member == 0:
            environment["parameters_per_member"] = sum(p.numel() for p in model.parameters())
            write_json(out / "environment.json", environment)
        models.append(model)
        histories.extend([dict(member=member, **row) for row in history])
        torch.save(
            {"input_dim": train.x.shape[-1], "state_dict": model.cpu().state_dict()},
            out / f"model_{member}.pt",
        )
        model.to(device)
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False)

    features, means, targets, units = [], [], [], []
    for scenario in cfg["selector_scenarios"]:
        windows = windows_for(parts["selector"], normalizer, cfg, scenario)
        ensemble, quantiles = predict_details(models, windows.x, device)
        features.append(selector_features(windows, ensemble, cfg["rul_cap"], quantiles))
        means.append(ensemble.mean(1))
        targets.append(windows.meta.rul_raw.to_numpy())
        units.append(windows.meta.unit.to_numpy())
    selectors = Selectors(
        cfg["risk_delta"],
        cfg["seed"],
        cfg["selector_iterations"],
        extended=cfg.get("extended_selectors", False),
        critical_rul=cfg["critical_rul"],
        critical_weight=cfg.get("critical_weight", 3.0),
    )
    with threadpool_limits(limits=cfg["threads"]):
        selectors.fit(
            np.concatenate(features),
            np.concatenate(means),
            np.concatenate(targets),
            np.concatenate(units),
        )
    joblib.dump(selectors, out / "selectors.joblib")
    write_json(
        out / "selector_fit.json",
        {
            "source": "held-out selector engines, never used to fit or early-stop the backbone",
            "asymmetric_event_prevalence": selectors.prevalence,
            "constant_asymmetric_score": selectors.constant_probability,
            "scenarios": cfg["selector_scenarios"],
        },
    )
    calibration = windows_for(parts["calibration"], normalizer, cfg, "clean")
    with threadpool_limits(limits=cfg["threads"]):
        _, ties, scores, _ = score_windows(calibration, models, selectors, cfg, device)
    thresholds = {
        method: {
            str(c): fit_threshold(score, ties, calibration.meta.unit.to_numpy(), c)
            for c in cfg["coverages"]
        }
        for method, score in scores.items()
    }
    write_json(out / "thresholds.json", thresholds)
    if cfg.get("extended_selectors", False):
        # Keep clean calibration scores so label-using threshold rules (e.g. SGR/LTT)
        # can be studied offline without refitting anything.
        (out / "calibration").mkdir()
        calib = calibration.meta.copy()
        calib["rul_capped"] = calibration.y
        with threadpool_limits(limits=cfg["threads"]):
            ensemble, _ = predict_details(models, calibration.x, device)
        calib["prediction"] = ensemble.mean(1)
        calib["tie_breaker"] = ties
        for method, score in scores.items():
            calib[f"score_{method}"] = score
        calib.to_csv(out / "calibration/predictions_clean.csv", index=False)
    with threadpool_limits(limits=cfg["threads"]):
        evaluate_frame(
            parts["development"],
            None,
            normalizer,
            models,
            selectors,
            thresholds,
            cfg,
            device,
            out / "development",
            "development",
        )
    write_json(out / "complete.json", {"completed_at": datetime.now(timezone.utc).isoformat()})
    print(f"Run complete: {out.resolve()}", flush=True)
    return out


def evaluate_official(run, raw, device_override=None):
    """Explicit opt-in: do not use this during model or selector development."""
    if not (run / "complete.json").exists():
        raise ValueError("Training run is incomplete")
    validate_data(raw)
    if (
        json.loads((run / "data_manifest.json").read_text())["files"]
        != json.loads((raw.parent.parent / "manifest.json").read_text())["files"]
    ):
        raise ValueError("Evaluation data differs from the training run's manifest")
    cfg = json.loads((run / "config.json").read_text())
    device = resolve_device(device_override or cfg["device"])
    torch.set_num_threads(cfg["threads"])
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cfg["threads"]))
    normalizer = joblib.load(run / "normalizer.joblib")
    selectors = joblib.load(run / "selectors.joblib")
    models = []
    for member in range(cfg["ensemble_size"]):
        checkpoint = torch.load(run / f"model_{member}.pt", map_location="cpu", weights_only=True)
        model = build_model(checkpoint["input_dim"], cfg)
        model.load_state_dict(checkpoint["state_dict"])
        models.append(model.to(device))
    thresholds = json.loads((run / "thresholds.json").read_text())
    frame = read_frame(raw / f"test_{cfg['subset']}.txt")
    labels = np.loadtxt(raw / f"RUL_{cfg['subset']}.txt", ndmin=1)
    endpoint_rul = dict(enumerate(labels, start=1))
    with threadpool_limits(limits=cfg["threads"]):
        evaluate_frame(
            frame,
            endpoint_rul,
            normalizer,
            models,
            selectors,
            thresholds,
            cfg,
            device,
            run / "official_test",
            "official_test",
        )
    write_json(
        run / "official_test_used.json", {"evaluated_at": datetime.now(timezone.utc).isoformat()}
    )
