"""Selective RUL on external run-to-failure data (N-CMAPSS, FEMTO/PRONOSTIA, XJTU-SY).

Protocol (fixed before any external result was seen, 2026-11-09):
- Target: remaining-life FRACTION r = rul_steps / (last step index) in [0, 1] (1 at the first
  record, 0 at the last). It is bounded by construction, so there is no clipping/cap.
- Risk event: prediction - r > delta, delta = 0.16 (= 20/125 on C-MAPSS); critical r <= 0.24
  (= 30/125). Metrics are unit-balanced, exactly as for C-MAPSS.
- ncmapss: units of the official dev split are divided into fit/validation/selector/calibration;
  the 39 official test units are evaluated once (one fold, "test").
- bearings: FEMTO + XJTU-SY pooled (32 bearings, 6 operating conditions); leave-one-condition-out.
  The held-out condition is never used for fitting, selector training or thresholds.
- Corruptions mirror C-MAPSS, with onset/duration relative to the unit's life.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import engine_weights
from .metrics import selective_metrics
from .model import predict_details, train_member
from .selection import Selectors, apply_threshold, fit_threshold, tie_breakers

SCALE_FEATURES = ("rms", "peak", "p2p")  # bearing amplitude features -> log scale


# ------------------------------------------------------------------ loading
def load_dataset(name, data_dir):
    data_dir = Path(data_dir)
    if name == "ncmapss":
        df = pd.read_csv(data_dir / "ncmapss.csv")
        features = ["alt", "Mach", "TRA", "T2", "T24", "T30", "T48", "T50", "P15", "P2", "P21",
                    "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf", "flight_seconds"]
        frame = pd.DataFrame({
            "unit_name": df.unit_id, "step": df.step.astype(int),
            "rul_steps": df.rul_steps.astype(float), "regime": df.Fc.astype(int).astype(str),
            "fold": np.where(df.split == "test", "test", "dev"),
        })
        frame[features] = df[features].to_numpy()
        mode = "regime"
    elif name == "bearings":
        parts = []
        for source in ("femto", "xjtu_sy"):
            df = pd.read_csv(data_dir / f"{source}.csv")
            df["unit_name"] = source + "_" + df.unit.astype(str)
            df["regime"] = source + "_c" + df.condition.astype(str)
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        features = [c for c in df.columns if c[:2] in ("h_", "v_")]
        for col in features:
            if col.split("_", 1)[1] in SCALE_FEATURES:
                df[col] = np.log(df[col].clip(lower=1e-9))
        frame = df[["unit_name", "step", "rul_steps", "regime"]].copy()
        frame["fold"] = frame.regime
        frame[features] = df[features].to_numpy()
        mode = "baseline"
    elif name == "battery":
        df = pd.read_csv(data_dir / "battery.csv")
        df = df[df.cycle <= df.cycle_life].copy()  # label = capacity-defined end of life
        features = ["QDischarge", "QCharge", "IR", "Tavg", "Tmin", "Tmax", "chargetime"]
        frame = pd.DataFrame({
            "unit_name": df.cell, "step": df.cycle.astype(int),
            "rul_steps": (df.cycle_life - df.cycle).astype(float), "regime": df.policy,
            "fold": np.where(df.batch == 3, "batch3", "dev"),
        })
        frame[features] = df[features].to_numpy()
        mode = "baseline"
    else:
        raise ValueError(f"Unknown external dataset: {name}")
    frame = frame.sort_values(["unit_name", "step"]).reset_index(drop=True)
    last = frame.groupby("unit_name").rul_steps.transform("max")
    frame["life_steps"] = last
    frame["rul"] = frame.rul_steps / last
    frame["position"] = 1.0 - frame.rul  # elapsed fraction of life (corruption timing only)
    codes = {u: i + 1 for i, u in enumerate(sorted(frame.unit_name.unique()))}
    frame["unit"] = frame.unit_name.map(codes).astype(int)
    return frame, features, mode


def folds_for(name, frame):
    if name == "ncmapss":
        return ["test"]
    if name == "battery":
        return ["batch3"]
    return sorted(frame.fold.unique())


def split_units(frame, name, fold, seed):
    fractions = (0.15, 0.20, 0.15)  # validation, selector, calibration
    if name == "battery":  # larger calibration set for the finite-sample risk study
        pool = np.array(sorted(frame[frame.fold == "dev"].unit.unique()))
        evaluation = sorted(frame[frame.fold == "batch3"].unit.unique())
        fractions = (0.10, 0.20, 0.30)
    elif name == "ncmapss":
        pool = np.array(sorted(frame[frame.fold == "dev"].unit.unique()))
        evaluation = sorted(frame[frame.fold == "test"].unit.unique())
    else:
        pool = np.array(sorted(frame[frame.fold != fold].unit.unique()))
        evaluation = sorted(frame[frame.fold == fold].unit.unique())
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    pool = rng.permutation(pool)
    n = len(pool)
    sizes = [max(2, round(f * n)) for f in fractions]
    v, s, c = sizes
    out = {
        "validation": pool[:v], "selector": pool[v:v + s], "calibration": pool[v + s:v + s + c],
        "fit": pool[v + s + c:], "evaluation": np.array(evaluation),
    }
    if len(out["fit"]) < 4:
        raise ValueError("Too few fit units")
    return {k: sorted(int(u) for u in v) for k, v in out.items()}


# ------------------------------------------------------------------ normalization / corruption
class ExternalNormalizer:
    """regime: per-regime z-score fitted on fit units (unseen regime -> global statistics).
    baseline: subtract each unit's own mean over its first k records (label-free and causal for
    any window that ends at or after record k), divide by the fit-set spread of that difference."""

    def __init__(self, features, mode, baseline_k=10):
        self.features, self.mode, self.k = list(features), mode, baseline_k

    def _baseline(self, df):
        values = df[self.features].to_numpy(float)
        rank = df.groupby("unit").cumcount().to_numpy()
        early = pd.DataFrame(np.where((rank < self.k)[:, None], values, np.nan))
        early["unit"] = df.unit.to_numpy()
        base = early.groupby("unit").transform("mean").fillna(0.0)
        return values - base.to_numpy(float)

    def fit(self, df):
        values = df[self.features].to_numpy(float)
        self.raw_scale = np.nanstd(values, axis=0) + 1e-9
        if self.mode == "regime":
            self.global_mean = np.nanmean(values, axis=0)
            self.stats = {
                r: (np.nanmean(g[self.features].to_numpy(float), axis=0),
                    np.nanstd(g[self.features].to_numpy(float), axis=0) + 1e-9)
                for r, g in df.groupby("regime")
            }
        else:
            self.scale = np.nanstd(self._baseline(df), axis=0) + 1e-9
        return self

    def transform(self, df):
        if self.mode == "regime":
            out = np.empty((len(df), len(self.features)))
            regimes = df.regime.to_numpy()
            for r in np.unique(regimes):
                mean, std = self.stats.get(r, (self.global_mean, self.raw_scale))
                mask = regimes == r
                out[mask] = (df.loc[mask, self.features].to_numpy(float) - mean) / std
            return out
        return self._baseline(df) / self.scale


def corrupt_external(df, features, scale, scenario, seed):
    if scenario == "clean":
        return df
    from .faults import apply_fault, parse_fault

    fault = parse_fault(scenario)
    if fault:
        kind, magnitude, n_channels = fault
        out = df.copy()
        for unit, group in out.groupby("unit", sort=True):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(unit)]))
            channels = rng.choice(len(features), size=min(n_channels, len(features)),
                                  replace=False)
            onset = rng.uniform(0.2, 0.4)
            extra = np.random.default_rng(np.random.SeedSequence([seed, int(unit), 77]))
            pos = group.position.to_numpy()
            hit = (pos >= onset) & (pos < onset + 0.3)
            for j in channels:
                values = group[features[j]].to_numpy(float)
                before = values[pos < onset]
                level = np.nanmean(before) if np.isfinite(before).any() else np.nanmean(values)
                out.loc[group.index, features[j]] = apply_fault(
                    values, hit, kind, np.full(len(values), scale[j]),
                    np.full(len(values), level), extra, magnitude)
        return out
    if scenario not in ("dropout", "drift", "dropout_strong", "drift_strong"):
        raise ValueError(scenario)
    out = df.copy()
    n = len(features)
    strong = scenario.endswith("_strong")
    k = min(n, max(3, round(3 * n / 14)) if strong else max(1, round(n / 14)))
    for unit, group in out.groupby("unit", sort=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(unit)]))
        channels = rng.choice(n, size=k, replace=False)
        onset = rng.uniform(0.2, 0.4)
        pos = group.position.to_numpy()
        if scenario.startswith("dropout"):
            hit = (pos >= onset) & (pos < onset + (0.4 if strong else 0.2))
            out.loc[group.index[hit], [features[j] for j in channels]] = np.nan
        else:
            ramp = np.clip((pos - onset) / 0.5, 0.0, 1.0)
            for j in channels:
                sign = rng.choice([-1, 1])
                out.loc[group.index, features[j]] = (
                    group[features[j]].to_numpy() + sign * (4.0 if strong else 2.0) * ramp * scale[j]
                )
    return out


# ------------------------------------------------------------------ windows / features
@dataclass
class ExtWindows:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def make_external_windows(df, normalizer, window, stride):
    values = normalizer.transform(df)
    missing = np.isnan(values)
    values = np.concatenate([np.nan_to_num(values, nan=0.0), missing.astype(float)], axis=1)
    xs, ys, rows = [], [], []
    unit_arr, step_arr, rul_arr = df.unit.to_numpy(), df.step.to_numpy(), df.rul.to_numpy()
    for unit in np.unique(unit_arr):
        index = np.flatnonzero(unit_arr == unit)
        if len(index) < window:
            raise ValueError(f"Unit {unit} shorter than the window")
        ends = list(range(window - 1, len(index), stride))
        if ends[-1] != len(index) - 1:
            ends.append(len(index) - 1)
        for end in ends:
            xs.append(values[index[end - window + 1:end + 1]])
            last = index[end]
            ys.append(rul_arr[last])
            rows.append({"unit": int(unit), "cycle": int(step_arr[last]),
                         "rul_raw": float(rul_arr[last])})
    meta = pd.DataFrame(rows)
    return ExtWindows(np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32), meta)


def external_features(windows, ensemble, n_features):
    sensors = windows.x[:, :, :n_features]
    missing = windows.x[:, :, n_features:]
    return np.column_stack([
        ensemble.mean(1), ensemble.std(1), np.log1p(windows.meta.cycle.to_numpy()),
        sensors.mean(1), sensors.std(1), sensors[:, -1] - sensors[:, 0], missing.mean(1),
    ]).astype(np.float32)


# ------------------------------------------------------------------ one run
def run_external(cfg, data_dir, out):
    """cfg: dataset, fold, model, seed + training/selector settings. Writes into out/."""
    from .experiment import resolve_device

    device = resolve_device(cfg["device"])
    torch.set_num_threads(cfg["threads"])
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cfg["threads"]))
    name = cfg["dataset"]
    frame, features, mode = load_dataset(name, data_dir)
    window, stride = cfg["window"][name], cfg["stride"][name]
    splits = split_units(frame, name, cfg["fold"], cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    parts = {k: frame[frame.unit.isin(v)].reset_index(drop=True) for k, v in splits.items()}
    normalizer = ExternalNormalizer(features, mode, cfg["baseline_k"]).fit(parts["fit"])
    joblib.dump(normalizer, out / "normalizer.joblib")

    def windows(part, scenario):
        altered = corrupt_external(parts[part], features, normalizer.raw_scale, scenario,
                                   cfg["seed"] + 10000)
        return make_external_windows(altered, normalizer, window, stride)

    model_cfg = dict(cfg, rul_cap=1.0, head="point", point_readout="sigmoid",
                     point_target="capped", window=window)
    train, validation = windows("fit", "clean"), windows("validation", "clean")
    models = []
    for member in range(cfg["ensemble_size"]):
        model, _ = train_member(train, validation, model_cfg, device, cfg["seed"] + member * 100)
        models.append(model)
        torch.save({"input_dim": train.x.shape[-1], "state_dict": model.cpu().state_dict()},
                   out / f"model_{member}.pt")
        model.to(device)

    def score(w):
        ensemble, _ = predict_details(models, w.x, device)
        feats = external_features(w, ensemble, len(features))
        ties = tie_breakers(w.meta, cfg["seed"] + 20000)
        return ensemble, ties, feats

    sel_feats, sel_pred, sel_y, sel_units = [], [], [], []
    for scenario in cfg["selector_scenarios"]:
        w = windows("selector", scenario)
        ensemble, _, feats = score(w)
        sel_feats.append(feats)
        sel_pred.append(ensemble.mean(1))
        sel_y.append(w.y)
        sel_units.append(w.meta.unit.to_numpy())
    selectors = Selectors(cfg["risk_delta"], cfg["seed"], cfg["selector_iterations"],
                          extended=True, critical_rul=cfg["critical_rul"],
                          critical_weight=cfg["critical_weight"])
    selectors.fit(np.concatenate(sel_feats), np.concatenate(sel_pred), np.concatenate(sel_y),
                  np.concatenate(sel_units))
    joblib.dump(selectors, out / "selectors.joblib")

    calib = windows("calibration", "clean")
    ensemble, ties, feats = score(calib)
    scores = selectors.scores(feats, ensemble, ties)
    units = calib.meta.unit.to_numpy()
    thresholds = {m: {str(c): fit_threshold(s, ties, units, c) for c in cfg["coverages"]}
                  for m, s in scores.items()}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")

    rows = []
    for scenario in cfg["eval_scenarios"]:
        w = windows("evaluation", scenario)
        ensemble, ties, feats = score(w)
        scores = selectors.scores(feats, ensemble, ties)
        pred = ensemble.mean(1)
        meta = w.meta.copy()
        meta["prediction"], meta["ensemble_std"], meta["tie_breaker"] = pred, ensemble.std(1), ties
        for method, s in scores.items():
            meta[f"score_{method}"] = s
        meta.to_csv(out / f"predictions_{scenario}.csv", index=False)
        units = meta.unit.to_numpy()

        def record(method, protocol, coverage, accepted):
            metrics = selective_metrics(pred, w.y, w.y, units, accepted, cfg["risk_delta"],
                                        cfg["critical_rul"])
            rows.append(dict(scenario=scenario, method=method, protocol=protocol,
                             target_coverage=coverage, **metrics))

        record("always", "always", 1.0, np.ones(len(pred), bool))
        for method, s in scores.items():
            for c in cfg["coverages"]:
                record(method, "frozen_calibration", c,
                       apply_threshold(s, ties, thresholds[method][str(c)]))
                record(method, "matched_coverage_diagnostic", c,
                       apply_threshold(s, ties, fit_threshold(s, ties, units, c)))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "complete.json").write_text(
        json.dumps({"completed_at": datetime.now(timezone.utc).isoformat(),
                    "n_eval_units": len(splits["evaluation"]),
                    "units": {k: len(v) for k, v in splits.items()}}, indent=2) + "\n")
    return out


def pooled_acceptance_metrics(run_dirs, target, delta, critical):
    """Pool out-of-fold predictions of several folds (same dataset/model/seed); every unit is
    evaluated once. Acceptance uses each fold's own frozen threshold, or a per-fold matched one."""
    rows = []
    by_scenario = {}
    for run in run_dirs:
        thresholds = json.loads((run / "thresholds.json").read_text())
        cfg = json.loads((run / "config.json").read_text())
        for scenario in cfg["eval_scenarios"]:
            p = pd.read_csv(run / f"predictions_{scenario}.csv")
            p["fold"] = cfg["fold"]
            methods = [c[6:] for c in p.columns if c.startswith("score_")]
            ties = p.tie_breaker.to_numpy()
            for m in methods:
                s = p[f"score_{m}"].to_numpy()
                p[f"acc_frozen_{m}"] = apply_threshold(s, ties, thresholds[m][str(target)])
                p[f"acc_matched_{m}"] = apply_threshold(
                    s, ties, fit_threshold(s, ties, p.unit.to_numpy(), target))
            by_scenario.setdefault(scenario, []).append(p)
    for scenario, parts in by_scenario.items():
        p = pd.concat(parts, ignore_index=True)
        pred, y, units = p.prediction.to_numpy(), p.rul_raw.to_numpy(), p.unit.to_numpy()
        rows.append(dict(scenario=scenario, method="always", protocol="always", **selective_metrics(
            pred, y, y, units, np.ones(len(p), bool), delta, critical)))
        for col in [c for c in p.columns if c.startswith("acc_")]:
            _, protocol, method = col.split("_", 2)
            rows.append(dict(scenario=scenario, method=method, protocol=protocol,
                             **selective_metrics(pred, y, y, units, p[col].to_numpy(), delta,
                                                 critical)))
    return pd.DataFrame(rows)


def unit_weights(units):
    return engine_weights(units)
