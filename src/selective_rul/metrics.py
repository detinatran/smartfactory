"""Engine-balanced selective metrics and engine-cluster bootstrap intervals."""

import numpy as np

from .data import engine_weights


def selective_metrics(
    predictions, raw_targets, capped_targets, units, accepted, delta, critical_rul
):
    w = engine_weights(units)
    accepted = np.asarray(accepted, dtype=bool)
    error = predictions - raw_targets
    bad = error > delta
    near = raw_targets <= critical_rul
    selected_w = w * accepted
    selected_mass = selected_w.sum()

    def selected_mean(values):
        return float(np.sum(selected_w * values) / selected_mass) if selected_mass else None

    capped_mse = selected_mean((predictions - capped_targets) ** 2)
    raw_mse = selected_mean(error**2)
    critical_mass = (w * near).sum()
    critical_selected_mass = (selected_w * near).sum()
    # Raw-label NASA score, SUM over accepted windows; meaningful comparisons need equal N.
    penalty = np.expm1(np.where(error < 0, -error / 13.0, error / 10.0))
    return {
        "n_windows": int(len(predictions)),
        "n_engines": int(len(np.unique(units))),
        "accepted_windows": int(accepted.sum()),
        "coverage": float(selected_mass / w.sum()),
        "window_coverage": float(accepted.mean()),
        "optimistic_error_rate": selected_mean(bad),
        "mean_overestimation": selected_mean(np.maximum(error, 0)),
        "rmse_capped": float(np.sqrt(capped_mse)) if capped_mse is not None else None,
        "rmse_raw": float(np.sqrt(raw_mse)) if raw_mse is not None else None,
        "mae_raw": selected_mean(np.abs(error)),
        "nasa_score_raw_sum": float(penalty[accepted].sum()) if accepted.any() else None,
        "nasa_score_raw_engine_mean": selected_mean(penalty),
        "critical_coverage": float(critical_selected_mass / critical_mass)
        if critical_mass
        else None,
        "critical_optimistic_error_rate": (
            float((selected_w * near * bad).sum() / critical_selected_mass)
            if critical_selected_mass
            else None
        ),
    }


def bootstrap_risk(predictions, raw_targets, units, accepted, delta, seed, repeats=200):
    """Resample complete engines, retaining within-engine dependence and frozen selectors."""
    if repeats <= 0:
        return {"risk_ci_low": None, "risk_ci_high": None}
    numerator, denominator = [], []
    for unit in np.unique(units):
        index = units == unit
        keep = accepted[index]
        bad = predictions[index] - raw_targets[index] > delta
        numerator.append(np.mean(keep & bad))
        denominator.append(np.mean(keep))
    numerator, denominator = np.array(numerator), np.array(denominator)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerator), (repeats, len(numerator)))
    den = denominator[indices].sum(1)
    num = numerator[indices].sum(1)
    values = num[den > 0] / den[den > 0]
    if not len(values):
        return {"risk_ci_low": None, "risk_ci_high": None}
    low, high = np.quantile(values, [0.025, 0.975])
    return {"risk_ci_low": float(low), "risk_ci_high": float(high)}
