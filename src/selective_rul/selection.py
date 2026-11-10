"""Held-out error predictors and label-free acceptance thresholds (no risk guarantee)."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import SENSORS, engine_weights


def selector_features(windows, ensemble_predictions, cap, quantiles=None):
    sensors = windows.x[:, :, : len(SENSORS)]
    missing = windows.x[:, :, -len(SENSORS) :]
    features = np.column_stack(
        [
            ensemble_predictions.mean(1) / cap,
            ensemble_predictions.std(1) / cap,
            windows.meta.cycle.to_numpy() / cap,
            sensors.mean(1),
            sensors.std(1),
            sensors[:, -1] - sensors[:, 0],
            missing.mean(1),
        ]
    ).astype(np.float32)
    if quantiles is not None:
        extra = (
            np.column_stack(
                [
                    quantiles,
                    quantiles[:, 1] - quantiles[:, 0],
                    quantiles[:, 2] - quantiles[:, 1],
                    ensemble_predictions.mean(1) - quantiles[:, 0],
                ]
            )
            / cap
        )
        features = np.column_stack([features, extra]).astype(np.float32)
    return features


class Selectors:
    def __init__(
        self, delta, seed, iterations=100, extended=False, critical_rul=30.0, critical_weight=3.0
    ):
        self.delta = delta
        self.seed = seed
        self.extended = extended
        self.critical_rul = critical_rul
        self.critical_weight = critical_weight
        params = dict(
            max_iter=iterations,
            max_leaf_nodes=7,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=seed,
        )
        self.asymmetric = HistGradientBoostingClassifier(**params)
        self.absolute = HistGradientBoostingRegressor(**params)
        self.constant_probability = None
        if extended:
            # Additional baselines/candidates; v1 estimators above are untouched.
            self.symmetric = HistGradientBoostingClassifier(**params)
            self.signed_q90 = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.9, **params
            )
            self.critical_weighted = HistGradientBoostingClassifier(**params)
            self.extended_constants = {}

    def fit(self, features, predictions, raw_targets, units):
        error = predictions - raw_targets
        bad = (error > self.delta).astype(int)
        weights = engine_weights(units)
        self.prevalence = float(np.average(bad, weights=weights))
        if len(np.unique(bad)) == 1:
            self.constant_probability = float(bad[0])
        else:
            self.asymmetric.fit(features, bad, sample_weight=weights)
        self.absolute.fit(features, np.abs(error), sample_weight=weights)
        if self.extended:
            symmetric = (np.abs(error) > self.delta).astype(int)
            near = raw_targets <= self.critical_rul
            boosted = weights * np.where(near, self.critical_weight, 1.0)
            for name, labels, w in (
                ("symmetric", symmetric, weights),
                ("critical_weighted", bad, boosted),
            ):
                if len(np.unique(labels)) == 1:
                    self.extended_constants[name] = float(labels[0])
                else:
                    getattr(self, name).fit(features, labels, sample_weight=w)
            self.signed_q90.fit(features, error, sample_weight=weights)
        return self

    def _probability(self, name, features):
        if name in self.extended_constants:
            return np.full(len(features), self.extended_constants[name])
        return getattr(self, name).predict_proba(features)[:, 1]

    def scores(self, features, ensemble_predictions, tie_breaker, quantiles=None):
        if self.constant_probability is None:
            asymmetric = self.asymmetric.predict_proba(features)[:, 1]
        else:
            asymmetric = np.full(len(features), self.constant_probability)
        scores = {
            "random": tie_breaker,
            "ensemble_std": ensemble_predictions.std(1),
            "absolute_error": self.absolute.predict(features),
            "asymmetric_risk": asymmetric,
        }
        if getattr(self, "extended", False):
            scores["symmetric_risk"] = self._probability("symmetric", features)
            scores["signed_error_q90"] = self.signed_q90.predict(features)
            scores["critical_weighted_risk"] = self._probability("critical_weighted", features)
        if quantiles is not None:
            scores["quantile_width"] = quantiles[:, 2] - quantiles[:, 0]
            scores["lower_tail_gap"] = ensemble_predictions.mean(1) - quantiles[:, 0]
        return scores


def tie_breakers(meta, seed):
    """Stable per (seed, unit, cycle), identical across corruption scenarios."""
    return np.array(
        [
            np.random.default_rng(np.random.SeedSequence([seed, int(u), int(c)])).random()
            for u, c in zip(meta.unit, meta.cycle, strict=True)
        ]
    )


def fit_threshold(scores, ties, units, coverage):
    """Weighted score quantile; does not inspect outcomes or certify a risk budget."""
    if not 0 < coverage <= 1:
        raise ValueError("Coverage must be in (0,1]")
    if coverage == 1:
        return {"score": None, "tie": None}
    order = np.lexsort((ties, scores))
    weights = engine_weights(units)[order]
    index = min(np.searchsorted(np.cumsum(weights) / weights.sum(), coverage), len(order) - 1)
    row = order[index]
    return {"score": float(scores[row]), "tie": float(ties[row])}


def apply_threshold(scores, ties, threshold):
    if threshold["score"] is None:
        return np.ones(len(scores), dtype=bool)
    return (scores < threshold["score"]) | (
        (scores == threshold["score"]) & (ties <= threshold["tie"])
    )
