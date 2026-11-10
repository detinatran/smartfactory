import numpy as np
import pandas as pd
import pytest
import torch

from selective_rul.data import (
    COLUMNS,
    SENSORS,
    Normalizer,
    Windows,
    corrupt,
    engine_weights,
    make_windows,
    split_engines,
)
from selective_rul.metrics import bootstrap_risk, selective_metrics
from selective_rul.model import RULModel
from selective_rul.selection import (
    Selectors,
    apply_threshold,
    fit_threshold,
    selector_features,
    tie_breakers,
)


def trajectory(unit=1, length=120):
    cycles = np.arange(1, length + 1)
    values = np.column_stack(
        [
            np.full(length, unit),
            cycles,
            np.zeros((length, 3)),
            *[cycles * (j + 1) + unit for j in range(21)],
        ]
    )
    frame = pd.DataFrame(values, columns=COLUMNS)
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
    return frame


def test_all_engines_are_disjoint_and_split_is_reproducible():
    splits = split_engines(np.repeat(np.arange(1, 101), 5), seed=42)
    assert splits == split_engines(np.arange(1, 101), seed=42)
    all_ids = [i for group in splits.values() for i in group]
    assert len(all_ids) == len(set(all_ids)) == 100
    assert [len(ids) for ids in splits.values()] == [50, 10, 15, 10, 15]
    assert splits != split_engines(np.arange(1, 101), seed=43)


def test_scaler_transform_cannot_learn_from_held_out_values():
    train = trajectory()
    normalizer = Normalizer().fit(train)
    old_mean, old_scale = normalizer.means.copy(), normalizer.scales.copy()
    held_out = trajectory(unit=999)
    held_out.loc[:, SENSORS] += 10000
    transformed = normalizer.transform(held_out)
    np.testing.assert_array_equal(normalizer.means, old_mean)
    np.testing.assert_array_equal(normalizer.scales, old_scale)
    assert transformed[:, :14].mean() > 10


def test_windows_use_only_past_and_current_cycles():
    original = trajectory(length=60)
    normalizer = Normalizer().fit(original)
    first = make_windows(original, normalizer, window=30, stride=5)
    changed = original.copy()
    changed.loc[changed.cycle > 30, SENSORS] += 99999
    second = make_windows(changed, normalizer, window=30, stride=5)
    np.testing.assert_array_equal(first.x[0], second.x[0])
    assert first.meta.cycle.iloc[0] == 30
    assert first.y[0] == 30
    np.testing.assert_allclose(first.x[0], normalizer.transform(original.iloc[:30]))
    assert not np.array_equal(first.x[1], second.x[1])


def test_windows_never_mix_engines_and_keep_raw_targets():
    frame = pd.concat([trajectory(1, 200), trajectory(2, 200)], ignore_index=True)
    normalizer = Normalizer().fit(frame)
    windows = make_windows(frame, normalizer, window=30, stride=10, cap=125)
    first_second_engine = np.flatnonzero(windows.meta.unit.to_numpy() == 2)[0]
    expected = normalizer.transform(frame[frame.unit == 2].iloc[:30])
    np.testing.assert_array_equal(windows.x[first_second_engine], expected)
    assert windows.y[0] == 125
    assert windows.meta.rul_raw.iloc[0] == 170


def test_short_official_history_is_padded_and_labels_are_not_shifted():
    normalizer = Normalizer().fit(trajectory())
    short = trajectory(7, 5)
    windows = make_windows(short, normalizer, window=30, endpoint_rul={7: 73})
    assert windows.x.shape == (1, 30, 31)
    assert windows.y.tolist() == [73]
    assert windows.meta.unit.tolist() == [7]
    np.testing.assert_array_equal(windows.x[0, 0], windows.x[0, 25])
    np.testing.assert_array_equal(windows.x[0, -1], normalizer.transform(short)[-1])


@pytest.mark.parametrize("scenario", ["dropout", "drift", "dropout_strong", "drift_strong"])
def test_corruption_is_prefix_consistent_and_never_changes_labels(scenario):
    frame = trajectory(length=180)
    normalizer = Normalizer().fit(frame)
    full = corrupt(frame, normalizer, scenario, seed=42)
    prefix = corrupt(frame.iloc[:100], normalizer, scenario, seed=42)
    pd.testing.assert_frame_equal(full.iloc[:100], prefix)
    pd.testing.assert_frame_equal(full[["unit", "cycle"]], frame[["unit", "cycle"]])
    assert np.isfinite(normalizer.transform(full)).all()
    if scenario.startswith("dropout"):
        assert full[SENSORS].isna().any().any()
        assert normalizer.transform(full)[:, -14:].sum() > 0


def test_model_encoder_is_causal_and_predictions_are_in_target_range():
    torch.manual_seed(42)
    model = RULModel(31, hidden=8, dropout=0, cap=125).eval()
    first = torch.randn(2, 30, 31)
    second = first.clone()
    second[:, 16:] += 100
    with torch.no_grad():
        a = model.encoder(first.transpose(1, 2))[:, :, :16]
        b = model.encoder(second.transpose(1, 2))[:, :, :16]
        torch.testing.assert_close(a, b)
        pred = model(first)
    assert pred.shape == (2,)
    assert ((pred > 0) & (pred < 125)).all()


def test_selector_features_cannot_read_rul_labels():
    frame = trajectory()
    windows = make_windows(frame, Normalizer().fit(frame))
    predictions = np.ones((len(windows.y), 2)) * 40
    first = selector_features(windows, predictions, 125)
    modified = Windows(windows.x, windows.y + 999, windows.meta.assign(rul_raw=999))
    second = selector_features(modified, predictions, 125)
    np.testing.assert_array_equal(first, second)


def test_weights_give_each_engine_equal_total_mass():
    units = np.array([1, 2, 2, 2])
    weights = engine_weights(units)
    assert weights[0] == pytest.approx(weights[1:].sum())


def test_threshold_ties_are_reproducible_without_accepting_everything():
    meta = pd.DataFrame({"unit": [1] * 10, "cycle": np.arange(10)})
    ties = tie_breakers(meta, 42)
    scores = np.ones(10)
    threshold = fit_threshold(scores, ties, meta.unit.to_numpy(), 0.8)
    accepted = apply_threshold(scores, ties, threshold)
    assert accepted.sum() == 8
    np.testing.assert_array_equal(ties, tie_breakers(meta, 42))
    assert apply_threshold(scores, ties, fit_threshold(scores, ties, meta.unit.to_numpy(), 1)).all()


def test_risk_sign_raw_labels_and_empty_acceptance():
    predictions = np.array([40.0, 5.0, 125.0])
    raw = np.array([10.0, 10.0, 200.0])
    capped = np.minimum(raw, 125)
    units = np.array([1, 2, 3])
    metrics = selective_metrics(predictions, raw, capped, units, np.ones(3, bool), 20, 30)
    assert metrics["optimistic_error_rate"] == pytest.approx(1 / 3)
    assert metrics["critical_optimistic_error_rate"] == 0.5
    assert metrics["rmse_raw"] > metrics["rmse_capped"]
    empty = selective_metrics(predictions, raw, capped, units, np.zeros(3, bool), 20, 30)
    assert empty["optimistic_error_rate"] is None
    assert empty["coverage"] == 0


def test_nasa_penalty_direction_and_engine_bootstrap():
    pred = np.array([87.0, 110.0])
    target = np.array([100.0, 100.0])
    units = np.array([1, 2])
    metrics = selective_metrics(pred, target, target, units, np.ones(2, bool), 5, 30)
    assert metrics["nasa_score_raw_sum"] == pytest.approx(2 * np.expm1(1))
    boot = bootstrap_risk(pred, target, units, np.ones(2, bool), 5, 42, 100)
    assert boot["risk_ci_low"] <= 0.5 <= boot["risk_ci_high"]


@pytest.mark.parametrize("target", [0.0, 100.0])
def test_selector_handles_single_class_without_fabricating_discrimination(target):
    features = np.zeros((100, 4))
    pred = np.ones(100) * 50
    selectors = Selectors(20, 42, iterations=2).fit(
        features, pred, np.full(100, target), np.repeat(np.arange(10), 10)
    )
    scores = selectors.scores(features, np.column_stack([pred, pred]), np.linspace(0, 1, 100))
    assert np.unique(scores["asymmetric_risk"]).tolist() == [float(50 - target > 20)]


def test_extended_selectors_add_scores_without_changing_v1_scores():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 5))
    pred = 60 + 30 * features[:, 0]
    target = np.clip(pred - 25 * features[:, 1] + rng.normal(0, 5, 400), 0, None)
    units = np.repeat(np.arange(40), 10)
    ties = np.linspace(0, 1, 400)
    ensemble = np.column_stack([pred, pred + 1])
    base = Selectors(20, 42, iterations=5).fit(features, pred, target, units)
    ext = Selectors(20, 42, iterations=5, extended=True).fit(features, pred, target, units)
    s_base, s_ext = base.scores(features, ensemble, ties), ext.scores(features, ensemble, ties)
    for name, value in s_base.items():
        np.testing.assert_array_equal(value, s_ext[name])
    extra = set(s_ext) - set(s_base)
    assert extra == {"symmetric_risk", "signed_error_q90", "critical_weighted_risk"}
    assert all(np.isfinite(s_ext[name]).all() for name in extra)
