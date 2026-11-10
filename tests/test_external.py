import numpy as np
import pandas as pd
import pytest

from selective_rul.external import (
    ExternalNormalizer,
    corrupt_external,
    make_external_windows,
    split_units,
)


def synthetic(n_units=30, length=60, conditions=3):
    rows = []
    for u in range(1, n_units + 1):
        for step in range(1, length + 1):
            rul_steps = length - step
            rows.append(dict(unit=u, unit_name=f"b{u}", step=step, rul_steps=rul_steps,
                             regime=f"c{u % conditions}", fold=f"c{u % conditions}",
                             f0=step * 0.1, f1=np.sin(step)))
    df = pd.DataFrame(rows)
    df["life_steps"] = length - 1
    df["rul"] = df.rul_steps / (length - 1)
    df["position"] = 1 - df.rul
    return df


def test_leave_condition_out_split_is_disjoint_and_excludes_held_out_condition():
    df = synthetic()
    splits = split_units(df, "bearings", "c1", seed=42)
    held = set(df[df.fold == "c1"].unit)
    assert set(splits["evaluation"]) == held
    seen = set()
    for name in ("fit", "validation", "selector", "calibration"):
        assert not set(splits[name]) & held
        assert not set(splits[name]) & seen
        seen |= set(splits[name])


def test_windows_target_is_life_fraction_and_dropout_creates_missingness():
    df = synthetic()
    norm = ExternalNormalizer(["f0", "f1"], "baseline", baseline_k=5).fit(df)
    clean = make_external_windows(df, norm, window=10, stride=5)
    assert clean.y.min() == pytest.approx(0.0) and clean.y.max() <= 1.0
    assert clean.x.shape[-1] == 4 and clean.x[:, :, 2:].sum() == 0
    dropped = corrupt_external(df, ["f0", "f1"], norm.raw_scale, "dropout_strong", 7)
    assert dropped[["f0", "f1"]].isna().any().any()
    noisy = make_external_windows(dropped, norm, window=10, stride=5)
    assert noisy.x[:, :, 2:].sum() > 0 and np.isfinite(noisy.x).all()


@pytest.mark.parametrize("kind", ["stuck", "bias", "noise", "intermittent", "spike", "gain"])
def test_unseen_faults_change_only_the_fault_interval(kind):
    df = synthetic()
    norm = ExternalNormalizer(["f0", "f1"], "baseline", baseline_k=5).fit(df)
    faulty = corrupt_external(df, ["f0", "f1"], norm.raw_scale, kind, 3)
    changed = ~np.isclose(faulty[["f0", "f1"]].to_numpy(), df[["f0", "f1"]].to_numpy(),
                          equal_nan=False)
    assert changed.any()
    assert not changed[df.position.to_numpy() < 0.2].any()
    assert not changed[df.position.to_numpy() >= 0.7].any()
