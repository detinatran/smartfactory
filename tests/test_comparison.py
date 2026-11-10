import json
from pathlib import Path

import pandas as pd
import pytest

from selective_rul.comparison import METRICS, collect_runs, summarize

ROOT = Path(__file__).resolve().parents[1]


def saved_run(root, name, seed=42, model="tcn", window=30, smoke=False, risk=0.2):
    run = root / name
    (run / "development").mkdir(parents=True)
    cfg = json.loads((ROOT / "configs/smoke.json").read_text())
    cfg.update(seed=seed, model=model, window=window, smoke=smoke)
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "complete.json").write_text(json.dumps({"completed_at": "2026-10-17T10:01:00+00:00"}))
    (run / "environment.json").write_text(json.dumps({"created_at": "2026-10-17T10:00:00+00:00"}))
    (run / "data_manifest.json").write_text(json.dumps({"files": {"train": "checksum"}}))
    (run / "splits.json").write_text(json.dumps({"fit": [seed], "development": [seed + 1]}))
    row = dict.fromkeys(METRICS, 0.5)
    row.update(
        optimistic_error_rate=risk,
        scenario="clean",
        method="asymmetric_risk",
        protocol="matched_coverage_diagnostic",
        target_coverage=0.9,
    )
    pd.DataFrame([row]).to_csv(run / "development/metrics.csv", index=False)
    # This conflicting official-test score must never enter model selection.
    (run / "official_test").mkdir()
    row["optimistic_error_rate"] = 0
    pd.DataFrame([row]).to_csv(run / "official_test/metrics.csv", index=False)
    return run


def test_summary_preserves_seeds_and_never_uses_official_test(tmp_path):
    saved_run(tmp_path, "a", seed=42, risk=0.2)
    saved_run(tmp_path, "b", seed=43, risk=0.4)
    results = summarize(collect_runs(tmp_path))
    assert len(results) == 1
    row = results.iloc[0]
    assert row.optimistic_error_rate_mean == pytest.approx(0.3)
    assert row.n_seeds == 2 and row.seeds == "42,43"
    assert row.optimistic_error_rate_std > 0


def test_different_windows_and_smoke_are_never_silently_pooled(tmp_path):
    saved_run(tmp_path, "a", window=30)
    saved_run(tmp_path, "b", window=60)
    saved_run(tmp_path, "smoke", model="gru", smoke=True)
    results = collect_runs(tmp_path)
    assert len(results) == 2
    assert results.protocol_id.nunique() == 2
    assert len(collect_runs(tmp_path, include_smoke=True)) == 3


def test_duplicate_seeds_and_mismatched_engine_splits_are_rejected(tmp_path):
    saved_run(tmp_path, "a")
    second = saved_run(tmp_path, "b")
    with pytest.raises(ValueError, match="Duplicate"):
        collect_runs(tmp_path)
    cfg = json.loads((second / "config.json").read_text())
    cfg["model"] = "gru"
    (second / "config.json").write_text(json.dumps(cfg))
    (second / "splits.json").write_text(json.dumps({"fit": [99]}))
    with pytest.raises(ValueError, match="splits"):
        collect_runs(tmp_path)
