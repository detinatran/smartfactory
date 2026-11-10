"""Exercise freezing/resume using synthetic artifacts; never open NASA test labels."""

import importlib.metadata
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_eval", ROOT / "scripts/evaluate_frozen.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def frozen_inputs(tmp_path):
    batch, out = tmp_path / "batch", tmp_path / "evaluation"
    raw = tmp_path / "data/raw/CMAPSSData"
    batch.mkdir()
    raw.mkdir(parents=True)
    runner.write_json(raw.parent.parent / "manifest.json", {"files": {}})
    source_digest, source_files = runner.source_hashes()
    jobs = []
    for head in ["point", "quantile"]:
        cfg = {
            "model": "gru",
            "head": head,
            "subset": "FD001",
            "seed": 42,
            "ensemble_size": 2,
            "smoke": False,
            "coverages": [0.9, 1.0],
            "eval_scenarios": ["clean"],
            "risk_delta": 20,
            "critical_rul": 30,
            "rul_cap": 125,
        }
        name = f"gru_{head}_w30_FD001_seed42"
        run = batch / name / "attempt_001"
        run.mkdir(parents=True)
        for name_on_disk in runner.ARTIFACTS:
            (run / name_on_disk).write_text("{}\n")
        runner.write_json(run / "config.json", cfg)
        runner.write_json(run / "data_manifest.json", {"files": {}})
        runner.write_json(
            run / "environment.json",
            {
                "official_test_used": False,
                "source_sha256": source_files,
                "packages": {
                    p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn"]
                },
            },
        )
        for i in range(2):
            (run / f"model_{i}.pt").write_bytes(b"synthetic checkpoint; never deserialize")
        jobs.append({"name": name, "config": cfg})
    runner.write_json(batch / "plan.json", {"jobs": jobs})
    spec = {
        "training_plan_sha256": runner.sha(batch / "plan.json"),
        "training_source_sha256": source_digest,
        "primary": {"model": "gru", "head": "point", "selector": "asymmetric_risk"},
        "control": {"model": "gru", "head": "quantile", "selector": "asymmetric_risk"},
        "subsets": ["FD001"],
        "seeds": [42],
        "expected_jobs": 2,
        "report_coverages": [0.9, 1.0],
        "scenarios": ["clean"],
        "risk_delta_cycles": 20,
        "critical_rul_cycles": 30,
        "rul_cap_cycles": 125,
        "primary_target_coverage": 0.9,
    }
    return batch, out, spec, raw


def fake_evaluator(run, raw, device):
    # Tiny fake final metrics test orchestration, not model performance.
    cfg = runner.read_json(run / "config.json")
    methods = ["random", "ensemble_std", "absolute_error", "asymmetric_risk"]
    if cfg["head"] == "quantile":
        methods += ["quantile_width", "lower_tail_gap"]
    rows = []
    for scenario in cfg["eval_scenarios"]:
        keys = [("always", "always", 1.0)]
        keys += [
            (m, p, c)
            for m in methods
            for p in ["frozen_calibration", "matched_coverage_diagnostic"]
            for c in cfg["coverages"]
        ]
        for method, protocol, coverage in keys:
            rows.append(
                dict(
                    scenario=scenario,
                    method=method,
                    protocol=protocol,
                    target_coverage=coverage,
                    n_windows=100,
                    n_engines=100,
                    **{key: 0.5 for key in runner.METRICS},
                )
            )
    (run / "official_test").mkdir()
    pd.DataFrame(rows).to_csv(run / "official_test/metrics.csv", index=False)
    runner.write_json(run / "official_test_used.json", {"synthetic": True})


def test_freeze_copies_inputs_without_reading_test_data_or_modifying_training(frozen_inputs):
    batch, out, spec, raw = frozen_inputs
    before = {str(p.relative_to(batch)): runner.sha(p) for p in batch.rglob("*") if p.is_file()}
    manifest = runner.prepare(batch, out, spec, raw)
    freeze_hash = runner.sha(out / "frozen_manifest.json")
    assert runner.prepare(batch, out, spec, raw) == manifest
    assert runner.sha(out / "frozen_manifest.json") == freeze_hash
    assert not list(raw.iterdir())  # No test data exists in this fixture.
    assert not list(out.rglob("official_test"))
    assert before == {
        str(p.relative_to(batch)): runner.sha(p) for p in batch.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("artifact", ["thresholds.json", "model_0.pt", "selectors.joblib"])
def test_changed_frozen_artifacts_fail_before_evaluation(frozen_inputs, artifact):
    batch, out, spec, raw = frozen_inputs
    manifest = runner.prepare(batch, out, spec, raw)
    (out / "frozen_runs" / manifest["runs"][0]["name"] / artifact).write_text("changed")
    with pytest.raises(ValueError, match="changed frozen artifact"):
        runner.evaluate_all(out, manifest, raw, "cpu", evaluator=fake_evaluator)
    assert not list(out.rglob("official_test"))


def test_changed_training_input_rejects_resume(frozen_inputs):
    batch, out, spec, raw = frozen_inputs
    manifest = runner.prepare(batch, out, spec, raw)
    (batch / manifest["runs"][0]["source_run"] / "thresholds.json").write_text("changed")
    with pytest.raises(ValueError, match="changed after freezing"):
        runner.prepare(batch, out, spec, raw)


def test_prepare_resumes_an_interrupted_checkpoint_copy(frozen_inputs, monkeypatch):
    batch, out, spec, raw = frozen_inputs
    original_copy = runner.shutil.copy2

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated disconnected Drive")

    monkeypatch.setattr(runner.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="disconnected"):
        runner.prepare(batch, out, spec, raw)
    assert list(out.rglob("*.copying"))
    monkeypatch.setattr(runner.shutil, "copy2", original_copy)
    manifest = runner.prepare(batch, out, spec, raw)
    assert not list(out.rglob("*.copying"))
    for record in manifest["runs"]:
        runner.verify_files(out / "frozen_runs" / record["name"], record["files"])


def test_resume_skips_completed_outputs_and_preserves_interrupted_files(frozen_inputs):
    batch, out, spec, raw = frozen_inputs
    manifest = runner.prepare(batch, out, spec, raw)
    run = out / "frozen_runs" / manifest["runs"][0]["name"]
    (run / "official_test").mkdir()
    (run / "official_test/partial.txt").write_text("interrupted")
    runner.evaluate_all(out, manifest, raw, "cpu", evaluator=fake_evaluator)
    assert (run / "official_test_interrupted_001/partial.txt").read_text() == "interrupted"

    def forbidden(*args):
        raise AssertionError("Completed evaluation must not run again")

    runner.evaluate_all(out, manifest, raw, "cpu", evaluator=forbidden)
    result = runner.summarize(out, manifest)
    assert runner.read_json(result / "evaluation_audit.json")["complete_jobs"] == 2
    assert not list(batch.rglob("official_test"))
    assert set(pd.read_csv(result / "primary_frozen90.csv").protocol) == {"frozen_calibration"}
    (run / "official_test/metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed frozen artifact"):
        runner.evaluate_all(out, manifest, raw, "cpu", evaluator=forbidden)


def test_summary_rejects_non_endpoint_evaluation(frozen_inputs):
    batch, out, spec, raw = frozen_inputs
    manifest = runner.prepare(batch, out, spec, raw)

    def wrong_evaluator(run, raw, device):
        fake_evaluator(run, raw, device)
        path = run / "official_test/metrics.csv"
        frame = pd.read_csv(path)
        frame["n_windows"] = 200
        frame.to_csv(path, index=False)

    runner.evaluate_all(out, manifest, raw, "cpu", evaluator=wrong_evaluator)
    with pytest.raises(ValueError, match="one window per engine"):
        runner.summarize(out, manifest)


def test_smoke_source_and_changed_plan_are_rejected(frozen_inputs):
    batch, out, spec, raw = frozen_inputs
    plan = runner.read_json(batch / "plan.json")
    plan["jobs"][0]["config"]["smoke"] = True
    runner.write_json(batch / "plan.json", plan)
    with pytest.raises(ValueError, match="Training plan differs"):
        runner.prepare(batch, out, spec, raw)
    spec["training_plan_sha256"] = runner.sha(batch / "plan.json")
    run = batch / plan["jobs"][0]["name"] / "attempt_001"
    runner.write_json(run / "config.json", plan["jobs"][0]["config"])
    with pytest.raises(ValueError, match="smoke config"):
        runner.prepare(batch, out, spec, raw)
