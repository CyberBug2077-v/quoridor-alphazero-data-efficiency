from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.Adaptive.experiment_runtime import RuntimeRequest, run_experiment
from experiments.scripts import run_adaptive
from experiments.scripts.verify_adaptive import (
    AdaptiveVerificationError,
    verify_adaptive,
    verify_run,
)
from experiments.tests.test_experiment_runtime import (
    SimulatedInterrupt,
    make_protocol,
    runtime_factory,
    write_yaml,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def add_pilot_gate(config_path: Path) -> None:
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol["adaptive_scheduler"]["baseline_games"] = 75
    protocol["pilot_gate"] = {
        "require_completed_run": True,
        "require_resume_equivalence": True,
        "require_finite_metrics": True,
        "require_scheduler_direction_correct": True,
        "require_fresh_state_inflow_change": True,
        "maximum_clipping_fraction": 1.0,
        "maximum_target_error": 1.0,
        "maximum_scheduler_overhead_fraction": 1.0,
    }
    write_yaml(config_path, protocol)


def completed_run(root: Path, *, resumed: bool) -> tuple[Path, Path]:
    root.mkdir()
    config_path, run_dir = make_protocol(root)
    add_pilot_gate(config_path)
    if resumed:
        with pytest.raises(SimulatedInterrupt):
            run_experiment(
                RuntimeRequest("fresh", config_path, run_dir),
                runtime_builder=runtime_factory([], interrupt_after=1),
            )
        run_experiment(
            RuntimeRequest("resume", config_path, run_dir),
            runtime_builder=runtime_factory([], interrupt_after=None),
        )
    else:
        run_experiment(
            RuntimeRequest("fresh", config_path, run_dir),
            runtime_builder=runtime_factory([], interrupt_after=None),
        )
    return config_path, run_dir


def test_run_adaptive_cli_dry_run_and_stable_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = SOURCE_ROOT / "experiments" / "configs" / "adaptive_pilot_v2.yaml"
    assert run_adaptive.main(["dry-run", "--config", str(pilot)]) == 0

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    config_path, run_dir = make_protocol(invalid_root)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol["initialization"]["checkpoint_sha256"] = "0" * 64
    write_yaml(config_path, protocol)
    assert (
        run_adaptive.main(
            ["fresh", "--config", str(config_path), "--run-dir", str(run_dir)]
        )
        == run_adaptive.EXIT_PROTOCOL_FAILURE
    )

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    assert (
        run_adaptive.main(["resume", "--run-dir", str(incomplete)])
        == run_adaptive.EXIT_RESUME_INCOMPLETE
    )

    monkeypatch.setattr(
        run_adaptive,
        "run_experiment",
        lambda request: (_ for _ in ()).throw(RuntimeError("technical")),
    )
    assert (
        run_adaptive.main(["resume", "--run-dir", str(incomplete)])
        == run_adaptive.EXIT_RUNTIME_FAILURE
    )


def test_verify_run_checks_complete_recovery_boundary(tmp_path: Path) -> None:
    _, run_dir = completed_run(tmp_path / "verified", resumed=True)

    result = verify_run(run_dir)

    assert result.report["status"] == "verified"
    assert result.report["checks"]["resume_state_complete"] is True
    assert result.report["checks"]["scheduler_plan_continuity"] is True
    assert result.report["completed_iterations"] == 2


def test_pilot_gate_requires_and_accepts_resume_equivalence(tmp_path: Path) -> None:
    _, reference_dir = completed_run(tmp_path / "reference", resumed=False)
    _, resumed_dir = completed_run(tmp_path / "resumed", resumed=True)

    without_reference = verify_adaptive(resumed_dir)
    assert without_reference["status"] == "failed"
    assert "resume_equivalence" in without_reference["failures"]

    summary = verify_adaptive(
        resumed_dir,
        resume_reference_run_dir=reference_dir,
    )
    assert summary["status"] == "passed"
    assert summary["resume_equivalence"]["passed"] is True
    persisted = json.loads(
        (resumed_dir / "pilot_gate_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "passed"


def test_verifier_rejects_scheduler_plan_discontinuity(tmp_path: Path) -> None:
    _, run_dir = completed_run(tmp_path / "tampered", resumed=False)
    metrics_path = run_dir / "metrics.jsonl"
    metrics = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    metrics[1]["games_planned"] += 1
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metrics),
        encoding="utf-8",
    )

    with pytest.raises(AdaptiveVerificationError, match="previous scheduler decision"):
        verify_run(run_dir)


def test_evaluation_restores_training_state_and_is_auditable(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    root.mkdir()
    config_path, run_dir = make_protocol(root)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol["budget"]["max_iterations"] = 1
    protocol["evaluation"]["evaluate_every_iterations"] = 1
    write_yaml(config_path, protocol)
    created = []

    def mutating_evaluation(resolved, runtime, checkpoint_path, output_path):
        runtime.network.value = 999
        runtime.network.optimizer.step = 999
        np.random.seed(999)
        output_path.write_text(
            json.dumps({"schema_version": 1, "checkpoint_path": str(checkpoint_path)}),
            encoding="utf-8",
        )
        return 0.01

    run_experiment(
        RuntimeRequest("fresh", config_path, run_dir),
        runtime_builder=runtime_factory(created, interrupt_after=None),
        evaluation_runner=mutating_evaluation,
    )

    assert created[-1].network.value == 1
    assert created[-1].network.optimizer.step == 1
    verified = verify_run(run_dir)
    assert verified.report["evaluated_iterations"] == [1]
    evaluation = json.loads(
        (run_dir / "evaluations" / "evaluation_checkpoint_1.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation["training_state_preserved"] is True
