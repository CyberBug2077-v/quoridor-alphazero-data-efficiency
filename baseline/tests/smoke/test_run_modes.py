from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import pytest

import run_smoke


def test_dry_run_creates_only_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "dry-run"

    def model_creation_is_forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run created the model runtime")

    monkeypatch.setattr(run_smoke, "create_runtime", model_creation_is_forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_smoke.py",
            "dry-run",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert run_smoke.main() == 0
    assert (output_dir / "resolved_config.yaml").is_file()
    assert not (output_dir / "metrics.jsonl").exists()
    assert not (output_dir / "checkpoints").exists()


def test_fresh_refuses_existing_formal_artifact_before_model_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

    def model_creation_is_forbidden(*_args, **_kwargs):
        raise AssertionError("fresh created a model before conflict validation")

    monkeypatch.setattr(run_smoke, "create_runtime", model_creation_is_forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_smoke.py", "fresh", "--output-dir", str(output_dir)],
    )

    assert run_smoke.main() == 2
    assert (output_dir / "metrics.jsonl").read_text(encoding="utf-8") == "{}\n"


def build_completed_boundary(
    resolved_config: dict,
    output_dir: Path,
    iterations: int = 2,
) -> tuple[dict, dict]:
    original = run_smoke.rebase_output_directory(resolved_config, output_dir)
    original["mode"] = "fresh"
    run_smoke.save_resolved_config(original)
    checkpoint_dir = Path(original["checkpoint"]["directory"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.jsonl"
    with run_smoke.MetricsJsonlWriter(metrics_path) as writer:
        for iteration in range(1, iterations + 1):
            checkpoint = checkpoint_dir / f"checkpoint_{iteration}.pth.tar"
            checkpoint.write_bytes(f"checkpoint-{iteration}".encode("ascii"))
            writer(
                {
                    "schema_version": 1,
                    "iteration": iteration,
                    "optimizer_steps": 1,
                    "checkpoint_path": checkpoint.as_posix(),
                }
            )
    with (checkpoint_dir / "latest.examples").open("wb") as examples_file:
        pickle.dump(
            {"iteration": iterations, "examples": [["sample"]] * iterations},
            examples_file,
        )
    requested = copy.deepcopy(resolved_config)
    return original, requested


def test_resume_boundary_requires_matching_metrics_checkpoint_and_examples(
    resolved_config: dict,
    tmp_path: Path,
) -> None:
    original, requested = build_completed_boundary(
        resolved_config,
        tmp_path / "resume-run",
    )

    iteration, checkpoint = run_smoke.validate_resume_state(original, requested)
    assert iteration == 2
    assert checkpoint.name == "checkpoint_2.pth.tar"

    with (Path(original["checkpoint"]["directory"]) / "latest.examples").open(
        "wb"
    ) as examples_file:
        pickle.dump({"iteration": 1, "examples": []}, examples_file)
    with pytest.raises(run_smoke.ConfigError, match="latest.examples iteration"):
        run_smoke.validate_resume_state(original, requested)


def test_resume_rejects_a_preexisting_next_checkpoint(
    resolved_config: dict,
    tmp_path: Path,
) -> None:
    original, requested = build_completed_boundary(
        resolved_config,
        tmp_path / "resume-run",
    )
    checkpoint_dir = Path(original["checkpoint"]["directory"])
    (checkpoint_dir / "checkpoint_3.pth.tar").write_bytes(b"partial")

    with pytest.raises(run_smoke.ConfigError, match="do not match metrics"):
        run_smoke.validate_resume_state(original, requested)


def test_resume_rejects_changed_critical_training_config(
    resolved_config: dict,
    tmp_path: Path,
) -> None:
    original, requested = build_completed_boundary(
        resolved_config,
        tmp_path / "resume-run",
    )
    requested["training"]["learning_rate"] *= 2

    with pytest.raises(run_smoke.ConfigError, match="critical training parameters"):
        run_smoke.validate_resume_state(original, requested)


def test_summary_documents_iteration_boundary_resume(
    resolved_config: dict,
    tmp_path: Path,
) -> None:
    resolved = run_smoke.rebase_output_directory(resolved_config, tmp_path / "run")
    checkpoint_dir = Path(resolved["checkpoint"]["directory"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint_1.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    with run_smoke.MetricsJsonlWriter(resolved["_output_path"] / "metrics.jsonl") as writer:
        writer(
            {
                "schema_version": 1,
                "iteration": 1,
                "optimizer_steps": 1,
                "checkpoint_path": checkpoint.as_posix(),
            }
        )

    summary_path = run_smoke.write_summary(
        resolved,
        mode="resume",
        status="stopped",
        evaluation_path=None,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed_iterations"] == [1]
    assert summary["status"] == "stopped"
    assert "optimizer, scheduler, and RNG state are not restored" in summary[
        "resume_semantics"
    ]
