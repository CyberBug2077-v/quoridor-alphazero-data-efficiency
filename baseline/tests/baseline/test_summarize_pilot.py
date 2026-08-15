from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

import summarize_pilot
from runtime.artifacts import atomic_write_json
from runtime.metadata import sha256_file


def test_summarize_pilot_reports_timings_and_projection_without_changing_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "baseline_pilot_seed1001"
    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "mode": "baseline",
        "run": {"id": run_dir.name, "seed": 1001},
        "budget": {"max_gpu_hours": None, "max_iterations": 3},
        "logging": {"metrics_file": "metrics.jsonl"},
        "evaluation": {
            "enabled": True,
            "evaluate_every_iterations": 2,
        },
    }
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    summary_path = run_dir / "summary.json"
    atomic_write_json(
        summary_path,
        {"status": "completed", "completed_iterations": [1, 2, 3]},
    )
    metrics = [
        {
            "iteration": 1,
            "iteration_seconds": 120.0,
            "self_play_seconds": 80.0,
            "training_seconds": 30.0,
            "games_completed": 75,
            "positions_generated": 400,
            "optimizer_steps": 10,
            "replay_buffer_size": 100,
            "peak_gpu_memory_mb": 1000.0,
            "cumulative_gpu_hours": 120.0 / 3600.0,
        },
        {
            "iteration": 2,
            "iteration_seconds": 100.0,
            "self_play_seconds": 60.0,
            "training_seconds": 30.0,
            "games_completed": 75,
            "positions_generated": 300,
            "optimizer_steps": 11,
            "replay_buffer_size": 220,
            "peak_gpu_memory_mb": 900.0,
            "cumulative_gpu_hours": 220.0 / 3600.0,
        },
        {
            "iteration": 3,
            "iteration_seconds": 140.0,
            "self_play_seconds": 90.0,
            "training_seconds": 40.0,
            "games_completed": 75,
            "positions_generated": 450,
            "optimizer_steps": 12,
            "replay_buffer_size": 330,
            "peak_gpu_memory_mb": 1100.0,
            "cumulative_gpu_hours": 360.0 / 3600.0,
        },
    ]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metrics),
        encoding="utf-8",
        newline="\n",
    )
    evaluation_path = evaluation_dir / "evaluation_checkpoint_2.json"
    atomic_write_json(
        evaluation_path,
        {
            "schema_version": 1,
            "checkpoint_path": "checkpoints/checkpoint_2.pth.tar",
            "opponents": {},
            "timing": {
                "evaluation_seconds": 20.0,
                "games_evaluated": 4,
                "seconds_per_game": 5.0,
            },
        },
    )
    source_hashes = {
        path: sha256_file(path)
        for path in (config_path, metrics_path, summary_path, evaluation_path)
    }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pilot.py",
            "--run-dir",
            str(run_dir),
            "--gpu-hours",
            "1",
        ],
    )
    assert summarize_pilot.main() == 0

    report_path = run_dir / "pilot_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "summarized"
    assert report["completed_iterations"] == [1, 2, 3]
    assert [row["timing"]["total_seconds"] for row in report["iterations"]] == [
        120.0,
        100.0,
        140.0,
    ]
    assert [row["timing"]["evaluation_seconds"] for row in report["iterations"]] == [
        0.0,
        20.0,
        0.0,
    ]
    assert [row["replay"]["growth"] for row in report["iterations"]] == [
        100,
        120,
        110,
    ]
    assert report["iterations"][0]["self_play"]["seconds_per_game"] == pytest.approx(
        80.0 / 75.0
    )
    assert report["iterations"][0]["self_play"]["seconds_per_position"] == 0.2
    steady = report["steady_state_excluding_iteration_1"]["median"]
    assert steady["total_seconds"] == 120.0
    assert steady["self_play_seconds"] == 75.0
    comparison = report["first_iteration_vs_later"]
    assert comparison["absolute_difference_first_minus_later_median"][
        "self_play_seconds"
    ] == 5.0
    projection = report["projection"]
    assert projection["median_base_iteration_seconds"] == 120.0
    assert projection["evaluation_amortized_seconds_per_iteration"] == 10.0
    assert projection["estimated_seconds_per_iteration"] == 130.0
    assert projection["estimated_iterations_per_gpu_hour"] == pytest.approx(
        3600.0 / 130.0
    )
    assert projection["requested_budgets"] == [
        {
            "gpu_hours": 1.0,
            "estimated_iterations": 27,
            "unrounded_iterations": pytest.approx(3600.0 / 130.0),
        }
    ]
    assert projection["modifies_formal_configuration"] is False
    assert report["warnings"] == []
    assert {path: sha256_file(path) for path in source_hashes} == source_hashes


def test_summarize_pilot_rejects_nonpositive_projection_budget(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": "baseline",
                "run": {"id": "pilot"},
                "logging": {"metrics_file": "metrics.jsonl"},
                "evaluation": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "iteration_seconds": 1.0,
                "self_play_seconds": 1.0,
                "training_seconds": 0.0,
                "games_completed": 1,
                "positions_generated": 1,
                "optimizer_steps": 1,
                "replay_buffer_size": 1,
                "peak_gpu_memory_mb": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    atomic_write_json(
        run_dir / "summary.json",
        {"status": "completed", "completed_iterations": [1]},
    )

    with pytest.raises(
        summarize_pilot.PilotSummaryError,
        match="budgets must be > 0",
    ):
        summarize_pilot.summarize_pilot(run_dir, gpu_hour_budgets=[0.0])
