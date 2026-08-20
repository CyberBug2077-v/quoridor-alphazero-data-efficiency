from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments.Adaptive import experiment_runtime
from experiments.Adaptive.experiment_runtime import (
    BaselineRuntime,
    ExperimentRuntimeError,
    RuntimeRequest,
    resolve_adaptive_protocol,
    run_experiment,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def test_real_pilot_protocol_resolves_without_writing_outputs() -> None:
    config_path = SOURCE_ROOT / "experiments" / "configs" / "adaptive_pilot_v2.yaml"
    run_dir = (
        SOURCE_ROOT
        / "experiments"
        / "outputs"
        / "adaptive_pilot_seed2001_4090_v2"
    )

    result = run_experiment(RuntimeRequest("dry-run", config_path, run_dir))
    resolved = result["resolved_config"]
    manifest = result["input_manifest"]

    assert result["files_written"] is False
    assert resolved["config_id"] == "adaptive_pilot_v2"
    assert resolved["model"]["board_size"] == 9
    assert resolved["self_play"]["games_per_iteration"] == 75
    assert resolved["adaptive_scheduler"]["target_states"] == 2516
    assert resolved["resource_accounting"]["max_gpu_hours"] == 6
    assert manifest["inputs"]["pretrained_checkpoint"][
        "sha256"
    ] == "4824a2a8ba1c1ebb5a38a992af075a45a033b87b403973b583ab98a079f35667"
    assert not (run_dir / "resolved_config.yaml").exists()


def test_real_preflight_protocol_resolves_as_frozen_five_iteration_run() -> None:
    config_path = (
        SOURCE_ROOT
        / "experiments"
        / "configs"
        / "adaptive_preflight_seed1001_v2.yaml"
    )
    run_dir = SOURCE_ROOT / "experiments" / "outputs" / "adaptive_short" / "v2"

    result = run_experiment(RuntimeRequest("dry-run", config_path, run_dir))

    assert result["resolved_config"]["config_id"] == "adaptive_preflight_seed1001_v2"
    assert result["resolved_config"]["status"] == "frozen"
    assert result["resolved_config"]["budget"]["max_iterations"] == 5
    assert result["resolved_config"]["adaptive_scheduler"]["target_states"] == 2516
    assert result["resolved_config"]["replay_instrumentation"]["schema_version"] == 2
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert protocol["adaptive_scheduler"]["observation_policy"] == {
        "accepted": "structurally_valid_nonempty_games_with_length_1_to_150",
        "truncated_at_max_length": (
            "include_for_scheduler_and_record_separately"
        ),
        "empty_game": "exclude_and_record",
        "malformed_game": "exclude_and_record",
        "abnormal_length": "exclude_and_record",
    }


def test_formal_protocol_records_production_preflight_launch_policy() -> None:
    config_path = SOURCE_ROOT / "experiments" / "configs" / "adaptive_formal_v2.yaml"
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen"
    assert protocol["freeze_gate"]["current_state"] == "production_preflight_required"
    assert protocol["freeze_gate"]["decision"] == {
        "original_pilot": {
            "planned_duration_hours": [4, 8],
            "status_before_formal_start": "not_completed",
            "required_before_formal_start": False,
        },
        "replacement_launch_gate": "five_iteration_production_preflight",
        "scheduler_parameters": "frozen_before_production_preflight",
    }
    assert protocol["freeze_gate"]["production_preflight"][
        "required_iterations"
    ] == 5
    assert protocol["freeze_gate"]["production_preflight"][
        "required_config_id"
    ] == "adaptive_preflight_seed1001_v2"
    assert protocol["freeze_gate"]["pilot_gate_summary"][
        "required_before_formal_start"
    ] is False
    assert "experiment_code_commit" not in protocol["run"]


class FakeOptimizer:
    def __init__(self) -> None:
        self.step = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step = state["step"]


class FakeNetwork:
    def __init__(self) -> None:
        self.value = 0
        self.optimizer = FakeOptimizer()

    def save_checkpoint(self, folder: str, filename: str) -> None:
        path = Path(folder) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"value": self.value}, path)

    def load_checkpoint(self, folder: str, filename: str) -> None:
        payload = torch.load(
            Path(folder) / filename,
            map_location="cpu",
            weights_only=False,
        )
        self.value = int(payload["value"])


class SimulatedInterrupt(RuntimeError):
    pass


class FakeCoach:
    def __init__(
        self,
        network: FakeNetwork,
        args: SimpleNamespace,
        *,
        interrupt_after: int | None,
    ) -> None:
        self.network = network
        self.args = args
        self.interrupt_after = interrupt_after
        self.iteration_callback = None
        self.trainExamplesHistory: list[list[tuple]] = []
        self.current_iteration = 0
        self.skipFirstSelfPlay = False
        self.save_every_n_iterations = 100

    def executeEpisode(self) -> list[tuple]:
        return [
            (
                np.zeros((4, 17, 17), dtype=np.int8),
                np.array([1.0], dtype=np.float32),
                1.0,
                np.array([1], dtype=np.int8),
                1,
                1,
            )
        ]

    def saveTrainExamples(self, iteration: int) -> None:
        raise AssertionError("Runtime must replace Coach replay persistence")

    def learn(self) -> None:
        for iteration in range(self.current_iteration + 1, self.args.numIters + 1):
            bucket = []
            for _ in range(self.args.numEps):
                bucket.extend(self.executeEpisode())
            self.trainExamplesHistory.append(bucket)
            self.network.value += 1
            self.network.optimizer.step += 1
            lengths = [1] * self.args.numEps
            callback = self.iteration_callback
            assert callback is not None
            callback(
                {
                    "schema_version": 1,
                    "iteration": iteration,
                    "games_completed": self.args.numEps,
                    "positions_generated": self.args.numEps,
                    "illegal_action_count": 0,
                    "mean_game_length": 1.0,
                    "min_game_length": 1,
                    "max_game_length": 1,
                    "replay_buffer_size": sum(
                        len(item) for item in self.trainExamplesHistory
                    ),
                    "available_examples": len(bucket),
                    "examples_used": len(bucket),
                    "samples_seen": len(bucket),
                    "training_batches": 1,
                    "optimizer_steps": 1,
                    "effective_batch_size": 1,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "micro_batches_processed": 1,
                    "policy_loss": 0.1,
                    "value_loss": 0.2,
                    "total_loss": 0.3,
                    "mean_grad_norm": 0.4,
                    "max_grad_norm": 0.4,
                    "self_play_seconds": 0.01,
                    "training_seconds": 0.02,
                    "iteration_seconds": 0.04,
                    "peak_gpu_memory_mb": None,
                    "checkpoint_path": None,
                }
            )
            if self.interrupt_after == iteration:
                raise SimulatedInterrupt(f"interrupt after iteration {iteration}")


def baseline_payload() -> dict:
    return {
        "run": {
            "id": "baseline",
            "seed": 1,
            "device": "cpu",
            "gpu_index": 0,
            "deterministic": True,
            "output_dir": "unused",
        },
        "initialization": {},
        "model": {
            "board_size": 9,
            "num_channels": 8,
            "num_res_blocks": 1,
            "attn_depth": 0,
            "num_heads": 1,
            "se_enabled": False,
            "dropout": 0.0,
        },
        "budget": {"max_gpu_hours": 1, "max_wall_clock_hours": None, "max_iterations": 2},
        "self_play": {
            "iterations": 2,
            "games_per_iteration": 1,
            "mcts_simulations": 2,
            "eval_mcts_in_batch": 1,
            "temperature_threshold": 1,
            "cpuct": 1.25,
            "dirichlet_noise": True,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.25,
            "max_game_length": 150,
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "micro_batch_size": 1,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "amp": False,
            "amp_dtype": "bf16",
            "update_gating": False,
        },
        "replay": {
            "max_queue_size": 100,
            "max_train_samples": 100,
            "history_iterations": 4,
        },
        "checkpoint": {
            "directory": "unused",
            "save_every_iterations": 2,
            "save_replay_state": True,
            "save_instrumentation_state": True,
            "save_rng_state": True,
            "compute_sha256": True,
        },
        "instrumentation": {},
        "logging": {},
        "evaluation": {
            "enabled": False,
            "evaluate_every_iterations": None,
            "games_per_opponent": 2,
            "opponents": ["random"],
            "model_mcts_simulations": 2,
            "eval_mcts_in_batch": 1,
            "temperature": 0,
            "dirichlet_noise": False,
            "max_game_length": 150,
            "alternate_starting_player": True,
        },
    }


def make_protocol(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "adaptive_run"
    baseline_path = tmp_path / "baseline.yaml"
    matched_path = tmp_path / "matched.yaml"
    checkpoint_path = tmp_path / "checkpoint_0.pth.tar"
    config_path = tmp_path / "adaptive.yaml"
    torch.save({"value": 0}, checkpoint_path)
    baseline = baseline_payload()
    baseline["initialization"] = {
        "mode": "pretrained_checkpoint",
        "checkpoint_path": checkpoint_path.as_posix(),
        "expected_sha256": sha256(checkpoint_path),
        "load_weights": True,
        "load_replay": False,
        "load_optimizer_state": False,
        "load_tracker_state": False,
        "start_iteration": 1,
    }
    write_yaml(baseline_path, baseline)
    write_yaml(
        matched_path,
        {
            "schema_version": 1,
            "config_id": "matched_compute_v1",
            "status": "frozen",
            "provenance": {
                "baseline_config": baseline_path.as_posix(),
                "baseline_config_sha256": sha256(baseline_path),
            },
            "common_initialization": {
                "mode": "pretrained_checkpoint",
                "checkpoint_path": checkpoint_path.as_posix(),
                "checkpoint_sha256": sha256(checkpoint_path),
                "load_weights": True,
                "load_replay": False,
                "load_optimizer_state": False,
                "load_tracker_state": False,
                "start_iteration": 1,
            },
            "allowed_condition_differences": [
                "self_play.games_per_iteration",
                "adaptive_scheduler",
                "resulting_iterations",
                "resulting_optimizer_steps",
                "resulting_resource_allocation",
            ],
            "compute_budget": {},
            "checkpoint_alignment": {},
        },
    )
    protocol = {
        "schema_version": 1,
        "config_id": "adaptive_pilot_test_v1",
        "status": "frozen",
        "provenance": {
            "baseline_config": baseline_path.as_posix(),
            "baseline_config_sha256": sha256(baseline_path),
            "baseline_git_commit": "test",
            "require_clean_worktree": False,
        },
        "run": {
            "id": "adaptive_test",
            "purpose": "scheduler_validation_only",
            "seed": 7,
            "device": "cpu",
            "gpu_index": 0,
            "deterministic": True,
            "output_dir": run_dir.as_posix(),
        },
        "shared_algorithm_parameters": {
            "equality_contract": {
                "path": matched_path.as_posix(),
                "sha256": sha256(matched_path),
            }
        },
        "initialization": {
            "mode": "pretrained_checkpoint",
            "checkpoint_path": checkpoint_path.as_posix(),
            "checkpoint_sha256": sha256(checkpoint_path),
            "load_weights": True,
            "load_replay": False,
            "load_optimizer_state": False,
            "load_tracker_state": False,
            "start_iteration": 1,
        },
        "adaptive_scheduler": {
            "method": "target_states",
            "target_states": 1,
            "length_estimator": {
                "method": "ema",
                "alpha": 1.0,
                "minimum_observations": 1,
            },
            "warm_start": {
                "first_iteration_games": 1,
                "initial_length_value": 1.0,
            },
            "bounds": {"min_games": 1, "max_games": 2},
            "rounding": "ceil",
        },
        "budget": {
            "max_gpu_hours": 100.0,
            "max_iterations": 2,
            "max_wall_clock_hours": 100.0,
            "allocated_gpu_count": 1,
            "evaluation_time_included": False,
            "instrumentation_overhead_included": True,
            "crossing_policy": "keep_crossing_iteration",
        },
        "checkpoint": {
            "save_every_iterations": 2,
            "save_replay_state": True,
            "save_scheduler_state": True,
            "save_instrumentation_state": True,
            "save_rng_state": True,
            "compute_sha256": True,
            "gpu_hour_analysis_milestones": [],
        },
        "instrumentation": {"tracker_file": "tracker.json"},
        "metrics": {
            "required_fields": [
                "iteration",
                "games_planned",
                "games_completed",
                "positions_generated",
                "scheduler_planned_games",
                "optimizer_steps",
                "iteration_seconds",
                "cumulative_gpu_hours",
                "checkpoint_path",
                "checkpoint_sha256",
            ]
        },
        "evaluation": {"evaluate_every_iterations": None},
    }
    write_yaml(config_path, protocol)
    return config_path, run_dir


def test_formal_gate_requires_exactly_five_completed_preflight_iterations(
    tmp_path: Path,
) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_path = Path(protocol["provenance"]["baseline_config"])
    preflight_summary = tmp_path / "preflight_summary.json"
    summary = {
        "status": "completed",
        "config_id": "adaptive_preflight_seed1001",
        "run_id": "adaptive_preflight_seed1001",
        "completed_iterations": [1, 2, 3, 4, 5],
        "final_iteration": 5,
    }
    preflight_summary.write_text(json.dumps(summary), encoding="utf-8")
    protocol["config_id"] = "adaptive_formal_test_v1"
    protocol["freeze_gate"] = {
        "current_state": "production_preflight_required",
        "production_preflight": {
            "summary_path": preflight_summary.as_posix(),
            "required_status": "completed",
            "required_config_id": "adaptive_preflight_seed1001",
            "required_run_id": "adaptive_preflight_seed1001",
            "required_iterations": 5,
        },
    }
    input_reference = {
        "path": baseline_path.as_posix(),
        "sha256": sha256(baseline_path),
    }
    protocol["inputs"] = {
        "accepted_pilot_config": input_reference,
        "shared_holdout_contract": input_reference,
    }
    write_yaml(config_path, protocol)

    resolution = resolve_adaptive_protocol(config_path, run_dir)
    assert resolution.input_manifest["inputs"]["production_preflight_summary"][
        "sha256"
    ] == sha256(preflight_summary)

    summary["completed_iterations"] = [1, 2, 3, 4]
    summary["final_iteration"] = 4
    preflight_summary.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ExperimentRuntimeError, match="exactly 5 iterations"):
        resolve_adaptive_protocol(config_path, run_dir)


def runtime_factory(
    created: list[BaselineRuntime],
    *,
    interrupt_after: int | None,
):
    def build(resolved: dict) -> BaselineRuntime:
        network = FakeNetwork()
        args = SimpleNamespace(
            numIters=int(resolved["budget"]["max_iterations"]) + 1,
            numEps=int(resolved["adaptive_scheduler"]["first_iteration_games"]),
        )
        coach = FakeCoach(network, args, interrupt_after=interrupt_after)
        runtime = BaselineRuntime(None, network, coach, args)
        created.append(runtime)
        return runtime

    return build


def test_fresh_interruption_and_resume_use_last_complete_commit(tmp_path: Path) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    created: list[BaselineRuntime] = []

    with pytest.raises(SimulatedInterrupt):
        run_experiment(
            RuntimeRequest("fresh", config_path, run_dir),
            runtime_builder=runtime_factory(created, interrupt_after=1),
        )

    first_metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["iteration"] for record in first_metrics] == [1]
    assert first_metrics[0]["checkpoint_path"] is None
    assert first_metrics[0]["checkpoint_sha256"] is None
    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    input_manifest = json.loads(
        (run_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    run_metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert resolved["git"] == input_manifest["git"] == run_metadata["git"]
    assert resolved["run"]["experiment_code_commit"] == resolved["git"]["commit"]
    assert resolved["git"]["commit"]
    latest = json.loads(
        (run_dir / "recovery" / "latest_commit.json").read_text(encoding="utf-8")
    )
    assert latest["iteration"] == 1
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_manifest["last_committed_iteration"] = 999
    checkpoint_manifest["checkpoints"].append(
        {
            "iteration": 999,
            "path": "uncommitted.pth.tar",
            "sha256": "0" * 64,
            "actual_gpu_hours": 999.0,
            "is_final": False,
            "is_milestone": False,
            "milestones": [],
        }
    )
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )

    result = run_experiment(
        RuntimeRequest("resume", config_path, run_dir),
        runtime_builder=runtime_factory(created, interrupt_after=None),
    )

    final_metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["iteration"] for record in final_metrics] == [1, 2]
    assert final_metrics[0]["games_planned"] == 1
    assert final_metrics[1]["budget_stop_reason"] == "max_iterations_reached"
    assert final_metrics[1]["checkpoint_sha256"] == sha256(
        Path(final_metrics[1]["checkpoint_path"])
    )
    assert created[-1].network.value == 2
    assert created[-1].network.optimizer.step == 2
    recovery_directories = sorted(
        path.name for path in (run_dir / "recovery").glob("iteration_*")
    )
    assert recovery_directories == ["iteration_000002"]
    committed_checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    assert committed_checkpoint_manifest["last_committed_iteration"] == 2
    assert all(
        entry["iteration"] != 999
        for entry in committed_checkpoint_manifest["checkpoints"]
    )
    assert result["status"] == "completed"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_iteration"] == 2
    assert summary["final_checkpoint"] == final_metrics[1]["checkpoint_path"]


def test_resume_rejects_mismatched_recovery_artifact_hash(tmp_path: Path) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    created: list[BaselineRuntime] = []
    with pytest.raises(SimulatedInterrupt):
        run_experiment(
            RuntimeRequest("fresh", config_path, run_dir),
            runtime_builder=runtime_factory(created, interrupt_after=1),
        )
    tracker_path = run_dir / "recovery" / "iteration_000001" / "tracker_state.json"
    tracker_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ExperimentRuntimeError, match="SHA-256 mismatch"):
        run_experiment(
            RuntimeRequest("resume", config_path, run_dir),
            runtime_builder=runtime_factory(created, interrupt_after=None),
        )


def test_gpu_hour_milestone_selects_latest_checkpoint_not_after_target(
    tmp_path: Path,
) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = 1e-12
    protocol["checkpoint"]["gpu_hour_analysis_milestones"] = [0.0, target]
    write_yaml(config_path, protocol)

    with pytest.raises(SimulatedInterrupt):
        run_experiment(
            RuntimeRequest("fresh", config_path, run_dir),
            runtime_builder=runtime_factory([], interrupt_after=1),
        )

    manifest = json.loads(
        (run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    selected = [
        entry for entry in manifest["checkpoints"] if target in entry["milestones"]
    ]
    assert len(selected) == 1
    assert selected[0]["iteration"] == 0
    assert selected[0]["actual_gpu_hours"] <= target


def test_fresh_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ExperimentRuntimeError, match="must not exist or must be empty"):
        run_experiment(
            RuntimeRequest("fresh", config_path, run_dir),
            runtime_builder=lambda resolved: pytest.fail("builder should not run"),
        )


def test_fresh_records_actual_head_without_yaml_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol["budget"]["max_iterations"] = 1
    assert "experiment_code_commit" not in protocol["run"]
    write_yaml(config_path, protocol)
    git = {"commit": "a" * 40, "branch": "test", "dirty": False}
    monkeypatch.setattr(experiment_runtime, "_git_metadata", lambda root: git.copy())

    run_experiment(
        RuntimeRequest("fresh", config_path, run_dir),
        runtime_builder=runtime_factory([], interrupt_after=None),
    )

    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert resolved["run"]["experiment_code_commit"] == git["commit"]
    assert resolved["git"] == manifest["git"] == metadata["git"] == git


def test_fresh_rejects_dirty_worktree_when_protocol_requires_clean_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol["provenance"]["require_clean_worktree"] = False
    protocol["run"]["require_clean_worktree"] = True
    write_yaml(config_path, protocol)
    monkeypatch.setattr(
        experiment_runtime,
        "_git_metadata",
        lambda root: {"commit": "b" * 40, "branch": "test", "dirty": True},
    )

    with pytest.raises(ExperimentRuntimeError, match="requires a clean worktree"):
        run_experiment(
            RuntimeRequest("fresh", config_path, run_dir),
            runtime_builder=lambda resolved: pytest.fail("builder should not run"),
        )


def test_evaluation_time_includes_runner_and_training_state_restore(
    tmp_path: Path,
) -> None:
    config_path, run_dir = make_protocol(tmp_path)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol["budget"]["max_iterations"] = 1
    protocol["evaluation"]["evaluate_every_iterations"] = 1
    write_yaml(config_path, protocol)
    created: list[BaselineRuntime] = []

    def evaluation_runner(
        resolved: dict,
        runtime: BaselineRuntime,
        checkpoint_path: Path,
        output_path: Path,
    ) -> float:
        runtime.network.value = 999
        runtime.network.optimizer.step = 999
        output_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        return 3600.0

    result = run_experiment(
        RuntimeRequest("fresh", config_path, run_dir),
        runtime_builder=runtime_factory(created, interrupt_after=None),
        evaluation_runner=evaluation_runner,
    )

    metric = json.loads(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert result["status"] == "completed"
    assert 0.0 <= metric["evaluation_seconds"] < 3600.0
    assert metric["checkpoint_seconds"] >= 0.0
    assert metric["evaluation_training_state_preserved"] is True
    assert created[-1].network.value == 1
    assert created[-1].network.optimizer.step == 1
