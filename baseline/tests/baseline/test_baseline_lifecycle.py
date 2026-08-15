from __future__ import annotations

import copy
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

import run_baseline
from runtime.artifacts import atomic_write_json
from runtime.checkpointing import load_run_state, save_run_state
from runtime.metadata import sha256_file


BASELINE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = BASELINE_ROOT / "configs"


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_baseline_entrypoint_lifecycle_with_minimal_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise fresh -> stopped -> resume -> completed -> evaluate-only cheaply."""

    test_root = tmp_path / "baseline"
    config_dir = test_root / "configs"
    config_dir.mkdir(parents=True)
    run_dir = test_root / "outputs" / "baseline_lifecycle_seed1001"
    checkpoint_dir = run_dir / "checkpoints"
    initial_checkpoint = (
        test_root
        / "outputs"
        / "pretraining_reproduction_seed1001"
        / "checkpoints"
        / "checkpoint_0.pth.tar"
    )
    initial_checkpoint.parent.mkdir(parents=True)

    pretrained_weight = torch.tensor([[7.0]])
    torch.save(
        {
            "state_dict": {"weight": pretrained_weight.clone()},
            # These sentinels must never become online state during fresh.
            "replay": [["pretraining-replay-must-not-load"]],
            "optimizer_state": {"pretraining-step": 999},
            "tracker_state": {"pretraining-tracker": 999},
        },
        initial_checkpoint,
    )
    initial_hash = sha256_file(initial_checkpoint)

    config = yaml.safe_load(
        (CONFIG_ROOT / "baseline_pilot.yaml").read_text(encoding="utf-8")
    )
    config["run"].update(
        {
            "id": run_dir.name,
            "device": "cuda",
            "output_dir": "outputs/baseline_lifecycle_seed1001",
        }
    )
    config["initialization"].update(
        {
            "checkpoint_path": (
                "outputs/pretraining_reproduction_seed1001/"
                "checkpoints/checkpoint_0.pth.tar"
            ),
            "expected_sha256": initial_hash,
        }
    )
    config["self_play"]["iterations"] = 3
    config["budget"]["max_iterations"] = 3
    config["checkpoint"]["directory"] = (
        "outputs/baseline_lifecycle_seed1001/checkpoints"
    )
    config_path = config_dir / "baseline_lifecycle.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )

    created_networks: list[Any] = []
    coach_construction: list[dict[str, Any]] = []
    replay_restores: list[dict[str, Any]] = []
    executed_iterations: list[list[int]] = []
    checkpoint_loads: list[Path] = []
    restored_rng_states: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []

    class FakeNetwork:
        def __init__(self) -> None:
            self.nnet = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.nnet.weight.fill_(-100.0)

    class FakeCoach:
        def __init__(self, game, network, args, iteration_callback=None) -> None:
            self.game = game
            self.nnet = network
            self.args = args
            self.iteration_callback = iteration_callback
            self.trainExamplesHistory: list[list[dict[str, Any]]] = []
            self.current_iteration = 0
            self.invocation_iterations: list[int] = []
            executed_iterations.append(self.invocation_iterations)
            coach_construction.append(
                {
                    "weight": self.nnet.nnet.weight.detach().clone(),
                    "replay": copy.deepcopy(self.trainExamplesHistory),
                    "current_iteration": self.current_iteration,
                    "load_model": self.args.load_model,
                    "load_examples": self.args.load_examples,
                }
            )

        def loadTrainExamples(self) -> None:
            replay_path = Path(self.args.checkpoint) / "latest.examples"
            with replay_path.open("rb") as source:
                replay = pickle.load(source)
            self.current_iteration = int(replay["iteration"])
            self.trainExamplesHistory = replay["examples"]
            replay_restores.append(copy.deepcopy(replay))

        def learn(self) -> None:
            checkpoint_path = Path(self.args.checkpoint)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            for iteration in range(self.current_iteration + 1, self.args.numIters + 1):
                self.invocation_iterations.append(iteration)
                with torch.no_grad():
                    self.nnet.nnet.weight.add_(1.0)
                self.trainExamplesHistory.append(
                    [{"source": "online", "iteration": iteration}]
                )
                numbered = checkpoint_path / f"checkpoint_{iteration}.pth.tar"
                numbered_tmp = numbered.with_name(numbered.name + ".tmp")
                torch.save({"state_dict": self.nnet.nnet.state_dict()}, numbered_tmp)
                os.replace(numbered_tmp, numbered)
                replay_path = checkpoint_path / "latest.examples"
                replay_tmp = replay_path.with_name(replay_path.name + ".tmp")
                with replay_tmp.open("wb") as output:
                    pickle.dump(
                        {
                            "iteration": iteration,
                            "examples": self.trainExamplesHistory,
                        },
                        output,
                    )
                os.replace(replay_tmp, replay_path)
                self.current_iteration = iteration
                self.iteration_callback(
                    {
                        "schema_version": 1,
                        "iteration": iteration,
                        "games_completed": 75,
                        "positions_generated": 1,
                        "illegal_action_count": 0,
                        "replay_buffer_size": len(self.trainExamplesHistory),
                        "available_examples": len(self.trainExamplesHistory),
                        "examples_used": 1,
                        "samples_seen": 1,
                        "training_batches": 1,
                        "optimizer_steps": 1,
                        "effective_batch_size": 1,
                        "micro_batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "micro_batches_processed": 1,
                        "policy_loss": 1.0,
                        "value_loss": 1.0,
                        "total_loss": 2.0,
                        "mean_grad_norm": 1.0,
                        "max_grad_norm": 1.0,
                        "self_play_seconds": 18.0,
                        "training_seconds": 18.0,
                        "iteration_seconds": 36.0,
                        "peak_gpu_memory_mb": 1.0,
                        "checkpoint_path": numbered.as_posix(),
                    }
                )

    def fake_create_runtime(resolved: dict[str, Any]):
        network = FakeNetwork()
        created_networks.append(network)
        train_args = SimpleNamespace(
            **copy.deepcopy(resolved["mapped_args"]["train_args"])
        )
        return object(), network, train_args, FakeCoach

    real_strict_load = run_baseline.strict_load_initial_weights

    def recording_strict_load(network, checkpoint_path, expected_sha256):
        checkpoint_loads.append(Path(checkpoint_path).resolve())
        return real_strict_load(network, checkpoint_path, expected_sha256)

    def fake_write_run_metadata(
        path,
        *,
        project_root,
        resolved_config=None,
        input_hashes=None,
        extra=None,
    ):
        payload = {
            "schema_version": 1,
            "git": {"commit": "lifecycle-test-commit", "dirty": False},
            "input_hashes": input_hashes or {},
            "resolved_config": resolved_config,
        }
        payload.update(extra or {})
        return atomic_write_json(path, payload)

    def fake_run_evaluation(resolved, game, network, checkpoint: Path) -> Path:
        checkpoint = Path(checkpoint).resolve()
        label = checkpoint.name.removeprefix("checkpoint_").removesuffix(
            ".pth.tar"
        )
        output = resolved["_output_path"] / "evaluations" / (
            f"evaluation_checkpoint_{label}.json"
        )
        before = {
            name: value.detach().clone()
            for name, value in network.nnet.state_dict().items()
        }
        expected = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )["state_dict"]
        assert all(torch.equal(before[name], expected[name]) for name in expected)
        atomic_write_json(
            output,
            {
                "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "opponents": {},
            },
        )
        after = network.nnet.state_dict()
        assert all(torch.equal(before[name], after[name]) for name in before)
        evaluations.append(
            {
                "checkpoint": checkpoint,
                "output": output,
                "parameters_unchanged": True,
            }
        )
        return output

    monkeypatch.setattr(run_baseline, "BASELINE_ROOT", test_root)
    monkeypatch.setattr(run_baseline, "SOURCE_ROOT", test_root)
    monkeypatch.setattr(run_baseline, "initialize_cuda", lambda resolved: None)
    monkeypatch.setattr(run_baseline, "set_seed", lambda seed, deterministic: None)
    monkeypatch.setattr(run_baseline, "create_runtime", fake_create_runtime)
    monkeypatch.setattr(run_baseline, "strict_load_initial_weights", recording_strict_load)
    monkeypatch.setattr(run_baseline, "write_run_metadata", fake_write_run_metadata)
    monkeypatch.setattr(
        run_baseline,
        "collect_git_metadata",
        lambda project_root: {"commit": "lifecycle-test-commit", "dirty": False},
    )
    monkeypatch.setattr(
        run_baseline,
        "capture_rng_state",
        lambda: {
            "python": "captured-python-rng",
            "numpy": "captured-numpy-rng",
            "torch_cpu": torch.tensor([1], dtype=torch.uint8),
            "torch_cuda": [],
        },
    )
    monkeypatch.setattr(
        run_baseline,
        "restore_rng_state",
        lambda state: restored_rng_states.append(copy.deepcopy(state)),
    )
    monkeypatch.setattr(run_baseline, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(run_baseline.torch.cuda, "empty_cache", lambda: None)

    def invoke(*arguments: str) -> int:
        monkeypatch.setattr(
            sys, "argv", ["run_baseline.py", *arguments]
        )
        return run_baseline.main()

    # Fresh loads checkpoint_0 weights but starts a brand-new online replay.
    assert invoke(
        "fresh",
        "--config",
        str(config_path),
        "--stop-after-iteration",
        "2",
    ) == 0
    stopped_summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    stopped_metrics = _read_metrics(run_dir / "metrics.jsonl")
    stopped_replay = pickle.loads((checkpoint_dir / "latest.examples").read_bytes())
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert stopped_summary["status"] == "stopped"
    assert stopped_summary["completed_iterations"] == [1, 2]
    assert [record["iteration"] for record in stopped_metrics] == [1, 2]
    assert coach_construction[0]["current_iteration"] == 0
    assert coach_construction[0]["replay"] == []
    assert coach_construction[0]["load_model"] is False
    assert coach_construction[0]["load_examples"] is False
    assert torch.equal(coach_construction[0]["weight"], pretrained_weight)
    assert all(
        sample["source"] == "online"
        for history in stopped_replay["examples"]
        for sample in history
    )
    assert config["initialization"]["load_replay"] is False
    assert config["initialization"]["load_optimizer_state"] is False
    assert config["initialization"]["load_tracker_state"] is False
    assert metadata["initialization_type"] == "pretrained"
    assert metadata["initial_replay_size"] == 0
    assert metadata["starting_iteration"] == 1
    assert metadata["initial_checkpoint_sha256"] == initial_hash
    assert checkpoint_loads[:2] == [
        initial_checkpoint.resolve(),
        initial_checkpoint.resolve(),
    ]

    # A second fresh invocation must fail before touching existing results.
    before_duplicate_fresh = _file_hashes(run_dir)
    assert invoke(
        "fresh",
        "--config",
        str(config_path),
        "--stop-after-iteration",
        "2",
    ) == 2
    assert _file_hashes(run_dir) == before_duplicate_fresh

    # Inject non-empty future instrumentation state and a known cumulative
    # budget at the stopped boundary, then prove resume carries both forward.
    stopped_state_path = checkpoint_dir / "latest.state.pt"
    stopped_state = load_run_state(stopped_state_path)
    stopped_state.update(
        {
            "python_rng_state": "resume-python-rng",
            "numpy_rng_state": "resume-numpy-rng",
            "torch_rng_state": "resume-torch-rng",
            "cuda_rng_state": "resume-cuda-rng",
            "cumulative_gpu_hours": 1.25,
            "instrumentation_state": {
                "tracker": {"last_iteration": 2, "samples": 17}
            },
        }
    )
    save_run_state(stopped_state_path, stopped_state)

    assert invoke("resume", "--run-dir", str(run_dir)) == 0
    completed_summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    completed_metrics = _read_metrics(run_dir / "metrics.jsonl")
    completed_state = load_run_state(stopped_state_path)

    assert completed_summary["status"] == "completed"
    assert completed_summary["completed_iterations"] == [1, 2, 3]
    assert [record["iteration"] for record in completed_metrics] == [1, 2, 3]
    assert executed_iterations == [[1, 2], [3]]
    assert checkpoint_loads[2] == (checkpoint_dir / "checkpoint_2.pth.tar").resolve()
    assert replay_restores[0]["iteration"] == 2
    assert [
        history[0]["iteration"] for history in replay_restores[0]["examples"]
    ] == [1, 2]
    assert restored_rng_states == [
        {
            "python": "resume-python-rng",
            "numpy": "resume-numpy-rng",
            "torch_cpu": "resume-torch-rng",
            "torch_cuda": "resume-cuda-rng",
        }
    ]
    assert completed_state["iteration"] == 3
    assert completed_state["instrumentation_state"] == {
        "tracker": {"last_iteration": 2, "samples": 17}
    }
    assert completed_state["cumulative_gpu_hours"] == 1.26
    assert completed_metrics[-1]["cumulative_gpu_hours"] == 1.26

    # Completion evaluated checkpoint 3. Evaluate-only targets checkpoint 2,
    # leaving that existing result and every training artifact untouched.
    existing_evaluation = run_dir / "evaluations" / "evaluation_checkpoint_3.json"
    assert existing_evaluation.is_file()
    protected_before_evaluation = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in (
            run_dir / "resolved_config.yaml",
            run_dir / "run_metadata.json",
            run_dir / "metrics.jsonl",
            run_dir / "summary.json",
            run_dir / "run.log",
            checkpoint_dir / "latest.examples",
            checkpoint_dir / "latest.state.pt",
            checkpoint_dir / "checkpoint_1.pth.tar",
            checkpoint_dir / "checkpoint_2.pth.tar",
            checkpoint_dir / "checkpoint_3.pth.tar",
            existing_evaluation,
        )
    }
    checkpoint_names_before = sorted(path.name for path in checkpoint_dir.glob("*"))

    assert invoke(
        "evaluate-only",
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        str(checkpoint_dir / "checkpoint_2.pth.tar"),
    ) == 0

    protected_after_evaluation = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in (
            run_dir / "resolved_config.yaml",
            run_dir / "run_metadata.json",
            run_dir / "metrics.jsonl",
            run_dir / "summary.json",
            run_dir / "run.log",
            checkpoint_dir / "latest.examples",
            checkpoint_dir / "latest.state.pt",
            checkpoint_dir / "checkpoint_1.pth.tar",
            checkpoint_dir / "checkpoint_2.pth.tar",
            checkpoint_dir / "checkpoint_3.pth.tar",
            existing_evaluation,
        )
    }
    assert protected_after_evaluation == protected_before_evaluation
    assert sorted(path.name for path in checkpoint_dir.glob("*")) == checkpoint_names_before
    assert (run_dir / "evaluations" / "evaluation_checkpoint_2.json").is_file()
    assert evaluations[-1]["parameters_unchanged"] is True
    assert evaluations[-1]["checkpoint"] == (
        checkpoint_dir / "checkpoint_2.pth.tar"
    ).resolve()
    assert checkpoint_loads[-1] == (
        checkpoint_dir / "checkpoint_2.pth.tar"
    ).resolve()
