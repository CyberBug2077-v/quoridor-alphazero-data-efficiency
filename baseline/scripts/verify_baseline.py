#!/usr/bin/env python3
"""Read-only verification of a completed baseline run."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from runtime.checkpointing import load_run_state, validate_checkpoint_hash
from runtime.config import ConfigError


BASELINE_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"non-finite number at {path}")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            require_finite(nested, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            require_finite(nested, f"{path}[{index}]")


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VerificationError(f"invalid YAML {path}: {exc}") from exc
    require(isinstance(payload, dict), f"YAML must contain a mapping: {path}")
    return payload


def load_json_mapping(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(payload, dict), f"JSON must contain an object: {path}")
    require_finite(payload, path.name)
    return payload


def load_metrics(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing metrics file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines, "metrics file is empty")
    records = []
    for line_number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"blank metrics line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid metrics line {line_number}: {exc}") from exc
        require(isinstance(record, dict), f"metrics line {line_number} is not an object")
        require_finite(record, f"metrics[{line_number}]")
        records.append(record)
    iterations = [record.get("iteration") for record in records]
    require(iterations == list(range(1, len(records) + 1)), "metrics iterations are not contiguous from 1")
    return records


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    require(path.is_file(), f"missing checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise VerificationError(f"checkpoint cannot be loaded on CPU: {path}: {exc}") from exc
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    require(isinstance(state_dict, dict) and bool(state_dict), f"empty checkpoint state dict: {path}")
    for name, tensor in state_dict.items():
        require(isinstance(tensor, torch.Tensor), f"non-tensor checkpoint value: {path}:{name}")
        require(bool(torch.isfinite(tensor).all()), f"non-finite checkpoint weight: {path}:{name}")
    return state_dict


def verify_model_structure(
    state_dict: dict[str, torch.Tensor], model: dict[str, Any]
) -> None:
    stem_weights = [
        tensor
        for name, tensor in state_dict.items()
        if name.endswith("stem.0.weight") or name == "stem.0.weight"
    ]
    if stem_weights:
        require(
            int(stem_weights[0].shape[0]) == int(model["num_channels"]),
            "checkpoint num_channels differs from model configuration",
        )


def expected_model_shapes(model: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    alphazero_root = BASELINE_ROOT / "external" / "alphazero"
    pathfinder_root = alphazero_root / "quoridor" / "pathFinder-module"
    for path in (alphazero_root, pathfinder_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from quoridor.QuoridorGame import QuoridorGame
        from quoridor.pytorch.QuoridorNNet import QuoridorNNet
        from utils import dotdict

        game = QuoridorGame(int(model["board_size"]))
        network = QuoridorNNet(game, dotdict(dict(model)))
    except Exception as exc:
        raise VerificationError(f"could not construct CPU model from configuration: {exc}") from exc
    return {name: tuple(tensor.shape) for name, tensor in network.state_dict().items()}
    block_indices = {
        int(match.group(1))
        for name in state_dict
        if (match := re.match(r"res_blocks\.(\d+)\.", name))
    }
    if block_indices:
        require(
            max(block_indices) + 1 == int(model["num_res_blocks"]),
            "checkpoint residual-block count differs from model configuration",
        )


def verify_baseline(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    require(run_dir.is_dir(), f"run directory not found: {run_dir}")
    require(run_dir.is_relative_to(BASELINE_ROOT), "run directory must be inside baseline")
    resolved = load_yaml_mapping(run_dir / "resolved_config.yaml")
    require(resolved.get("mode") == "baseline", "resolved configuration mode must be baseline")
    run = resolved.get("run", {})
    require(run.get("id") == run_dir.name, "run ID does not match directory")
    require(isinstance(run.get("seed"), int), "run seed is missing")
    initialization = resolved.get("initialization", {})
    require(initialization.get("load_replay") is False, "restore_replay/load_replay must be false")
    require(initialization.get("mode") == "pretrained_checkpoint", "initialization must be pretrained")

    logging = resolved["logging"]
    metadata = load_json_mapping(run_dir / logging["metadata_file"])
    require(bool(metadata.get("git", {}).get("commit")), "Git commit is missing from metadata")
    initial_path = Path(metadata.get("initial_checkpoint_path", ""))
    initial_hash = metadata.get("initial_checkpoint_sha256")
    require(initial_path.is_file(), "initial checkpoint path is missing or no longer exists")
    require(isinstance(initial_hash, str) and len(initial_hash) == 64, "initial checkpoint hash is missing")
    configured_initial = Path(initialization["checkpoint_path"])
    if not configured_initial.is_absolute():
        configured_initial = BASELINE_ROOT / configured_initial
    require(initial_path.resolve() == configured_initial.resolve(), "initial checkpoint path differs from resolved config")
    try:
        validate_checkpoint_hash(initial_path, initial_hash)
    except (OSError, ValueError) as exc:
        raise VerificationError(str(exc)) from exc
    require(initialization.get("expected_sha256") == initial_hash, "initial checkpoint hash differs from config")
    initial_state = load_checkpoint(initial_path)
    verify_model_structure(initial_state, resolved["model"])
    configured_shapes = expected_model_shapes(resolved["model"])
    require(
        {name: tuple(tensor.shape) for name, tensor in initial_state.items()}
        == configured_shapes,
        "initial checkpoint does not match the configured CPU model",
    )

    metrics = load_metrics(run_dir / logging["metrics_file"])
    expected_iterations = resolved["self_play"].get("iterations")
    if expected_iterations is None:
        expected_iterations = resolved.get("budget", {}).get("max_iterations")
    require(isinstance(expected_iterations, int) and expected_iterations > 0, "expected iterations are unresolved")
    require(len(metrics) == expected_iterations, "completed iterations differ from resolved config")
    games_per_iteration = int(resolved["self_play"]["games_per_iteration"])
    checkpoint_frequency = resolved["checkpoint"]["save_every_iterations"]
    require(isinstance(checkpoint_frequency, int) and checkpoint_frequency > 0, "checkpoint frequency is unresolved")
    required_checkpoint_iterations = set(range(checkpoint_frequency, expected_iterations + 1, checkpoint_frequency))
    required_checkpoint_iterations.add(expected_iterations)
    recorded_checkpoint_iterations = set()
    for record in metrics:
        iteration = int(record["iteration"])
        require(record.get("games_completed") == games_per_iteration, f"wrong game count at iteration {iteration}")
        require(record.get("optimizer_steps", 0) > 0, f"invalid optimizer steps at iteration {iteration}")
        require(record.get("illegal_action_count") == 0, f"illegal actions at iteration {iteration}")
        checkpoint_value = record.get("checkpoint_path")
        if checkpoint_value is not None:
            require(Path(checkpoint_value or "").name == f"checkpoint_{iteration}.pth.tar", f"wrong checkpoint path at iteration {iteration}")
            recorded_checkpoint_iterations.add(iteration)
        require(
            iteration not in required_checkpoint_iterations or checkpoint_value is not None,
            f"missing required checkpoint at iteration {iteration}",
        )

    checkpoint_dir = run_dir / "checkpoints"
    actual_checkpoints = {
        int(match.group(1)): path
        for path in checkpoint_dir.glob("checkpoint_*.pth.tar")
        if (match := CHECKPOINT_PATTERN.match(path.name))
    }
    require(set(actual_checkpoints) == recorded_checkpoint_iterations, "numbered checkpoints do not match metrics")
    require(required_checkpoint_iterations <= recorded_checkpoint_iterations, "required checkpoint cadence is incomplete")
    expected_keys = set(initial_state)
    expected_shapes = {name: tuple(tensor.shape) for name, tensor in initial_state.items()}
    for iteration, path in actual_checkpoints.items():
        state = load_checkpoint(path)
        require(set(state) == expected_keys, f"checkpoint parameter names differ at iteration {iteration}")
        require({name: tuple(tensor.shape) for name, tensor in state.items()} == expected_shapes, f"checkpoint shapes differ at iteration {iteration}")
    last_checkpoint = actual_checkpoints[expected_iterations]
    require(Path(metrics[-1]["checkpoint_path"]).resolve() == last_checkpoint.resolve(), "final metrics/checkpoint mismatch")

    examples_path = checkpoint_dir / "latest.examples"
    try:
        with examples_path.open("rb") as source:
            replay = pickle.load(source)
    except Exception as exc:
        raise VerificationError(f"could not load replay: {exc}") from exc
    require(isinstance(replay, dict), "replay artifact must be a mapping")
    require(replay.get("iteration") == expected_iterations, "replay iteration mismatch")
    history = replay.get("examples")
    require(isinstance(history, list) and bool(history), "replay history is empty or invalid")
    require(len(history) <= int(resolved["replay"]["history_iterations"]), "replay history exceeds configured window")
    require(all(hasattr(iteration, "__len__") and len(iteration) > 0 for iteration in history), "replay contains an empty/corrupt iteration")
    replay_size = sum(len(iteration) for iteration in history)
    require(replay_size == metrics[-1].get("replay_buffer_size"), "replay size differs from final metrics")

    try:
        state = load_run_state(checkpoint_dir / "latest.state.pt")
    except (OSError, ValueError) as exc:
        raise VerificationError(str(exc)) from exc
    require(state.get("iteration") == expected_iterations, "resume-state iteration mismatch")
    for field in ("python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state"):
        require(field in state and state[field] is not None, f"resume state missing {field}")
    cumulative = state.get("cumulative_gpu_hours")
    require(isinstance(cumulative, (int, float)) and math.isfinite(cumulative) and cumulative >= 0, "invalid cumulative GPU-hours")
    require(isinstance(state.get("instrumentation_state"), dict), "invalid instrumentation state")

    evaluation_path = run_dir / "evaluations" / f"evaluation_checkpoint_{expected_iterations}.json"
    evaluation = load_json_mapping(evaluation_path)
    require(Path(evaluation.get("checkpoint_path", "")).resolve() == last_checkpoint.resolve(), "evaluation checkpoint mismatch")
    configured_opponents = resolved["evaluation"]["opponents"]
    require(set(evaluation.get("opponents", {})) == set(configured_opponents), "evaluation opponent set mismatch")
    games_per_opponent = resolved["evaluation"]["games_per_opponent"]
    require(isinstance(games_per_opponent, int) and games_per_opponent % 2 == 0, "invalid configured evaluation games")
    for opponent in configured_opponents:
        result = evaluation["opponents"][opponent]
        games = result.get("games", [])
        require(len(games) == games_per_opponent, f"wrong evaluation game count for {opponent}")
        first = sum(game.get("model_side") == "first" for game in games)
        second = sum(game.get("model_side") == "second" for game in games)
        require(first == second == games_per_opponent // 2, f"unbalanced sides for {opponent}")
        require(result.get("wins", 0) + result.get("draws", 0) + result.get("losses", 0) == games_per_opponent, f"W/D/L mismatch for {opponent}")
        require(result.get("illegal_actions") == 0, f"illegal evaluation actions for {opponent}")

    summary = load_json_mapping(run_dir / logging["summary_file"])
    require(summary.get("status") == "completed", "summary status must be completed")
    require(summary.get("completed_iterations") == list(range(1, expected_iterations + 1)), "summary iterations mismatch")
    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": run["id"],
        "completed_iterations": expected_iterations,
        "initial_checkpoint_verified": True,
        "metrics_verified": True,
        "checkpoints_verified": True,
        "replay_verified": True,
        "evaluation_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(json.dumps(verify_baseline(args.run_dir), indent=2, ensure_ascii=False))
        return 0
    except (VerificationError, ConfigError, OSError, ValueError) as exc:
        print(f"Baseline verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
