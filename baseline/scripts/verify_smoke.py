#!/usr/bin/env python3
"""Read-only verification of a completed baseline smoke run."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "smoke_gpu"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")
REQUIRED_ROOT_FILES = {
    "resolved_config.yaml",
    "metrics.jsonl",
    "summary.json",
    "evaluation.json",
    "run.log",
}
REQUIRED_CHECKPOINT_FILES = {
    "checkpoint_1.pth.tar",
    "checkpoint_2.pth.tar",
    "checkpoint_3.pth.tar",
    "checkpoint_4.pth.tar",
    "best.pth.tar",
    "latest.examples",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_finite(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(value), f"non-finite number at {path}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{path}[{index}]")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(loaded, dict), f"JSON root must be an object: {path}")
    return loaded


def load_metrics(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing file: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        require(bool(line.strip()), f"blank metrics line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"invalid metrics JSON on line {line_number}: {exc}"
            ) from exc
        require(isinstance(record, dict), f"metrics line {line_number} is not an object")
        records.append(record)
    return records


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    require(path.is_file(), f"missing checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise VerificationError(f"checkpoint cannot be reloaded: {path}: {exc}") from exc
    require(isinstance(checkpoint, dict), f"checkpoint is not a mapping: {path}")
    state_dict = checkpoint.get("state_dict")
    require(isinstance(state_dict, dict) and state_dict, f"missing state_dict: {path}")
    for name, tensor in state_dict.items():
        require(torch.is_tensor(tensor), f"non-tensor state entry {name} in {path}")
        require(torch.isfinite(tensor).all().item(), f"non-finite weight {name} in {path}")
    return state_dict


def verify_artifact_tree(run_dir: Path) -> Path:
    require(run_dir.is_dir(), f"run directory not found: {run_dir}")
    root_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    require(
        root_files == REQUIRED_ROOT_FILES,
        f"unexpected root artifacts: expected={sorted(REQUIRED_ROOT_FILES)}, "
        f"actual={sorted(root_files)}",
    )
    root_directories = {path.name for path in run_dir.iterdir() if path.is_dir()}
    require(root_directories == {"checkpoints"}, f"unexpected directories: {root_directories}")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_files = {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
    require(
        checkpoint_files == REQUIRED_CHECKPOINT_FILES,
        f"unexpected checkpoint artifacts: expected={sorted(REQUIRED_CHECKPOINT_FILES)}, "
        f"actual={sorted(checkpoint_files)}",
    )
    require(
        not any(path.is_dir() for path in checkpoint_dir.iterdir()),
        "unexpected directory inside checkpoints",
    )
    return checkpoint_dir


def verify_smoke(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    checkpoint_dir = verify_artifact_tree(run_dir)

    resolved_path = run_dir / "resolved_config.yaml"
    try:
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VerificationError(f"invalid resolved config: {exc}") from exc
    require(isinstance(resolved, dict), "resolved config must be a mapping")
    require(resolved.get("mode") == "smoke", "resolved config mode must be smoke")
    expected_iterations = int(resolved["self_play"]["iterations"])
    expected_games_per_iteration = int(resolved["self_play"]["games_per_iteration"])
    require(expected_iterations == 4, f"expected 4 iterations, got {expected_iterations}")
    require(expected_games_per_iteration == 2, "expected 2 self-play games per iteration")

    configured_output = Path(resolved["run"]["output_dir"])
    if not configured_output.is_absolute():
        configured_output = (PROJECT_ROOT / configured_output).resolve()
    require(configured_output == run_dir, "resolved output directory does not match verified run")
    require(
        Path(resolved["checkpoint"]["directory"]).name == "checkpoints",
        "resolved checkpoint directory is inconsistent",
    )

    metrics = load_metrics(run_dir / "metrics.jsonl")
    require(len(metrics) == expected_iterations, "metrics.jsonl must contain 4 records")
    require(
        [record.get("iteration") for record in metrics] == [1, 2, 3, 4],
        "metrics iterations must be exactly [1, 2, 3, 4]",
    )
    total_games = 0
    for record in metrics:
        iteration = record["iteration"]
        require(record.get("schema_version") == 1, f"invalid metrics schema at {iteration}")
        games = int(record.get("games_completed", 0))
        total_games += games
        require(games == expected_games_per_iteration, f"wrong game count at iteration {iteration}")
        require(record.get("positions_generated", 0) > 0, f"no samples at iteration {iteration}")
        require(record.get("optimizer_steps", 0) > 0, f"no optimizer step at iteration {iteration}")
        require(record.get("illegal_action_count") == 0, f"illegal action at iteration {iteration}")
        for field in (
            "policy_loss",
            "value_loss",
            "total_loss",
            "mean_grad_norm",
            "max_grad_norm",
        ):
            value = record.get(field)
            require(isinstance(value, (int, float)), f"missing {field} at iteration {iteration}")
            require(math.isfinite(value), f"non-finite {field} at iteration {iteration}")
        checkpoint_name = Path(str(record.get("checkpoint_path", "")).replace("\\", "/")).name
        require(
            checkpoint_name == f"checkpoint_{iteration}.pth.tar",
            f"metrics checkpoint mismatch at iteration {iteration}",
        )
        require_finite(record, f"metrics[{iteration}]")
    require(total_games == 8, f"expected 8 self-play games, got {total_games}")

    examples_path = checkpoint_dir / "latest.examples"
    try:
        with examples_path.open("rb") as examples_file:
            replay = pickle.load(examples_file)
    except Exception as exc:
        raise VerificationError(f"latest.examples cannot be loaded: {exc}") from exc
    require(isinstance(replay, dict), "latest.examples must contain a mapping")
    require(replay.get("iteration") == 4, "latest.examples must belong to iteration 4")
    history = replay.get("examples")
    require(isinstance(history, list), "replay history must be a list")
    expected_history = min(
        expected_iterations,
        int(resolved["training"]["replay_history_iterations"]),
    )
    require(len(history) == expected_history, f"expected {expected_history} replay rounds")
    require(all(len(iteration_examples) > 0 for iteration_examples in history), "empty replay round")
    replay_size = sum(len(iteration_examples) for iteration_examples in history)
    require(
        replay_size == metrics[-1]["replay_buffer_size"],
        "replay history size does not match final metrics",
    )

    checkpoint_states = {}
    expected_keys = None
    for iteration in range(1, expected_iterations + 1):
        path = checkpoint_dir / f"checkpoint_{iteration}.pth.tar"
        state = load_checkpoint_state(path)
        keys = set(state)
        if expected_keys is None:
            expected_keys = keys
        require(keys == expected_keys, f"checkpoint state keys differ: {path.name}")
        checkpoint_states[iteration] = state
    best_state = load_checkpoint_state(checkpoint_dir / "best.pth.tar")
    require(set(best_state) == expected_keys, "best checkpoint state keys differ")
    for name, tensor in checkpoint_states[4].items():
        require(torch.equal(tensor, best_state[name]), f"best checkpoint differs at {name}")

    evaluation = load_json(run_dir / "evaluation.json")
    require_finite(evaluation, "evaluation")
    require(
        Path(evaluation.get("checkpoint_path", "")).name == "checkpoint_4.pth.tar",
        "evaluation did not use checkpoint_4",
    )
    opponents = evaluation.get("opponents")
    require(isinstance(opponents, dict), "evaluation opponents missing")
    require(set(opponents) == {"random", "greedy"}, "evaluation opponents must be random/greedy")
    for opponent in ("random", "greedy"):
        result = opponents[opponent]
        games = result.get("games")
        require(isinstance(games, list) and len(games) == 2, f"{opponent} must have 2 games")
        require(
            result.get("wins", 0) + result.get("draws", 0) + result.get("losses", 0) == 2,
            f"{opponent} W/D/L does not total 2",
        )
        require(
            [game.get("model_side") for game in games] == ["first", "second"],
            f"{opponent} did not swap sides",
        )
        require(result.get("illegal_actions") == 0, f"{opponent} evaluation had illegal actions")
        require(
            all(game.get("illegal_actions") == 0 for game in games),
            f"{opponent} game contained an illegal action",
        )

    summary = load_json(run_dir / "summary.json")
    require_finite(summary, "summary")
    require(summary.get("status") == "completed", "summary status must be completed")
    require(summary.get("completed_iterations") == [1, 2, 3, 4], "summary iteration list mismatch")
    require(summary.get("target_iterations") == 4, "summary target mismatch")
    require(
        Path(str(summary.get("final_checkpoint", "")).replace("\\", "/")).name
        == "checkpoint_4.pth.tar",
        "summary final checkpoint mismatch",
    )

    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    fresh_headers = [
        line for line in run_log.splitlines() if line.startswith("[run] mode=fresh run_id=")
    ]
    require(len(fresh_headers) == 1, "run.log must contain exactly one fresh-run header")
    require(
        f"run_id={resolved['run']['id']}" in fresh_headers[0],
        "run.log run_id does not match resolved config",
    )
    require(
        f"output={run_dir.as_posix()}" in fresh_headers[0],
        "run.log output path does not match verified run",
    )

    return {
        "run_dir": run_dir.as_posix(),
        "iterations": expected_iterations,
        "self_play_games": total_games,
        "metrics_records": len(metrics),
        "checkpoints": expected_iterations,
        "replay_history_iterations": len(history),
        "evaluation_games": {
            opponent: len(opponents[opponent]["games"])
            for opponent in ("random", "greedy")
        },
        "status": "verified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Completed smoke run directory (default: {DEFAULT_RUN_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_smoke(args.run_dir)
    except (VerificationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Smoke verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
