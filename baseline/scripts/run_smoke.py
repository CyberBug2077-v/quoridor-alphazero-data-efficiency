#!/usr/bin/env python3
"""Validate, resolve, and run the baseline GPU smoke test."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pickle
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from runtime.artifacts import (
    JsonlWriter as MetricsJsonlWriter,
    atomic_write_yaml,
    find_existing_run_artifacts as find_formal_artifacts,
    write_summary,
)
from runtime.checkpointing import load_model_checkpoint
from runtime.config import (
    ConfigError,
    load_yaml,
    map_baseline_to_train_args,
    map_model_to_nn_args,
    merge_defaults,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "smoke_gpu.yaml"

DEFAULTS: dict[str, Any] = {
    "run": {
        "device": "cuda",
        "gpu_index": 0,
        "deterministic": True,
        "output_dir": None,
    },
    "self_play": {
        "temperature_threshold": 15,
        "cpuct": 1.25,
        "dirichlet_noise": False,
        "dirichlet_alpha": 0.15,
        "dirichlet_epsilon": 0.25,
        "max_game_length": 150,
        "eval_mcts_in_batch": 4,
    },
    "training": {
        "amp": False,
        "amp_dtype": "bf16",
        "max_queue_size": 500_000,
        "num_channels": 128,
        "num_res_blocks": 6,
        "attn_depth": 1,
        "num_heads": 8,
        "se_enabled": False,
        "fast_opts": False,
        "gradient_clip": 1.0,
        "update_gating": False,
        "update_threshold": -0.51,
    },
    "checkpoint": {
        "save_every_iterations": 1,
    },
    "evaluation": {
        "enabled": False,
        "games_per_opponent": 0,
        "opponents": [],
        "model_mcts_simulations": 8,
        "temperature": 0,
        "dirichlet_noise": False,
        "max_game_length": 150,
    },
}

REQUIRED_FIELDS = {
    "run": {"id", "seed"},
    "self_play": {"iterations", "games_per_iteration", "mcts_simulations"},
    "training": {
        "epochs",
        "batch_size",
        "learning_rate",
        "max_samples",
        "replay_history_iterations",
    },
    "checkpoint": set(),
    "evaluation": set(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and optionally run the baseline GPU smoke test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("dry-run", "fresh", "resume", "evaluate-only"),
        help="Execution mode (default: fresh; --dry-run remains supported).",
    )
    parser.add_argument(
        "--mode",
        dest="mode_option",
        choices=("dry-run", "fresh", "resume", "evaluate-only"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Backward-compatible alias for dry-run mode.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Complete run directory to resume or evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the configured output directory (fresh/dry-run/evaluate-only/tests).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit checkpoint for evaluate-only mode.",
    )
    parser.add_argument(
        "--stop-after-iteration",
        type=int,
        help="Stop after this completed iteration without final evaluation.",
    )
    args = parser.parse_args()
    selected = [mode for mode in (args.mode, args.mode_option) if mode]
    if len(set(selected)) > 1:
        parser.error("positional mode and --mode disagree")
    explicit_mode = selected[0] if selected else None
    args.mode = explicit_mode or "fresh"
    if args.dry_run:
        if explicit_mode not in (None, "dry-run"):
            parser.error("--dry-run cannot be combined with another mode")
        args.mode = "dry-run"
    if args.stop_after_iteration is not None:
        if args.mode not in {"fresh", "resume"}:
            parser.error("--stop-after-iteration is only valid for fresh/resume")
        if args.stop_after_iteration < 1:
            parser.error("--stop-after-iteration must be >= 1")
    if args.mode == "resume" and args.run_dir is None:
        parser.error("resume mode requires --run-dir")
    if args.mode in {"dry-run", "fresh"} and args.run_dir is not None:
        parser.error("--run-dir is only valid for resume/evaluate-only")
    if args.mode == "evaluate-only" and args.run_dir is None and args.checkpoint is None:
        parser.error("evaluate-only requires --run-dir or --checkpoint")
    if args.mode == "evaluate-only" and args.run_dir is not None and args.output_dir is not None:
        parser.error("evaluate-only with --run-dir writes into that run directory")
    if args.mode != "evaluate-only" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid for evaluate-only")
    if args.mode == "resume" and args.output_dir is not None:
        parser.error("resume uses --run-dir and does not accept --output-dir")
    return args


def load_config(path: Path) -> dict[str, Any]:
    loaded = load_yaml(path)

    missing_sections = sorted(set(REQUIRED_FIELDS) - set(loaded))
    if missing_sections:
        raise ConfigError(
            f"missing required section(s): {', '.join(missing_sections)}"
        )

    for section, required_fields in REQUIRED_FIELDS.items():
        section_value = loaded.get(section)
        if not isinstance(section_value, dict):
            raise ConfigError(f"{section} must be a mapping")
        allowed_fields = set(DEFAULTS[section]) | required_fields
        unknown = sorted(set(section_value) - allowed_fields)
        if unknown:
            raise ConfigError(
                f"unknown field(s) in {section}: {', '.join(unknown)}"
            )
        missing_fields = sorted(required_fields - set(section_value))
        if missing_fields:
            qualified = ", ".join(
                f"{section}.{field}" for field in missing_fields
            )
            raise ConfigError(f"missing required field(s): {qualified}")

    unknown_sections = sorted(set(loaded) - set(DEFAULTS))
    if unknown_sections:
        raise ConfigError(
            f"unknown top-level section: {unknown_sections[0]}"
        )
    return merge_defaults(DEFAULTS, loaded)


def require_int(
    section: dict[str, Any], field: str, *, path: str, minimum: int
) -> int:
    value = section[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path}.{field} must be an integer >= {minimum}")
    return value


def require_float(
    section: dict[str, Any], field: str, *, path: str, minimum: float
) -> float:
    value = section[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{field} must be a number >= {minimum}")
    result = float(value)
    if result < minimum:
        raise ConfigError(f"{path}.{field} must be a number >= {minimum}")
    return result


def require_bool(section: dict[str, Any], field: str, *, path: str) -> bool:
    value = section[field]
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{field} must be true or false")
    return value


def require_string(section: dict[str, Any], field: str, *, path: str) -> str:
    value = section[field]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{field} must be a non-empty string")
    return value.strip()


def resolve_output_dir(run: dict[str, Any], run_id: str) -> tuple[Path, str]:
    raw_output_dir = run["output_dir"] or f"outputs/{run_id}"
    if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
        raise ConfigError("run.output_dir must be a non-empty path string")

    requested = Path(raw_output_dir)
    output_dir = (
        requested.resolve()
        if requested.is_absolute()
        else (PROJECT_ROOT / requested).resolve()
    )
    try:
        relative_output = output_dir.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ConfigError("run.output_dir must be inside the baseline directory") from exc

    return output_dir, relative_output.as_posix()


def set_random_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    run = config["run"]
    self_play = config["self_play"]
    training = config["training"]
    checkpoint = config["checkpoint"]
    evaluation = config["evaluation"]

    run_id = require_string(run, "id", path="run")
    seed = require_int(run, "seed", path="run", minimum=0)
    if seed > np.iinfo(np.uint32).max:
        raise ConfigError("run.seed must be <= 4294967295 for NumPy compatibility")
    deterministic = require_bool(run, "deterministic", path="run")
    gpu_index = require_int(run, "gpu_index", path="run", minimum=0)

    device = require_string(run, "device", path="run")
    if device != "cuda":
        raise ConfigError("smoke_gpu.yaml requires run.device: cuda")
    if not torch.cuda.is_available():
        raise ConfigError(
            "CUDA is not available to PyTorch; check the selected interpreter, "
            "PyTorch build, and NVIDIA driver"
        )
    if gpu_index >= torch.cuda.device_count():
        raise ConfigError(
            f"run.gpu_index {gpu_index} is invalid; detected "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )

    iterations = require_int(
        self_play, "iterations", path="self_play", minimum=1
    )
    games_per_iteration = require_int(
        self_play, "games_per_iteration", path="self_play", minimum=1
    )
    mcts_simulations = require_int(
        self_play, "mcts_simulations", path="self_play", minimum=1
    )
    temperature_threshold = require_int(
        self_play, "temperature_threshold", path="self_play", minimum=0
    )
    max_game_length = require_int(
        self_play, "max_game_length", path="self_play", minimum=1
    )
    eval_mcts_in_batch = require_int(
        self_play, "eval_mcts_in_batch", path="self_play", minimum=1
    )
    if mcts_simulations < eval_mcts_in_batch:
        raise ConfigError(
            "self_play.mcts_simulations must be >= self_play.eval_mcts_in_batch"
        )
    if mcts_simulations % eval_mcts_in_batch != 0:
        raise ConfigError(
            "self_play.mcts_simulations must be divisible by "
            "self_play.eval_mcts_in_batch"
        )
    cpuct = require_float(self_play, "cpuct", path="self_play", minimum=0.0)
    dirichlet_noise = require_bool(
        self_play, "dirichlet_noise", path="self_play"
    )
    dirichlet_alpha = require_float(
        self_play, "dirichlet_alpha", path="self_play", minimum=0.0
    )
    configured_dirichlet_epsilon = require_float(
        self_play, "dirichlet_epsilon", path="self_play", minimum=0.0
    )
    if configured_dirichlet_epsilon > 1.0:
        raise ConfigError("self_play.dirichlet_epsilon must be <= 1.0")
    dirichlet_epsilon = (
        configured_dirichlet_epsilon if dirichlet_noise else 0.0
    )

    epochs = require_int(training, "epochs", path="training", minimum=1)
    batch_size = require_int(training, "batch_size", path="training", minimum=1)
    max_samples = require_int(
        training, "max_samples", path="training", minimum=1
    )
    if max_samples < batch_size:
        raise ConfigError("training.max_samples must be >= training.batch_size")
    replay_history_iterations = require_int(
        training, "replay_history_iterations", path="training", minimum=1
    )
    learning_rate = require_float(
        training, "learning_rate", path="training", minimum=0.0
    )
    amp = require_bool(training, "amp", path="training")
    amp_dtype = require_string(training, "amp_dtype", path="training")
    if amp_dtype not in {"fp16", "bf16"}:
        raise ConfigError("training.amp_dtype must be fp16 or bf16")
    max_queue_size = require_int(
        training, "max_queue_size", path="training", minimum=1
    )
    num_channels = require_int(
        training, "num_channels", path="training", minimum=1
    )
    num_res_blocks = require_int(
        training, "num_res_blocks", path="training", minimum=1
    )
    attn_depth = require_int(training, "attn_depth", path="training", minimum=0)
    num_heads = require_int(training, "num_heads", path="training", minimum=1)
    se_enabled = require_bool(training, "se_enabled", path="training")
    fast_opts = require_bool(training, "fast_opts", path="training")
    gradient_clip = require_float(
        training, "gradient_clip", path="training", minimum=0.0
    )
    update_gating = require_bool(
        training, "update_gating", path="training"
    )
    update_threshold = require_float(
        training, "update_threshold", path="training", minimum=-1.0
    )
    if update_threshold > 1.0:
        raise ConfigError("training.update_threshold must be <= 1.0")

    save_every_iterations = require_int(
        checkpoint, "save_every_iterations", path="checkpoint", minimum=1
    )

    evaluation_enabled = require_bool(evaluation, "enabled", path="evaluation")
    games_per_opponent = require_int(
        evaluation, "games_per_opponent", path="evaluation", minimum=0
    )
    opponents = evaluation["opponents"]
    if not isinstance(opponents, list) or not all(
        isinstance(opponent, str) and opponent.strip() for opponent in opponents
    ):
        raise ConfigError("evaluation.opponents must be a list of non-empty strings")
    opponents = [opponent.strip() for opponent in opponents]
    unsupported_opponents = sorted(set(opponents) - {"random", "greedy"})
    if unsupported_opponents:
        raise ConfigError(
            "unsupported evaluation opponent(s): "
            + ", ".join(unsupported_opponents)
        )
    model_mcts_simulations = require_int(
        evaluation,
        "model_mcts_simulations",
        path="evaluation",
        minimum=1,
    )
    evaluation_temperature = require_float(
        evaluation,
        "temperature",
        path="evaluation",
        minimum=0.0,
    )
    evaluation_dirichlet_noise = require_bool(
        evaluation,
        "dirichlet_noise",
        path="evaluation",
    )
    evaluation_max_game_length = require_int(
        evaluation,
        "max_game_length",
        path="evaluation",
        minimum=1,
    )
    if evaluation_enabled and (games_per_opponent < 1 or not opponents):
        raise ConfigError(
            "enabled evaluation requires games_per_opponent >= 1 and at least "
            "one opponent"
        )
    if games_per_opponent % 2 != 0:
        raise ConfigError(
            "evaluation.games_per_opponent must be even so opponents can swap sides"
        )
    if evaluation_dirichlet_noise:
        raise ConfigError("evaluation.dirichlet_noise must be false")

    torch.cuda.set_device(gpu_index)
    set_random_seed(seed, deterministic)
    gpu_name = torch.cuda.get_device_name(gpu_index)
    output_dir, output_display = resolve_output_dir(run, run_id)
    checkpoint_dir = (Path(output_display) / "checkpoints").as_posix()

    try:
        source_config = config_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_config = str(config_path.resolve())

    resolved = {
        "schema_version": 1,
        "mode": "dry-run",
        "source_config": source_config,
        "run": {
            "id": run_id,
            "seed": seed,
            "device": device,
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "deterministic": deterministic,
            "output_dir": output_display,
        },
        "self_play": {
            "iterations": iterations,
            "games_per_iteration": games_per_iteration,
            "mcts_simulations": mcts_simulations,
            "temperature_threshold": temperature_threshold,
            "cpuct": cpuct,
            "dirichlet_noise": dirichlet_noise,
            "dirichlet_alpha": dirichlet_alpha,
            "dirichlet_epsilon": dirichlet_epsilon,
            "max_game_length": max_game_length,
            "eval_mcts_in_batch": eval_mcts_in_batch,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "replay_history_iterations": replay_history_iterations,
            "max_queue_size": max_queue_size,
            "amp": amp,
            "amp_dtype": amp_dtype,
            "num_channels": num_channels,
            "num_res_blocks": num_res_blocks,
            "attn_depth": attn_depth,
            "num_heads": num_heads,
            "se_enabled": se_enabled,
            "fast_opts": fast_opts,
            "gradient_clip": gradient_clip,
            "update_gating": update_gating,
            "update_threshold": update_threshold,
        },
        "checkpoint": {
            "directory": checkpoint_dir,
            "save_every_iterations": save_every_iterations,
        },
        "evaluation": {
            "enabled": evaluation_enabled,
            "games_per_opponent": games_per_opponent,
            "opponents": list(opponents),
            "model_mcts_simulations": model_mcts_simulations,
            "temperature": evaluation_temperature,
            "dirichlet_noise": evaluation_dirichlet_noise,
            "max_game_length": evaluation_max_game_length,
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
        },
        "resolved_config_path": (
            Path(output_display) / "resolved_config.yaml"
        ).as_posix(),
        "_output_path": output_dir,
    }
    resolved["mapped_args"] = {
        "train_args": map_baseline_to_train_args(resolved),
        "nn_args": map_model_to_nn_args(resolved),
        "evaluation_args": copy.deepcopy(resolved["evaluation"]),
    }
    return resolved


def save_resolved_config(resolved: dict[str, Any]) -> Path:
    output_dir = resolved["_output_path"]
    serializable = {
        key: value for key, value in resolved.items() if key != "_output_path"
    }
    resolved_path = output_dir / "resolved_config.yaml"
    return atomic_write_yaml(resolved_path, serializable)


def print_resolved_config(resolved: dict[str, Any]) -> None:
    run = resolved["run"]
    self_play = resolved["self_play"]
    training = resolved["training"]

    print(f"Device: {run['device']}")
    print(f"GPU: {run['gpu_name']}")
    print(f"Seed: {run['seed']}")
    print(f"Iterations: {self_play['iterations']}")
    print(f"Games/iteration: {self_play['games_per_iteration']}")
    print(f"MCTS simulations: {self_play['mcts_simulations']}")
    print(f"AMP: {str(training['amp']).lower()}")
    print(f"Dirichlet noise: {str(self_play['dirichlet_noise']).lower()}")
    print(f"Output: {run['output_dir']}")


CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")


def rebase_output_directory(resolved: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rebased = copy.deepcopy(resolved)
    absolute_output = output_dir.expanduser().resolve()
    checkpoint_dir = absolute_output / "checkpoints"
    output_display = absolute_output.as_posix()
    checkpoint_display = checkpoint_dir.as_posix()
    rebased["_output_path"] = absolute_output
    rebased["run"]["output_dir"] = output_display
    rebased["checkpoint"]["directory"] = checkpoint_display
    rebased["resolved_config_path"] = (absolute_output / "resolved_config.yaml").as_posix()
    train_args = rebased["mapped_args"]["train_args"]
    train_args["checkpoint"] = checkpoint_display
    train_args["load_folder_file"] = [checkpoint_display, "best.pth.tar"]
    train_args["load_folder_examples_file"] = [checkpoint_display, "latest.examples"]
    return rebased


def check_runtime_dependencies() -> None:
    missing = [
        name
        for name in ("numpy", "torch", "yaml", "progress", "einops", "psutil")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ConfigError(f"missing Python dependencies: {', '.join(missing)}")
    required_paths = [
        PROJECT_ROOT / "external" / "alphazero",
        PROJECT_ROOT / "external" / "alphazero" / "quoridor",
        PROJECT_ROOT / "external" / "alphazero" / "quoridor" / "pathFinder-module",
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_dir()]
    if missing_paths:
        raise ConfigError(f"missing runtime path(s): {', '.join(missing_paths)}")
    native_dir = required_paths[-1]
    native_suffix = ".pyd" if sys.platform == "win32" else ".so"
    native_modules = list(native_dir.glob(f"pathFinder*{native_suffix}"))
    if not native_modules:
        raise ConfigError(f"native pathFinder module not found in {native_dir}")


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigError(f"metrics file not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ConfigError(f"blank metrics line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid metrics JSON on line {line_number}: {exc}") from exc
        records.append(record)
    iterations = [record.get("iteration") for record in records]
    if iterations != list(range(1, len(records) + 1)):
        raise ConfigError(f"metrics iterations are not contiguous: {iterations}")
    return records


def load_run_resolved_config(run_dir: Path) -> dict[str, Any]:
    absolute_run_dir = run_dir.expanduser().resolve()
    path = absolute_run_dir / "resolved_config.yaml"
    try:
        loaded = load_yaml(path)
    except ConfigError as exc:
        raise ConfigError(f"could not load resolved config {path}: {exc}") from exc
    loaded["_output_path"] = absolute_run_dir
    return rebase_output_directory(loaded, absolute_run_dir)


def initialize_resolved_environment(resolved: dict[str, Any]) -> None:
    if resolved["run"]["device"] != "cuda" or not torch.cuda.is_available():
        raise ConfigError("CUDA is required by the resolved smoke configuration")
    gpu_index = int(resolved["run"]["gpu_index"])
    if gpu_index >= torch.cuda.device_count():
        raise ConfigError(
            f"run.gpu_index {gpu_index} is invalid; detected "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )
    torch.cuda.set_device(gpu_index)
    set_random_seed(
        int(resolved["run"]["seed"]),
        bool(resolved["run"]["deterministic"]),
    )


def critical_training_config(resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": {
            key: resolved["run"][key]
            for key in ("seed", "device", "gpu_index", "deterministic")
        },
        "self_play": resolved["self_play"],
        "training": resolved["training"],
        "checkpoint": {
            "save_every_iterations": resolved["checkpoint"]["save_every_iterations"]
        },
        "nn_args": resolved["mapped_args"]["nn_args"],
    }


def validate_resume_state(
    original: dict[str, Any],
    requested: dict[str, Any],
) -> tuple[int, Path]:
    if critical_training_config(original) != critical_training_config(requested):
        raise ConfigError("current config does not match the run's critical training parameters")

    output_dir = original["_output_path"]
    checkpoint_dir = Path(original["checkpoint"]["directory"])
    metrics = read_metrics(output_dir / "metrics.jsonl")
    if not metrics:
        raise ConfigError("resume requires at least one completed metrics record")
    last_iteration = int(metrics[-1]["iteration"])

    numbered = {}
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.iterdir():
            match = CHECKPOINT_PATTERN.match(path.name)
            if match:
                numbered[int(match.group(1))] = path
    expected_iterations = list(range(1, last_iteration + 1))
    if sorted(numbered) != expected_iterations:
        raise ConfigError(
            "numbered checkpoints do not match metrics iterations: "
            f"checkpoints={sorted(numbered)}, metrics={expected_iterations}"
        )
    checkpoint_path = numbered[last_iteration]
    if Path(metrics[-1].get("checkpoint_path", "")).name != checkpoint_path.name:
        raise ConfigError("last metrics record does not reference the matching checkpoint")

    examples_path = checkpoint_dir / "latest.examples"
    if not examples_path.is_file():
        raise ConfigError(f"training examples not found: {examples_path}")
    try:
        with examples_path.open("rb") as examples_file:
            examples = pickle.load(examples_file)
    except Exception as exc:
        raise ConfigError(f"could not load training examples: {exc}") from exc
    if not isinstance(examples, dict) or examples.get("iteration") != last_iteration:
        raise ConfigError(
            "latest.examples iteration does not match metrics/checkpoint iteration"
        )
    next_iteration = last_iteration + 1
    if next_iteration in numbered:
        raise ConfigError(f"next iteration checkpoint already exists: {numbered[next_iteration]}")
    return last_iteration, checkpoint_path


def create_runtime(resolved: dict[str, Any]):
    alphazero_root = PROJECT_ROOT / "external" / "alphazero"
    alphazero_path = str(alphazero_root)
    if alphazero_path not in sys.path:
        sys.path.insert(0, alphazero_path)

    from Coach import Coach
    from quoridor.QuoridorGame import QuoridorGame
    from quoridor.pytorch import NNet as nnet_module
    from quoridor.pytorch.NNet import NNetWrapper
    from utils import dotdict

    train_args = dotdict(copy.deepcopy(resolved["mapped_args"]["train_args"]))
    nn_args = copy.deepcopy(resolved["mapped_args"]["nn_args"])
    nnet_module.args.update(nn_args)

    game = QuoridorGame(9)
    nnet = NNetWrapper(game)
    return game, nnet, train_args, Coach


class _TeeStream:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.original.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.original.isatty()

    def fileno(self) -> int:
        return self.original.fileno()

    @property
    def encoding(self):
        return self.original.encoding


class RunLog:
    def __init__(
        self,
        path: Path,
        *,
        append: bool,
        mode: str,
        resolved: dict[str, Any],
    ):
        self.path = path
        self.append = append
        self.mode = mode
        self.resolved = resolved

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open(
            "a" if self.append else "w",
            encoding="utf-8",
            newline="\n",
        )
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._file)
        sys.stderr = _TeeStream(self._stderr, self._file)
        print(
            "[run] "
            f"mode={self.mode} "
            f"run_id={self.resolved['run']['id']} "
            f"output={self.resolved['_output_path'].as_posix()} "
            f"started_at_utc={datetime.now(timezone.utc).isoformat()}"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        status = "failed" if exc_type is not None else "finished"
        print(
            f"[run] mode={self.mode} status={status} "
            f"ended_at_utc={datetime.now(timezone.utc).isoformat()}"
        )
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()


def run_training(
    resolved: dict[str, Any],
    *,
    mode: str = "fresh",
    resume_checkpoint: Path | None = None,
    stop_after_iteration: int | None = None,
) -> tuple[Path, Path | None, Path]:
    game, nnet, train_args, Coach = create_runtime(resolved)
    total_iterations = resolved["self_play"]["iterations"]
    target_iteration = min(stop_after_iteration or total_iterations, total_iterations)
    train_args.numIters = target_iteration

    if mode == "resume":
        if resume_checkpoint is None:
            raise ConfigError("resume checkpoint was not supplied")
        train_args.load_model = True
        train_args.load_examples = True
        train_args.load_folder_file = [str(resume_checkpoint.parent), resume_checkpoint.name]
        load_model_checkpoint(nnet, resume_checkpoint)

    metrics_path = resolved["_output_path"] / "metrics.jsonl"
    with MetricsJsonlWriter(metrics_path, append=(mode == "resume")) as metrics_writer:
        coach = Coach(
            game,
            nnet,
            train_args,
            iteration_callback=metrics_writer,
        )
        if train_args.load_examples:
            coach.loadTrainExamples()
        coach.learn()

    scratch_checkpoint = Path(train_args.checkpoint) / "temp.pth.tar"
    if scratch_checkpoint.is_file():
        scratch_checkpoint.unlink()

    completed_iterations = len(read_metrics(metrics_path))
    stopped_early = completed_iterations < total_iterations
    evaluation_suppressed = stop_after_iteration is not None
    evaluation_path = None
    if (
        not stopped_early
        and not evaluation_suppressed
        and resolved["evaluation"]["enabled"]
    ):
        from evaluate import evaluate_checkpoint

        _, evaluation_path = evaluate_checkpoint(resolved, game, nnet)
    summary_path = write_summary(
        resolved,
        mode=mode,
        status="stopped" if stopped_early or evaluation_suppressed else "completed",
        evaluation_path=evaluation_path,
    )
    return metrics_path, evaluation_path, summary_path


def run_evaluate_only(
    resolved: dict[str, Any],
    checkpoint: Path | None,
) -> Path:
    game, nnet, _train_args, _Coach = create_runtime(resolved)
    from evaluate import evaluate_checkpoint

    _, output_path = evaluate_checkpoint(resolved, game, nnet, checkpoint=checkpoint)
    return output_path


def prepare_requested_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    resolved = resolve_config(config, args.config)
    if args.output_dir is not None:
        resolved = rebase_output_directory(resolved, args.output_dir)
    return resolved


def main() -> int:
    args = parse_args()
    try:
        check_runtime_dependencies()
        requested = (
            None
            if args.mode == "evaluate-only" and args.run_dir is not None
            else prepare_requested_config(args)
        )
        if (
            args.stop_after_iteration is not None
            and requested is not None
            and args.stop_after_iteration > requested["self_play"]["iterations"]
        ):
            raise ConfigError(
                "--stop-after-iteration cannot exceed configured iterations"
            )

        if args.mode == "dry-run":
            assert requested is not None
            requested["mode"] = "dry-run"
            conflicts = find_formal_artifacts(requested["_output_path"])
            print_resolved_config(requested)
            print(yaml.safe_dump(requested["mapped_args"], sort_keys=False, allow_unicode=True))
            if conflicts:
                print("Output conflicts detected (resolved config not overwritten):")
                for conflict in conflicts:
                    print(f"  - {conflict}")
            else:
                path = save_resolved_config(requested)
                print(f"Resolved config: {path}")
            return 0

        if args.mode == "fresh":
            assert requested is not None
            requested["mode"] = "smoke"
            conflicts = find_formal_artifacts(requested["_output_path"])
            if conflicts:
                joined = "\n  - ".join(str(path) for path in conflicts)
                raise ConfigError(f"fresh run would overwrite existing artifacts:\n  - {joined}")
            if requested["checkpoint"]["save_every_iterations"] != 1:
                raise ConfigError("fresh smoke runs require checkpoint.save_every_iterations: 1")
            save_resolved_config(requested)
            with RunLog(
                requested["_output_path"] / "run.log",
                append=False,
                mode="fresh",
                resolved=requested,
            ):
                print_resolved_config(requested)
                metrics_path, evaluation_path, summary_path = run_training(
                    requested,
                    mode="fresh",
                    stop_after_iteration=args.stop_after_iteration,
                )
                print(f"Metrics: {metrics_path}")
                if evaluation_path:
                    print(f"Evaluation: {evaluation_path}")
                print(f"Summary: {summary_path}")
            return 0

        if args.mode == "resume":
            assert requested is not None
            original = load_run_resolved_config(args.run_dir)
            last_iteration, checkpoint_path = validate_resume_state(original, requested)
            if last_iteration >= original["self_play"]["iterations"]:
                raise ConfigError("run already completed all configured iterations")
            if args.stop_after_iteration is not None and args.stop_after_iteration <= last_iteration:
                raise ConfigError(
                    f"--stop-after-iteration must be greater than completed iteration {last_iteration}"
                )
            with RunLog(
                original["_output_path"] / "run.log",
                append=True,
                mode="resume",
                resolved=original,
            ):
                print_resolved_config(original)
                metrics_path, evaluation_path, summary_path = run_training(
                    original,
                    mode="resume",
                    resume_checkpoint=checkpoint_path,
                    stop_after_iteration=args.stop_after_iteration,
                )
                print(f"Metrics: {metrics_path}")
                if evaluation_path:
                    print(f"Evaluation: {evaluation_path}")
                print(f"Summary: {summary_path}")
            return 0

        if args.run_dir is not None:
            resolved = load_run_resolved_config(args.run_dir)
            initialize_resolved_environment(resolved)
        else:
            assert requested is not None
            resolved = requested
        resolved["mode"] = "evaluate-only"
        checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
        with RunLog(
            resolved["_output_path"] / "run.log",
            append=(resolved["_output_path"] / "run.log").exists(),
            mode="evaluate-only",
            resolved=resolved,
        ):
            output_path = run_evaluate_only(resolved, checkpoint)
            print(f"Evaluation: {output_path}")
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
