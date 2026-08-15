from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_yaml(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("the YAML document must contain a top-level mapping")
    return loaded


def merge_defaults(
    defaults: dict[str, Any], supplied: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge supplied values over an independent copy of defaults."""
    if not isinstance(defaults, dict) or not isinstance(supplied, dict):
        raise ConfigError("defaults and supplied configuration must be mappings")
    merged = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _project_root(path: Path | str) -> Path:
    config_path = Path(path).expanduser().resolve()
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a mapping")
    return value


def _require_fields(section: dict[str, Any], name: str, fields: set[str]) -> None:
    missing = sorted(fields - set(section))
    if missing:
        qualified = ", ".join(f"{name}.{field}" for field in missing)
        raise ConfigError(f"missing required field(s): {qualified}")


def _reject_unknown(section: dict[str, Any], name: str, allowed: set[str]) -> None:
    unknown = sorted(
        key for key in set(section) - allowed if not str(key).startswith("_")
    )
    if unknown:
        raise ConfigError(f"unknown field(s) in {name}: {', '.join(unknown)}")


def _require_bool(section: dict[str, Any], field: str, section_name: str) -> bool:
    value = section.get(field)
    if not isinstance(value, bool):
        raise ConfigError(f"{section_name}.{field} must be true or false")
    return value


def _require_positive_int(
    section: dict[str, Any], field: str, section_name: str, *, nullable: bool = False
) -> int | None:
    value = section.get(field)
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        suffix = " or null" if nullable else ""
        raise ConfigError(f"{section_name}.{field} must be an integer >= 1{suffix}")
    return value


def _validate_batch_configuration(
    training: dict[str, Any],
    section_name: str,
) -> None:
    batch_size = _require_positive_int(
        training, "batch_size", section_name
    )
    micro_batch_size = _require_positive_int(
        training, "micro_batch_size", section_name
    )

    if micro_batch_size > batch_size:
        raise ConfigError(
            f"{section_name}.micro_batch_size must be <= "
            f"{section_name}.batch_size"
        )

    if batch_size % micro_batch_size != 0:
        raise ConfigError(
            f"{section_name}.batch_size must be divisible by "
            f"{section_name}.micro_batch_size"
        )


def _require_nonnegative_number(
    section: dict[str, Any], field: str, section_name: str, *, nullable: bool = False
) -> float | None:
    value = section.get(field)
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        suffix = " or null" if nullable else ""
        raise ConfigError(f"{section_name}.{field} must be a non-negative number{suffix}")
    return float(value)


def _require_string(section: dict[str, Any], field: str, section_name: str) -> str:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section_name}.{field} must be a non-empty string")
    return value.strip()


def _validate_run(run: dict[str, Any]) -> None:
    _require_fields(run, "run", {"id", "seed", "device", "gpu_index", "deterministic", "output_dir"})
    _require_string(run, "id", "run")
    seed = run["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ConfigError("run.seed must be an integer between 0 and 4294967295")
    _require_string(run, "device", "run")
    gpu_index = run["gpu_index"]
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ConfigError("run.gpu_index must be an integer >= 0")
    _require_bool(run, "deterministic", "run")
    _require_string(run, "output_dir", "run")


def _validate_model(model: dict[str, Any]) -> None:
    _require_fields(
        model,
        "model",
        {
            "board_size",
            "num_channels",
            "num_res_blocks",
            "attn_depth",
            "num_heads",
            "se_enabled",
            "dropout",
        },
    )
    for field in ("board_size", "num_channels", "num_res_blocks", "num_heads"):
        _require_positive_int(model, field, "model")
    attn_depth = model["attn_depth"]
    if isinstance(attn_depth, bool) or not isinstance(attn_depth, int) or attn_depth < 0:
        raise ConfigError("model.attn_depth must be an integer >= 0")
    _require_bool(model, "se_enabled", "model")
    dropout = _require_nonnegative_number(model, "dropout", "model")
    if dropout is not None and dropout > 1:
        raise ConfigError("model.dropout must be <= 1")


def _validate_logging(logging: dict[str, Any]) -> None:
    for field in ("metrics_file", "metadata_file", "evaluation_file", "summary_file"):
        _require_string(logging, field, "logging")


def _annotate_paths(config: dict[str, Any], path: Path | str) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    config_path = Path(path).expanduser().resolve()
    project_root = _project_root(config_path)
    output = Path(resolved["run"]["output_dir"])
    output_path = output.resolve() if output.is_absolute() else (project_root / output).resolve()
    resolved["_config_path"] = config_path
    resolved["_project_root"] = project_root
    resolved["_output_path"] = output_path
    return resolved


def resolve_pretraining_config(
    config: dict[str, Any], path: Path | str
) -> dict[str, Any]:
    resolved = _annotate_paths(config, path)
    project_root = resolved["_project_root"]
    data = _require_mapping(resolved, "data")
    for field in ("archive_path", "extracted_path"):
        requested = Path(_require_string(data, field, "data"))
        data[f"_{field}"] = (
            requested.resolve() if requested.is_absolute() else (project_root / requested).resolve()
        )
    checkpoint = _require_mapping(resolved, "checkpoint")
    checkpoint_dir = Path(_require_string(checkpoint, "directory", "checkpoint"))
    checkpoint["_directory_path"] = (
        checkpoint_dir.resolve()
        if checkpoint_dir.is_absolute()
        else (project_root / checkpoint_dir).resolve()
    )
    validate_pretraining_config(resolved)
    resolved["mapped_args"] = {
        "nn_args": map_model_to_nn_args(resolved),
    }
    return resolved


def resolve_baseline_config(
    config: dict[str, Any], path: Path | str
) -> dict[str, Any]:
    resolved = _annotate_paths(config, path)
    project_root = resolved["_project_root"]
    initialization = _require_mapping(resolved, "initialization")
    checkpoint_path = Path(_require_string(initialization, "checkpoint_path", "initialization"))
    initialization["_checkpoint_path"] = (
        checkpoint_path.resolve()
        if checkpoint_path.is_absolute()
        else (project_root / checkpoint_path).resolve()
    )
    checkpoint = _require_mapping(resolved, "checkpoint")
    checkpoint_dir = Path(_require_string(checkpoint, "directory", "checkpoint"))
    checkpoint["_directory_path"] = (
        checkpoint_dir.resolve()
        if checkpoint_dir.is_absolute()
        else (project_root / checkpoint_dir).resolve()
    )
    validate_baseline_config(resolved)
    resolved["mapped_args"] = {
        "train_args": map_baseline_to_train_args(resolved),
        "nn_args": map_model_to_nn_args(resolved),
        "evaluation_args": copy.deepcopy(resolved["evaluation"]),
    }
    return resolved


def validate_pretraining_config(resolved: dict[str, Any]) -> None:
    allowed = {"run", "data", "model", "pretraining", "checkpoint", "logging"}
    public_sections = {key for key in resolved if not key.startswith("_") and key != "mapped_args"}
    unknown = sorted(public_sections - allowed)
    if unknown:
        raise ConfigError(f"pretraining configuration contains forbidden section(s): {', '.join(unknown)}")
    run = _require_mapping(resolved, "run")
    data = _require_mapping(resolved, "data")
    model = _require_mapping(resolved, "model")
    training = _require_mapping(resolved, "pretraining")
    checkpoint = _require_mapping(resolved, "checkpoint")
    logging = _require_mapping(resolved, "logging")
    _validate_run(run)
    _validate_model(model)
    _reject_unknown(run, "run", {"id", "seed", "device", "gpu_index", "deterministic", "output_dir"})
    _reject_unknown(model, "model", {"board_size", "num_channels", "num_res_blocks", "attn_depth", "num_heads", "se_enabled", "dropout"})
    _reject_unknown(data, "data", {"archive_path", "extracted_path", "expected_sha256", "validate_before_training"})
    _reject_unknown(training, "pretraining", {"epochs", "batch_size", "micro_batch_size", "learning_rate", "optimizer", "weight_decay", "gradient_clip", "amp", "amp_dtype"})
    _reject_unknown(checkpoint, "checkpoint", {"directory", "checkpoint_0_filename", "best_filename", "save_rng_state", "compute_sha256"})
    _reject_unknown(logging, "logging", {"metrics_file", "metadata_file", "evaluation_file", "summary_file", "log_every_optimizer_steps"})
    _require_fields(data, "data", {"archive_path", "extracted_path", "expected_sha256", "validate_before_training"})
    digest = data["expected_sha256"]
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest.lower()):
        raise ConfigError("data.expected_sha256 must be a 64-character SHA-256 digest")
    _require_bool(data, "validate_before_training", "data")
    _require_fields(training, "pretraining", {"epochs", "batch_size", "micro_batch_size", "learning_rate", "optimizer", "weight_decay", "gradient_clip", "amp", "amp_dtype"})
    _require_positive_int(training, "epochs", "pretraining")
    _validate_batch_configuration(training, "pretraining")
    _require_nonnegative_number(training, "learning_rate", "pretraining")
    if _require_string(training, "optimizer", "pretraining").lower() != "adamw":
        raise ConfigError("pretraining.optimizer must be adamw")
    _require_nonnegative_number(training, "weight_decay", "pretraining")
    _require_nonnegative_number(training, "gradient_clip", "pretraining")
    _require_bool(training, "amp", "pretraining")
    if _require_string(training, "amp_dtype", "pretraining") not in {"fp16", "bf16"}:
        raise ConfigError("pretraining.amp_dtype must be fp16 or bf16")
    for field in ("directory", "checkpoint_0_filename", "best_filename"):
        _require_string(checkpoint, field, "checkpoint")
    if checkpoint["checkpoint_0_filename"] != "checkpoint_0.pth.tar":
        raise ConfigError("checkpoint.checkpoint_0_filename must be checkpoint_0.pth.tar")
    if checkpoint["best_filename"] != "best.pth.tar":
        raise ConfigError("checkpoint.best_filename must be best.pth.tar")
    for field in ("save_rng_state", "compute_sha256"):
        _require_bool(checkpoint, field, "checkpoint")
    _validate_logging(logging)
    _require_positive_int(logging, "log_every_optimizer_steps", "logging")


def validate_baseline_config(resolved: dict[str, Any]) -> None:
    required_sections = {
        "run", "initialization", "model", "budget", "self_play", "training",
        "replay", "checkpoint", "instrumentation", "logging", "evaluation",
    }
    missing = sorted(required_sections - set(resolved))
    if missing:
        raise ConfigError(f"missing required section(s): {', '.join(missing)}")
    public_sections = {key for key in resolved if not key.startswith("_") and key != "mapped_args"}
    unknown_sections = sorted(public_sections - required_sections)
    if unknown_sections:
        raise ConfigError(f"unknown top-level section(s): {', '.join(unknown_sections)}")
    run = _require_mapping(resolved, "run")
    initialization = _require_mapping(resolved, "initialization")
    model = _require_mapping(resolved, "model")
    budget = _require_mapping(resolved, "budget")
    self_play = _require_mapping(resolved, "self_play")
    training = _require_mapping(resolved, "training")
    replay = _require_mapping(resolved, "replay")
    checkpoint = _require_mapping(resolved, "checkpoint")
    instrumentation = _require_mapping(resolved, "instrumentation")
    logging = _require_mapping(resolved, "logging")
    evaluation = _require_mapping(resolved, "evaluation")
    _validate_run(run)
    _validate_model(model)
    _reject_unknown(run, "run", {"id", "seed", "device", "gpu_index", "deterministic", "output_dir"})
    _reject_unknown(initialization, "initialization", {"mode", "checkpoint_path", "expected_sha256", "load_weights", "load_replay", "load_optimizer_state", "load_tracker_state", "start_iteration"})
    _reject_unknown(model, "model", {"board_size", "num_channels", "num_res_blocks", "attn_depth", "num_heads", "se_enabled", "dropout"})
    _reject_unknown(budget, "budget", {"max_gpu_hours", "max_wall_clock_hours", "max_iterations"})
    _reject_unknown(self_play, "self_play", {"iterations", "games_per_iteration", "mcts_simulations", "eval_mcts_in_batch", "temperature_threshold", "cpuct", "dirichlet_noise", "dirichlet_alpha", "dirichlet_epsilon", "max_game_length"})
    _reject_unknown(training, "training", {"epochs", "batch_size", "micro_batch_size", "learning_rate", "optimizer", "weight_decay", "gradient_clip", "amp", "amp_dtype", "update_gating"})
    _reject_unknown(replay, "replay", {"max_queue_size", "max_train_samples", "history_iterations"})
    _reject_unknown(checkpoint, "checkpoint", {"directory", "save_every_iterations", "save_replay_state", "save_instrumentation_state", "save_rng_state", "compute_sha256"})
    _reject_unknown(instrumentation, "instrumentation", {"enabled", "tracker_file", "track_iteration_timing", "track_resource_usage", "persist_tracker_state", "verify_resume_continuity", "verify_evaluation_read_only", "measure_overhead"})
    _reject_unknown(logging, "logging", {"metrics_file", "metadata_file", "evaluation_file", "summary_file"})
    _reject_unknown(evaluation, "evaluation", {"enabled", "evaluate_every_iterations", "games_per_opponent", "opponents", "model_mcts_simulations", "eval_mcts_in_batch", "temperature", "dirichlet_noise", "max_game_length", "alternate_starting_player"})
    _require_fields(initialization, "initialization", {"mode", "checkpoint_path", "expected_sha256", "load_weights", "load_replay", "load_optimizer_state", "load_tracker_state", "start_iteration"})
    if initialization["mode"] != "pretrained_checkpoint":
        raise ConfigError("initialization.mode must be pretrained_checkpoint")
    for field in ("load_weights", "load_replay", "load_optimizer_state", "load_tracker_state"):
        _require_bool(initialization, field, "initialization")
    if not initialization["load_weights"] or any(initialization[field] for field in ("load_replay", "load_optimizer_state", "load_tracker_state")):
        raise ConfigError("pretrained initialization must load only model weights")
    if initialization["start_iteration"] != 1:
        raise ConfigError("initialization.start_iteration must be 1")
    expected = initialization["expected_sha256"]
    if expected is not None and (not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected.lower())):
        raise ConfigError("initialization.expected_sha256 must be null or a SHA-256 digest")
    for field in ("max_gpu_hours", "max_wall_clock_hours"):
        if field in budget:
            _require_nonnegative_number(budget, field, "budget", nullable=True)
    _require_positive_int(budget, "max_iterations", "budget", nullable=True)
    _require_positive_int(self_play, "iterations", "self_play", nullable=True)
    for field in ("games_per_iteration", "mcts_simulations", "eval_mcts_in_batch", "max_game_length"):
        _require_positive_int(self_play, field, "self_play")
    _require_positive_int(self_play, "temperature_threshold", "self_play")
    if self_play["mcts_simulations"] % self_play["eval_mcts_in_batch"]:
        raise ConfigError("self_play.mcts_simulations must be divisible by eval_mcts_in_batch")
    for field in ("cpuct", "dirichlet_alpha", "dirichlet_epsilon"):
        _require_nonnegative_number(self_play, field, "self_play")
    _require_bool(self_play, "dirichlet_noise", "self_play")
    _require_positive_int(training, "epochs", "training")
    _validate_batch_configuration(training, "training")
    for field in ("learning_rate", "weight_decay", "gradient_clip"):
        _require_nonnegative_number(training, field, "training")
    if _require_string(training, "optimizer", "training").lower() != "adamw":
        raise ConfigError("training.optimizer must be adamw")
    _require_bool(training, "amp", "training")
    _require_bool(training, "update_gating", "training")
    if training["update_gating"]:
        raise ConfigError("baseline training.update_gating must be false")
    if _require_string(training, "amp_dtype", "training") not in {"fp16", "bf16"}:
        raise ConfigError("training.amp_dtype must be fp16 or bf16")
    for field in ("max_queue_size", "max_train_samples", "history_iterations"):
        _require_positive_int(replay, field, "replay")
    if replay["max_train_samples"] < training["batch_size"]:
        raise ConfigError("replay.max_train_samples must be >= training.batch_size")
    _require_string(checkpoint, "directory", "checkpoint")
    _require_positive_int(checkpoint, "save_every_iterations", "checkpoint", nullable=True)
    for field in ("save_replay_state", "save_instrumentation_state", "save_rng_state", "compute_sha256"):
        _require_bool(checkpoint, field, "checkpoint")
    for field, value in instrumentation.items():
        if field == "tracker_file":
            _require_string(instrumentation, field, "instrumentation")
        elif not isinstance(value, bool):
            raise ConfigError(f"instrumentation.{field} must be true or false")
    _validate_logging(logging)
    _require_bool(evaluation, "enabled", "evaluation")
    _require_positive_int(evaluation, "evaluate_every_iterations", "evaluation", nullable=True)
    games = _require_positive_int(evaluation, "games_per_opponent", "evaluation", nullable=True)
    if games is not None and games % 2:
        raise ConfigError("evaluation.games_per_opponent must be even")
    opponents = evaluation.get("opponents")
    if not isinstance(opponents, list) or not opponents or not all(isinstance(item, str) and item.strip() for item in opponents):
        raise ConfigError("evaluation.opponents must be a non-empty list of strings")
    model_mcts = _require_positive_int(evaluation, "model_mcts_simulations", "evaluation")
    eval_batch = _require_positive_int(evaluation, "eval_mcts_in_batch", "evaluation")
    if model_mcts is not None and eval_batch is not None and model_mcts % eval_batch:
        raise ConfigError("evaluation.model_mcts_simulations must be divisible by eval_mcts_in_batch")
    _require_nonnegative_number(evaluation, "temperature", "evaluation")
    _require_positive_int(evaluation, "max_game_length", "evaluation")
    _require_bool(evaluation, "dirichlet_noise", "evaluation")
    _require_bool(evaluation, "alternate_starting_player", "evaluation")


def map_baseline_to_train_args(resolved: dict[str, Any]) -> dict[str, Any]:
    run = resolved["run"]
    self_play = resolved["self_play"]
    training = resolved["training"]
    replay = resolved.get("replay", {})
    checkpoint = resolved["checkpoint"]
    iterations = self_play.get("iterations")
    if iterations is None:
        iterations = resolved.get("budget", {}).get("max_iterations")
    max_queue = replay.get("max_queue_size", training.get("max_queue_size"))
    max_samples = replay.get("max_train_samples", training.get("max_samples"))
    history = replay.get("history_iterations", training.get("replay_history_iterations"))
    update_gating = bool(training.get("update_gating", False))
    update_threshold = training.get("update_threshold", -0.51)
    micro_batch_size = training.get("micro_batch_size", training["batch_size"])
    resolved_checkpoint_dir = checkpoint.get("_directory_path", checkpoint["directory"])
    checkpoint_dir = (
        resolved_checkpoint_dir.as_posix()
        if isinstance(resolved_checkpoint_dir, Path)
        else str(resolved_checkpoint_dir)
    )
    return {
        "exp_name": run["id"],
        "seed": run["seed"],
        "numIters": iterations,
        "numEps": self_play["games_per_iteration"],
        "numMCTSSims": self_play["mcts_simulations"],
        "tempThreshold": self_play["temperature_threshold"],
        "cpuct": self_play["cpuct"],
        "dirichlet_alpha": self_play["dirichlet_alpha"],
        "dirichlet_epsilon": self_play["dirichlet_epsilon"] if self_play.get("dirichlet_noise", True) else 0.0,
        "max_game_length": self_play["max_game_length"],
        "eval_mcts_in_batch": self_play["eval_mcts_in_batch"],
        "maxlenOfQueue": max_queue,
        "max_train_size": max_samples,
        "batch_size": training["batch_size"],
        "micro_batch_size": micro_batch_size,
        "lr": training["learning_rate"],
        "checkpoint": checkpoint_dir,
        "save_every_n_iterations": checkpoint["save_every_iterations"],
        "load_model": False,
        "load_examples": False,
        "load_folder_file": [checkpoint_dir, "best.pth.tar"],
        "load_folder_examples_file": [checkpoint_dir, "latest.examples"],
        "numItersForTrainExamplesHistory": history,
        "arenaCompare": 4 if update_gating else 0,
        "updateThreshold": update_threshold if update_gating else -0.51,
        "print_summary": True,
        "expert_examples_data": None,
        "fill_with_expert_data": False,
        "heuristic_alpha": 0.0,
        "heuristic_decay_iters": 1,
        "heuristic_rollouts": 0,
    }


def map_model_to_nn_args(resolved: dict[str, Any]) -> dict[str, Any]:
    training = resolved.get("training", resolved.get("pretraining", {}))
    model = resolved.get("model", training)
    run = resolved["run"]
    micro_batch_size = training.get("micro_batch_size", training["batch_size"])
    return {
        "cuda": run.get("device", "cuda") == "cuda",
        "lr": training["learning_rate"],
        "dropout": model.get("dropout", 0.3),
        "weight_decay": training.get("weight_decay", 0.0001),
        "lr_decay_gamma": 1.0,
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": (
            training["batch_size"] // micro_batch_size
        ),
        "num_channels": model["num_channels"],
        "num_res_blocks": model["num_res_blocks"],
        "attn_depth": model["attn_depth"],
        "num_heads": model["num_heads"],
        "se_enabled": model.get("se_enabled", False),
        "fast_opts": training.get("fast_opts", False),
        "clip": training.get("gradient_clip", 1.0),
        "use_amp": training.get("amp", False),
        "amp_dtype": training.get("amp_dtype", "bf16"),
    }
