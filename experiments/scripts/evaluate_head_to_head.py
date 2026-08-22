#!/usr/bin/env python3
"""Evaluate final common-horizon Baseline and Adaptive checkpoints head-to-head."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
BASELINE_ROOT = SOURCE_ROOT / "baseline"
BASELINE_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for import_root in (SOURCE_ROOT, BASELINE_ROOT, BASELINE_SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from arena.arena import MatchResult, play_game  # noqa: E402
from evaluate_fixed_basket import (  # noqa: E402
    EvaluationLogger,
    ScheduledTemperatureAlphaZeroBot,
    _cleanup_bot,
    _prepare_bot,
    _serialize_moves,
)
from head_to_head_stats import (  # noqa: E402
    HeadToHeadStatsError,
    colour_stratified_bootstrap,
    paired_seed,
    stable_game_key,
)


DEFAULT_CONFIG = EXPERIMENTS_ROOT / "configs" / "head_to_head_v2.yaml"
DEFAULT_OUTPUT = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "adaptive_seed1001_4090_v2_analysis"
    / "head_to_head_v2"
)

PAIR_FIELDS = (
    "seed_pair_index",
    "game_seed",
    "baseline_iteration",
    "baseline_gpu_hours",
    "baseline_checkpoint_sha256",
    "adaptive_iteration",
    "adaptive_gpu_hours",
    "adaptive_checkpoint_sha256",
    "adaptive_white_game_key",
    "adaptive_white_result",
    "adaptive_black_game_key",
    "adaptive_black_result",
    "pair_complete",
)

SUMMARY_FIELDS = (
    "common_horizon_gpu_hours",
    "baseline_iteration",
    "baseline_gpu_hours",
    "baseline_checkpoint_sha256",
    "adaptive_iteration",
    "adaptive_gpu_hours",
    "adaptive_checkpoint_sha256",
    "technically_valid_games",
    "adaptive_white_games",
    "adaptive_black_games",
    "adaptive_wins",
    "draws",
    "adaptive_losses",
    "adaptive_score_rate",
    "adaptive_score_rate_ci95_low",
    "adaptive_score_rate_ci95_high",
    "bootstrap_method",
    "bootstrap_strata",
    "bootstrap_iterations",
    "bootstrap_seed",
    "h3_head_to_head_support",
    "status",
)


class HeadToHeadError(ValueError):
    """Raised when a head-to-head v2 invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadToHeadError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), f"{label} must contain a mapping")
    return loaded


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HeadToHeadError(f"invalid {label}: {exc}") from exc
    _require(isinstance(loaded, dict), f"{label} must contain an object")
    return loaded


def _load_jsonl(path: Path, label: str, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists() and not required:
        return []
    _require(path.is_file(), f"{label} not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HeadToHeadError(
                    f"invalid JSON in {label} line {line_number}: {exc}"
                ) from exc
            _require(isinstance(record, dict), f"{label} line {line_number} is not an object")
            records.append(record)
    return records


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path, rows: list[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl_fsync(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(encoded + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_experiments_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (EXPERIMENTS_ROOT / path).resolve()


def _verified_hash(path: Path, expected: Any, label: str) -> str:
    _require(path.is_file(), f"{label} not found: {path}")
    actual = sha256_file(path)
    if expected is not None:
        _require(
            isinstance(expected, str) and actual == expected.lower(),
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}",
        )
    return actual


def common_horizon_and_baseline_grid(
    matched: dict[str, Any],
) -> tuple[float, dict[int, float]]:
    try:
        targets = matched["pairing_and_randomness"]["checkpoint_grid"]["targets"]
    except (KeyError, TypeError) as exc:
        raise HeadToHeadError("matched-compute checkpoint grid is missing") from exc
    _require(isinstance(targets, list) and targets, "matched-compute checkpoint grid is empty")
    grid: dict[int, float] = {}
    previous = -math.inf
    for index, target in enumerate(targets):
        _require(isinstance(target, dict), f"matched target {index} is invalid")
        iteration = target.get("baseline_checkpoint_iteration")
        gpu_hours = target.get("gpu_hours")
        _require(_is_int(iteration), f"matched target {index} iteration is invalid")
        _require(_finite_number(gpu_hours), f"matched target {iteration} GPU-hours are invalid")
        hours = float(gpu_hours)
        _require(hours > previous, "matched GPU-hour grid is not strictly increasing")
        _require(int(iteration) not in grid, f"duplicate matched Baseline iteration {iteration}")
        grid[int(iteration)] = hours
        previous = hours
    return float(targets[-1]["gpu_hours"]), grid


def _checkpoint_entries(
    payload: dict[str, Any],
    *,
    condition: str,
    baseline_grid: dict[int, float],
) -> list[dict[str, Any]]:
    raw = payload.get("checkpoints")
    _require(isinstance(raw, list) and raw, f"{condition} checkpoint manifest is empty")
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        _require(isinstance(item, dict), f"{condition} checkpoint entry {index} is invalid")
        iteration = item.get("iteration")
        digest = item.get("sha256")
        recorded_path = item.get("path")
        _require(_is_int(iteration) and iteration >= 0, f"{condition} checkpoint iteration is invalid")
        _require(iteration not in seen, f"duplicate {condition} checkpoint iteration {iteration}")
        _require(isinstance(digest, str) and len(digest) == 64, f"{condition} checkpoint {iteration} SHA is invalid")
        _require(isinstance(recorded_path, str) and recorded_path, f"{condition} checkpoint {iteration} path is invalid")
        gpu_hours = (
            baseline_grid.get(int(iteration))
            if condition == "baseline"
            else item.get("actual_gpu_hours")
        )
        _require(
            _finite_number(gpu_hours) and float(gpu_hours) >= 0.0,
            f"{condition} checkpoint {iteration} lacks valid GPU-hours",
        )
        seen.add(int(iteration))
        entries.append(
            {
                "condition": condition,
                "iteration": int(iteration),
                "actual_gpu_hours": float(gpu_hours),
                "sha256": digest.lower(),
                "recorded_path": recorded_path,
            }
        )
    entries.sort(key=lambda item: (item["actual_gpu_hours"], item["iteration"]))
    return entries


def select_final_checkpoint(
    entries: list[dict[str, Any]], common_horizon: float
) -> dict[str, Any]:
    """Select the latest checkpoint on the GPU-hour axis without outcome access."""
    eligible = [item for item in entries if item["actual_gpu_hours"] <= common_horizon]
    _require(bool(eligible), "no checkpoint is available by the common horizon")
    return max(eligible, key=lambda item: (item["actual_gpu_hours"], item["iteration"]))


def _resolve_checkpoint_path(item: dict[str, Any], run_root: Path) -> Path:
    recorded = Path(item["recorded_path"])
    candidates = [recorded, run_root / "checkpoints" / recorded.name]
    if item["iteration"] == 0:
        candidates.append(
            SOURCE_ROOT
            / "baseline"
            / "outputs"
            / "pretraining_reproduction_seed1001"
            / "checkpoints"
            / recorded.name
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise HeadToHeadError(
        f"{item['condition']} checkpoint {item['iteration']} cannot be resolved below {run_root}"
    )


@dataclass(frozen=True)
class HeadToHeadContext:
    config_path: Path
    config: dict[str, Any]
    matched_path: Path
    matched: dict[str, Any]
    common_horizon: float
    baseline_manifest_path: Path
    adaptive_manifest_path: Path
    baseline_resolved_path: Path
    adaptive_resolved_path: Path
    baseline: dict[str, Any]
    adaptive: dict[str, Any]
    board_size: int
    output_dir: Path

    @property
    def attempts_path(self) -> Path:
        return self.output_dir / "attempts.jsonl"

    @property
    def games_path(self) -> Path:
        return self.output_dir / "games.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "evaluation_manifest.json"


def resolve_context(args: argparse.Namespace) -> HeadToHeadContext:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path, "head-to-head protocol")
    _require(config.get("config_id") == "head_to_head_v2", "config_id must be head_to_head_v2")
    conditions = config.get("conditions")
    _require(isinstance(conditions, dict), "conditions are missing")
    baseline_config = conditions.get("baseline")
    adaptive_config = conditions.get("adaptive")
    _require(isinstance(baseline_config, dict), "Baseline condition is missing")
    _require(isinstance(adaptive_config, dict), "Adaptive condition is missing")

    matched_path = (
        args.matched_compute.expanduser().resolve()
        if args.matched_compute is not None
        else _resolve_experiments_path(config["matched_compute"]["path"])
    )
    _verified_hash(
        matched_path,
        config["matched_compute"].get("expected_sha256"),
        "matched-compute protocol",
    )
    matched = _load_yaml(matched_path, "matched-compute protocol")
    horizon, baseline_grid = common_horizon_and_baseline_grid(matched)

    baseline_run_root = (
        args.baseline_run_dir.expanduser().resolve()
        if args.baseline_run_dir is not None
        else _resolve_experiments_path(baseline_config["run_root"])
    )
    adaptive_run_root = (
        args.adaptive_run_dir.expanduser().resolve()
        if args.adaptive_run_dir is not None
        else _resolve_experiments_path(adaptive_config["run_root"])
    )
    baseline_manifest_path = (
        args.baseline_checkpoint_manifest.expanduser().resolve()
        if args.baseline_checkpoint_manifest is not None
        else _resolve_experiments_path(baseline_config["checkpoint_manifest"])
    )
    adaptive_manifest_path = (
        args.adaptive_checkpoint_manifest.expanduser().resolve()
        if args.adaptive_checkpoint_manifest is not None
        else _resolve_experiments_path(adaptive_config["checkpoint_manifest"])
    )
    _verified_hash(
        baseline_manifest_path,
        baseline_config.get("checkpoint_manifest_expected_sha256"),
        "Baseline checkpoint manifest",
    )
    _verified_hash(
        adaptive_manifest_path,
        adaptive_config.get("checkpoint_manifest_expected_sha256"),
        "Adaptive checkpoint manifest",
    )
    baseline_manifest = _load_json(baseline_manifest_path, "Baseline checkpoint manifest")
    adaptive_manifest = _load_json(adaptive_manifest_path, "Adaptive checkpoint manifest")
    baseline = select_final_checkpoint(
        _checkpoint_entries(
            baseline_manifest, condition="baseline", baseline_grid=baseline_grid
        ),
        horizon,
    )
    adaptive = select_final_checkpoint(
        _checkpoint_entries(
            adaptive_manifest, condition="adaptive", baseline_grid=baseline_grid
        ),
        horizon,
    )
    baseline_path = _resolve_checkpoint_path(baseline, baseline_run_root)
    adaptive_path = _resolve_checkpoint_path(adaptive, adaptive_run_root)
    baseline["path"] = baseline_path.as_posix()
    adaptive["path"] = adaptive_path.as_posix()
    _verified_hash(baseline_path, baseline["sha256"], "selected Baseline checkpoint")
    _verified_hash(adaptive_path, adaptive["sha256"], "selected Adaptive checkpoint")
    _require(baseline["actual_gpu_hours"] <= horizon, "selected Baseline checkpoint exceeds common horizon")
    _require(adaptive["actual_gpu_hours"] <= horizon, "selected Adaptive checkpoint exceeds common horizon")

    baseline_resolved_path = _resolve_experiments_path(baseline_config["resolved_config"])
    adaptive_resolved_path = _resolve_experiments_path(adaptive_config["resolved_config"])
    baseline_resolved = _load_yaml(baseline_resolved_path, "Baseline resolved config")
    adaptive_resolved = _load_yaml(adaptive_resolved_path, "Adaptive resolved config")
    try:
        baseline_board_size = int(baseline_resolved["model"]["board_size"])
        adaptive_board_size = int(adaptive_resolved["model"]["board_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HeadToHeadError("condition resolved config lacks model.board_size") from exc
    _require(baseline_board_size == adaptive_board_size, "condition board sizes differ")
    _require(baseline_board_size > 0, "board size must be positive")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_experiments_path(config["outputs"]["root"])
    )
    return HeadToHeadContext(
        config_path=config_path,
        config=config,
        matched_path=matched_path,
        matched=matched,
        common_horizon=horizon,
        baseline_manifest_path=baseline_manifest_path,
        adaptive_manifest_path=adaptive_manifest_path,
        baseline_resolved_path=baseline_resolved_path,
        adaptive_resolved_path=adaptive_resolved_path,
        baseline=baseline,
        adaptive=adaptive,
        board_size=baseline_board_size,
        output_dir=output_dir,
    )


def validate_protocol(config: dict[str, Any]) -> None:
    games = config.get("games")
    model = config.get("model_protocol")
    selection = config.get("model_selection")
    analysis = config.get("analysis")
    _require(isinstance(games, dict), "games protocol is missing")
    _require(isinstance(model, dict), "model protocol is missing")
    _require(isinstance(selection, dict), "model-selection protocol is missing")
    _require(isinstance(analysis, dict), "analysis protocol is missing")
    _require(games.get("seed_pairs") == 50, "head-to-head requires 50 seed pairs")
    _require(games.get("games_per_seed") == 2, "head-to-head requires two games per seed")
    _require(games.get("technically_valid_games") == 100, "head-to-head requires 100 valid games")
    _require(games.get("max_turns") == 150, "max_turns must be 150")
    required_model = {
        "use_mcts": True,
        "mcts_simulations": 200,
        "eval_mcts_in_batch": 4,
        "cpuct": 1.25,
        "temperature": 0.0,
        "dirichlet_noise": False,
        "clear_tree_each_move": True,
        "reset_tree_before_each_game": True,
        "identical_for_both_conditions": True,
    }
    for field, expected in required_model.items():
        _require(model.get(field) == expected, f"model_protocol.{field} must be {expected!r}")
    _require(selection.get("rule") == "latest_completed_checkpoint_not_after_common_horizon", "checkpoint selection rule is invalid")
    _require(selection.get("allow_best_checkpoint") is False, "best-checkpoint selection must be forbidden")
    _require(selection.get("interpolation") == "forbidden", "checkpoint interpolation must be forbidden")
    bootstrap = analysis.get("bootstrap")
    _require(isinstance(bootstrap, dict), "bootstrap protocol is missing")
    _require(bootstrap.get("method") == "nonparametric_stratified_bootstrap", "bootstrap method is invalid")
    _require(bootstrap.get("strata") == ["adaptive_colour"], "bootstrap must be stratified by Adaptive colour")
    _require(bootstrap.get("resamples") == 10000, "bootstrap resamples must be 10,000")


def build_tasks(context: HeadToHeadContext) -> list[dict[str, Any]]:
    validate_protocol(context.config)
    games = context.config["games"]
    base_seed = int(games["seed"]["base_seed"])
    tasks: list[dict[str, Any]] = []
    for seed_pair_index in range(int(games["seed_pairs"])):
        game_seed = paired_seed(
            context.config["config_id"],
            base_seed,
            context.baseline["sha256"],
            context.adaptive["sha256"],
            seed_pair_index,
        )
        for adaptive_color in ("white", "black"):
            baseline_color = "black" if adaptive_color == "white" else "white"
            key = stable_game_key(
                context.baseline["sha256"],
                context.adaptive["sha256"],
                seed_pair_index,
                adaptive_color,
            )
            tasks.append(
                {
                    "stable_game_key": key,
                    "seed_pair_index": seed_pair_index,
                    "game_seed": game_seed,
                    "baseline_color": baseline_color,
                    "adaptive_color": adaptive_color,
                    "baseline_iteration": context.baseline["iteration"],
                    "baseline_gpu_hours": context.baseline["actual_gpu_hours"],
                    "baseline_checkpoint_path": context.baseline["path"],
                    "baseline_checkpoint_sha256": context.baseline["sha256"],
                    "adaptive_iteration": context.adaptive["iteration"],
                    "adaptive_gpu_hours": context.adaptive["actual_gpu_hours"],
                    "adaptive_checkpoint_path": context.adaptive["path"],
                    "adaptive_checkpoint_sha256": context.adaptive["sha256"],
                }
            )
    _require(len(tasks) == 100, "head-to-head task manifest is not exactly 100 games")
    _require(len({task["stable_game_key"] for task in tasks}) == 100, "head-to-head task keys are not unique")
    seeds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        seeds[int(task["seed_pair_index"])].append(task)
    for seed_pair_index, pair in seeds.items():
        _require(len(pair) == 2, f"seed pair {seed_pair_index} does not contain two games")
        _require(len({item["game_seed"] for item in pair}) == 1, f"seed pair {seed_pair_index} does not share one seed")
        _require({item["adaptive_color"] for item in pair} == {"white", "black"}, f"seed pair {seed_pair_index} does not swap colours")
    return tasks


def _input_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _checkpoint_public(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": item["condition"],
        "iteration": item["iteration"],
        "actual_gpu_hours": item["actual_gpu_hours"],
        "path": item["path"],
        "sha256": item["sha256"],
    }


def _initial_pair_rows(
    context: HeadToHeadContext, tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for task in tasks:
        by_seed[int(task["seed_pair_index"])][str(task["adaptive_color"])] = task
    rows: list[dict[str, Any]] = []
    for seed_pair_index in range(50):
        white = by_seed[seed_pair_index]["white"]
        black = by_seed[seed_pair_index]["black"]
        rows.append(
            {
                "seed_pair_index": seed_pair_index,
                "game_seed": white["game_seed"],
                "baseline_iteration": context.baseline["iteration"],
                "baseline_gpu_hours": context.baseline["actual_gpu_hours"],
                "baseline_checkpoint_sha256": context.baseline["sha256"],
                "adaptive_iteration": context.adaptive["iteration"],
                "adaptive_gpu_hours": context.adaptive["actual_gpu_hours"],
                "adaptive_checkpoint_sha256": context.adaptive["sha256"],
                "adaptive_white_game_key": white["stable_game_key"],
                "adaptive_white_result": "",
                "adaptive_black_game_key": black["stable_game_key"],
                "adaptive_black_result": "",
                "pair_complete": False,
            }
        )
    return rows


def build_preparation(
    context: HeadToHeadContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    tasks = build_tasks(context)
    resolved = copy.deepcopy(context.config)
    resolved["runtime_resolution"] = {
        "common_horizon_gpu_hours": context.common_horizon,
        "selection_rule": "latest completed checkpoint with actual_gpu_hours <= common_horizon",
        "best_checkpoint_selection_used": False,
        "baseline": _checkpoint_public(context.baseline),
        "adaptive": _checkpoint_public(context.adaptive),
        "board_size": context.board_size,
        "output_dir": context.output_dir.as_posix(),
    }
    implementations = {
        "evaluate_head_to_head": Path(__file__).resolve(),
        "verify_head_to_head": Path(__file__).resolve().with_name("verify_head_to_head.py"),
        "head_to_head_stats": Path(__file__).resolve().with_name("head_to_head_stats.py"),
        "evaluate_fixed_basket": BASELINE_SCRIPTS / "evaluate_fixed_basket.py",
        "arena": SOURCE_ROOT / "baseline" / "arena" / "arena.py",
        "bot_alphazero": SOURCE_ROOT / "baseline" / "arena" / "bot_alphazero.py",
    }
    input_manifest = {
        "schema_version": 2,
        "config_id": context.config["config_id"],
        "inputs": {
            "protocol": _input_entry(context.config_path),
            "matched_compute": _input_entry(context.matched_path),
            "baseline_checkpoint_manifest": _input_entry(context.baseline_manifest_path),
            "adaptive_checkpoint_manifest": _input_entry(context.adaptive_manifest_path),
            "baseline_resolved_config": _input_entry(context.baseline_resolved_path),
            "adaptive_resolved_config": _input_entry(context.adaptive_resolved_path),
            "selected_baseline_checkpoint": _input_entry(Path(context.baseline["path"])),
            "selected_adaptive_checkpoint": _input_entry(Path(context.adaptive["path"])),
            "implementations": {
                name: _input_entry(path) for name, path in implementations.items()
            },
        },
    }
    implementation_set = input_manifest["inputs"]["implementations"]
    input_manifest["implementation_revisions"] = [
        {
            "sha256": _canonical_sha256(implementation_set),
            "implementations": implementation_set,
        }
    ]
    manifest = {
        "schema_version": 2,
        "config_id": context.config["config_id"],
        "status": "prepared",
        "common_horizon_gpu_hours": context.common_horizon,
        "checkpoint_selection_rule": "latest_completed_checkpoint_not_after_common_horizon",
        "best_checkpoint_selection_used": False,
        "parameter_interpolation": False,
        "baseline_checkpoint": _checkpoint_public(context.baseline),
        "adaptive_checkpoint": _checkpoint_public(context.adaptive),
        "expected_seed_pairs": 50,
        "expected_technically_valid_games": 100,
        "technically_valid_games": 0,
        "tasks": tasks,
        "task_manifest_sha256": _canonical_sha256(tasks),
        "technical_failure_policy": {
            "attempts_file": context.attempts_path.as_posix(),
            "formal_games_file": context.games_path.as_posix(),
            "technical_terminations": ["invalid_move", "bot_error"],
            "retry_same_key_only": True,
        },
        "outputs": {
            "resolved_config": (context.output_dir / "resolved_config.yaml").as_posix(),
            "input_manifest": (context.output_dir / "input_manifest.json").as_posix(),
            "evaluation_manifest": context.manifest_path.as_posix(),
            "attempts": context.attempts_path.as_posix(),
            "games": context.games_path.as_posix(),
            "checkpoint_pairs": (context.output_dir / "checkpoint_pairs.csv").as_posix(),
            "checkpoint_summary": (context.output_dir / "checkpoint_summary.csv").as_posix(),
            "summary": (context.output_dir / "summary.json").as_posix(),
            "evaluation_log": (context.output_dir / "evaluation.log").as_posix(),
        },
    }
    return resolved, input_manifest, manifest, _initial_pair_rows(context, tasks)


def write_preparation(
    context: HeadToHeadContext,
    resolved: dict[str, Any],
    input_manifest: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    pair_rows: list[dict[str, Any]],
) -> None:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_yaml(context.output_dir / "resolved_config.yaml", resolved)
    _atomic_write_json(context.output_dir / "input_manifest.json", input_manifest)
    _atomic_write_json(context.manifest_path, evaluation_manifest)
    _atomic_write_csv(context.output_dir / "checkpoint_pairs.csv", pair_rows, PAIR_FIELDS)


def _task_map(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(task["stable_game_key"]): task for task in tasks}


def _validate_task_fields(record: dict[str, Any], task: dict[str, Any], label: str) -> None:
    for field in (
        "stable_game_key",
        "seed_pair_index",
        "game_seed",
        "baseline_color",
        "adaptive_color",
        "baseline_iteration",
        "baseline_gpu_hours",
        "baseline_checkpoint_path",
        "baseline_checkpoint_sha256",
        "adaptive_iteration",
        "adaptive_gpu_hours",
        "adaptive_checkpoint_path",
        "adaptive_checkpoint_sha256",
    ):
        _require(record.get(field) == task.get(field), f"{label} {field} mismatch for {task['stable_game_key']}")


def _technically_valid(record: dict[str, Any]) -> bool:
    return (
        record.get("termination") not in {"invalid_move", "bot_error"}
        and record.get("fault") is None
    )


def _game_from_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    game = dict(attempt)
    game["record_type"] = "technically_valid_game"
    return game


def load_and_validate_state(
    context: HeadToHeadContext,
    tasks: list[dict[str, Any]],
    *,
    recover: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    tasks_by_key = _task_map(tasks)
    attempts = _load_jsonl(context.attempts_path, "attempts.jsonl", required=False)
    games = _load_jsonl(context.games_path, "games.jsonl", required=False)
    attempts_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_attempt_by_key: dict[str, dict[str, Any]] = {}
    for record_number, record in enumerate(attempts, start=1):
        key = record.get("stable_game_key")
        _require(isinstance(key, str) and key in tasks_by_key, f"attempt {record_number} has an unknown game key")
        _validate_task_fields(record, tasks_by_key[key], f"attempt {record_number}")
        expected_attempt_index = len(attempts_by_key[key]) + 1
        _require(record.get("attempt_index") == expected_attempt_index, f"attempt index is not sequential for {key}")
        valid = _technically_valid(record)
        _require(record.get("technically_valid") is valid, f"attempt technical-validity flag mismatch for {key}")
        if valid:
            _require(key not in valid_attempt_by_key, f"multiple technically valid attempts exist for {key}")
            valid_attempt_by_key[key] = record
        else:
            _require(record.get("termination") in {"invalid_move", "bot_error"}, f"attempt {key} has an unclassified technical failure")
        attempts_by_key[key].append(record)

    games_by_key: dict[str, dict[str, Any]] = {}
    for record_number, record in enumerate(games, start=1):
        key = record.get("stable_game_key")
        _require(isinstance(key, str) and key in tasks_by_key, f"game {record_number} has an unknown game key")
        _require(key not in games_by_key, f"duplicate technically valid game key {key}")
        _validate_task_fields(record, tasks_by_key[key], f"game {record_number}")
        _require(record.get("technically_valid") is True and _technically_valid(record), f"games.jsonl contains a technical failure for {key}")
        _require(key in valid_attempt_by_key, f"game {key} has no corresponding technically valid attempt")
        expected = _game_from_attempt(valid_attempt_by_key[key])
        _require(record == expected, f"game {key} differs from its technically valid attempt")
        games_by_key[key] = record

    missing_promotions = [
        key for key in valid_attempt_by_key if key not in games_by_key
    ]
    if missing_promotions and not recover:
        raise HeadToHeadError(
            "technically valid attempts are missing from games.jsonl: "
            + ", ".join(sorted(missing_promotions))
        )
    for key in sorted(missing_promotions):
        game = _game_from_attempt(valid_attempt_by_key[key])
        _append_jsonl_fsync(context.games_path, game)
        games_by_key[key] = game
    return attempts_by_key, games_by_key


def parse_retry_keys(
    values: Sequence[str], tasks: list[dict[str, Any]]
) -> set[str]:
    tasks_by_key = _task_map(tasks)
    by_short_key = {
        f"{task['seed_pair_index']}:{task['adaptive_color']}": task["stable_game_key"]
        for task in tasks
    }
    aliases = {
        **by_short_key,
        **{
            key.replace(":white", ":adaptive_white").replace(":black", ":adaptive_black"): value
            for key, value in by_short_key.items()
        },
    }
    parsed: set[str] = set()
    for value in values:
        key = value if value in tasks_by_key else aliases.get(value)
        _require(key is not None, f"unknown --retry-game value: {value}")
        _require(key not in parsed, f"duplicate --retry-game value: {value}")
        parsed.add(str(key))
    return parsed


def _set_all_seeds(game_seed: int) -> None:
    random.seed(game_seed)
    np.random.seed(game_seed & 0xFFFFFFFF)
    torch_seed = game_seed & 0x7FFFFFFFFFFFFFFF
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def make_model(
    checkpoint: dict[str, Any], context: HeadToHeadContext
) -> ScheduledTemperatureAlphaZeroBot:
    path = Path(checkpoint["path"])
    model = context.config["model_protocol"]
    return ScheduledTemperatureAlphaZeroBot(
        "white",
        str(path.parent),
        path.name,
        board_size=context.board_size,
        use_mcts=True,
        clear_tree_each_move=True,
        numMCTSSims=int(model["mcts_simulations"]),
        cpuct=float(model["cpuct"]),
        eval_mcts_in_batch=int(model["eval_mcts_in_batch"]),
        early_temp=0.0,
        early_moves=0,
        later_temp=0.0,
    )


def _condition_for_color(task: dict[str, Any], color: Any) -> str | None:
    if color == task["adaptive_color"]:
        return "adaptive"
    if color == task["baseline_color"]:
        return "baseline"
    return None


def _adaptive_result(result: MatchResult, task: dict[str, Any]) -> str | None:
    if result.termination in {"invalid_move", "bot_error"} or result.fault is not None:
        return None
    if result.winner == task["adaptive_color"]:
        return "win"
    if result.winner == task["baseline_color"]:
        return "loss"
    return "draw"


def build_attempt_record(
    task: dict[str, Any],
    result: MatchResult,
    *,
    attempt_index: int,
    baseline_bot: ScheduledTemperatureAlphaZeroBot,
    adaptive_bot: ScheduledTemperatureAlphaZeroBot,
    max_turns: int,
    implementation_set_sha256: str,
) -> dict[str, Any]:
    valid = result.termination not in {"invalid_move", "bot_error"} and result.fault is None
    invalid_actor = (
        _condition_for_color(task, result.fault)
        if result.termination == "invalid_move"
        else None
    )
    bot_error_actor = (
        _condition_for_color(task, result.fault)
        if result.termination == "bot_error"
        else None
    )
    record = {
        "schema_version": 2,
        "config_id": "head_to_head_v2",
        "record_type": "attempt",
        **task,
        "attempt_index": attempt_index,
        "implementation_set_sha256": implementation_set_sha256,
        "winner": result.winner,
        "winner_condition": _condition_for_color(task, result.winner),
        "adaptive_result": _adaptive_result(result, task),
        "termination": result.termination,
        "fault": result.fault,
        "technically_valid": valid,
        "invalid_move_actor": invalid_actor,
        "bot_error_actor": bot_error_actor,
        "turns": result.turns,
        "total_moves": result.total_moves,
        "duration_seconds": result.match_duration,
        "baseline_move_seconds": result.move_times.get(task["baseline_color"], 0.0),
        "adaptive_move_seconds": result.move_times.get(task["adaptive_color"], 0.0),
        "baseline_moves": result.move_counts.get(task["baseline_color"], 0),
        "adaptive_moves": result.move_counts.get(task["adaptive_color"], 0),
        "baseline_temperature_history": list(baseline_bot.temperature_history),
        "adaptive_temperature_history": list(adaptive_bot.temperature_history),
        "baseline_model_fallback_count": len(baseline_bot.fallback_events),
        "adaptive_model_fallback_count": len(adaptive_bot.fallback_events),
        "baseline_model_fallback_events": list(baseline_bot.fallback_events),
        "adaptive_model_fallback_events": list(adaptive_bot.fallback_events),
        "max_turns": max_turns,
        "message": result.message,
        "moves": _serialize_moves(result),
    }
    _require(
        all(value == 0.0 for value in record["baseline_temperature_history"]),
        "Baseline temperature history is not identically zero",
    )
    _require(
        all(value == 0.0 for value in record["adaptive_temperature_history"]),
        "Adaptive temperature history is not identically zero",
    )
    return record


def _merge_resume_inputs(
    existing: dict[str, Any], current: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    existing_inputs = existing.get("inputs")
    current_inputs = current.get("inputs")
    _require(isinstance(existing_inputs, dict), "existing input manifest lacks inputs")
    _require(isinstance(current_inputs, dict), "current input manifest lacks inputs")
    existing_core = {
        key: value for key, value in existing_inputs.items() if key != "implementations"
    }
    current_core = {
        key: value for key, value in current_inputs.items() if key != "implementations"
    }
    _require(existing_core == current_core, "resume changes protocol, checkpoint, or model inputs")
    implementations = current_inputs.get("implementations")
    _require(isinstance(implementations, dict), "current implementation manifest is invalid")
    digest = _canonical_sha256(implementations)
    merged = copy.deepcopy(existing)
    revisions = merged.get("implementation_revisions")
    _require(isinstance(revisions, list) and revisions, "existing input manifest lacks implementation revisions")
    known = {item.get("sha256") for item in revisions if isinstance(item, dict)}
    if digest not in known:
        revisions.append({"sha256": digest, "implementations": implementations})
    merged["inputs"]["implementations"] = implementations
    return merged, digest


def evaluate_pending_tasks(
    context: HeadToHeadContext,
    tasks: list[dict[str, Any]],
    attempts_by_key: dict[str, list[dict[str, Any]]],
    games_by_key: dict[str, dict[str, Any]],
    *,
    retry_keys: set[str],
    implementation_set_sha256: str,
    logger: EvaluationLogger,
) -> int:
    unresolved = {
        key for key, values in attempts_by_key.items() if values and key not in games_by_key
    }
    _require(retry_keys <= unresolved, "--retry-game must identify unresolved technical failures")
    runnable = [
        task
        for task in tasks
        if task["stable_game_key"] not in games_by_key
        and (
            not attempts_by_key.get(task["stable_game_key"])
            or task["stable_game_key"] in retry_keys
        )
    ]
    if not runnable:
        logger.write("No runnable head-to-head tasks remain in this invocation")
        return 0

    logger.write(
        "Loading final common-horizon models: "
        f"Baseline iteration={context.baseline['iteration']} "
        f"Adaptive iteration={context.adaptive['iteration']}"
    )
    baseline_bot = make_model(context.baseline, context)
    try:
        adaptive_bot = make_model(context.adaptive, context)
    except Exception:
        _cleanup_bot(baseline_bot)
        del baseline_bot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    attempts_completed = 0
    try:
        for task in runnable:
            key = str(task["stable_game_key"])
            _set_all_seeds(int(task["game_seed"]))
            prepared_baseline = _prepare_bot(baseline_bot, task["baseline_color"])
            prepared_adaptive = _prepare_bot(adaptive_bot, task["adaptive_color"])
            if task["adaptive_color"] == "white":
                result = play_game(
                    prepared_adaptive,
                    prepared_baseline,
                    max_turns=int(context.config["games"]["max_turns"]),
                )
            else:
                result = play_game(
                    prepared_baseline,
                    prepared_adaptive,
                    max_turns=int(context.config["games"]["max_turns"]),
                )
            attempt = build_attempt_record(
                task,
                result,
                attempt_index=len(attempts_by_key.get(key, [])) + 1,
                baseline_bot=prepared_baseline,
                adaptive_bot=prepared_adaptive,
                max_turns=int(context.config["games"]["max_turns"]),
                implementation_set_sha256=implementation_set_sha256,
            )
            _append_jsonl_fsync(context.attempts_path, attempt)
            attempts_by_key.setdefault(key, []).append(attempt)
            attempts_completed += 1
            if attempt["technically_valid"]:
                game = _game_from_attempt(attempt)
                _append_jsonl_fsync(context.games_path, game)
                games_by_key[key] = game
                logger.write(
                    "Technically valid game completed and fsynced: "
                    f"seed_pair={task['seed_pair_index']} "
                    f"adaptive_color={task['adaptive_color']} "
                    f"result={attempt['adaptive_result']}"
                )
            else:
                logger.write(
                    "Technical failure retained in attempts.jsonl: "
                    f"key={key} termination={attempt['termination']} "
                    f"actor={attempt['invalid_move_actor'] or attempt['bot_error_actor']}"
                )
    finally:
        _cleanup_bot(baseline_bot)
        _cleanup_bot(adaptive_bot)
        del baseline_bot
        del adaptive_bot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return attempts_completed


def _pair_rows_with_results(
    context: HeadToHeadContext,
    tasks: list[dict[str, Any]],
    games_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _initial_pair_rows(context, tasks)
    for row in rows:
        white = games_by_key.get(str(row["adaptive_white_game_key"]))
        black = games_by_key.get(str(row["adaptive_black_game_key"]))
        row["adaptive_white_result"] = white["adaptive_result"] if white else ""
        row["adaptive_black_result"] = black["adaptive_result"] if black else ""
        row["pair_complete"] = white is not None and black is not None
    return rows


def summarize_results(
    context: HeadToHeadContext,
    tasks: list[dict[str, Any]],
    attempts_by_key: dict[str, list[dict[str, Any]]],
    games_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    games = [games_by_key[key] for key in sorted(games_by_key)]
    pair_rows = _pair_rows_with_results(context, tasks, games_by_key)
    _atomic_write_csv(context.output_dir / "checkpoint_pairs.csv", pair_rows, PAIR_FIELDS)
    complete = len(games) == 100 and all(bool(row["pair_complete"]) for row in pair_rows)
    wins = sum(record.get("adaptive_result") == "win" for record in games)
    draws = sum(record.get("adaptive_result") == "draw" for record in games)
    losses = sum(record.get("adaptive_result") == "loss" for record in games)
    white_games = sum(record.get("adaptive_color") == "white" for record in games)
    black_games = sum(record.get("adaptive_color") == "black" for record in games)
    bootstrap_config = context.config["analysis"]["bootstrap"]
    interval: dict[str, Any] | None = None
    if complete:
        _require(white_games == black_games == 50, "complete evaluation does not contain 50 games per Adaptive colour")
        interval = colour_stratified_bootstrap(
            games,
            resamples=int(bootstrap_config["resamples"]),
            seed=int(bootstrap_config["seed"]),
            expected_per_colour=int(bootstrap_config["preserve_games_per_colour"]),
        )
        _require(
            math.isclose(
                float(interval["score_rate"]),
                (wins + 0.5 * draws) / 100.0,
                abs_tol=1.0e-12,
            ),
            "bootstrap point estimate differs from the registered score formula",
        )
    unresolved_keys = sorted(
        key
        for key, values in attempts_by_key.items()
        if values and key not in games_by_key
    )
    technical_attempts = sum(
        not bool(record["technically_valid"])
        for values in attempts_by_key.values()
        for record in values
    )
    support = (
        bool(float(interval["ci95_low"]) > 0.5) if interval is not None else None
    )
    summary_row = {
        "common_horizon_gpu_hours": context.common_horizon,
        "baseline_iteration": context.baseline["iteration"],
        "baseline_gpu_hours": context.baseline["actual_gpu_hours"],
        "baseline_checkpoint_sha256": context.baseline["sha256"],
        "adaptive_iteration": context.adaptive["iteration"],
        "adaptive_gpu_hours": context.adaptive["actual_gpu_hours"],
        "adaptive_checkpoint_sha256": context.adaptive["sha256"],
        "technically_valid_games": len(games),
        "adaptive_white_games": white_games,
        "adaptive_black_games": black_games,
        "adaptive_wins": wins,
        "draws": draws,
        "adaptive_losses": losses,
        "adaptive_score_rate": interval["score_rate"] if interval else "",
        "adaptive_score_rate_ci95_low": interval["ci95_low"] if interval else "",
        "adaptive_score_rate_ci95_high": interval["ci95_high"] if interval else "",
        "bootstrap_method": bootstrap_config["method"],
        "bootstrap_strata": "adaptive_colour",
        "bootstrap_iterations": bootstrap_config["resamples"],
        "bootstrap_seed": bootstrap_config["seed"],
        "h3_head_to_head_support": support if support is not None else "",
        "status": "completed" if complete else "incomplete",
    }
    _atomic_write_csv(
        context.output_dir / "checkpoint_summary.csv",
        [summary_row],
        SUMMARY_FIELDS,
    )
    termination_counts = {
        value: sum(record.get("termination") == value for record in games)
        for value in sorted({str(record.get("termination")) for record in games})
    }
    summary = {
        "schema_version": 2,
        "config_id": context.config["config_id"],
        "status": "completed" if complete else "incomplete",
        "common_horizon_gpu_hours": context.common_horizon,
        "checkpoint_selection": {
            "rule": "latest_completed_checkpoint_not_after_common_horizon",
            "best_checkpoint_selection_used": False,
            "baseline": _checkpoint_public(context.baseline),
            "adaptive": _checkpoint_public(context.adaptive),
        },
        "games": {
            "expected_technically_valid": 100,
            "technically_valid": len(games),
            "seed_pairs_expected": 50,
            "seed_pairs_complete": sum(bool(row["pair_complete"]) for row in pair_rows),
            "unique_game_keys": len(games_by_key),
            "adaptive_white": white_games,
            "adaptive_black": black_games,
            "adaptive_wins": wins,
            "draws": draws,
            "adaptive_losses": losses,
            "termination_counts": termination_counts,
        },
        "attempts": {
            "total": sum(len(values) for values in attempts_by_key.values()),
            "technical_failures": technical_attempts,
            "unresolved_technical_failure_keys": unresolved_keys,
        },
        "adaptive_score": {
            "formula": "(adaptive_wins + 0.5 * draws) / 100",
            "score_rate": interval["score_rate"] if interval else None,
            "ci95_low": interval["ci95_low"] if interval else None,
            "ci95_high": interval["ci95_high"] if interval else None,
            "bootstrap": {
                "method": bootstrap_config["method"],
                "strata": ["adaptive_colour"],
                "resamples": bootstrap_config["resamples"],
                "seed": bootstrap_config["seed"],
                "preserve_games_per_colour": bootstrap_config["preserve_games_per_colour"],
            },
        },
        "h3_head_to_head_support": {
            "rule": "adaptive_score_rate_ci95_low > 0.5",
            "supported": support,
        },
        "outputs": {
            "attempts": context.attempts_path.as_posix(),
            "games": context.games_path.as_posix(),
            "checkpoint_pairs": (context.output_dir / "checkpoint_pairs.csv").as_posix(),
            "checkpoint_summary": (context.output_dir / "checkpoint_summary.csv").as_posix(),
        },
    }
    summary["output_sha256"] = {
        "checkpoint_pairs": sha256_file(context.output_dir / "checkpoint_pairs.csv"),
        "checkpoint_summary": sha256_file(context.output_dir / "checkpoint_summary.csv"),
    }
    if context.attempts_path.is_file():
        summary["output_sha256"]["attempts"] = sha256_file(context.attempts_path)
    if context.games_path.is_file():
        summary["output_sha256"]["games"] = sha256_file(context.games_path)
    _atomic_write_json(context.output_dir / "summary.json", summary)
    manifest = _load_json(context.manifest_path, "evaluation manifest")
    manifest["status"] = summary["status"]
    manifest["technically_valid_games"] = len(games)
    manifest["attempts_recorded"] = summary["attempts"]["total"]
    manifest["technical_failures"] = technical_attempts
    manifest["unresolved_technical_failure_keys"] = unresolved_keys
    manifest["games_sha256"] = (
        sha256_file(context.games_path) if context.games_path.is_file() else None
    )
    manifest["attempts_sha256"] = (
        sha256_file(context.attempts_path) if context.attempts_path.is_file() else None
    )
    manifest["summary"] = summary
    _atomic_write_json(context.manifest_path, manifest)
    return summary


def _validate_resume_manifest(
    existing: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for field in (
        "config_id",
        "common_horizon_gpu_hours",
        "checkpoint_selection_rule",
        "best_checkpoint_selection_used",
        "baseline_checkpoint",
        "adaptive_checkpoint",
        "expected_seed_pairs",
        "expected_technically_valid_games",
        "tasks",
        "task_manifest_sha256",
    ):
        _require(existing.get(field) == candidate.get(field), f"resume manifest {field} mismatch")


def execute(args: argparse.Namespace, logger: EvaluationLogger) -> str:
    context = resolve_context(args)
    resolved, current_inputs, candidate_manifest, pair_rows = build_preparation(context)
    if args.summarize_only:
        _require(context.manifest_path.is_file(), "--summarize-only requires evaluation_manifest.json")
        attempts_by_key, games_by_key = load_and_validate_state(
            context, candidate_manifest["tasks"], recover=True
        )
        summary = summarize_results(
            context, candidate_manifest["tasks"], attempts_by_key, games_by_key
        )
        logger.write(
            f"Head-to-head summary status: {summary['status']}; "
            f"valid_games={summary['games']['technically_valid']}"
        )
        return str(summary["status"])

    if args.retry_game and not args.resume:
        raise HeadToHeadError("--retry-game requires --resume")
    if not args.resume and (
        context.attempts_path.exists() or context.games_path.exists()
    ):
        raise HeadToHeadError(
            "attempts.jsonl or games.jsonl already exists; pass --resume"
        )

    if args.resume:
        _require(context.manifest_path.is_file(), "--resume requires evaluation_manifest.json")
        existing_manifest = _load_json(context.manifest_path, "existing evaluation manifest")
        _validate_resume_manifest(existing_manifest, candidate_manifest)
        existing_inputs = _load_json(
            context.output_dir / "input_manifest.json", "existing input manifest"
        )
        merged_inputs, implementation_digest = _merge_resume_inputs(
            existing_inputs, current_inputs
        )
        _atomic_write_json(context.output_dir / "input_manifest.json", merged_inputs)
        manifest = existing_manifest
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["implementation_revision_sha256"] = implementation_digest
        manifest["implementation_revision_count"] = len(
            merged_inputs["implementation_revisions"]
        )
        _atomic_write_json(context.manifest_path, manifest)
    else:
        write_preparation(context, resolved, current_inputs, candidate_manifest, pair_rows)
        manifest = candidate_manifest
        implementation_digest = current_inputs["implementation_revisions"][0]["sha256"]

    logger.write(
        "Head-to-head prepared: "
        f"horizon={context.common_horizon:.12f} "
        f"Baseline={context.baseline['iteration']}@{context.baseline['actual_gpu_hours']:.12f} "
        f"Adaptive={context.adaptive['iteration']}@{context.adaptive['actual_gpu_hours']:.12f}"
    )
    if args.prepare_only:
        logger.write("Evaluation status: prepared; no head-to-head games started")
        return "prepared"

    tasks = candidate_manifest["tasks"]
    attempts_by_key, games_by_key = load_and_validate_state(
        context, tasks, recover=True
    )
    retry_keys = parse_retry_keys(args.retry_game, tasks)
    manifest = _load_json(context.manifest_path, "evaluation manifest")
    manifest["status"] = "running"
    manifest["implementation_revision_sha256"] = implementation_digest
    manifest["requested_retry_keys"] = sorted(retry_keys)
    manifest.pop("failure", None)
    _atomic_write_json(context.manifest_path, manifest)
    new_attempts = evaluate_pending_tasks(
        context,
        tasks,
        attempts_by_key,
        games_by_key,
        retry_keys=retry_keys,
        implementation_set_sha256=implementation_digest,
        logger=logger,
    )
    summary = summarize_results(
        context, tasks, attempts_by_key, games_by_key
    )
    logger.write(
        "Head-to-head invocation finished: "
        f"new_attempts={new_attempts} "
        f"valid_games={summary['games']['technically_valid']} "
        f"status={summary['status']}"
    )
    if summary["status"] == "incomplete":
        logger.write(
            "Unresolved technical failures require diagnosis and explicit "
            "--resume --retry-game with the same key"
        )
    return str(summary["status"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matched-compute", type=Path)
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--adaptive-run-dir", type=Path)
    parser.add_argument("--baseline-checkpoint-manifest", type=Path)
    parser.add_argument("--adaptive-checkpoint-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve/check checkpoints and write the 100-game task manifest only.",
    )
    modes.add_argument(
        "--resume",
        action="store_true",
        help="Continue unattempted games and explicitly requested technical retries.",
    )
    modes.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summaries from existing attempts and valid games.",
    )
    modes.add_argument(
        "--verify-only",
        action="store_true",
        help="Run the read-only head-to-head v2 acceptance verifier.",
    )
    parser.add_argument(
        "--retry-game",
        action="append",
        default=[],
        metavar="KEY|SEED:COLOR",
        help="Retry an unresolved technical failure using its stable key.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        if args.retry_game:
            print("ERROR: --retry-game cannot be combined with --verify-only", file=sys.stderr)
            return 2
        try:
            from verify_head_to_head import verify_head_to_head

            report = verify_head_to_head(
                config_path=args.config,
                output_dir=args.output_dir,
                matched_compute_path=args.matched_compute,
                baseline_run_dir=args.baseline_run_dir,
                adaptive_run_dir=args.adaptive_run_dir,
                baseline_checkpoint_manifest=args.baseline_checkpoint_manifest,
                adaptive_checkpoint_manifest=args.adaptive_checkpoint_manifest,
            )
            print(
                "Head-to-head verification passed: "
                f"games={report['games']['technically_valid']} "
                f"score_rate={report['adaptive_score']['score_rate']:.6f}"
            )
            return 0
        except (HeadToHeadError, HeadToHeadStatsError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    log_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT
    )
    with EvaluationLogger(log_dir / "evaluation.log") as logger:
        try:
            status = execute(args, logger)
            return 0 if status in {"prepared", "completed"} else 3
        except (HeadToHeadError, HeadToHeadStatsError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            logger.write(f"Head-to-head evaluation failed: {exc}")
            manifest_path = log_dir / "evaluation_manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = _load_json(manifest_path, "evaluation manifest")
                    manifest["status"] = "failed"
                    manifest["failure"] = str(exc)
                    _atomic_write_json(manifest_path, manifest)
                except Exception:
                    pass
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
