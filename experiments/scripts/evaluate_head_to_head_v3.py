#!/usr/bin/env python3
"""Run the paired, seeded, trajectory-audited head-to-head v3 evaluation."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
BASELINE_ROOT = SOURCE_ROOT / "baseline"
BASELINE_SCRIPTS = BASELINE_ROOT / "analysis" / "scripts"
for import_root in (SOURCE_ROOT, BASELINE_ROOT, BASELINE_SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_head_to_head as v2  # noqa: E402
from arena.arena import MatchResult, play_game  # noqa: E402
from evaluate_fixed_basket import (  # noqa: E402
    EvaluationLogger,
    ScheduledTemperatureAlphaZeroBot,
    _cleanup_bot,
    _prepare_bot,
    _serialize_moves,
)
from head_to_head_stats import paired_seed, stable_game_key  # noqa: E402
from head_to_head_stats_v3 import (  # noqa: E402
    HeadToHeadStatsV3Error,
    seed_pair_bootstrap,
    trajectory_diversity,
    trajectory_sha256,
)


DEFAULT_CONFIG = EXPERIMENTS_ROOT / "configs" / "head_to_head_v3.yaml"
DEFAULT_OUTPUT = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "adaptive_seed1001_4090_v2_analysis"
    / "head_to_head_v3"
)

SUMMARY_FIELDS = v2.SUMMARY_FIELDS + (
    "adaptive_white_unique_trajectories",
    "adaptive_black_unique_trajectories",
    "trajectory_diversity_passed",
)

HeadToHeadError = v2.HeadToHeadError


def _require(condition: bool, message: str) -> None:
    v2._require(condition, message)


def resolve_context(args: argparse.Namespace) -> v2.HeadToHeadContext:
    config_path = args.config.expanduser().resolve()
    config = v2._load_yaml(config_path, "head-to-head v3 protocol")
    _require(config.get("config_id") == "head_to_head_v3", "config_id must be head_to_head_v3")
    conditions = config.get("conditions")
    _require(isinstance(conditions, dict), "conditions are missing")
    baseline_config = conditions.get("baseline")
    adaptive_config = conditions.get("adaptive")
    _require(isinstance(baseline_config, dict), "Baseline condition is missing")
    _require(isinstance(adaptive_config, dict), "Adaptive condition is missing")

    matched_path = (
        args.matched_compute.expanduser().resolve()
        if args.matched_compute is not None
        else v2._resolve_experiments_path(config["matched_compute"]["path"])
    )
    v2._verified_hash(
        matched_path,
        config["matched_compute"].get("expected_sha256"),
        "matched-compute protocol",
    )
    matched = v2._load_yaml(matched_path, "matched-compute protocol")
    horizon, baseline_grid = v2.common_horizon_and_baseline_grid(matched)

    baseline_run_root = (
        args.baseline_run_dir.expanduser().resolve()
        if args.baseline_run_dir is not None
        else v2._resolve_experiments_path(baseline_config["run_root"])
    )
    adaptive_run_root = (
        args.adaptive_run_dir.expanduser().resolve()
        if args.adaptive_run_dir is not None
        else v2._resolve_experiments_path(adaptive_config["run_root"])
    )
    baseline_manifest_path = (
        args.baseline_checkpoint_manifest.expanduser().resolve()
        if args.baseline_checkpoint_manifest is not None
        else v2._resolve_experiments_path(baseline_config["checkpoint_manifest"])
    )
    adaptive_manifest_path = (
        args.adaptive_checkpoint_manifest.expanduser().resolve()
        if args.adaptive_checkpoint_manifest is not None
        else v2._resolve_experiments_path(adaptive_config["checkpoint_manifest"])
    )
    v2._verified_hash(
        baseline_manifest_path,
        baseline_config.get("checkpoint_manifest_expected_sha256"),
        "Baseline checkpoint manifest",
    )
    v2._verified_hash(
        adaptive_manifest_path,
        adaptive_config.get("checkpoint_manifest_expected_sha256"),
        "Adaptive checkpoint manifest",
    )
    baseline_manifest = v2._load_json(
        baseline_manifest_path, "Baseline checkpoint manifest"
    )
    adaptive_manifest = v2._load_json(
        adaptive_manifest_path, "Adaptive checkpoint manifest"
    )
    baseline = v2.select_final_checkpoint(
        v2._checkpoint_entries(
            baseline_manifest, condition="baseline", baseline_grid=baseline_grid
        ),
        horizon,
    )
    adaptive = v2.select_final_checkpoint(
        v2._checkpoint_entries(
            adaptive_manifest, condition="adaptive", baseline_grid=baseline_grid
        ),
        horizon,
    )
    baseline_path = v2._resolve_checkpoint_path(baseline, baseline_run_root)
    adaptive_path = v2._resolve_checkpoint_path(adaptive, adaptive_run_root)
    baseline["path"] = baseline_path.as_posix()
    adaptive["path"] = adaptive_path.as_posix()
    v2._verified_hash(baseline_path, baseline["sha256"], "selected Baseline checkpoint")
    v2._verified_hash(adaptive_path, adaptive["sha256"], "selected Adaptive checkpoint")
    _require(baseline["actual_gpu_hours"] <= horizon, "selected Baseline checkpoint exceeds common horizon")
    _require(adaptive["actual_gpu_hours"] <= horizon, "selected Adaptive checkpoint exceeds common horizon")

    baseline_resolved_path = v2._resolve_experiments_path(
        baseline_config["resolved_config"]
    )
    adaptive_resolved_path = v2._resolve_experiments_path(
        adaptive_config["resolved_config"]
    )
    baseline_resolved = v2._load_yaml(
        baseline_resolved_path, "Baseline resolved config"
    )
    adaptive_resolved = v2._load_yaml(
        adaptive_resolved_path, "Adaptive resolved config"
    )
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
        else v2._resolve_experiments_path(config["outputs"]["root"])
    )
    return v2.HeadToHeadContext(
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
        "dirichlet_noise": False,
        "clear_tree_each_move": True,
        "reset_tree_before_each_game": True,
        "identical_for_both_conditions": True,
    }
    for field, expected in required_model.items():
        _require(model.get(field) == expected, f"model_protocol.{field} must be {expected!r}")
    schedule = model.get("temperature_schedule")
    _require(isinstance(schedule, dict), "temperature schedule is missing")
    _require(schedule.get("unit") == "own_model_turns", "temperature schedule must count own-model turns")
    _require(schedule.get("early_temperature") == 0.18, "early temperature must be 0.18")
    _require(schedule.get("early_moves_per_player") == 5, "early temperature must cover five own turns")
    _require(schedule.get("later_temperature") == 0.0, "later temperature must be zero")
    _require(selection.get("rule") == "latest_completed_checkpoint_not_after_common_horizon", "checkpoint selection rule is invalid")
    _require(selection.get("allow_best_checkpoint") is False, "best-checkpoint selection must be forbidden")
    _require(selection.get("interpolation") == "forbidden", "checkpoint interpolation must be forbidden")
    bootstrap = analysis.get("bootstrap")
    _require(isinstance(bootstrap, dict), "bootstrap protocol is missing")
    _require(bootstrap.get("method") == "nonparametric_seed_pair_bootstrap", "bootstrap must resample seed pairs")
    _require(bootstrap.get("resampling_unit") == "seed_pair", "bootstrap unit must be seed_pair")
    _require(bootstrap.get("games_per_unit") == 2, "each bootstrap unit must contain two games")
    _require(bootstrap.get("resamples") == 10000, "bootstrap resamples must be 10,000")
    diversity = analysis.get("trajectory_diversity")
    _require(isinstance(diversity, dict), "trajectory-diversity protocol is missing")
    _require(diversity.get("minimum_unique_trajectories_per_colour") == 2, "each colour must require at least two unique trajectories")


def build_tasks(context: v2.HeadToHeadContext) -> list[dict[str, Any]]:
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
            tasks.append(
                {
                    "stable_game_key": stable_game_key(
                        context.baseline["sha256"],
                        context.adaptive["sha256"],
                        seed_pair_index,
                        adaptive_color,
                    ),
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
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_pair[int(task["seed_pair_index"])].append(task)
    for seed_pair_index, pair in by_pair.items():
        _require(len(pair) == 2, f"seed pair {seed_pair_index} does not contain two games")
        _require(len({item["game_seed"] for item in pair}) == 1, f"seed pair {seed_pair_index} does not share one seed")
        _require({item["adaptive_color"] for item in pair} == {"white", "black"}, f"seed pair {seed_pair_index} does not swap colours")
    return tasks


def build_preparation(
    context: v2.HeadToHeadContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    tasks = build_tasks(context)
    resolved = copy.deepcopy(context.config)
    resolved["runtime_resolution"] = {
        "common_horizon_gpu_hours": context.common_horizon,
        "selection_rule": "latest completed checkpoint with actual_gpu_hours <= common_horizon",
        "best_checkpoint_selection_used": False,
        "baseline": v2._checkpoint_public(context.baseline),
        "adaptive": v2._checkpoint_public(context.adaptive),
        "board_size": context.board_size,
        "output_dir": context.output_dir.as_posix(),
    }
    implementations = {
        "evaluate_head_to_head_v3": Path(__file__).resolve(),
        "verify_head_to_head_v3": Path(__file__).resolve().with_name("verify_head_to_head_v3.py"),
        "head_to_head_stats_v3": Path(__file__).resolve().with_name("head_to_head_stats_v3.py"),
        "head_to_head_v2_common": Path(v2.__file__).resolve(),
        "evaluate_fixed_basket": BASELINE_SCRIPTS / "evaluate_fixed_basket.py",
        "arena": SOURCE_ROOT / "baseline" / "arena" / "arena.py",
        "bot_alphazero": SOURCE_ROOT / "baseline" / "arena" / "bot_alphazero.py",
    }
    input_manifest = {
        "schema_version": 3,
        "config_id": context.config["config_id"],
        "inputs": {
            "protocol": v2._input_entry(context.config_path),
            "matched_compute": v2._input_entry(context.matched_path),
            "baseline_checkpoint_manifest": v2._input_entry(context.baseline_manifest_path),
            "adaptive_checkpoint_manifest": v2._input_entry(context.adaptive_manifest_path),
            "baseline_resolved_config": v2._input_entry(context.baseline_resolved_path),
            "adaptive_resolved_config": v2._input_entry(context.adaptive_resolved_path),
            "selected_baseline_checkpoint": v2._input_entry(Path(context.baseline["path"])),
            "selected_adaptive_checkpoint": v2._input_entry(Path(context.adaptive["path"])),
            "implementations": {
                name: v2._input_entry(path) for name, path in implementations.items()
            },
        },
    }
    implementation_set = input_manifest["inputs"]["implementations"]
    input_manifest["implementation_revisions"] = [
        {
            "sha256": v2._canonical_sha256(implementation_set),
            "implementations": implementation_set,
        }
    ]
    manifest = {
        "schema_version": 3,
        "config_id": context.config["config_id"],
        "status": "prepared",
        "common_horizon_gpu_hours": context.common_horizon,
        "checkpoint_selection_rule": "latest_completed_checkpoint_not_after_common_horizon",
        "best_checkpoint_selection_used": False,
        "parameter_interpolation": False,
        "baseline_checkpoint": v2._checkpoint_public(context.baseline),
        "adaptive_checkpoint": v2._checkpoint_public(context.adaptive),
        "expected_seed_pairs": 50,
        "expected_technically_valid_games": 100,
        "technically_valid_games": 0,
        "tasks": tasks,
        "task_manifest_sha256": v2._canonical_sha256(tasks),
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
    return resolved, input_manifest, manifest, v2._initial_pair_rows(context, tasks)


def make_model(
    checkpoint: dict[str, Any], context: v2.HeadToHeadContext
) -> ScheduledTemperatureAlphaZeroBot:
    path = Path(checkpoint["path"])
    model = context.config["model_protocol"]
    schedule = model["temperature_schedule"]
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
        early_temp=float(schedule["early_temperature"]),
        early_moves=int(schedule["early_moves_per_player"]),
        later_temp=float(schedule["later_temperature"]),
    )


def expected_temperature_history(
    moves: int, model_protocol: dict[str, Any]
) -> list[float]:
    schedule = model_protocol["temperature_schedule"]
    early_moves = int(schedule["early_moves_per_player"])
    return [float(schedule["early_temperature"])] * min(moves, early_moves) + [
        float(schedule["later_temperature"])
    ] * max(0, moves - early_moves)


def build_attempt_record(
    context: v2.HeadToHeadContext,
    task: dict[str, Any],
    result: MatchResult,
    *,
    attempt_index: int,
    baseline_bot: ScheduledTemperatureAlphaZeroBot,
    adaptive_bot: ScheduledTemperatureAlphaZeroBot,
    implementation_set_sha256: str,
) -> dict[str, Any]:
    valid = result.termination not in {"invalid_move", "bot_error"} and result.fault is None
    invalid_actor = v2._condition_for_color(task, result.fault) if result.termination == "invalid_move" else None
    bot_error_actor = v2._condition_for_color(task, result.fault) if result.termination == "bot_error" else None
    moves = _serialize_moves(result)
    record = {
        "schema_version": 3,
        "config_id": context.config["config_id"],
        "record_type": "attempt",
        **task,
        "attempt_index": attempt_index,
        "implementation_set_sha256": implementation_set_sha256,
        "winner": result.winner,
        "winner_condition": v2._condition_for_color(task, result.winner),
        "adaptive_result": v2._adaptive_result(result, task),
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
        "max_turns": int(context.config["games"]["max_turns"]),
        "message": result.message,
        "moves": moves,
    }
    record["trajectory_sha256"] = trajectory_sha256(record)
    for condition in ("baseline", "adaptive"):
        history = record[f"{condition}_temperature_history"]
        expected = expected_temperature_history(
            int(record[f"{condition}_moves"]), context.config["model_protocol"]
        )
        _require(history == expected, f"{condition} temperature history differs from the v3 schedule")
    return record


def evaluate_pending_tasks(
    context: v2.HeadToHeadContext,
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
        logger.write("No runnable head-to-head v3 tasks remain in this invocation")
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
        raise
    attempts_completed = 0
    try:
        for task in runnable:
            key = str(task["stable_game_key"])
            v2._set_all_seeds(int(task["game_seed"]))
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
                context,
                task,
                result,
                attempt_index=len(attempts_by_key.get(key, [])) + 1,
                baseline_bot=prepared_baseline,
                adaptive_bot=prepared_adaptive,
                implementation_set_sha256=implementation_set_sha256,
            )
            v2._append_jsonl_fsync(context.attempts_path, attempt)
            attempts_by_key.setdefault(key, []).append(attempt)
            attempts_completed += 1
            if attempt["technically_valid"]:
                game = v2._game_from_attempt(attempt)
                v2._append_jsonl_fsync(context.games_path, game)
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


def summarize_results(
    context: v2.HeadToHeadContext,
    tasks: list[dict[str, Any]],
    attempts_by_key: dict[str, list[dict[str, Any]]],
    games_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    games = [games_by_key[key] for key in sorted(games_by_key)]
    pair_rows = v2._pair_rows_with_results(context, tasks, games_by_key)
    v2._atomic_write_csv(
        context.output_dir / "checkpoint_pairs.csv", pair_rows, v2.PAIR_FIELDS
    )
    structurally_complete = len(games) == 100 and all(
        bool(row["pair_complete"]) for row in pair_rows
    )
    wins = sum(record.get("adaptive_result") == "win" for record in games)
    draws = sum(record.get("adaptive_result") == "draw" for record in games)
    losses = sum(record.get("adaptive_result") == "loss" for record in games)
    white_games = sum(record.get("adaptive_color") == "white" for record in games)
    black_games = sum(record.get("adaptive_color") == "black" for record in games)
    bootstrap_config = context.config["analysis"]["bootstrap"]
    diversity_config = context.config["analysis"]["trajectory_diversity"]
    interval: dict[str, Any] | None = None
    diversity: dict[str, Any] | None = None
    if structurally_complete:
        _require(white_games == black_games == 50, "complete evaluation does not contain 50 games per Adaptive colour")
        interval = seed_pair_bootstrap(
            games,
            resamples=int(bootstrap_config["resamples"]),
            seed=int(bootstrap_config["seed"]),
            expected_pairs=int(bootstrap_config["preserve_seed_pairs"]),
        )
        diversity = trajectory_diversity(
            games,
            minimum_unique_per_colour=int(
                diversity_config["minimum_unique_trajectories_per_colour"]
            ),
        )
        _require(
            math.isclose(
                float(interval["score_rate"]),
                (wins + 0.5 * draws) / 100.0,
                abs_tol=1.0e-12,
            ),
            "paired-bootstrap point estimate differs from the score formula",
        )
    diversity_passed = bool(diversity and diversity["acceptance_passed"])
    if not structurally_complete:
        status = "incomplete"
    elif not diversity_passed:
        status = "data_quality_failed"
    else:
        status = "completed"
    support = (
        bool(float(interval["ci95_low"]) > 0.5)
        if interval is not None and diversity_passed
        else None
    )
    unresolved_keys = sorted(
        key for key, values in attempts_by_key.items() if values and key not in games_by_key
    )
    technical_attempts = sum(
        not bool(record["technically_valid"])
        for values in attempts_by_key.values()
        for record in values
    )
    white_unique = diversity["segments"]["white"]["unique_trajectories"] if diversity else ""
    black_unique = diversity["segments"]["black"]["unique_trajectories"] if diversity else ""
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
        "bootstrap_strata": "seed_pair",
        "bootstrap_iterations": bootstrap_config["resamples"],
        "bootstrap_seed": bootstrap_config["seed"],
        "h3_head_to_head_support": support if support is not None else "",
        "status": status,
        "adaptive_white_unique_trajectories": white_unique,
        "adaptive_black_unique_trajectories": black_unique,
        "trajectory_diversity_passed": diversity_passed if diversity else "",
    }
    v2._atomic_write_csv(
        context.output_dir / "checkpoint_summary.csv",
        [summary_row],
        SUMMARY_FIELDS,
    )
    termination_counts = {
        value: sum(record.get("termination") == value for record in games)
        for value in sorted({str(record.get("termination")) for record in games})
    }
    summary = {
        "schema_version": 3,
        "config_id": context.config["config_id"],
        "status": status,
        "common_horizon_gpu_hours": context.common_horizon,
        "checkpoint_selection": {
            "rule": "latest_completed_checkpoint_not_after_common_horizon",
            "best_checkpoint_selection_used": False,
            "baseline": v2._checkpoint_public(context.baseline),
            "adaptive": v2._checkpoint_public(context.adaptive),
        },
        "temperature_schedule": context.config["model_protocol"]["temperature_schedule"],
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
        "trajectory_diversity": diversity,
        "adaptive_score": {
            "formula": "mean of 50 seed-pair means; each pair mean is the mean Adaptive score over its colour-swapped games",
            "score_rate": interval["score_rate"] if interval else None,
            "ci95_low": interval["ci95_low"] if interval else None,
            "ci95_high": interval["ci95_high"] if interval else None,
            "bootstrap": {
                "method": bootstrap_config["method"],
                "resampling_unit": "seed_pair",
                "games_per_unit": 2,
                "resamples": bootstrap_config["resamples"],
                "seed": bootstrap_config["seed"],
                "preserve_seed_pairs": bootstrap_config["preserve_seed_pairs"],
            },
        },
        "h3_head_to_head_support": {
            "rule": "paired_bootstrap_score_rate_ci95_low > 0.5 and trajectory_diversity.acceptance_passed",
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
        "checkpoint_pairs": v2.sha256_file(context.output_dir / "checkpoint_pairs.csv"),
        "checkpoint_summary": v2.sha256_file(context.output_dir / "checkpoint_summary.csv"),
    }
    if context.attempts_path.is_file():
        summary["output_sha256"]["attempts"] = v2.sha256_file(context.attempts_path)
    if context.games_path.is_file():
        summary["output_sha256"]["games"] = v2.sha256_file(context.games_path)
    v2._atomic_write_json(context.output_dir / "summary.json", summary)
    manifest = v2._load_json(context.manifest_path, "evaluation manifest")
    manifest["status"] = status
    manifest["technically_valid_games"] = len(games)
    manifest["attempts_recorded"] = summary["attempts"]["total"]
    manifest["technical_failures"] = technical_attempts
    manifest["unresolved_technical_failure_keys"] = unresolved_keys
    manifest["games_sha256"] = v2.sha256_file(context.games_path) if context.games_path.is_file() else None
    manifest["attempts_sha256"] = v2.sha256_file(context.attempts_path) if context.attempts_path.is_file() else None
    manifest["summary"] = summary
    v2._atomic_write_json(context.manifest_path, manifest)
    return summary


def execute(args: argparse.Namespace, logger: EvaluationLogger) -> str:
    context = resolve_context(args)
    resolved, current_inputs, candidate_manifest, pair_rows = build_preparation(context)
    if args.summarize_only:
        _require(context.manifest_path.is_file(), "--summarize-only requires evaluation_manifest.json")
        attempts_by_key, games_by_key = v2.load_and_validate_state(
            context, candidate_manifest["tasks"], recover=True
        )
        summary = summarize_results(
            context, candidate_manifest["tasks"], attempts_by_key, games_by_key
        )
        logger.write(
            f"Head-to-head v3 summary status: {summary['status']}; "
            f"valid_games={summary['games']['technically_valid']}"
        )
        return str(summary["status"])
    if args.retry_game and not args.resume:
        raise HeadToHeadError("--retry-game requires --resume")
    if not args.resume and (context.attempts_path.exists() or context.games_path.exists()):
        raise HeadToHeadError("attempts.jsonl or games.jsonl already exists; pass --resume")
    if args.resume:
        _require(context.manifest_path.is_file(), "--resume requires evaluation_manifest.json")
        existing_manifest = v2._load_json(context.manifest_path, "existing evaluation manifest")
        v2._validate_resume_manifest(existing_manifest, candidate_manifest)
        existing_inputs = v2._load_json(
            context.output_dir / "input_manifest.json", "existing input manifest"
        )
        merged_inputs, implementation_digest = v2._merge_resume_inputs(
            existing_inputs, current_inputs
        )
        v2._atomic_write_json(context.output_dir / "input_manifest.json", merged_inputs)
        manifest = existing_manifest
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["implementation_revision_sha256"] = implementation_digest
        manifest["implementation_revision_count"] = len(merged_inputs["implementation_revisions"])
        v2._atomic_write_json(context.manifest_path, manifest)
    else:
        v2.write_preparation(
            context, resolved, current_inputs, candidate_manifest, pair_rows
        )
        manifest = candidate_manifest
        implementation_digest = current_inputs["implementation_revisions"][0]["sha256"]
    logger.write(
        "Head-to-head v3 prepared: "
        f"horizon={context.common_horizon:.12f} "
        f"Baseline={context.baseline['iteration']}@{context.baseline['actual_gpu_hours']:.12f} "
        f"Adaptive={context.adaptive['iteration']}@{context.adaptive['actual_gpu_hours']:.12f}"
    )
    if args.prepare_only:
        logger.write("Evaluation status: prepared; no head-to-head v3 games started")
        return "prepared"
    tasks = candidate_manifest["tasks"]
    attempts_by_key, games_by_key = v2.load_and_validate_state(
        context, tasks, recover=True
    )
    retry_keys = v2.parse_retry_keys(args.retry_game, tasks)
    manifest = v2._load_json(context.manifest_path, "evaluation manifest")
    manifest["status"] = "running"
    manifest["implementation_revision_sha256"] = implementation_digest
    manifest["requested_retry_keys"] = sorted(retry_keys)
    manifest.pop("failure", None)
    v2._atomic_write_json(context.manifest_path, manifest)
    new_attempts = evaluate_pending_tasks(
        context,
        tasks,
        attempts_by_key,
        games_by_key,
        retry_keys=retry_keys,
        implementation_set_sha256=implementation_digest,
        logger=logger,
    )
    summary = summarize_results(context, tasks, attempts_by_key, games_by_key)
    logger.write(
        "Head-to-head v3 invocation finished: "
        f"new_attempts={new_attempts} "
        f"valid_games={summary['games']['technically_valid']} "
        f"status={summary['status']}"
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
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--retry-game", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        try:
            from verify_head_to_head_v3 import verify_head_to_head_v3

            report = verify_head_to_head_v3(
                config_path=args.config,
                output_dir=args.output_dir,
                matched_compute_path=args.matched_compute,
                baseline_run_dir=args.baseline_run_dir,
                adaptive_run_dir=args.adaptive_run_dir,
                baseline_checkpoint_manifest=args.baseline_checkpoint_manifest,
                adaptive_checkpoint_manifest=args.adaptive_checkpoint_manifest,
            )
            print(
                "Head-to-head v3 verification passed: "
                f"games={report['games']['technically_valid']} "
                f"score_rate={report['adaptive_score']['score_rate']:.6f}"
            )
            return 0
        except (HeadToHeadError, HeadToHeadStatsV3Error, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
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
        except (HeadToHeadError, HeadToHeadStatsV3Error, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            logger.write(f"Head-to-head v3 evaluation failed: {exc}")
            manifest_path = log_dir / "evaluation_manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = v2._load_json(manifest_path, "evaluation manifest")
                    manifest["status"] = "failed"
                    manifest["failure"] = str(exc)
                    v2._atomic_write_json(manifest_path, manifest)
                except Exception:
                    pass
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
