from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (BASELINE_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from arena.utils import calculate_stable_elo_ratings
from evaluate_fixed_basket import (
    EvaluationLogger,
    FixedBasketError,
    load_protocol,
    resolve_output_dir,
    stable_seed,
)


OPPONENT_SUMMARY_FIELDS = (
    "checkpoint",
    "opponent",
    "games",
    "model_white_games",
    "model_black_games",
    "wins",
    "draws",
    "losses",
    "score",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "mean_turns",
    "mean_duration_seconds",
    "faults",
)

CHECKPOINT_SUMMARY_FIELDS = (
    "checkpoint",
    "gpu_hours",
    "total_games",
    "wins",
    "losses",
    "draws",
    "score_rate",
    "win_rate",
    "draw_rate",
    "heuristic_20_score",
    "heuristic_200_score",
    "greedy_random_50_score",
    "random_score",
    "mean_game_length",
    "mean_move_time",
    "invalid_moves",
    "bot_errors",
    "max_turn_draws",
    "js_invalid_proposals",
    "model_fallbacks",
    "score_rate_ci95_low",
    "score_rate_ci95_high",
    "bootstrap_iterations",
)

ELO_SUMMARY_FIELDS = (
    "participant",
    "participant_type",
    "elo",
    "status",
    "fit_scope",
    "random_seed",
)

BOOTSTRAP_ITERATIONS = 10_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize fixed_basket_v1 game results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Protocol YAML; defaults to INPUT_DIR/protocol.resolved.yaml.",
    )
    parser.add_argument("--mode", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fieldnames})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FixedBasketError(f"required JSON file not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixedBasketError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise FixedBasketError(f"JSON root must be an object: {path}")
    return loaded


def _load_games(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FixedBasketError(f"fixed-basket games file not found: {path}")
    games = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixedBasketError(
                    f"invalid games JSONL line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise FixedBasketError(
                    f"games JSONL line {line_number} must contain an object"
                )
            games.append(record)
    return games


def _default_elo_calculator(
    results: list[dict[str, Any]], bot_names: list[str]
) -> dict[str, float]:
    return calculate_stable_elo_ratings(
        results,
        bot_names,
        k_factor=16.0,
        max_iterations=2000,
    )


def _result_score(record: dict[str, Any]) -> float:
    result = record.get("model_result")
    return 1.0 if result == "win" else 0.5 if result == "draw" else 0.0


def _stratified_score_interval(
    records: list[dict[str, Any]],
    *,
    protocol_id: str,
    base_seed: int,
    checkpoint: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float | None, float | None]:
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        strata[(str(record["opponent"]), str(record["model_color"]))].append(
            _result_score(record)
        )
    if not records or len(strata) != 8 or any(not values for values in strata.values()):
        return None, None
    seed = stable_seed(
        f"{protocol_id}:stratified_bootstrap",
        checkpoint,
        "opponent_x_model_color",
        iterations,
        base_seed=base_seed,
    )
    rng = np.random.default_rng(seed)
    bootstrap_scores = np.zeros(iterations, dtype=np.float64)
    total = len(records)
    for values in strata.values():
        scores = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(scores), size=(iterations, len(scores)))
        bootstrap_scores += scores[indices].sum(axis=1) / total
    lower, upper = np.quantile(bootstrap_scores, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_results(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    games: list[dict[str, Any]],
    output_dir: Path,
    *,
    games_path: Path,
    manifest_path: Path,
    mode: str = "formal",
    elo_calculator: Callable[
        [list[dict[str, Any]], list[str]], dict[str, float]
    ] = _default_elo_calculator,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    selected = manifest.get("selected_checkpoints", protocol["checkpoints"])
    if not isinstance(selected, list):
        raise FixedBasketError("manifest selected_checkpoints must be a list")
    selected_set = set(selected)
    expected_checkpoints = [
        checkpoint
        for checkpoint in protocol["checkpoints"]
        if checkpoint in selected_set
    ]
    opponent_ids = [opponent["id"] for opponent in protocol["opponents"]]
    games_per_opponent = int(protocol["games_per_opponent"])
    games_per_side = games_per_opponent // 2
    expected_total = len(expected_checkpoints) * len(opponent_ids) * games_per_opponent
    manifest_entries = {
        int(entry["iteration"]): entry for entry in manifest.get("checkpoints", [])
    }
    errors: list[str] = []
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        errors.append("manifest protocol_id mismatch")
    resolved_hash = manifest.get("protocol_resolved_sha256")
    if isinstance(resolved_hash, str) and resolved_hash != _sha256(protocol["_path"]):
        errors.append("resolved protocol hash mismatch")
    if manifest.get("evaluation_mode", mode) != mode:
        errors.append("manifest evaluation_mode mismatch")
    if not set(expected_checkpoints) <= set(manifest_entries):
        errors.append("manifest checkpoint coverage mismatch")
    if manifest.get("expected_evaluation_games", expected_total) != expected_total:
        errors.append("manifest expected_evaluation_games mismatch")
    if len(games) != expected_total:
        errors.append(f"expected {expected_total} games, found {len(games)}")

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[int, str, int]] = set()
    invalid_records = 0
    faults = 0
    invalid_record_details = []
    for record in games:
        try:
            checkpoint = int(record["checkpoint"])
            opponent_id = str(record["opponent"])
            game_index = int(record["game_index"])
            key = (checkpoint, opponent_id, game_index)
        except (KeyError, TypeError, ValueError):
            invalid_records += 1
            invalid_record_details.append({"key": None, "errors": ["invalid game key"]})
            continue
        record_errors = []
        if key in seen_keys:
            record_errors.append("duplicate game key")
        seen_keys.add(key)
        entry = manifest_entries.get(checkpoint)
        if checkpoint not in expected_checkpoints or opponent_id not in opponent_ids:
            record_errors.append("unconfigured checkpoint or opponent")
        if not 0 <= game_index < games_per_opponent:
            record_errors.append("game_index outside configured range")
        expected_seed = stable_seed(
            protocol["protocol_id"],
            checkpoint,
            opponent_id,
            game_index,
            base_seed=int(protocol["base_seed"]),
        )
        expected_color = "white" if game_index < games_per_side else "black"
        if record.get("game_seed") != expected_seed:
            record_errors.append("stable game seed mismatch")
        if record.get("model_color") != expected_color:
            record_errors.append("model color does not match 25/25 schedule")
        if record.get("max_turns") != protocol["max_turns"]:
            record_errors.append("max_turns mismatch")
        if record.get("protocol_id") != protocol["protocol_id"]:
            record_errors.append("protocol_id mismatch")
        if (
            entry is None
            or record.get("checkpoint_sha256") != entry.get("sha256")
        ):
            record_errors.append("checkpoint hash mismatch")
        if entry is None or record.get("checkpoint_path") != entry.get("path"):
            record_errors.append("checkpoint path mismatch")
        if record.get("model_result") not in {"win", "draw", "loss"}:
            record_errors.append("invalid model result")
        winner = record.get("winner")
        if winner not in {"white", "black", None}:
            record_errors.append("invalid winner")
        termination = record.get("termination")
        if not isinstance(termination, str) or not termination:
            record_errors.append("invalid termination")
        if not isinstance(record.get("moves"), list):
            record_errors.append("moves must be a list")
        temperatures = record.get("model_temperatures")
        if not isinstance(temperatures, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in temperatures if isinstance(temperatures, list)
        ):
            record_errors.append("invalid model temperature history")
        elif any(
            float(value)
            != (
                float(protocol["model"]["early_temperature"])
                if index < protocol["model"]["early_temperature_moves_per_player"]
                else float(protocol["model"]["later_temperature"])
            )
            for index, value in enumerate(temperatures)
        ):
            record_errors.append("model temperature schedule mismatch")
        for numeric_field in ("turns", "duration_seconds"):
            value = record.get(numeric_field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                record_errors.append(f"invalid {numeric_field}")
        if (
            isinstance(record.get("turns"), (int, float))
            and record["turns"] > protocol["max_turns"]
        ):
            record_errors.append("turns exceeds max_turns")
        expected_result = "draw"
        opponent_color = "black" if expected_color == "white" else "white"
        if record.get("fault") == expected_color:
            expected_result = "loss"
        elif record.get("fault") == opponent_color:
            expected_result = "win"
        elif winner == expected_color:
            expected_result = "win"
        elif winner == opponent_color:
            expected_result = "loss"
        if record.get("model_result") != expected_result:
            record_errors.append("model result is inconsistent with winner/fault")
        if record.get("fault") is not None:
            faults += 1
            record_errors.append("game fault present")
        if record_errors:
            invalid_records += 1
            invalid_record_details.append(
                {"key": list(key), "errors": sorted(set(record_errors))}
            )
        groups[(checkpoint, opponent_id)].append(record)

    opponent_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_metrics: dict[str, Any] = {}
    elo_games = []
    for checkpoint in expected_checkpoints:
        opponent_scores: dict[str, float] = {}
        checkpoint_group: list[dict[str, Any]] = []
        for opponent_id in opponent_ids:
            group = groups.get((checkpoint, opponent_id), [])
            if len(group) != games_per_opponent:
                errors.append(
                    f"checkpoint {checkpoint} opponent {opponent_id} has "
                    f"{len(group)} games"
                )
            white_games = sum(record.get("model_color") == "white" for record in group)
            black_games = sum(record.get("model_color") == "black" for record in group)
            if white_games != games_per_side or black_games != games_per_side:
                errors.append(
                    f"checkpoint {checkpoint} opponent {opponent_id} side split "
                    f"is {white_games}/{black_games}"
                )
            wins = sum(record.get("model_result") == "win" for record in group)
            draws = sum(record.get("model_result") == "draw" for record in group)
            losses = sum(record.get("model_result") == "loss" for record in group)
            count = len(group)
            score = (wins + 0.5 * draws) / count if count else 0.0
            opponent_scores[opponent_id] = score
            checkpoint_group.extend(group)
            opponent_rows.append(
                {
                    "checkpoint": checkpoint,
                    "opponent": opponent_id,
                    "games": count,
                    "model_white_games": white_games,
                    "model_black_games": black_games,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "score": score,
                    "win_rate": wins / count if count else 0.0,
                    "draw_rate": draws / count if count else 0.0,
                    "loss_rate": losses / count if count else 0.0,
                    "mean_turns": (
                        sum(float(record.get("turns", 0)) for record in group) / count
                        if count
                        else 0.0
                    ),
                    "mean_duration_seconds": (
                        sum(
                            float(record.get("duration_seconds", 0.0))
                            for record in group
                        )
                        / count
                        if count
                        else 0.0
                    ),
                    "faults": sum(record.get("fault") is not None for record in group),
                }
            )
            for record in group:
                outcome = record.get("model_result")
                elo_games.append(
                    {
                        "bot1": f"checkpoint_{checkpoint}",
                        "bot2": opponent_id,
                        "score_bot1": 1.0 if outcome == "win" else 0.5 if outcome == "draw" else 0.0,
                    }
                )
        checkpoint_games = len(checkpoint_group)
        checkpoint_wins = sum(
            record.get("model_result") == "win" for record in checkpoint_group
        )
        checkpoint_draws = sum(
            record.get("model_result") == "draw" for record in checkpoint_group
        )
        checkpoint_losses = sum(
            record.get("model_result") == "loss" for record in checkpoint_group
        )
        checkpoint_score = (
            (checkpoint_wins + 0.5 * checkpoint_draws) / checkpoint_games
            if checkpoint_games
            else 0.0
        )
        macro_score = (
            sum(opponent_scores.values()) / len(opponent_scores)
            if opponent_scores
            else 0.0
        )
        if (
            checkpoint_games == len(opponent_ids) * games_per_opponent
            and not math.isclose(checkpoint_score, macro_score, abs_tol=1e-12)
        ):
            errors.append(
                f"checkpoint {checkpoint} direct score and equal-opponent score differ"
            )
        gpu_seconds = sum(
            float(record.get("duration_seconds", 0.0))
            for record in checkpoint_group
        )
        total_model_move_seconds = sum(
            float(record.get("model_move_seconds", 0.0))
            for record in checkpoint_group
        )
        total_model_moves = sum(
            int(record.get("model_moves", 0)) for record in checkpoint_group
        )
        ci_low, ci_high = _stratified_score_interval(
            checkpoint_group,
            protocol_id=protocol["protocol_id"],
            base_seed=int(protocol["base_seed"]),
            checkpoint=checkpoint,
        )
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "gpu_hours": gpu_seconds / 3600.0,
                "total_games": checkpoint_games,
                "wins": checkpoint_wins,
                "losses": checkpoint_losses,
                "draws": checkpoint_draws,
                "score_rate": checkpoint_score,
                "win_rate": (
                    checkpoint_wins / checkpoint_games if checkpoint_games else 0.0
                ),
                "draw_rate": (
                    checkpoint_draws / checkpoint_games if checkpoint_games else 0.0
                ),
                "heuristic_20_score": opponent_scores.get("heuristic_20", 0.0),
                "heuristic_200_score": opponent_scores.get("heuristic_200", 0.0),
                "greedy_random_50_score": opponent_scores.get(
                    "greedy_random_50", 0.0
                ),
                "random_score": opponent_scores.get("random", 0.0),
                "mean_game_length": (
                    sum(float(record.get("turns", 0)) for record in checkpoint_group)
                    / checkpoint_games
                    if checkpoint_games
                    else 0.0
                ),
                "mean_move_time": (
                    total_model_move_seconds / total_model_moves
                    if total_model_moves
                    else 0.0
                ),
                "invalid_moves": sum(
                    record.get("termination") == "invalid_move"
                    for record in checkpoint_group
                ),
                "bot_errors": sum(
                    record.get("termination") == "bot_error"
                    for record in checkpoint_group
                ),
                "max_turn_draws": sum(
                    record.get("termination") == "max_turns"
                    and record.get("model_result") == "draw"
                    for record in checkpoint_group
                ),
                "js_invalid_proposals": sum(
                    int(record.get("opponent_invalid_proposals", 0))
                    for record in checkpoint_group
                ),
                "model_fallbacks": sum(
                    int(record.get("model_fallback_count", 0))
                    for record in checkpoint_group
                ),
                "score_rate_ci95_low": ci_low,
                "score_rate_ci95_high": ci_high,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
        checkpoint_metrics[str(checkpoint)] = {
            "total_games": checkpoint_games,
            "score_rate": checkpoint_score,
            "equal_opponent_score_rate": macro_score,
            "score_rate_ci95": [ci_low, ci_high],
        }

    if invalid_records:
        errors.append(f"{invalid_records} game records failed validation")
    bot_names = [
        f"checkpoint_{iteration}" for iteration in expected_checkpoints
    ] + opponent_ids
    random.seed(int(protocol["base_seed"]))
    elo_ratings = (
        elo_calculator(elo_games, bot_names)
        if mode == "formal" and elo_games
        else {}
    )
    status = "completed" if not errors and faults == 0 else "failed"
    checkpoint_path = output_dir / "checkpoint_summary.csv"
    opponent_path = output_dir / "opponent_summary.csv"
    elo_path = output_dir / "elo_summary.csv"
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": status,
        "expected_games": expected_total,
        "observed_games": len(games),
        "expected_games_per_checkpoint_opponent": games_per_opponent,
        "expected_model_games_per_side": games_per_side,
        "max_turns": protocol["max_turns"],
        "checkpoint_metrics": checkpoint_metrics,
        "metric_definitions": {
            "gpu_hours": "sum(duration_seconds) / 3600; one GPU used serially",
            "score_rate": "(wins + 0.5 * draws) / total_games",
            "fixed_basket_score": (
                "equal-weight mean of the four opponent scores; identical to the "
                "pooled score because every opponent contributes the same game count"
            ),
            "mean_game_length": "mean(turns)",
            "mean_move_time": "sum(model_move_seconds) / sum(model_moves)",
            "max_turn_draws": (
                "games with termination=max_turns and model_result=draw"
            ),
            "js_invalid_proposals": (
                "illegal raw JS proposals rejected before arena play; these are "
                "reported separately from game-level invalid_move terminations"
            ),
            "model_fallbacks": (
                "arena-legal deterministic model fallbacks after the AlphaZero "
                "policy cannot map positive mass to a legal pyquoridor move"
            ),
        },
        "confidence_intervals": {
            "method": "nonparametric stratified bootstrap",
            "strata": ["opponent", "model_color"],
            "confidence_level": 0.95,
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed_derivation": (
                "first 32 bits of SHA-256 using protocol, checkpoint, stratum label, "
                "iteration count, and base_seed"
            ),
        },
        "elo": {
            "status": "provisional" if mode == "formal" else "not_computed",
            "random_seed": int(protocol["base_seed"]),
            "fit_scope": "baseline checkpoints and fixed opponents",
            "ratings": dict(sorted(elo_ratings.items())),
            "finalization_note": (
                "After Adaptive evaluation, jointly fit Baseline checkpoints, "
                "Adaptive checkpoints, and fixed opponents in one final Elo model."
            ),
        },
        "data_quality": {
            "unique_game_keys": len(seen_keys),
            "invalid_records": invalid_records,
            "faults": faults,
            "termination_counts": dict(
                sorted(
                    (termination, sum(record.get("termination") == termination for record in games))
                    for termination in {
                        str(record.get("termination")) for record in games
                    }
                )
            ),
            "js_invalid_proposals": sum(
                int(record.get("opponent_invalid_proposals", 0))
                for record in games
            ),
            "model_fallbacks": sum(
                int(record.get("model_fallback_count", 0)) for record in games
            ),
            "errors": errors,
            "invalid_record_details": invalid_record_details,
        },
        "inputs": {
            "games_jsonl": games_path.as_posix(),
            "games_jsonl_sha256": _sha256(games_path),
            "evaluation_manifest": manifest_path.as_posix(),
            "protocol_config": protocol["_path"].as_posix(),
            "protocol_config_sha256": _sha256(protocol["_path"]),
        },
        "outputs": {"checkpoint_summary": checkpoint_path.as_posix()},
        "sanity_evaluation_modified": False,
    }
    _atomic_write_csv(
        checkpoint_path, checkpoint_rows, CHECKPOINT_SUMMARY_FIELDS
    )
    if mode == "formal":
        elo_rows = [
            {
                "participant": participant,
                "participant_type": (
                    "checkpoint" if participant.startswith("checkpoint_") else "opponent"
                ),
                "elo": rating,
                "status": "provisional",
                "fit_scope": "baseline checkpoints and fixed opponents",
                "random_seed": int(protocol["base_seed"]),
            }
            for participant, rating in sorted(elo_ratings.items())
        ]
        _atomic_write_csv(
            opponent_path, opponent_rows, OPPONENT_SUMMARY_FIELDS
        )
        _atomic_write_csv(elo_path, elo_rows, ELO_SUMMARY_FIELDS)
        summary["outputs"].update(
            {
                "opponent_summary": opponent_path.as_posix(),
                "elo_summary": elo_path.as_posix(),
            }
        )
    summary["output_sha256"] = {
        name: _sha256(Path(path)) for name, path in summary["outputs"].items()
    }
    updated_manifest = dict(manifest)
    updated_manifest["summary"] = summary
    outputs = dict(updated_manifest.get("outputs", {}))
    outputs.update(summary["outputs"])
    updated_manifest["outputs"] = outputs
    _atomic_write_json(manifest_path, updated_manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else resolve_output_dir(args.mode, None)
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir
    )
    with EvaluationLogger(output_dir / "evaluation.log") as logger:
        try:
            config_path = (
                args.config
                if args.config is not None
                else input_dir / "protocol.resolved.yaml"
            )
            protocol = load_protocol(
                config_path,
                allow_games_per_opponent_override=args.mode == "pilot",
            )
            manifest_path = input_dir / "evaluation_manifest.json"
            games_path = input_dir / "games.jsonl"
            manifest = _load_json(manifest_path)
            games = _load_games(games_path)
            summary = summarize_results(
                protocol,
                manifest,
                games,
                output_dir,
                games_path=games_path,
                manifest_path=manifest_path,
                mode=args.mode,
            )
            logger.write(f"Observed games: {summary['observed_games']}")
            logger.write(f"Expected games: {summary['expected_games']}")
            logger.write(f"Summary status: {summary['status']}")
            return 0 if summary["status"] == "completed" else 2
        except (FixedBasketError, OSError, ValueError, KeyError) as exc:
            logger.write(f"Fixed-basket summary failed: {exc}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
