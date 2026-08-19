from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_CONFIG = BASELINE_ROOT / "analysis" / "configs" / "baseline_gate2.yaml"
DEFAULT_BASKET_CONFIG = (
    BASELINE_ROOT / "analysis" / "configs" / "fixed_basket_v1.yaml"
)
DEFAULT_BASKET_OUTPUT = (
    BASELINE_ROOT / "outputs" / "baseline_seed1001_4090_analysis" / "fixed_basket_v1"
)
DEFAULT_GAMES = DEFAULT_BASKET_OUTPUT / "games.jsonl"
DEFAULT_CHECKPOINT_SUMMARY = DEFAULT_BASKET_OUTPUT / "checkpoint_summary.csv"
DEFAULT_MANIFEST = DEFAULT_BASKET_OUTPUT / "evaluation_manifest.json"
DEFAULT_OUTPUT_DIR = (
    BASELINE_ROOT / "outputs" / "baseline_seed1001_4090_analysis" / "h1_v1_1"
)

WINDOW_FIELDS = (
    "window_index",
    "start_checkpoint",
    "end_checkpoint",
    "checkpoint_iterations",
    "start_score",
    "end_score",
    "endpoint_score_change",
    "absolute_endpoint_score_change",
    "ols_slope_per_iteration",
    "ols_slope_per_20_iterations",
    "slope_ci95_low_per_20_iterations",
    "slope_ci95_high_per_20_iterations",
    "bootstrap_mean_slope_per_20_iterations",
    "bootstrap_seed",
    "bootstrap_resamples",
    "slope_ci_contains_zero",
    "endpoint_change_within_limit",
    "qualifies",
)


class PlateauInputError(ValueError):
    """Raised when fixed-basket inputs cannot support a formal plateau result."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fixed_basket_v1 and detect consecutive flat score windows."
        )
    )
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--basket-config", type=Path, default=DEFAULT_BASKET_CONFIG)
    parser.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    parser.add_argument(
        "--checkpoint-summary", type=Path, default=DEFAULT_CHECKPOINT_SUMMARY
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlateauInputError(f"required YAML file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PlateauInputError(f"cannot read YAML file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PlateauInputError(f"YAML root must be a mapping: {path}")
    return loaded


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlateauInputError(f"required JSON file not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlateauInputError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PlateauInputError(f"JSON root must be an object: {path}")
    return loaded


def _load_games(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PlateauInputError(f"required games JSONL file not found: {path}")
    games: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PlateauInputError(
                        f"invalid games JSONL line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise PlateauInputError(
                        f"games JSONL line {line_number} must contain an object"
                    )
                games.append(record)
    except (OSError, UnicodeError) as exc:
        raise PlateauInputError(f"cannot read games JSONL {path}: {exc}") from exc
    return games


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PlateauInputError(f"required checkpoint summary not found: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise PlateauInputError(f"checkpoint summary has no header: {path}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PlateauInputError(f"cannot read checkpoint summary {path}: {exc}") from exc


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PlateauInputError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise PlateauInputError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PlateauInputError(f"{label} must be an integer")
    return converted


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PlateauInputError(f"{label} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise PlateauInputError(f"{label} must be numeric") from exc
    if not math.isfinite(converted):
        raise PlateauInputError(f"{label} must be finite")
    return converted


def _checkpoint_values(protocol: Mapping[str, Any]) -> list[int]:
    values = protocol.get("checkpoints")
    if not isinstance(values, list):
        raise PlateauInputError("fixed-basket checkpoints must be a list")
    return [_as_int(value, "checkpoint") for value in values]


def _opponent_ids(protocol: Mapping[str, Any]) -> list[str]:
    opponents = protocol.get("opponents")
    if not isinstance(opponents, list):
        raise PlateauInputError("fixed-basket opponents must be a list")
    ids = []
    for opponent in opponents:
        if not isinstance(opponent, dict) or not isinstance(opponent.get("id"), str):
            raise PlateauInputError("every fixed-basket opponent must have an id")
        ids.append(opponent["id"])
    return ids


def _validate_fixed_basket_protocol(protocol: Mapping[str, Any]) -> None:
    expected_checkpoints = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 210]
    expected_opponents = [
        "heuristic_20",
        "heuristic_200",
        "greedy_random_50",
        "random",
    ]
    if protocol.get("schema_version") != 1:
        raise PlateauInputError("fixed-basket schema_version must be 1")
    if protocol.get("protocol_id") != "fixed_basket_v1":
        raise PlateauInputError("fixed-basket protocol_id must be fixed_basket_v1")
    if _checkpoint_values(protocol) != expected_checkpoints:
        raise PlateauInputError("fixed-basket must register the 12 formal checkpoints")
    if _opponent_ids(protocol) != expected_opponents:
        raise PlateauInputError("fixed-basket must register the four v1 opponents")
    if protocol.get("games_per_opponent") != 50:
        raise PlateauInputError("fixed-basket games_per_opponent must be 50")
    if protocol.get("alternate_sides") is not True:
        raise PlateauInputError("fixed-basket must alternate model sides")


def _plateau_settings(
    gate_config: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    gate_checkpoints = gate_config.get("checkpoints")
    if not isinstance(gate_checkpoints, dict) or gate_checkpoints.get(
        "iterations"
    ) != _checkpoint_values(protocol):
        raise PlateauInputError("gate2 and fixed-basket checkpoint grids differ")
    plateau = gate_config.get("plateau")
    if not isinstance(plateau, dict):
        raise PlateauInputError("baseline_gate2.yaml must contain a plateau mapping")
    if plateau.get("metric") != "fixed_basket_macro_score":
        raise PlateauInputError("plateau metric must be fixed_basket_macro_score")
    if plateau.get("method") != (
        "rolling_ols_slope_with_paired_stratified_bootstrap"
    ):
        raise PlateauInputError("unsupported plateau method")

    integer_fields = (
        "window_checkpoints",
        "bootstrap_seed",
        "bootstrap_resamples",
        "slope_unit_iterations",
        "consecutive_qualifying_windows",
    )
    settings = dict(plateau)
    for field in integer_fields:
        settings[field] = _as_int(plateau.get(field), f"plateau.{field}")
        if settings[field] <= 0:
            raise PlateauInputError(f"plateau.{field} must be positive")
    settings["confidence_level"] = _as_finite_float(
        plateau.get("confidence_level"), "plateau.confidence_level"
    )
    if not 0.0 < settings["confidence_level"] < 1.0:
        raise PlateauInputError("plateau.confidence_level must be between 0 and 1")
    settings["maximum_absolute_score_change_over_window"] = _as_finite_float(
        plateau.get("maximum_absolute_score_change_over_window"),
        "plateau.maximum_absolute_score_change_over_window",
    )
    if settings["maximum_absolute_score_change_over_window"] < 0.0:
        raise PlateauInputError(
            "plateau.maximum_absolute_score_change_over_window must be nonnegative"
        )
    if plateau.get("require_slope_confidence_interval_contains_zero") is not True:
        raise PlateauInputError("plateau must require its slope CI to contain zero")
    if plateau.get("plateau_iteration") != (
        "first_checkpoint_of_first_qualifying_window"
    ):
        raise PlateauInputError("unsupported plateau iteration definition")
    return settings


def _game_key(record: Mapping[str, Any]) -> tuple[int, str, int]:
    try:
        checkpoint = _as_int(record["checkpoint"], "game checkpoint")
        opponent = record["opponent"]
        game_index = _as_int(record["game_index"], "game_index")
    except KeyError as exc:
        raise PlateauInputError(f"game record missing key field: {exc.args[0]}") from exc
    if not isinstance(opponent, str) or not opponent:
        raise PlateauInputError("game opponent must be a non-empty string")
    return checkpoint, opponent, game_index


def validate_evaluation_status(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    checkpoints = _checkpoint_values(protocol)
    expected_total = (
        len(checkpoints)
        * len(_opponent_ids(protocol))
        * _as_int(protocol.get("games_per_opponent"), "games_per_opponent")
    )
    errors = []
    if manifest.get("status") != "completed":
        errors.append("manifest status is not completed")
    if manifest.get("evaluation_mode") != "formal":
        errors.append("manifest evaluation_mode is not formal")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        errors.append("manifest protocol_id does not match fixed_basket_v1")
    if manifest.get("selected_checkpoints") != checkpoints:
        errors.append("manifest does not contain the 12 formal checkpoints")
    if manifest.get("expected_evaluation_games") != expected_total:
        errors.append(f"manifest expected_evaluation_games is not {expected_total}")
    summary = manifest.get("summary")
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        errors.append("manifest summary status is not completed")
    elif summary.get("observed_games") != expected_total:
        errors.append(f"manifest summary observed_games is not {expected_total}")
    if errors:
        raise PlateauInputError("; ".join(errors))


def validate_game_coverage(
    games: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    checkpoints = _checkpoint_values(protocol)
    opponents = _opponent_ids(protocol)
    games_per_opponent = _as_int(
        protocol.get("games_per_opponent"), "games_per_opponent"
    )
    expected_total = len(checkpoints) * len(opponents) * games_per_opponent
    if len(games) != expected_total:
        raise PlateauInputError(
            f"expected {expected_total} games, found {len(games)}"
        )

    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for record in games:
        checkpoint, opponent, game_index = _game_key(record)
        if checkpoint not in checkpoints:
            raise PlateauInputError(f"unregistered checkpoint in games: {checkpoint}")
        if opponent not in opponents:
            raise PlateauInputError(f"unregistered opponent in games: {opponent}")
        groups[(checkpoint, opponent)].append(game_index)

    expected_indices = set(range(games_per_opponent))
    for checkpoint in checkpoints:
        for opponent in opponents:
            indices = groups.get((checkpoint, opponent), [])
            if len(indices) != games_per_opponent:
                raise PlateauInputError(
                    f"checkpoint {checkpoint} opponent {opponent} has "
                    f"{len(indices)} games; expected {games_per_opponent}"
                )
            if set(indices) != expected_indices:
                raise PlateauInputError(
                    f"checkpoint {checkpoint} opponent {opponent} does not cover "
                    f"game_index 0..{games_per_opponent - 1}"
                )


def validate_unique_game_keys(
    games: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    expected_total = (
        len(_checkpoint_values(protocol))
        * len(_opponent_ids(protocol))
        * _as_int(protocol.get("games_per_opponent"), "games_per_opponent")
    )
    keys = [_game_key(record) for record in games]
    unique_count = len(set(keys))
    if len(keys) != expected_total or unique_count != expected_total:
        raise PlateauInputError(
            f"expected {expected_total} unique game keys, found {unique_count} "
            f"across {len(keys)} records"
        )


def validate_side_balance(
    games: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    checkpoints = _checkpoint_values(protocol)
    opponents = _opponent_ids(protocol)
    games_per_opponent = _as_int(
        protocol.get("games_per_opponent"), "games_per_opponent"
    )
    games_per_side = games_per_opponent // 2
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in games:
        checkpoint, opponent, _ = _game_key(record)
        groups[(checkpoint, opponent)].append(record)
    for checkpoint in checkpoints:
        for opponent in opponents:
            group = groups.get((checkpoint, opponent), [])
            white = sum(record.get("model_color") == "white" for record in group)
            black = sum(record.get("model_color") == "black" for record in group)
            invalid = sum(
                record.get("model_color") not in {"white", "black"}
                for record in group
            )
            if white != games_per_side or black != games_per_side or invalid:
                raise PlateauInputError(
                    f"checkpoint {checkpoint} opponent {opponent} side balance is "
                    f"white={white}, black={black}; expected {games_per_side}/{games_per_side}"
                )


def validate_terminations(games: Sequence[Mapping[str, Any]]) -> None:
    allowed_terminations = {"win", "max_turns"}
    for record in games:
        key = _game_key(record)
        termination = record.get("termination")
        result = record.get("model_result")
        if termination not in allowed_terminations:
            raise PlateauInputError(
                f"game {key} has error termination {termination!r}"
            )
        if record.get("fault") is not None:
            raise PlateauInputError(f"game {key} has non-null fault")
        if result not in {"win", "draw", "loss"}:
            raise PlateauInputError(f"game {key} has invalid model_result {result!r}")
        if termination == "max_turns" and result != "draw":
            raise PlateauInputError(
                f"game {key} reached max_turns without a draw result"
            )
        if termination == "win" and result == "draw":
            raise PlateauInputError(f"game {key} has a draw with win termination")


def _outcome_score(record: Mapping[str, Any]) -> float:
    result = record.get("model_result")
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    if result == "loss":
        return 0.0
    raise PlateauInputError(f"game {_game_key(record)} has invalid model_result")


def reconstruct_checkpoint_scores(
    games: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    checkpoints = _checkpoint_values(protocol)
    opponents = _opponent_ids(protocol)
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in games:
        checkpoint, opponent, _ = _game_key(record)
        grouped[(checkpoint, opponent)].append(record)

    reconstructed: dict[int, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        opponent_scores: dict[str, float] = {}
        checkpoint_games: list[Mapping[str, Any]] = []
        for opponent in opponents:
            group = grouped.get((checkpoint, opponent), [])
            if not group:
                raise PlateauInputError(
                    f"cannot reconstruct checkpoint {checkpoint}: no games for {opponent}"
                )
            opponent_scores[opponent] = sum(
                _outcome_score(record) for record in group
            ) / len(group)
            checkpoint_games.extend(group)
        wins = sum(record.get("model_result") == "win" for record in checkpoint_games)
        draws = sum(
            record.get("model_result") == "draw" for record in checkpoint_games
        )
        losses = sum(
            record.get("model_result") == "loss" for record in checkpoint_games
        )
        reconstructed[checkpoint] = {
            "opponent_scores": opponent_scores,
            "macro_score": sum(opponent_scores.values()) / len(opponents),
            "total_games": len(checkpoint_games),
            "wins": wins,
            "draws": draws,
            "losses": losses,
        }
    return reconstructed


def validate_summary_scores(
    reconstructed: Mapping[int, Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-12,
) -> None:
    checkpoints = _checkpoint_values(protocol)
    opponents = _opponent_ids(protocol)
    rows_by_checkpoint: dict[int, Mapping[str, Any]] = {}
    for row in summary_rows:
        checkpoint = _as_int(row.get("checkpoint"), "summary checkpoint")
        if checkpoint in rows_by_checkpoint:
            raise PlateauInputError(
                f"checkpoint summary has duplicate checkpoint {checkpoint}"
            )
        rows_by_checkpoint[checkpoint] = row
    if set(rows_by_checkpoint) != set(checkpoints):
        raise PlateauInputError("checkpoint summary does not cover the 12 checkpoints")

    for checkpoint in checkpoints:
        expected = reconstructed[checkpoint]
        row = rows_by_checkpoint[checkpoint]
        reported_macro = _as_finite_float(
            row.get("score_rate"), f"checkpoint {checkpoint} score_rate"
        )
        if not math.isclose(
            reported_macro,
            float(expected["macro_score"]),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise PlateauInputError(
                f"checkpoint {checkpoint} summary score_rate {reported_macro} "
                f"does not match reconstructed macro score {expected['macro_score']}"
            )
        opponent_scores = expected["opponent_scores"]
        for opponent in opponents:
            column = f"{opponent}_score"
            reported = _as_finite_float(
                row.get(column), f"checkpoint {checkpoint} {column}"
            )
            if not math.isclose(
                reported,
                float(opponent_scores[opponent]),
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            ):
                raise PlateauInputError(
                    f"checkpoint {checkpoint} {column} does not match raw games"
                )
        for field in ("total_games", "wins", "draws", "losses"):
            reported_count = _as_int(
                row.get(field), f"checkpoint {checkpoint} {field}"
            )
            if reported_count != expected[field]:
                raise PlateauInputError(
                    f"checkpoint {checkpoint} {field} does not match raw games"
                )


def build_rolling_windows(
    checkpoints: Sequence[int], window_size: int = 4
) -> list[tuple[int, ...]]:
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise PlateauInputError("window_size must be an integer")
    if window_size < 2:
        raise PlateauInputError("window_size must be at least 2")
    values = [_as_int(value, "checkpoint") for value in checkpoints]
    if any(right <= left for left, right in zip(values, values[1:])):
        raise PlateauInputError("checkpoints must be strictly increasing")
    if len(values) < window_size:
        return []
    return [
        tuple(values[index : index + window_size])
        for index in range(len(values) - window_size + 1)
    ]


def ols_slope(x_values: Sequence[int | float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise PlateauInputError("OLS requires equal-length x/y with at least two points")
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise PlateauInputError("OLS inputs must be finite")
    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise PlateauInputError("OLS checkpoint axis has zero variance")
    return float(centered @ y / denominator)


def paired_stratified_bootstrap_slope(
    games: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[int],
    opponents: Sequence[str],
    *,
    bootstrap_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 81_001,
) -> dict[str, float | int]:
    checkpoint_values = [_as_int(value, "checkpoint") for value in checkpoints]
    if len(checkpoint_values) < 2:
        raise PlateauInputError("bootstrap slope requires at least two checkpoints")
    if bootstrap_resamples <= 0:
        raise PlateauInputError("bootstrap_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise PlateauInputError("confidence_level must be between 0 and 1")
    opponent_values = [str(opponent) for opponent in opponents]
    if not opponent_values or len(set(opponent_values)) != len(opponent_values):
        raise PlateauInputError("bootstrap opponents must be unique and non-empty")

    requested_checkpoints = set(checkpoint_values)
    requested_opponents = set(opponent_values)
    lookup: dict[tuple[int, str, str], dict[int, float]] = defaultdict(dict)
    for record in games:
        checkpoint, opponent, game_index = _game_key(record)
        if checkpoint not in requested_checkpoints or opponent not in requested_opponents:
            continue
        color = record.get("model_color")
        if color not in {"white", "black"}:
            raise PlateauInputError(
                f"game {(checkpoint, opponent, game_index)} has invalid model_color"
            )
        stratum = (checkpoint, opponent, color)
        if game_index in lookup[stratum]:
            raise PlateauInputError(
                f"duplicate bootstrap game key {(checkpoint, opponent, game_index)}"
            )
        lookup[stratum][game_index] = _outcome_score(record)

    strata = [
        (opponent, color)
        for opponent in opponent_values
        for color in ("white", "black")
    ]
    matrices: list[np.ndarray] = []
    common_size: int | None = None
    for opponent, color in strata:
        reference_indices = sorted(
            lookup.get((checkpoint_values[0], opponent, color), {})
        )
        if not reference_indices:
            raise PlateauInputError(
                f"bootstrap stratum {opponent}/{color} has no games"
            )
        for checkpoint in checkpoint_values[1:]:
            indices = sorted(lookup.get((checkpoint, opponent, color), {}))
            if indices != reference_indices:
                raise PlateauInputError(
                    f"bootstrap stratum {opponent}/{color} does not share game indices "
                    "across checkpoints"
                )
        if common_size is None:
            common_size = len(reference_indices)
        elif len(reference_indices) != common_size:
            raise PlateauInputError("bootstrap strata have unequal game counts")
        matrices.append(
            np.asarray(
                [
                    [
                        lookup[(checkpoint, opponent, color)][game_index]
                        for game_index in reference_indices
                    ]
                    for checkpoint in checkpoint_values
                ],
                dtype=np.float64,
            )
        )

    assert common_size is not None
    rng = np.random.default_rng(seed)
    bootstrap_scores = np.zeros(
        (bootstrap_resamples, len(checkpoint_values)), dtype=np.float64
    )
    for matrix in matrices:
        sampled_positions = rng.integers(
            0, common_size, size=(bootstrap_resamples, common_size)
        )
        sampled_means = matrix[:, sampled_positions].mean(axis=2).T
        bootstrap_scores += sampled_means / len(matrices)

    x = np.asarray(checkpoint_values, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    bootstrap_slopes = bootstrap_scores @ centered / denominator
    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(
        bootstrap_slopes, [alpha, 1.0 - alpha], method="linear"
    )
    observed_scores = np.mean(np.stack(matrices, axis=0), axis=(0, 2))
    return {
        "observed_slope": ols_slope(checkpoint_values, observed_scores.tolist()),
        "bootstrap_mean_slope": float(bootstrap_slopes.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_resamples": int(bootstrap_resamples),
        "seed": int(seed),
    }


def qualify_flat_window(
    slope_ci_low: float,
    slope_ci_high: float,
    start_score: float,
    end_score: float,
    maximum_absolute_score_change: float = 0.03,
) -> bool:
    values = [slope_ci_low, slope_ci_high, start_score, end_score]
    if not all(math.isfinite(float(value)) for value in values):
        raise PlateauInputError("flat-window inputs must be finite")
    if slope_ci_low > slope_ci_high:
        raise PlateauInputError("slope CI lower bound exceeds upper bound")
    if maximum_absolute_score_change < 0.0:
        raise PlateauInputError("maximum score change must be nonnegative")
    return (
        slope_ci_low <= 0.0 <= slope_ci_high
        and abs(end_score - start_score) <= maximum_absolute_score_change
    )


def detect_consecutive_plateau_windows(
    window_results: Sequence[Mapping[str, Any]],
    required_consecutive_windows: int = 2,
) -> dict[str, Any]:
    if required_consecutive_windows <= 0:
        raise PlateauInputError("required_consecutive_windows must be positive")
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    qualifying_indices: list[int] = []
    for index, window in enumerate(window_results):
        qualifies = window.get("qualifies") is True
        if qualifies:
            qualifying_indices.append(index)
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append((run_start, index - run_start))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(window_results) - run_start))

    qualifying_run = next(
        (run for run in runs if run[1] >= required_consecutive_windows), None
    )
    if qualifying_run is not None:
        first_index, run_length = qualifying_run
        first = window_results[first_index]
        confirmation_index = first_index + required_consecutive_windows - 1
        confirmation = window_results[confirmation_index]
        plateau_start = _as_int(
            first.get("start_checkpoint"), "window start_checkpoint"
        )
        return {
            "plateau_detected": True,
            "plateau_iteration": plateau_start,
            "plateau_start_checkpoint": plateau_start,
            "first_qualifying_window_index": first_index,
            "first_qualifying_window_start": plateau_start,
            "first_qualifying_window_end": _as_int(
                first.get("end_checkpoint"), "window end_checkpoint"
            ),
            "confirmation_window_start": _as_int(
                confirmation.get("start_checkpoint"),
                "confirmation window start_checkpoint",
            ),
            "confirmation_window_end": _as_int(
                confirmation.get("end_checkpoint"),
                "confirmation window end_checkpoint",
            ),
            "plateau_confirmation_checkpoint": _as_int(
                confirmation.get("end_checkpoint"),
                "confirmation window end_checkpoint",
            ),
            "consecutive_qualifying_windows": run_length,
            "qualifying_window_indices": qualifying_indices,
        }
    return {
        "plateau_detected": False,
        "plateau_iteration": None,
        "plateau_start_checkpoint": None,
        "first_qualifying_window_index": None,
        "first_qualifying_window_start": None,
        "first_qualifying_window_end": None,
        "confirmation_window_start": None,
        "confirmation_window_end": None,
        "plateau_confirmation_checkpoint": None,
        "consecutive_qualifying_windows": 0,
        "qualifying_window_indices": qualifying_indices,
    }


def _analyse_windows(
    games: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    checkpoint_scores: Mapping[int, Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints = _checkpoint_values(protocol)
    opponents = _opponent_ids(protocol)
    windows = build_rolling_windows(checkpoints, settings["window_checkpoints"])
    rows: list[dict[str, Any]] = []
    decision_windows: list[dict[str, Any]] = []
    slope_unit = settings["slope_unit_iterations"]
    max_change = settings["maximum_absolute_score_change_over_window"]
    for window_index, window in enumerate(windows):
        window_seed = settings["bootstrap_seed"] + window_index
        bootstrap = paired_stratified_bootstrap_slope(
            games,
            window,
            opponents,
            bootstrap_resamples=settings["bootstrap_resamples"],
            confidence_level=settings["confidence_level"],
            seed=window_seed,
        )
        scores = [float(checkpoint_scores[value]["macro_score"]) for value in window]
        observed_slope = ols_slope(window, scores)
        if not math.isclose(
            observed_slope,
            float(bootstrap["observed_slope"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise PlateauInputError(
                f"window {window_index} score reconstruction differs in bootstrap"
            )
        ci_low = float(bootstrap["ci_low"])
        ci_high = float(bootstrap["ci_high"])
        endpoint_change = scores[-1] - scores[0]
        qualifies = qualify_flat_window(
            ci_low,
            ci_high,
            scores[0],
            scores[-1],
            max_change,
        )
        decision_window = {
            "window_index": window_index,
            "start_checkpoint": window[0],
            "end_checkpoint": window[-1],
            "qualifies": qualifies,
        }
        decision_windows.append(decision_window)
        rows.append(
            {
                "window_index": window_index,
                "start_checkpoint": window[0],
                "end_checkpoint": window[-1],
                "checkpoint_iterations": "|".join(str(value) for value in window),
                "start_score": scores[0],
                "end_score": scores[-1],
                "endpoint_score_change": endpoint_change,
                "absolute_endpoint_score_change": abs(endpoint_change),
                "ols_slope_per_iteration": observed_slope,
                "ols_slope_per_20_iterations": observed_slope * slope_unit,
                "slope_ci95_low_per_20_iterations": ci_low * slope_unit,
                "slope_ci95_high_per_20_iterations": ci_high * slope_unit,
                "bootstrap_mean_slope_per_20_iterations": (
                    float(bootstrap["bootstrap_mean_slope"]) * slope_unit
                ),
                "bootstrap_seed": window_seed,
                "bootstrap_resamples": settings["bootstrap_resamples"],
                "slope_ci_contains_zero": ci_low <= 0.0 <= ci_high,
                "endpoint_change_within_limit": abs(endpoint_change) <= max_change,
                "qualifies": qualifies,
            }
        )
    decision = detect_consecutive_plateau_windows(
        decision_windows,
        settings["consecutive_qualifying_windows"],
    )
    return rows, decision


def _atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    gate_config = _load_yaml_mapping(args.gate_config)
    protocol = _load_yaml_mapping(args.basket_config)
    _validate_fixed_basket_protocol(protocol)
    settings = _plateau_settings(gate_config, protocol)
    games = _load_games(args.games)
    summary_rows = _load_summary_rows(args.checkpoint_summary)
    manifest = _load_json_mapping(args.manifest)

    validate_evaluation_status(manifest, protocol)
    validate_game_coverage(games, protocol)
    validate_unique_game_keys(games, protocol)
    validate_side_balance(games, protocol)
    validate_terminations(games)
    checkpoint_scores = reconstruct_checkpoint_scores(games, protocol)
    validate_summary_scores(checkpoint_scores, summary_rows, protocol)

    window_rows, decision = _analyse_windows(
        games, protocol, checkpoint_scores, settings
    )
    expected_windows = len(_checkpoint_values(protocol)) - settings["window_checkpoints"] + 1
    if len(window_rows) != expected_windows:
        raise PlateauInputError(
            f"expected {expected_windows} rolling windows, built {len(window_rows)}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    windows_path = output_dir / "plateau_windows.csv"
    plateau_path = output_dir / "plateau.json"
    payload = {
        "schema_version": 1,
        "analysis_id": "h1_v1_plateau",
        "protocol_id": protocol["protocol_id"],
        "status": "completed",
        "metric": settings["metric"],
        "checkpoint_scores": [
            {
                "checkpoint": checkpoint,
                "fixed_basket_macro_score": checkpoint_scores[checkpoint][
                    "macro_score"
                ],
                "opponent_scores": checkpoint_scores[checkpoint]["opponent_scores"],
            }
            for checkpoint in _checkpoint_values(protocol)
        ],
        "window_count": len(window_rows),
        "window_checkpoints": settings["window_checkpoints"],
        "bootstrap": {
            "method": "paired_stratified_by_opponent_and_model_color",
            "resamples": settings["bootstrap_resamples"],
            "base_seed": settings["bootstrap_seed"],
            "window_seed_formula": "base_seed + zero_based_window_index",
            "confidence_level": settings["confidence_level"],
            "paired_key": "game_index",
        },
        "qualification": {
            "slope_ci_must_contain_zero": True,
            "maximum_absolute_score_change_over_window": settings[
                "maximum_absolute_score_change_over_window"
            ],
            "required_consecutive_windows": settings[
                "consecutive_qualifying_windows"
            ],
            "plateau_iteration_definition": settings["plateau_iteration"],
        },
        **decision,
        "inputs": {
            "gate_config": args.gate_config.expanduser().resolve().as_posix(),
            "fixed_basket_protocol": args.basket_config.expanduser()
            .resolve()
            .as_posix(),
            "games": args.games.expanduser().resolve().as_posix(),
            "checkpoint_summary": args.checkpoint_summary.expanduser()
            .resolve()
            .as_posix(),
            "evaluation_manifest": args.manifest.expanduser().resolve().as_posix(),
        },
        "outputs": {
            "plateau_windows": windows_path.as_posix(),
            "plateau": plateau_path.as_posix(),
        },
    }
    _atomic_write_csv(windows_path, window_rows, WINDOW_FIELDS)
    _atomic_write_json(plateau_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_analysis(args)
    except PlateauInputError as exc:
        print(f"Input validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "Plateau detected at checkpoint "
        f"{payload['plateau_iteration']}"
        if payload["plateau_detected"]
        else "No qualifying plateau detected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
