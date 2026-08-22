#!/usr/bin/env python3
"""Audit H3 inputs, construct matched strength curves, and decide H3."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import yaml

from head_to_head_stats import colour_stratified_bootstrap


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
DEFAULT_CONFIG = EXPERIMENTS_ROOT / "configs" / "h3_v2.yaml"

OPPONENT_SCORE_FIELDS = (
    "heuristic_20_score",
    "heuristic_200_score",
    "greedy_random_50_score",
    "random_score",
)
EXPECTED_OPPONENTS = tuple(field.removesuffix("_score") for field in OPPONENT_SCORE_FIELDS)

ALIGNED_FIELDS = (
    "target_baseline_iteration",
    "target_gpu_hours",
    "baseline_checkpoint_iteration",
    "adaptive_checkpoint_iteration",
    "adaptive_checkpoint_gpu_hours",
    "baseline_macro_score",
    "adaptive_macro_score",
    "baseline_joint_elo",
    "baseline_joint_elo_ci95_low",
    "baseline_joint_elo_ci95_high",
    "adaptive_joint_elo",
    "adaptive_joint_elo_ci95_low",
    "adaptive_joint_elo_ci95_high",
    "adaptive_minus_baseline_joint_elo",
    "joint_elo_difference_ci95_low",
    "joint_elo_difference_ci95_high",
    "joint_fit_id",
    "joint_fit_scope",
    "observed_target",
    "within_common_horizon",
    "interpolation",
    "extrapolation",
)

UTILITY_FIELDS = (
    "interval_index",
    "left_target_baseline_iteration",
    "right_target_baseline_iteration",
    "left_gpu_hours",
    "right_gpu_hours",
    "interval_gpu_hours",
    "baseline_elo_left",
    "baseline_elo_right",
    "adaptive_elo_left",
    "adaptive_elo_right",
    "baseline_interval_area",
    "adaptive_interval_area",
    "adaptive_minus_baseline_interval_area",
    "baseline_normalized_contribution",
    "adaptive_normalized_contribution",
    "effect_normalized_contribution",
    "cumulative_baseline_area",
    "cumulative_adaptive_area",
    "cumulative_effect_area",
    "interpolation",
    "extrapolated",
)

EFFECT_FIELDS = (
    "metric",
    "decision_role",
    "baseline_value",
    "adaptive_value",
    "effect",
    "effect_definition",
    "practical_threshold",
    "ci95_low",
    "ci95_high",
    "baseline_target_iteration",
    "adaptive_target_iteration",
    "baseline_gpu_hours",
    "adaptive_gpu_hours",
    "baseline_right_censored",
    "adaptive_right_censored",
    "assessment",
    "assessable",
    "evidence_scope",
    "audit_status",
    "notes",
)


class H3Error(ValueError):
    """Raised when the H3 protocol cannot produce an assessable result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise H3Error(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing YAML input: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise H3Error(f"invalid YAML input {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"YAML input must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H3Error(f"invalid JSON input {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON input must contain an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                _require(bool(stripped), f"blank JSONL row at {path}:{line_number}")
                row = json.loads(stripped)
                _require(
                    isinstance(row, dict),
                    f"JSONL row is not an object at {path}:{line_number}",
                )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise H3Error(f"invalid JSONL input {path}: {exc}") from exc
    return rows


_INTEGER = re.compile(r"^-?\d+$")


def _coerce_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if _INTEGER.fullmatch(value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _load_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            _require(reader.fieldnames is not None, f"CSV input lacks a header: {path}")
            return [
                {key: _coerce_csv_value(value) for key, value in row.items()}
                for row in reader
            ]
    except OSError as exc:
        raise H3Error(f"cannot read CSV input {path}: {exc}") from exc


def _resolve_experiments_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (EXPERIMENTS_ROOT / path).resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            json.dump(
                payload,
                destination,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            yaml.safe_dump(payload, destination, sort_keys=False, allow_unicode=True)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            destination.write("\n".join(lines) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    _require(_finite_number(value), f"field {field} is missing or non-finite")
    return float(value)


def _integer(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"field {field} is not an integer",
    )
    return value


def _close(left: float, right: float, *, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, details: Any) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "passed" if passed else "failed",
                "details": details,
            }
        )

    @property
    def passed(self) -> bool:
        return all(item["status"] == "passed" for item in self.checks)

    @property
    def failures(self) -> list[str]:
        return [item["name"] for item in self.checks if item["status"] == "failed"]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "passed" if self.passed else "failed",
            "checks": self.checks,
            "failures": self.failures,
        }


T = TypeVar("T")


def _audited(audit: Audit, name: str, operation: Callable[[], T]) -> T:
    try:
        result = operation()
    except (H3Error, KeyError, TypeError, ValueError) as exc:
        audit.record(name, False, str(exc))
        raise H3Error(str(exc)) from exc
    audit.record(name, True, "validated")
    return result


def _input_paths_and_manifest(
    config: dict[str, Any],
    config_path: Path,
    audit: Audit,
) -> tuple[dict[str, Path], dict[str, Any]]:
    inputs = config.get("inputs")
    _require(isinstance(inputs, dict) and inputs, "H3 protocol inputs are empty")
    paths: dict[str, Path] = {}
    entries: dict[str, Any] = {}
    for name, specification in inputs.items():
        if not isinstance(specification, dict):
            audit.record(f"input.{name}.specification", False, "not a mapping")
            continue
        path = _resolve_experiments_path(specification.get("path", ""))
        paths[name] = path
        exists = path.is_file()
        actual = _sha256_file(path) if exists else None
        expected = specification.get("sha256")
        hash_matches = expected is None or (
            isinstance(expected, str) and actual == expected.lower()
        )
        passed = exists and hash_matches
        audit.record(
            f"input.{name}.integrity",
            passed,
            {
                "path": path.as_posix(),
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "generated_output_hash_recorded_only": expected is None,
            },
        )
        entries[name] = {
            "path": path.as_posix(),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "passed" if passed else "failed",
        }
    manifest = {
        "schema_version": 1,
        "config": {
            "path": config_path.as_posix(),
            "sha256": _sha256_file(config_path),
        },
        "inputs": entries,
    }
    return paths, manifest


def _common_grid(matched: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    try:
        raw_targets = matched["pairing_and_randomness"]["checkpoint_grid"]["targets"]
        cap = matched["compute_budget"]["common_horizon"]["maximum_gpu_hours"]
    except (KeyError, TypeError) as exc:
        raise H3Error("matched-compute protocol lacks checkpoint-grid horizon fields") from exc
    _require(isinstance(raw_targets, list) and len(raw_targets) >= 2, "common grid is too short")
    _require(_finite_number(cap) and float(cap) > 0.0, "common-horizon cap is invalid")
    targets: list[dict[str, Any]] = []
    for raw in raw_targets:
        _require(isinstance(raw, dict), "matched-compute target is not a mapping")
        iteration = raw.get("baseline_checkpoint_iteration")
        gpu_hours = raw.get("gpu_hours")
        _require(
            isinstance(iteration, int)
            and not isinstance(iteration, bool)
            and _finite_number(gpu_hours),
            "matched-compute target is invalid",
        )
        targets.append(
            {
                "target_baseline_iteration": iteration,
                "target_gpu_hours": float(gpu_hours),
            }
        )
    _require(len({item["target_baseline_iteration"] for item in targets}) == len(targets), "target iterations are not unique")
    _require(targets[0]["target_baseline_iteration"] == 0, "checkpoint 0 is absent")
    _require(_close(targets[0]["target_gpu_hours"], 0.0), "checkpoint 0 is not at zero GPU-hours")
    _require(
        all(
            targets[index]["target_gpu_hours"] < targets[index + 1]["target_gpu_hours"]
            for index in range(len(targets) - 1)
        ),
        "common-grid GPU-hours are not strictly increasing",
    )
    horizon = min(float(cap), targets[-1]["target_gpu_hours"])
    eligible = [item for item in targets if item["target_gpu_hours"] <= horizon]
    _require(len(eligible) == 12, f"common grid must contain 12 targets, found {len(eligible)}")
    _require(
        _close(eligible[-1]["target_gpu_hours"], horizon),
        "common horizon is not an observed target; extrapolation is forbidden",
    )
    return eligible, horizon


def _summary_by_target(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    grid: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    expected_targets = [int(item["target_baseline_iteration"]) for item in grid]
    _require(len(rows) == len(expected_targets), f"{condition} checkpoint summary must contain 12 rows")
    by_target: dict[int, dict[str, Any]] = {}
    for row in rows:
        target_field = "checkpoint" if condition == "baseline" else "target_baseline_iteration"
        target = _integer(row, target_field)
        _require(target not in by_target, f"duplicate {condition} checkpoint-summary target {target}")
        _require(_integer(row, "checkpoint") == target, f"{condition} checkpoint label differs from target {target}")
        _require(_integer(row, "total_games") == 200, f"{condition} target {target} does not contain 200 games")
        _require(_integer(row, "invalid_moves") == 0, f"{condition} target {target} has invalid moves")
        _require(_integer(row, "bot_errors") == 0, f"{condition} target {target} has bot errors")
        component_scores = [_number(row, field) for field in OPPONENT_SCORE_FIELDS]
        macro = sum(component_scores) / len(component_scores)
        _require(_close(macro, _number(row, "score_rate")), f"{condition} target {target} macro score mismatch")
        row["_macro_score"] = macro
        if condition == "adaptive":
            _require(
                _integer(row, "selected_adaptive_iteration") >= 0,
                f"Adaptive target {target} has invalid selected iteration",
            )
            actual_gpu = _number(row, "selected_adaptive_gpu_hours")
            target_gpu = _number(row, "target_gpu_hours")
            _require(actual_gpu <= target_gpu + 1.0e-12, f"Adaptive target {target} selects a checkpoint beyond the target")
        by_target[target] = row
    _require(list(sorted(by_target)) == sorted(expected_targets), f"{condition} target grid differs from matched compute")
    return by_target


def _fixed_game_statistics(
    records: list[dict[str, Any]],
    *,
    condition: str,
    expected_targets: list[int],
) -> tuple[dict[int, float], dict[tuple[int, str, int], tuple[int, str]], dict[int, set[str]]]:
    _require(len(records) == 2400, f"{condition} fixed basket must contain 2,400 games")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str, int]] = set()
    paired_inputs: dict[tuple[int, str, int], tuple[int, str]] = {}
    checkpoint_hashes: dict[int, set[str]] = defaultdict(set)
    for record in records:
        target = _integer(record, "checkpoint")
        opponent = record.get("opponent")
        game_index = _integer(record, "game_index")
        _require(target in expected_targets, f"{condition} fixed-basket target {target} is outside the grid")
        _require(opponent in EXPECTED_OPPONENTS, f"{condition} has unexpected opponent {opponent!r}")
        key = (target, str(opponent), game_index)
        _require(key not in seen, f"duplicate {condition} fixed-basket game key {key}")
        seen.add(key)
        color = record.get("model_color")
        _require(color in {"white", "black"}, f"{condition} has invalid model color at {key}")
        _require(record.get("model_result") in {"win", "draw", "loss"}, f"{condition} has invalid result at {key}")
        _require(record.get("fault") is None, f"{condition} has a game fault at {key}")
        _require(record.get("termination") not in {"invalid_move", "bot_error"}, f"{condition} has an invalid technical result at {key}")
        seed = record.get("game_seed")
        _require(isinstance(seed, int) and not isinstance(seed, bool), f"{condition} has invalid seed at {key}")
        paired_inputs[key] = (seed, str(color))
        checkpoint_hash = record.get("checkpoint_sha256")
        _require(isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64, f"{condition} has invalid checkpoint SHA at {key}")
        checkpoint_hashes[target].add(checkpoint_hash)
        groups[(target, str(opponent))].append(record)
    _require(len(seen) == 2400, f"{condition} fixed-basket game keys are incomplete")
    macros: dict[int, float] = {}
    for target in expected_targets:
        opponent_scores: list[float] = []
        for opponent in EXPECTED_OPPONENTS:
            group = groups[(target, opponent)]
            _require(len(group) == 50, f"{condition} target {target} vs {opponent} does not contain 50 games")
            white = sum(item["model_color"] == "white" for item in group)
            black = sum(item["model_color"] == "black" for item in group)
            _require(white == black == 25, f"{condition} target {target} vs {opponent} is not split 25/25 by color")
            points = sum(
                1.0 if item["model_result"] == "win" else 0.5 if item["model_result"] == "draw" else 0.0
                for item in group
            )
            opponent_scores.append(points / len(group))
        macros[target] = sum(opponent_scores) / len(opponent_scores)
        _require(len(checkpoint_hashes[target]) == 1, f"{condition} target {target} uses multiple checkpoint SHAs")
    return macros, paired_inputs, checkpoint_hashes


def _validate_fixed_inputs(
    baseline_games: list[dict[str, Any]],
    adaptive_games: list[dict[str, Any]],
    baseline_summary: dict[int, dict[str, Any]],
    adaptive_summary: dict[int, dict[str, Any]],
    targets: list[int],
) -> None:
    baseline_macros, baseline_pairing, baseline_hashes = _fixed_game_statistics(
        baseline_games, condition="baseline", expected_targets=targets
    )
    adaptive_macros, adaptive_pairing, adaptive_hashes = _fixed_game_statistics(
        adaptive_games, condition="adaptive", expected_targets=targets
    )
    _require(baseline_pairing == adaptive_pairing, "Baseline and Adaptive fixed baskets do not use identical seeds and colors")
    _require(
        baseline_hashes[0] == adaptive_hashes[0],
        "Baseline and Adaptive checkpoint 0 identities differ",
    )
    for target in targets:
        _require(
            _close(baseline_macros[target], float(baseline_summary[target]["_macro_score"])),
            f"Baseline raw-game macro score differs from checkpoint summary at target {target}",
        )
        _require(
            _close(adaptive_macros[target], float(adaptive_summary[target]["_macro_score"])),
            f"Adaptive raw-game macro score differs from checkpoint summary at target {target}",
        )


def _joint_by_target(
    rows: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    adaptive_summary: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], str, str]:
    _require(len(rows) == len(grid) == 12, "joint Elo summary must contain all 12 targets")
    by_target: dict[int, dict[str, Any]] = {}
    fit_ids: set[str] = set()
    fit_scopes: set[str] = set()
    for target_specification, row in zip(grid, rows):
        target = _integer(row, "target_baseline_iteration")
        expected_target = int(target_specification["target_baseline_iteration"])
        _require(target == expected_target, "joint Elo rows are not on the ordered matched-compute grid")
        _require(_close(_number(row, "target_gpu_hours"), float(target_specification["target_gpu_hours"])), f"joint Elo GPU-hours mismatch at target {target}")
        _require(target not in by_target, f"duplicate joint Elo target {target}")
        for field in (
            "baseline_joint_elo",
            "baseline_joint_elo_ci95_low",
            "baseline_joint_elo_ci95_high",
            "adaptive_joint_elo",
            "adaptive_joint_elo_ci95_low",
            "adaptive_joint_elo_ci95_high",
            "adaptive_minus_baseline_elo",
            "adaptive_minus_baseline_elo_ci95_low",
            "adaptive_minus_baseline_elo_ci95_high",
        ):
            _number(row, field)
        expected_effect = _number(row, "adaptive_joint_elo") - _number(row, "baseline_joint_elo")
        _require(_close(expected_effect, _number(row, "adaptive_minus_baseline_elo")), f"joint Elo difference mismatch at target {target}")
        _require(row.get("baseline_participant") == f"baseline_checkpoint_{target}", f"Baseline joint Elo participant mismatch at target {target}")
        _require(row.get("adaptive_participant") == f"adaptive_checkpoint_{target}", f"Adaptive joint Elo participant mismatch at target {target}")
        adaptive = adaptive_summary[target]
        _require(_integer(row, "selected_adaptive_iteration") == _integer(adaptive, "selected_adaptive_iteration"), f"selected Adaptive iteration mismatch at target {target}")
        _require(_close(_number(row, "selected_adaptive_gpu_hours"), _number(adaptive, "selected_adaptive_gpu_hours")), f"selected Adaptive GPU-hours mismatch at target {target}")
        fit_id = row.get("fit_id")
        fit_scope = row.get("fit_scope")
        _require(isinstance(fit_id, str) and bool(fit_id), "joint Elo fit_id is absent")
        _require(isinstance(fit_scope, str) and fit_scope == "one_joint_model_over_all_baseline_adaptive_and_shared_anchor_games", "joint Elo was not produced by the required single joint fit")
        fit_ids.add(fit_id)
        fit_scopes.add(fit_scope)
        by_target[target] = row
    _require(len(fit_ids) == 1 and len(fit_scopes) == 1, "joint Elo rows do not share one fit identity")
    return by_target, next(iter(fit_ids)), next(iter(fit_scopes))


def _validate_head_to_head(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    horizon: float,
    records_sha256: str,
) -> dict[str, float | int]:
    _require(summary.get("status") == "completed", "head-to-head summary is not completed")
    output_hashes = summary.get("output_sha256")
    _require(isinstance(output_hashes, dict), "head-to-head summary output hashes are absent")
    _require(
        output_hashes.get("games") == records_sha256,
        "head-to-head games SHA differs from summary.json",
    )
    _require(len(records) == 100, "head-to-head must contain 100 technically valid games")
    keys: set[str] = set()
    pairs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = record.get("stable_game_key")
        _require(isinstance(key, str) and bool(key), "head-to-head game lacks stable_game_key")
        _require(key not in keys, f"duplicate head-to-head game key {key}")
        keys.add(key)
        pair_index = _integer(record, "seed_pair_index")
        color = record.get("adaptive_color")
        _require(0 <= pair_index < 50, "head-to-head seed_pair_index is outside 0..49")
        _require(color in {"white", "black"}, "head-to-head Adaptive color is invalid")
        _require(color not in pairs[pair_index], f"duplicate color in head-to-head pair {pair_index}")
        _require(record.get("adaptive_result") in {"win", "draw", "loss"}, "head-to-head Adaptive result is invalid")
        _require(record.get("technically_valid") is True, "head-to-head formal record is not technically valid")
        pairs[pair_index][str(color)] = record
    _require(len(pairs) == 50, "head-to-head does not contain 50 seed pairs")
    for pair_index, pair in pairs.items():
        _require(set(pair) == {"white", "black"}, f"head-to-head pair {pair_index} lacks a color-swapped game")
        _require(pair["white"].get("game_seed") == pair["black"].get("game_seed"), f"head-to-head pair {pair_index} does not share one seed")
    checkpoint_selection = summary.get("checkpoint_selection")
    _require(isinstance(checkpoint_selection, dict), "head-to-head checkpoint selection is absent")
    _require(checkpoint_selection.get("best_checkpoint_selection_used") is False, "head-to-head selected a best checkpoint")
    for condition in ("baseline", "adaptive"):
        checkpoint = checkpoint_selection.get(condition)
        _require(isinstance(checkpoint, dict), f"head-to-head {condition} checkpoint selection is absent")
        _require(
            _number(checkpoint, "actual_gpu_hours") <= horizon + 1.0e-12,
            f"head-to-head {condition} checkpoint exceeds the common horizon",
        )
    adaptive_score = summary.get("adaptive_score")
    _require(isinstance(adaptive_score, dict), "head-to-head adaptive_score summary is absent")
    bootstrap = adaptive_score.get("bootstrap")
    _require(isinstance(bootstrap, dict), "head-to-head bootstrap specification is absent")
    interval = colour_stratified_bootstrap(
        records,
        resamples=int(bootstrap.get("resamples")),
        seed=int(bootstrap.get("seed")),
        expected_per_colour=int(bootstrap.get("preserve_games_per_colour")),
    )
    for result_field, summary_field in (
        ("score_rate", "score_rate"),
        ("ci95_low", "ci95_low"),
        ("ci95_high", "ci95_high"),
    ):
        _require(_close(float(interval[result_field]), _number(adaptive_score, summary_field)), f"head-to-head {summary_field} differs from an independent color-stratified bootstrap")
    _require(int(interval["adaptive_white_games"]) == int(interval["adaptive_black_games"]) == 50, "head-to-head is not color balanced")
    return interval


def _validate_h2_decision(payload: dict[str, Any]) -> str:
    _require(payload.get("hypothesis") == "H2", "H2 decision has the wrong hypothesis id")
    status = payload.get("status")
    _require(status in {"supported", "partially_supported", "not_supported", "not_assessable"}, "H2 decision status is invalid")
    return str(status)


def _aligned_rows(
    grid: list[dict[str, Any]],
    baseline_summary: dict[int, dict[str, Any]],
    adaptive_summary: dict[int, dict[str, Any]],
    joint: dict[int, dict[str, Any]],
    horizon: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_specification in grid:
        target = int(target_specification["target_baseline_iteration"])
        target_gpu = float(target_specification["target_gpu_hours"])
        baseline = baseline_summary[target]
        adaptive = adaptive_summary[target]
        elo = joint[target]
        rows.append(
            {
                "target_baseline_iteration": target,
                "target_gpu_hours": target_gpu,
                "baseline_checkpoint_iteration": target,
                "adaptive_checkpoint_iteration": _integer(adaptive, "selected_adaptive_iteration"),
                "adaptive_checkpoint_gpu_hours": _number(adaptive, "selected_adaptive_gpu_hours"),
                "baseline_macro_score": baseline["_macro_score"],
                "adaptive_macro_score": adaptive["_macro_score"],
                "baseline_joint_elo": _number(elo, "baseline_joint_elo"),
                "baseline_joint_elo_ci95_low": _number(elo, "baseline_joint_elo_ci95_low"),
                "baseline_joint_elo_ci95_high": _number(elo, "baseline_joint_elo_ci95_high"),
                "adaptive_joint_elo": _number(elo, "adaptive_joint_elo"),
                "adaptive_joint_elo_ci95_low": _number(elo, "adaptive_joint_elo_ci95_low"),
                "adaptive_joint_elo_ci95_high": _number(elo, "adaptive_joint_elo_ci95_high"),
                "adaptive_minus_baseline_joint_elo": _number(elo, "adaptive_minus_baseline_elo"),
                "joint_elo_difference_ci95_low": _number(elo, "adaptive_minus_baseline_elo_ci95_low"),
                "joint_elo_difference_ci95_high": _number(elo, "adaptive_minus_baseline_elo_ci95_high"),
                "joint_fit_id": elo["fit_id"],
                "joint_fit_scope": elo["fit_scope"],
                "observed_target": True,
                "within_common_horizon": target_gpu <= horizon + 1.0e-12,
                "interpolation": "none_at_observed_target",
                "extrapolation": "none",
            }
        )
    return rows


def _utility_rows(
    aligned: list[dict[str, Any]],
    horizon: float,
) -> tuple[list[dict[str, Any]], float, float]:
    _require(len(aligned) >= 2, "AULC needs at least two observed points")
    rows: list[dict[str, Any]] = []
    cumulative_baseline = 0.0
    cumulative_adaptive = 0.0
    for index, (left, right) in enumerate(zip(aligned, aligned[1:])):
        left_x = float(left["target_gpu_hours"])
        right_x = float(right["target_gpu_hours"])
        width = right_x - left_x
        _require(width > 0.0, "utility-curve interval has non-positive width")
        baseline_area = width * (float(left["baseline_joint_elo"]) + float(right["baseline_joint_elo"])) / 2.0
        adaptive_area = width * (float(left["adaptive_joint_elo"]) + float(right["adaptive_joint_elo"])) / 2.0
        cumulative_baseline += baseline_area
        cumulative_adaptive += adaptive_area
        rows.append(
            {
                "interval_index": index,
                "left_target_baseline_iteration": left["target_baseline_iteration"],
                "right_target_baseline_iteration": right["target_baseline_iteration"],
                "left_gpu_hours": left_x,
                "right_gpu_hours": right_x,
                "interval_gpu_hours": width,
                "baseline_elo_left": left["baseline_joint_elo"],
                "baseline_elo_right": right["baseline_joint_elo"],
                "adaptive_elo_left": left["adaptive_joint_elo"],
                "adaptive_elo_right": right["adaptive_joint_elo"],
                "baseline_interval_area": baseline_area,
                "adaptive_interval_area": adaptive_area,
                "adaptive_minus_baseline_interval_area": adaptive_area - baseline_area,
                "baseline_normalized_contribution": baseline_area / horizon,
                "adaptive_normalized_contribution": adaptive_area / horizon,
                "effect_normalized_contribution": (adaptive_area - baseline_area) / horizon,
                "cumulative_baseline_area": cumulative_baseline,
                "cumulative_adaptive_area": cumulative_adaptive,
                "cumulative_effect_area": cumulative_adaptive - cumulative_baseline,
                "interpolation": "piecewise_linear",
                "extrapolated": False,
            }
        )
    _require(_close(float(aligned[0]["target_gpu_hours"]), 0.0), "AULC curve does not start at checkpoint 0")
    _require(_close(float(aligned[-1]["target_gpu_hours"]), horizon), "AULC curve does not end at the common horizon")
    return rows, cumulative_baseline / horizon, cumulative_adaptive / horizon


def _best_observed(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    best_value = max(float(row[field]) for row in rows)
    return next(row for row in rows if _close(float(row[field]), best_value))


def _time_to_threshold(
    rows: list[dict[str, Any]], field: str, threshold: float, horizon: float
) -> tuple[float, bool, int | None]:
    for row in rows:
        if float(row[field]) >= threshold:
            return float(row["target_gpu_hours"]), False, int(row["target_baseline_iteration"])
    return horizon, True, None


def _maximum_drawdown(rows: list[dict[str, Any]], field: str) -> float:
    running_best = -math.inf
    maximum = 0.0
    for row in rows:
        value = float(row[field])
        running_best = max(running_best, value)
        maximum = max(maximum, running_best - value)
    return maximum


def _effect_rows(
    config: dict[str, Any],
    aligned: list[dict[str, Any]],
    baseline_aulc: float,
    adaptive_aulc: float,
    head_interval: dict[str, float | int],
    horizon: float,
) -> list[dict[str, Any]]:
    primary_threshold = float(config["primary_metric"]["practical_effect_tolerance"]["improvement_at_or_above_elo"])
    final_elo_threshold = float(config["other_strength_metrics"]["final_joint_elo"]["support_threshold_elo"])
    macro_threshold = float(config["other_strength_metrics"]["final_fixed_basket_macro_score"]["support_threshold_absolute_score"])
    drawdown_threshold = float(config["other_strength_metrics"]["observed_maximum_drawdown"]["support_threshold_elo"])
    score_target = float(config["other_strength_metrics"]["time_to_macro_score"]["threshold"])
    final = aligned[-1]
    best_baseline = _best_observed(aligned, "baseline_joint_elo")
    best_adaptive = _best_observed(aligned, "adaptive_joint_elo")
    baseline_time, baseline_censored, baseline_crossing = _time_to_threshold(aligned, "baseline_macro_score", score_target, horizon)
    adaptive_time, adaptive_censored, adaptive_crossing = _time_to_threshold(aligned, "adaptive_macro_score", score_target, horizon)
    baseline_drawdown = _maximum_drawdown(aligned, "baseline_joint_elo")
    adaptive_drawdown = _maximum_drawdown(aligned, "adaptive_joint_elo")
    primary_effect = adaptive_aulc - baseline_aulc
    final_elo_effect = float(final["adaptive_joint_elo"]) - float(final["baseline_joint_elo"])
    final_macro_effect = float(final["adaptive_macro_score"]) - float(final["baseline_macro_score"])
    drawdown_effect = baseline_drawdown - adaptive_drawdown
    h2h_score = float(head_interval["score_rate"])
    h2h_low = float(head_interval["ci95_low"])
    h2h_high = float(head_interval["ci95_high"])
    time_assessable = not baseline_censored and not adaptive_censored
    common = {
        "ci95_low": "",
        "ci95_high": "",
        "baseline_target_iteration": "",
        "adaptive_target_iteration": "",
        "baseline_gpu_hours": "",
        "adaptive_gpu_hours": "",
        "baseline_right_censored": False,
        "adaptive_right_censored": False,
        "assessable": True,
        "evidence_scope": "observed_shared_grid_within_common_horizon",
        "audit_status": "passed",
    }

    def row(**values: Any) -> dict[str, Any]:
        result = dict(common)
        result.update(values)
        return result

    return [
        row(
            metric="normalized_joint_elo_aulc",
            decision_role="primary",
            baseline_value=baseline_aulc,
            adaptive_value=adaptive_aulc,
            effect=primary_effect,
            effect_definition="adaptive_minus_baseline_elo",
            practical_threshold=primary_threshold,
            assessment="improved" if primary_effect >= primary_threshold else "not_improved",
            notes="trapezoidal area divided by common horizon; all interval contributions are in utility_curve.csv",
        ),
        row(
            metric="final_joint_elo",
            decision_role="corroborating_final",
            baseline_value=final["baseline_joint_elo"],
            adaptive_value=final["adaptive_joint_elo"],
            effect=final_elo_effect,
            effect_definition="adaptive_minus_baseline_elo",
            practical_threshold=final_elo_threshold,
            ci95_low=final["joint_elo_difference_ci95_low"],
            ci95_high=final["joint_elo_difference_ci95_high"],
            baseline_target_iteration=final["target_baseline_iteration"],
            adaptive_target_iteration=final["target_baseline_iteration"],
            baseline_gpu_hours=final["target_gpu_hours"],
            adaptive_gpu_hours=final["target_gpu_hours"],
            assessment="supporting" if final_elo_effect >= final_elo_threshold else "not_supporting",
            notes="final observed common-horizon target from the single joint Elo fit",
        ),
        row(
            metric="final_fixed_basket_macro_score",
            decision_role="corroborating_final",
            baseline_value=final["baseline_macro_score"],
            adaptive_value=final["adaptive_macro_score"],
            effect=final_macro_effect,
            effect_definition="adaptive_minus_baseline_absolute_score",
            practical_threshold=macro_threshold,
            baseline_target_iteration=final["target_baseline_iteration"],
            adaptive_target_iteration=final["target_baseline_iteration"],
            baseline_gpu_hours=final["target_gpu_hours"],
            adaptive_gpu_hours=final["target_gpu_hours"],
            assessment="supporting" if final_macro_effect >= macro_threshold else "not_supporting",
            notes="equal-weight mean of four opponent score rates",
        ),
        row(
            metric="best_observed_joint_elo",
            decision_role="descriptive",
            baseline_value=best_baseline["baseline_joint_elo"],
            adaptive_value=best_adaptive["adaptive_joint_elo"],
            effect=float(best_adaptive["adaptive_joint_elo"]) - float(best_baseline["baseline_joint_elo"]),
            effect_definition="adaptive_best_minus_baseline_best_elo",
            practical_threshold="",
            baseline_target_iteration=best_baseline["target_baseline_iteration"],
            adaptive_target_iteration=best_adaptive["target_baseline_iteration"],
            baseline_gpu_hours=best_baseline["target_gpu_hours"],
            adaptive_gpu_hours=best_adaptive["target_gpu_hours"],
            assessment="descriptive_only",
            notes="maximum observed point; ties resolved to earliest GPU-hour target; never used for checkpoint selection",
        ),
        row(
            metric="time_to_macro_score_0_86",
            decision_role="supplementary",
            baseline_value=baseline_time,
            adaptive_value=adaptive_time,
            effect=(baseline_time - adaptive_time) if time_assessable else "",
            effect_definition="baseline_minus_adaptive_gpu_hours; positive_is_faster_adaptive",
            practical_threshold="",
            baseline_target_iteration=baseline_crossing if baseline_crossing is not None else "",
            adaptive_target_iteration=adaptive_crossing if adaptive_crossing is not None else "",
            baseline_gpu_hours=baseline_time,
            adaptive_gpu_hours=adaptive_time,
            baseline_right_censored=baseline_censored,
            adaptive_right_censored=adaptive_censored,
            assessment="descriptive" if time_assessable else "right_censored",
            assessable=time_assessable,
            notes="first observed target at or above 0.86; no crossing interpolation",
        ),
        row(
            metric="observed_maximum_drawdown",
            decision_role="corroborating_stability",
            baseline_value=baseline_drawdown,
            adaptive_value=adaptive_drawdown,
            effect=drawdown_effect,
            effect_definition="baseline_minus_adaptive_maximum_drawdown_elo",
            practical_threshold=drawdown_threshold,
            assessment="supporting" if drawdown_effect >= drawdown_threshold else "not_supporting",
            notes="running-best minus current Elo on observed targets only",
        ),
        row(
            metric="final_head_to_head_score_rate",
            decision_role="corroborating_head_to_head",
            baseline_value=0.5,
            adaptive_value=h2h_score,
            effect=h2h_score - 0.5,
            effect_definition="adaptive_score_rate_minus_0.5",
            practical_threshold=0.5,
            ci95_low=h2h_low,
            ci95_high=h2h_high,
            assessment="supporting" if h2h_low > 0.5 else "not_supporting",
            evidence_scope="100_technically_valid_colour_swapped_games",
            notes="support depends on the color-stratified 95% CI lower bound, not the point estimate",
        ),
    ]


def _row_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped = {str(row["metric"]): row for row in rows}
    _require(len(mapped) == len(rows), "effects contain duplicate metrics")
    return mapped


def decide_h3(
    effects: list[dict[str, Any]],
    *,
    audit_passed: bool,
    h2_status: str,
) -> dict[str, Any]:
    if not audit_passed:
        status = "not_assessable"
        primary_improved = None
        support_flags: dict[str, bool | None] = {
            "final_joint_elo": None,
            "final_fixed_basket_macro_score": None,
            "observed_maximum_drawdown": None,
            "final_head_to_head": None,
        }
        any_support = None
        reason = "input_audit_failed"
    else:
        mapped = _row_by_metric(effects)
        required = {
            "normalized_joint_elo_aulc",
            "final_joint_elo",
            "final_fixed_basket_macro_score",
            "observed_maximum_drawdown",
            "final_head_to_head_score_rate",
        }
        _require(required <= set(mapped), "effects omit a decision metric")
        primary_improved = mapped["normalized_joint_elo_aulc"]["assessment"] == "improved"
        support_flags = {
            "final_joint_elo": mapped["final_joint_elo"]["assessment"] == "supporting",
            "final_fixed_basket_macro_score": mapped["final_fixed_basket_macro_score"]["assessment"] == "supporting",
            "observed_maximum_drawdown": mapped["observed_maximum_drawdown"]["assessment"] == "supporting",
            "final_head_to_head": mapped["final_head_to_head_score_rate"]["assessment"] == "supporting",
        }
        any_support = any(bool(value) for value in support_flags.values())
        if primary_improved and any_support:
            status = "supported"
            reason = "primary_aulc_and_consistent_support_met"
        elif primary_improved or any_support:
            status = "partially_supported"
            reason = "primary_aulc_and_consistent_support_disagree"
        else:
            status = "not_supported"
            reason = "primary_aulc_and_consistent_support_not_met"

    if status == "supported" and h2_status == "supported":
        attribution_status = "attributed_to_replay_adaptive_mechanism"
        attribution_reason = "H3 performance improvement and H2 mechanism evidence are both supported"
    elif status == "supported":
        attribution_status = "performance_improvement_without_replay_mechanism_attribution"
        attribution_reason = "H3 is supported but H2 does not support the replay/adaptive mechanism claim"
    else:
        attribution_status = "not_claimed_without_supported_h3_performance_improvement"
        attribution_reason = "mechanism attribution is not made unless H3 performance improvement is supported"
    return {
        "schema_version": 1,
        "hypothesis": "H3",
        "status": status,
        "reason": reason,
        "performance_improvement": {
            "status": status,
            "primary_aulc_improved": primary_improved,
            "support_flags": support_flags,
            "any_consistent_support": any_support,
            "independent_of_h2": True,
        },
        "mechanism_attribution": {
            "status": attribution_status,
            "h2_status": h2_status,
            "h2_can_change_h3_performance_decision": False,
            "reason": attribution_reason,
        },
        "recomputation_rule": {
            "performance_input": "effects.csv",
            "not_assessable": "input audit failed",
            "primary": "normalized_joint_elo_aulc assessment == improved",
            "consistent_support": "any of final_joint_elo, final_fixed_basket_macro_score, observed_maximum_drawdown, or final_head_to_head_score_rate has assessment == supporting",
            "supported": "primary and consistent_support",
            "partially_supported": "exactly one of primary and consistent_support",
            "not_supported": "neither primary nor consistent_support",
            "mechanism_attribution": "only when H3 status == supported; requires h2_v2 decision status == supported",
        },
    }


def _not_assessable_outputs(
    output_dir: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    audit: Audit,
    reason: str,
) -> None:
    _atomic_write_csv(output_dir / "aligned_strength.csv", ALIGNED_FIELDS, [])
    _atomic_write_csv(output_dir / "utility_curve.csv", UTILITY_FIELDS, [])
    _atomic_write_csv(output_dir / "effects.csv", EFFECT_FIELDS, [])
    decision = decide_h3([], audit_passed=False, h2_status="not_assessable")
    decision["audit_failures"] = audit.failures
    decision["detail"] = reason
    decision["effects_sha256"] = _sha256_file(output_dir / "effects.csv")
    _atomic_write_json(output_dir / "decision.json", decision)
    _atomic_write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "config_id": config.get("config_id"),
            "status": "not_assessable",
            "reason": reason,
            "audit_failures": audit.failures,
            "performance_improvement": decision["performance_improvement"],
            "mechanism_attribution": decision["mechanism_attribution"],
        },
    )
    _atomic_write_json(output_dir / "input_manifest.json", manifest)
    _atomic_write_json(output_dir / "input_audit.json", audit.payload())
    _atomic_write_text(
        output_dir / "run.log",
        [
            "H3 analysis status: not_assessable",
            f"Reason: {reason}",
            f"Audit failures: {', '.join(audit.failures)}",
        ],
    )


def execute(config_path: Path, output_override: Path | None = None) -> str:
    config_path = config_path.expanduser().resolve()
    config = _load_yaml(config_path)
    _require(config.get("config_id") == "h3_v2", "protocol config_id must be h3_v2")
    output_dir = (
        output_override.expanduser().resolve()
        if output_override is not None
        else _resolve_experiments_path(config["outputs"]["root"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(config)
    resolved["resolved_execution"] = {
        "config_path": config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
    }
    _atomic_write_yaml(output_dir / "resolved_config.yaml", resolved)
    audit = Audit()
    try:
        paths, manifest = _input_paths_and_manifest(config, config_path, audit)
    except H3Error as exc:
        manifest = {
            "schema_version": 1,
            "config": {"path": config_path.as_posix(), "sha256": _sha256_file(config_path)},
            "inputs": {},
        }
        audit.record("input_manifest", False, str(exc))
        _not_assessable_outputs(output_dir, config, manifest, audit, str(exc))
        return "not_assessable"
    if not audit.passed:
        _not_assessable_outputs(output_dir, config, manifest, audit, "one or more required inputs are missing or fail SHA-256 audit")
        return "not_assessable"

    try:
        matched = _audited(audit, "matched_compute.common_grid", lambda: _load_yaml(paths["matched_compute"]))
        grid, horizon = _audited(audit, "matched_compute.dynamic_horizon", lambda: _common_grid(matched))
        targets = [int(item["target_baseline_iteration"]) for item in grid]
        baseline_summary_rows = _load_csv(paths["baseline_fixed_basket_checkpoint_summary"])
        adaptive_summary_rows = _load_csv(paths["adaptive_fixed_basket_checkpoint_summary"])
        baseline_summary = _audited(
            audit,
            "baseline_fixed_basket.checkpoint_summary",
            lambda: _summary_by_target(baseline_summary_rows, condition="baseline", grid=grid),
        )
        adaptive_summary = _audited(
            audit,
            "adaptive_fixed_basket.checkpoint_summary",
            lambda: _summary_by_target(adaptive_summary_rows, condition="adaptive", grid=grid),
        )
        baseline_games = _load_jsonl(paths["baseline_fixed_basket_records"])
        adaptive_games = _load_jsonl(paths["adaptive_fixed_basket_records"])
        _audited(
            audit,
            "fixed_basket.raw_games_and_shared_design",
            lambda: _validate_fixed_inputs(
                baseline_games,
                adaptive_games,
                baseline_summary,
                adaptive_summary,
                targets,
            ),
        )
        joint_rows = _load_csv(paths["joint_elo_summary"])
        joint, fit_id, fit_scope = _audited(
            audit,
            "joint_elo.single_joint_fit_and_grid",
            lambda: _joint_by_target(joint_rows, grid, adaptive_summary),
        )
        head_records = _load_jsonl(paths["head_to_head_records"])
        head_summary = _load_json(paths["head_to_head_summary"])
        head_interval = _audited(
            audit,
            "head_to_head.valid_games_and_colour_stratified_ci",
            lambda: _validate_head_to_head(
                head_records,
                head_summary,
                horizon,
                str(manifest["inputs"]["head_to_head_records"]["actual_sha256"]),
            ),
        )
        h2_payload = _load_json(paths["h2_decision"])
        h2_status = _audited(
            audit,
            "h2_decision.valid_for_attribution_only",
            lambda: _validate_h2_decision(h2_payload),
        )
        _require(audit.passed, "input audit failed")
        aligned = _aligned_rows(grid, baseline_summary, adaptive_summary, joint, horizon)
        utility, baseline_aulc, adaptive_aulc = _audited(
            audit,
            "utility_curve.trapezoidal_recomputation",
            lambda: _utility_rows(aligned, horizon),
        )
        effects = _effect_rows(
            config,
            aligned,
            baseline_aulc,
            adaptive_aulc,
            head_interval,
            horizon,
        )
    except (H3Error, OSError, KeyError, TypeError, ValueError) as exc:
        if audit.passed:
            audit.record("analysis", False, str(exc))
        _not_assessable_outputs(output_dir, config, manifest, audit, str(exc))
        return "not_assessable"

    _atomic_write_csv(output_dir / "aligned_strength.csv", ALIGNED_FIELDS, aligned)
    _atomic_write_csv(output_dir / "utility_curve.csv", UTILITY_FIELDS, utility)
    _atomic_write_csv(output_dir / "effects.csv", EFFECT_FIELDS, effects)
    decision = decide_h3(effects, audit_passed=True, h2_status=h2_status)
    decision.update(
        {
            "common_horizon_gpu_hours": horizon,
            "shared_grid_points": len(aligned),
            "joint_elo_fit_id": fit_id,
            "joint_elo_fit_scope": fit_scope,
            "effects_sha256": _sha256_file(output_dir / "effects.csv"),
            "input_audit_status": "passed",
        }
    )
    _atomic_write_json(output_dir / "decision.json", decision)
    summary = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "status": decision["status"],
        "common_horizon_gpu_hours": horizon,
        "shared_grid_points": len(aligned),
        "checkpoint_0_present": aligned[0]["target_baseline_iteration"] == 0,
        "common_horizon_endpoint_present": _close(float(aligned[-1]["target_gpu_hours"]), horizon),
        "extrapolated_points": 0,
        "joint_elo": {
            "single_joint_fit": True,
            "fit_id": fit_id,
            "fit_scope": fit_scope,
            "baseline_normalized_aulc": baseline_aulc,
            "adaptive_normalized_aulc": adaptive_aulc,
            "adaptive_minus_baseline_normalized_aulc": adaptive_aulc - baseline_aulc,
        },
        "performance_improvement": decision["performance_improvement"],
        "mechanism_attribution": decision["mechanism_attribution"],
        "input_audit_status": "passed",
        "outputs": {
            "aligned_strength": (output_dir / "aligned_strength.csv").as_posix(),
            "utility_curve": (output_dir / "utility_curve.csv").as_posix(),
            "effects": (output_dir / "effects.csv").as_posix(),
            "decision": (output_dir / "decision.json").as_posix(),
        },
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    manifest["common_horizon_gpu_hours"] = horizon
    manifest["shared_grid_points"] = len(aligned)
    _atomic_write_json(output_dir / "input_manifest.json", manifest)
    _atomic_write_json(output_dir / "input_audit.json", audit.payload())
    _atomic_write_text(
        output_dir / "run.log",
        [
            f"H3 analysis status: {decision['status']}",
            f"Common horizon GPU-hours: {horizon}",
            f"Shared observed targets: {len(aligned)}",
            f"Joint Elo fit_id: {fit_id}",
            f"Baseline normalized AULC: {baseline_aulc}",
            f"Adaptive normalized AULC: {adaptive_aulc}",
            f"Adaptive minus Baseline normalized AULC: {adaptive_aulc - baseline_aulc}",
            f"H2 attribution input status: {h2_status}",
            f"Mechanism attribution: {decision['mechanism_attribution']['status']}",
        ],
    )
    return str(decision["status"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        status = execute(args.config, args.output_dir)
    except (H3Error, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"H3 analysis status: {status}")
    return 0 if status != "not_assessable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
