#!/usr/bin/env python3
"""Summarize and project a completed baseline pilot without changing its config."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

from runtime.artifacts import atomic_write_json


EVALUATION_PATTERN = re.compile(r"^evaluation_checkpoint_(\d+)\.json$")
COMPARISON_FIELDS = (
    "total_seconds",
    "self_play_seconds",
    "network_training_seconds",
    "evaluation_seconds",
    "gpu_hours_including_evaluation",
    "seconds_per_game",
    "seconds_per_position",
    "optimizer_steps",
    "replay_size_growth",
    "peak_gpu_memory_mb",
)


class PilotSummaryError(ValueError):
    """Raised when pilot artifacts cannot be summarized safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-hours",
        type=float,
        action="append",
        default=[],
        help=(
            "GPU-hour budget to project; may be supplied more than once. "
            "The report always includes iterations per single GPU-hour."
        ),
    )
    return parser.parse_args()


def _mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PilotSummaryError(f"{label} not found: {path}")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise PilotSummaryError(f"could not parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PilotSummaryError(f"{label} must contain a mapping")
    return payload


def _finite_number(
    value: Any,
    label: str,
    *,
    nonnegative: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotSummaryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise PilotSummaryError(f"{label} must be a finite non-negative number")
    return result


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label)


def _artifact_filename(config: dict[str, Any], field: str, default: str) -> str:
    value = config.get("logging", {}).get(field, default)
    if not isinstance(value, str) or not value:
        raise PilotSummaryError(f"logging.{field} must be a filename")
    requested = Path(value)
    if requested.is_absolute() or requested.name != value:
        raise PilotSummaryError(f"logging.{field} must be a filename")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotSummaryError(f"metrics file not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise PilotSummaryError(f"blank metrics line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotSummaryError(
                f"invalid metrics JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise PilotSummaryError(
                f"metrics line {line_number} must contain an object"
            )
        records.append(record)
    if not records:
        raise PilotSummaryError("metrics file is empty")
    iterations = [record.get("iteration") for record in records]
    expected = list(range(1, len(records) + 1))
    if iterations != expected:
        raise PilotSummaryError(
            f"metrics iterations must be contiguous from 1: {iterations}"
        )
    return records


def _evaluation_timings(
    run_dir: Path,
) -> tuple[dict[int, float], list[dict[str, Any]], list[str]]:
    evaluation_dir = run_dir / "evaluations"
    timings: dict[int, float] = {}
    files: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not evaluation_dir.is_dir():
        return timings, files, warnings
    for path in sorted(evaluation_dir.glob("evaluation_checkpoint_*.json")):
        match = EVALUATION_PATTERN.match(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        payload = _mapping(path, f"evaluation for checkpoint {iteration}")
        timing = payload.get("timing")
        duration = None
        if isinstance(timing, dict):
            duration = timing.get("evaluation_seconds")
        if duration is None:
            duration = payload.get("evaluation_seconds")
        if duration is None:
            warnings.append(
                f"{path.name} predates evaluation timing instrumentation"
            )
        else:
            timings[iteration] = _finite_number(
                duration, f"{path.name}.timing.evaluation_seconds"
            )
        files.append(
            {
                "iteration": iteration,
                "path": path.as_posix(),
                "evaluation_seconds": timings.get(iteration),
            }
        )
    return timings, files, warnings


def _median(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.median(finite) if finite else None


def _comparison(
    first: dict[str, Any],
    later: list[dict[str, Any]],
) -> dict[str, Any]:
    later_median = {
        field: _median([record.get(field) for record in later])
        for field in COMPARISON_FIELDS
    }
    first_values = {field: first.get(field) for field in COMPARISON_FIELDS}
    absolute: dict[str, float | None] = {}
    percent: dict[str, float | None] = {}
    ratio: dict[str, float | None] = {}
    for field in COMPARISON_FIELDS:
        first_value = first_values[field]
        median_value = later_median[field]
        if first_value is None or median_value is None:
            absolute[field] = None
            percent[field] = None
            ratio[field] = None
            continue
        difference = float(first_value) - median_value
        absolute[field] = difference
        if median_value == 0:
            percent[field] = None
            ratio[field] = None
        else:
            percent[field] = difference / median_value * 100.0
            ratio[field] = float(first_value) / median_value
    return {
        "first_iteration": first_values,
        "later_iterations": [record["iteration"] for record in later],
        "later_median": later_median,
        "absolute_difference_first_minus_later_median": absolute,
        "percent_difference_first_minus_later_median": percent,
        "ratio_first_to_later_median": ratio,
    }


def summarize_pilot(
    run_dir: Path | str,
    *,
    gpu_hour_budgets: list[float] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise PilotSummaryError(f"run directory not found: {run_path}")
    config_path = run_path / "resolved_config.yaml"
    config = _mapping(config_path, "resolved configuration")
    if config.get("mode") != "baseline":
        raise PilotSummaryError("resolved configuration mode must be baseline")
    run = config.get("run")
    if not isinstance(run, dict) or run.get("id") != run_path.name:
        raise PilotSummaryError("run ID must match the run directory name")
    metrics_path = run_path / _artifact_filename(
        config, "metrics_file", "metrics.jsonl"
    )
    metrics = _read_metrics(metrics_path)
    summary_path = run_path / _artifact_filename(
        config, "summary_file", "summary.json"
    )
    summary = _mapping(summary_path, "run summary")
    if summary.get("status") != "completed":
        raise PilotSummaryError("run summary status must be completed")
    completed_iterations = [int(record["iteration"]) for record in metrics]
    if summary.get("completed_iterations") != completed_iterations:
        raise PilotSummaryError(
            "run summary completed iterations do not match metrics"
        )
    evaluation_timings, evaluation_files, warnings = _evaluation_timings(run_path)

    evaluation = config.get("evaluation", {})
    evaluation_enabled = bool(evaluation.get("enabled", False))
    cadence = evaluation.get("evaluate_every_iterations")
    if evaluation_enabled:
        if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
            raise PilotSummaryError(
                "evaluation.evaluate_every_iterations must be an integer >= 1"
            )
    else:
        cadence = None

    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    previous_replay_size = 0
    previous_cumulative_gpu_hours = 0.0
    scheduled_missing: list[int] = []
    for index, record in enumerate(metrics, 1):
        iteration = int(record["iteration"])
        total_seconds = _finite_number(
            record.get("iteration_seconds"),
            f"metrics iteration {iteration}.iteration_seconds",
        )
        self_play_seconds = _finite_number(
            record.get("self_play_seconds"),
            f"metrics iteration {iteration}.self_play_seconds",
        )
        training_seconds = _finite_number(
            record.get("training_seconds"),
            f"metrics iteration {iteration}.training_seconds",
        )
        games = int(
            _finite_number(
                record.get("games_completed"),
                f"metrics iteration {iteration}.games_completed",
            )
        )
        positions = int(
            _finite_number(
                record.get("positions_generated"),
                f"metrics iteration {iteration}.positions_generated",
            )
        )
        optimizer_steps = int(
            _finite_number(
                record.get("optimizer_steps"),
                f"metrics iteration {iteration}.optimizer_steps",
            )
        )
        replay_size = int(
            _finite_number(
                record.get("replay_buffer_size"),
                f"metrics iteration {iteration}.replay_buffer_size",
            )
        )
        peak_memory = _optional_number(
            record.get("peak_gpu_memory_mb"),
            f"metrics iteration {iteration}.peak_gpu_memory_mb",
        )
        scheduled = bool(
            evaluation_enabled and cadence and iteration % cadence == 0
        )
        evaluation_seconds = evaluation_timings.get(iteration)
        if evaluation_seconds is None and not scheduled:
            evaluation_seconds = 0.0
        elif evaluation_seconds is None:
            scheduled_missing.append(iteration)
        total_with_evaluation = (
            total_seconds + evaluation_seconds
            if evaluation_seconds is not None
            else None
        )
        cumulative = _optional_number(
            record.get("cumulative_gpu_hours"),
            f"metrics iteration {iteration}.cumulative_gpu_hours",
        )
        reported_delta = (
            cumulative - previous_cumulative_gpu_hours
            if cumulative is not None
            else None
        )
        if reported_delta is not None and reported_delta < -1e-12:
            raise PilotSummaryError(
                f"cumulative GPU-hours decreased at iteration {iteration}"
            )
        if cumulative is not None:
            previous_cumulative_gpu_hours = cumulative
        replay_growth = replay_size - previous_replay_size
        previous_replay_size = replay_size
        seconds_per_game = self_play_seconds / games if games else None
        seconds_per_position = self_play_seconds / positions if positions else None
        other_seconds = max(0.0, total_seconds - self_play_seconds - training_seconds)
        iteration_gpu_hours = total_seconds / 3600.0
        gpu_hours_with_evaluation = (
            total_with_evaluation / 3600.0
            if total_with_evaluation is not None
            else None
        )
        row = {
            "iteration": iteration,
            "timing": {
                "total_seconds": total_seconds,
                "self_play_seconds": self_play_seconds,
                "network_training_seconds": training_seconds,
                "other_iteration_seconds": other_seconds,
                "evaluation_seconds": evaluation_seconds,
                "evaluation_timing_status": (
                    "measured"
                    if iteration in evaluation_timings
                    else "missing"
                    if scheduled
                    else "not_scheduled"
                ),
                "total_including_evaluation_seconds": total_with_evaluation,
            },
            "gpu_hours": {
                "iteration": iteration_gpu_hours,
                "evaluation": (
                    evaluation_seconds / 3600.0
                    if evaluation_seconds is not None
                    else None
                ),
                "including_evaluation": gpu_hours_with_evaluation,
                "reported_cumulative": cumulative,
                "reported_iteration_delta": reported_delta,
            },
            "self_play": {
                "games_completed": games,
                "positions_generated": positions,
                "seconds_per_game": seconds_per_game,
                "seconds_per_position": seconds_per_position,
            },
            "training": {"optimizer_steps": optimizer_steps},
            "replay": {"size": replay_size, "growth": replay_growth},
            "resources": {"peak_gpu_memory_mb": peak_memory},
        }
        rows.append(row)
        flat_rows.append(
            {
                "iteration": iteration,
                "total_seconds": total_seconds,
                "self_play_seconds": self_play_seconds,
                "network_training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "gpu_hours_including_evaluation": gpu_hours_with_evaluation,
                "seconds_per_game": seconds_per_game,
                "seconds_per_position": seconds_per_position,
                "optimizer_steps": optimizer_steps,
                "replay_size_growth": replay_growth,
                "peak_gpu_memory_mb": peak_memory,
            }
        )

    if scheduled_missing:
        warnings.append(
            "scheduled evaluation timing is missing for iteration(s): "
            + ", ".join(str(value) for value in scheduled_missing)
        )

    later = flat_rows[1:]
    steady_median = {
        field: _median([record.get(field) for record in later])
        for field in COMPARISON_FIELDS
    }
    measured_steady_evaluations = [
        duration
        for iteration, duration in evaluation_timings.items()
        if iteration > 1
    ]
    median_evaluation = _median(measured_steady_evaluations)
    amortized_evaluation = (
        median_evaluation / cadence
        if median_evaluation is not None and cadence is not None
        else 0.0
    )
    median_base_seconds = steady_median["total_seconds"]
    projected_seconds = (
        median_base_seconds + amortized_evaluation
        if median_base_seconds is not None
        else None
    )
    projected_gpu_hours = (
        projected_seconds / 3600.0 if projected_seconds else None
    )
    iterations_per_gpu_hour = (
        1.0 / projected_gpu_hours if projected_gpu_hours else None
    )

    requested_budgets = list(gpu_hour_budgets or [])
    configured_budget = config.get("budget", {}).get("max_gpu_hours")
    if configured_budget is not None:
        configured_budget = _finite_number(
            configured_budget, "budget.max_gpu_hours"
        )
        if configured_budget not in requested_budgets:
            requested_budgets.append(configured_budget)
    normalized_budgets: list[float] = []
    for index, budget in enumerate(requested_budgets):
        value = _finite_number(budget, f"gpu_hour_budgets[{index}]")
        if value <= 0:
            raise PilotSummaryError("GPU-hour projection budgets must be > 0")
        if value not in normalized_budgets:
            normalized_budgets.append(value)
    projections = [
        {
            "gpu_hours": budget,
            "estimated_iterations": (
                math.floor(budget / projected_gpu_hours)
                if projected_gpu_hours
                else None
            ),
            "unrounded_iterations": (
                budget / projected_gpu_hours if projected_gpu_hours else None
            ),
        }
        for budget in normalized_budgets
    ]

    return {
        "schema_version": 1,
        "status": "summarized",
        "run_id": run["id"],
        "run_dir": run_path.as_posix(),
        "completed_iterations": completed_iterations,
        "source_files": {
            "resolved_config": config_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "summary": summary_path.as_posix(),
            "evaluations": evaluation_files,
        },
        "iterations": rows,
        "evaluation": {
            "enabled": evaluation_enabled,
            "configured_frequency_iterations": cadence,
            "measured_iterations": sorted(evaluation_timings),
            "scheduled_missing_timing_iterations": scheduled_missing,
            "total_measured_seconds": sum(evaluation_timings.values()),
            "median_measured_seconds_excluding_iteration_1": median_evaluation,
        },
        "first_iteration_vs_later": (
            _comparison(flat_rows[0], later) if later else None
        ),
        "steady_state_excluding_iteration_1": {
            "iterations": [record["iteration"] for record in later],
            "median": steady_median,
        },
        "projection": {
            "basis": "median iteration time excluding iteration 1 plus amortized evaluation",
            "median_base_iteration_seconds": median_base_seconds,
            "median_evaluation_seconds": median_evaluation,
            "evaluation_amortized_seconds_per_iteration": amortized_evaluation,
            "estimated_seconds_per_iteration": projected_seconds,
            "estimated_gpu_hours_per_iteration": projected_gpu_hours,
            "estimated_iterations_per_gpu_hour": iterations_per_gpu_hour,
            "requested_budgets": projections,
            "modifies_formal_configuration": False,
        },
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    try:
        report = summarize_pilot(
            args.run_dir,
            gpu_hour_budgets=args.gpu_hours,
        )
        output_path = args.run_dir.expanduser().resolve() / "pilot_report.json"
        atomic_write_json(output_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Pilot report: {output_path}")
        return 0
    except (PilotSummaryError, OSError, ValueError) as exc:
        print(f"Pilot summary error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
