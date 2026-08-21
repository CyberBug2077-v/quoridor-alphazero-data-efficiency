#!/usr/bin/env python3
"""Audit, aggregate, and formally decide H2 at matched GPU-hours."""

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
from pathlib import Path
from typing import Any, Sequence

import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
DEFAULT_CONFIG = EXPERIMENTS_ROOT / "configs" / "h2_v2.yaml"

ALIGNED_FIELDS = (
    "target_baseline_iteration",
    "target_gpu_hours",
    "metric",
    "source",
    "baseline_selected_iteration",
    "baseline_selected_gpu_hours",
    "baseline_value",
    "baseline_status",
    "adaptive_selected_iteration",
    "adaptive_selected_gpu_hours",
    "adaptive_value",
    "adaptive_status",
    "interpolation",
)

EFFECT_FIELDS = (
    "metric",
    "scope",
    "baseline_value",
    "adaptive_value",
    "raw_relative_effect",
    "direction_adjusted_effect",
    "practical_threshold",
    "valid_points",
    "coverage_gpu_hours",
    "assessment",
    "decision_role",
    "audit_status",
    "aggregation",
    "baseline_valid_points",
    "adaptive_valid_points",
    "baseline_coverage_gpu_hours",
    "adaptive_coverage_gpu_hours",
    "baseline_aggregation_numerator",
    "baseline_aggregation_denominator",
    "adaptive_aggregation_numerator",
    "adaptive_aggregation_denominator",
    "baseline_excluded_rows",
    "adaptive_excluded_rows",
)

RESOURCE_FIELDS = (
    "condition",
    "scope",
    "audit_status",
    "scope_start_gpu_hours_exclusive",
    "scope_end_gpu_hours_inclusive",
    "selected_final_iteration",
    "selected_final_gpu_hours",
    "valid_iterations",
    "coverage_gpu_hours",
    "positions_generated",
    "games_completed",
    "optimizer_steps",
    "self_play_fraction",
    "training_fraction",
    "scheduler_overhead_fraction",
    "instrumentation_overhead_fraction",
)

SCOPES = ("full_interval", "endpoint_interval")


class H2Error(ValueError):
    """Raised when the H2 protocol itself cannot be executed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise H2Error(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing H2 protocol: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise H2Error(f"invalid H2 protocol: {exc}") from exc
    _require(isinstance(payload, dict), "H2 protocol must contain a mapping")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise H2Error(f"JSON input must contain an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                raise H2Error(f"blank JSONL row at {path}:{line_number}")
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise H2Error(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
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
        parsed = float(value)
    except ValueError:
        return value
    return parsed


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise H2Error(f"CSV input lacks a header: {path}")
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


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
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
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
    if not _finite_number(value):
        raise H2Error(f"field {field} is missing or non-finite at iteration {row.get('iteration')}")
    return float(value)


def _integer(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise H2Error(f"field {field} is not an integer")
    return value


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


def _input_paths_and_hashes(
    config: dict[str, Any], audit: Audit
) -> tuple[dict[str, Path], dict[str, Any]]:
    raw_inputs = config.get("inputs")
    _require(isinstance(raw_inputs, dict) and raw_inputs, "H2 protocol inputs are empty")
    paths: dict[str, Path] = {}
    manifest: dict[str, Any] = {}
    for name, specification in raw_inputs.items():
        if not isinstance(specification, dict):
            audit.record(f"input.{name}.specification", False, "not a mapping")
            continue
        path = _resolve_experiments_path(specification.get("path", ""))
        paths[name] = path
        expected = specification.get("sha256")
        exists = path.is_file()
        actual = _sha256_file(path) if exists else None
        passed = (
            exists
            and isinstance(expected, str)
            and actual == expected.lower()
        )
        audit.record(
            f"input.{name}.sha256",
            passed,
            {"path": path.as_posix(), "expected": expected, "actual": actual},
        )
        manifest[name] = {
            "path": path.as_posix(),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "passed" if passed else "failed",
        }
    return paths, manifest


def common_horizon_from_matched_compute(
    matched: dict[str, Any],
) -> tuple[float, float, list[dict[str, Any]]]:
    try:
        raw_targets = matched["pairing_and_randomness"]["checkpoint_grid"]["targets"]
        cap = matched["compute_budget"]["common_horizon"]["maximum_gpu_hours"]
    except (KeyError, TypeError) as exc:
        raise H2Error("matched-compute protocol lacks the horizon fields") from exc
    _require(isinstance(raw_targets, list) and len(raw_targets) >= 2, "checkpoint grid is too short")
    targets: list[dict[str, Any]] = []
    for raw in raw_targets:
        _require(isinstance(raw, dict), "checkpoint-grid target is not a mapping")
        iteration = raw.get("baseline_checkpoint_iteration")
        gpu_hours = raw.get("gpu_hours")
        _require(
            isinstance(iteration, int)
            and not isinstance(iteration, bool)
            and _finite_number(gpu_hours),
            "checkpoint-grid target is invalid",
        )
        targets.append(
            {
                "target_baseline_iteration": iteration,
                "target_gpu_hours": float(gpu_hours),
            }
        )
    _require(
        all(
            targets[index]["target_gpu_hours"]
            < targets[index + 1]["target_gpu_hours"]
            for index in range(len(targets) - 1)
        ),
        "checkpoint-grid GPU-hours are not strictly increasing",
    )
    _require(_finite_number(cap) and float(cap) > 0.0, "common-horizon cap is invalid")
    horizon = min(targets[-1]["target_gpu_hours"], float(cap))
    eligible_grid = [item for item in targets if item["target_gpu_hours"] <= horizon]
    _require(len(eligible_grid) >= 2, "common horizon leaves fewer than two grid targets")
    endpoint_start = eligible_grid[-2]["target_gpu_hours"]
    return horizon, endpoint_start, eligible_grid


def _validate_metrics(
    condition: str, rows: list[dict[str, Any]], audit: Audit
) -> dict[int, dict[str, Any]] | None:
    try:
        _require(bool(rows), f"{condition} metrics are empty")
        iterations = [_integer(row, "iteration") for row in rows]
        _require(iterations == list(range(1, len(rows) + 1)), f"{condition} metrics are not contiguous")
        required = (
            "cumulative_gpu_hours",
            "iteration_seconds",
            "positions_generated",
            "games_completed",
            "optimizer_steps",
            "replay_buffer_size",
            "examples_used",
            "samples_seen",
            "self_play_seconds",
            "training_seconds",
        )
        previous_gpu_hours = -1.0
        for row in rows:
            for field in required:
                _number(row, field)
            cumulative = _number(row, "cumulative_gpu_hours")
            _require(cumulative > previous_gpu_hours, f"{condition} GPU-hours are not increasing")
            _require(_number(row, "iteration_seconds") > 0.0, f"{condition} has non-positive iteration_seconds")
            _require(_number(row, "optimizer_steps") > 0.0, f"{condition} has zero optimizer_steps")
            row["iteration_gpu_hours"] = _number(row, "iteration_seconds") / 3600.0
            previous_gpu_hours = cumulative
        output = {_integer(row, "iteration"): row for row in rows}
        audit.record(
            f"state.{condition}.metrics",
            True,
            {
                "rows": len(rows),
                "iteration_start": iterations[0],
                "iteration_end": iterations[-1],
                "final_gpu_hours": previous_gpu_hours,
            },
        )
        return output
    except (H2Error, TypeError, ValueError) as exc:
        audit.record(f"state.{condition}.metrics", False, str(exc))
        return None


def _join_analysis_rows(
    condition: str,
    label: str,
    rows: list[dict[str, Any]],
    metrics: dict[int, dict[str, Any]],
    *,
    require_complete: bool,
    audit: Audit,
) -> list[dict[str, Any]] | None:
    try:
        iterations = [_integer(row, "iteration") for row in rows]
        _require(len(iterations) == len(set(iterations)), f"{condition} {label} has duplicate iterations")
        if require_complete:
            _require(set(iterations) == set(metrics), f"{condition} {label} iteration coverage differs from metrics")
        else:
            _require(set(iterations) <= set(metrics), f"{condition} {label} contains an unknown iteration")
        joined: list[dict[str, Any]] = []
        for row in rows:
            iteration = _integer(row, "iteration")
            metric_row = metrics[iteration]
            enriched = dict(row)
            enriched["cumulative_gpu_hours"] = _number(metric_row, "cumulative_gpu_hours")
            enriched["iteration_seconds"] = _number(metric_row, "iteration_seconds")
            enriched["iteration_gpu_hours"] = _number(metric_row, "iteration_gpu_hours")
            joined.append(enriched)
        joined.sort(key=lambda row: _integer(row, "iteration"))
        audit.record(
            f"state.{condition}.{label}",
            True,
            {
                "rows": len(joined),
                "iteration_start": iterations[0] if iterations else None,
                "iteration_end": iterations[-1] if iterations else None,
            },
        )
        return joined
    except (H2Error, TypeError, ValueError) as exc:
        audit.record(f"state.{condition}.{label}", False, str(exc))
        return None


def _validate_analysis_fields(
    condition: str,
    derived: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    audit: Audit,
) -> None:
    try:
        for row in derived:
            for field in (
                "turnover_fraction",
                "evicted_states",
                "mean_sample_age",
                "p90_sample_age",
            ):
                _number(row, field)
        for row in replay:
            for field in (
                "states",
                "incoming_unique_states",
                "duplicate_hash_occurrences",
            ):
                _number(row, field)
            _require(row.get("count_matches_metrics") is True, f"{condition} replay count mismatch")
        censored = [
            _integer(row, "iteration")
            for row in replay
            if row.get("incoming_ratio_left_censored") is True
        ]
        _require(
            len(censored) == 1 and censored[0] == _integer(replay[0], "iteration"),
            f"{condition} replay left-censor marker is invalid",
        )
        audit.record(
            f"state.{condition}.analysis_fields",
            True,
            {"left_censored_incoming_ratio_iterations": censored},
        )
    except (H2Error, TypeError, ValueError, IndexError) as exc:
        audit.record(f"state.{condition}.analysis_fields", False, str(exc))


def _validate_quality_report(condition: str, report: dict[str, Any], audit: Audit) -> None:
    checks = report.get("checks")
    checks_passed = isinstance(checks, dict) and all(value is True for value in checks.values())
    errors = report.get("global_errors", report.get("findings", []))
    passed = report.get("status") == "passed" and checks_passed and errors == []
    audit.record(
        f"quality.{condition}.data_quality_report",
        passed,
        {"status": report.get("status"), "checks": checks, "errors": errors},
    )


def _validate_replay_summary(condition: str, summary: dict[str, Any], audit: Audit) -> None:
    validations = summary.get("validations")
    anomalies = summary.get("anomaly_counts")
    passed = (
        summary.get("status") == "completed"
        and isinstance(validations, dict)
        and all(value is True for value in validations.values())
        and isinstance(anomalies, dict)
        and sum(int(value) for value in anomalies.values()) == 0
    )
    audit.record(
        f"quality.{condition}.replay_summary",
        passed,
        {
            "status": summary.get("status"),
            "validations": validations,
            "anomaly_counts": anomalies,
        },
    )


def _validate_holdout_summary(condition: str, summary: dict[str, Any], audit: Audit) -> None:
    if condition == "baseline":
        passed = (
            summary.get("status") == "completed"
            and summary.get("checkpoint_count") == 12
            and summary.get("states") == 9259
            and summary.get("games") == 200
        )
        details = {
            "status": summary.get("status"),
            "checkpoint_count": summary.get("checkpoint_count"),
            "states": summary.get("states"),
            "games": summary.get("games"),
        }
    else:
        acceptance = summary.get("acceptance")
        passed = (
            summary.get("status") == "completed"
            and isinstance(acceptance, dict)
            and acceptance.get("target_count") == 12
            and acceptance.get("states_per_target") == 9259
            and acceptance.get("trajectories_per_target") == 200
            and acceptance.get("all_metrics_finite") is True
            and acceptance.get("all_selected_checkpoints_not_after_target") is True
            and acceptance.get("checkpoint_0_identity_verified") is True
        )
        details = {"status": summary.get("status"), "acceptance": acceptance}
    audit.record(f"quality.{condition}.holdout_summary", passed, details)


def _parse_inputs(paths: dict[str, Path], audit: Audit) -> dict[str, Any]:
    loaders = {
        "matched_compute": _load_yaml,
        "baseline_metrics": _load_jsonl,
        "baseline_derived_metrics": _load_csv,
        "baseline_replay_iteration_metrics": _load_csv,
        "baseline_replay_summary": _load_json,
        "baseline_data_quality_report": _load_json,
        "baseline_holdout_summary": _load_json,
        "adaptive_metrics": _load_jsonl,
        "adaptive_derived_metrics": _load_csv,
        "adaptive_replay_iteration_metrics": _load_csv,
        "adaptive_replay_summary": _load_json,
        "adaptive_data_quality_report": _load_json,
        "adaptive_holdout_summary": _load_json,
    }
    parsed: dict[str, Any] = {}
    for name, loader in loaders.items():
        path = paths.get(name)
        if path is None or not path.is_file():
            audit.record(f"parse.{name}", False, "input is missing")
            continue
        try:
            parsed[name] = loader(path)
            audit.record(f"parse.{name}", True, {"path": path.as_posix()})
        except (H2Error, OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
            audit.record(f"parse.{name}", False, str(exc))
    return parsed


def _build_condition(
    condition: str, parsed: dict[str, Any], audit: Audit
) -> dict[str, Any] | None:
    required = (
        f"{condition}_metrics",
        f"{condition}_derived_metrics",
        f"{condition}_replay_iteration_metrics",
        f"{condition}_replay_summary",
        f"{condition}_data_quality_report",
        f"{condition}_holdout_summary",
    )
    if any(name not in parsed for name in required):
        audit.record(f"state.{condition}.inputs_complete", False, "one or more parsed inputs are missing")
        return None
    audit.record(f"state.{condition}.inputs_complete", True, list(required))
    metrics = _validate_metrics(condition, parsed[f"{condition}_metrics"], audit)
    if metrics is None:
        return None
    derived = _join_analysis_rows(
        condition,
        "derived_metrics",
        parsed[f"{condition}_derived_metrics"],
        metrics,
        require_complete=True,
        audit=audit,
    )
    replay = _join_analysis_rows(
        condition,
        "replay_iteration_metrics",
        parsed[f"{condition}_replay_iteration_metrics"],
        metrics,
        require_complete=False,
        audit=audit,
    )
    if derived is None or replay is None:
        return None
    _validate_analysis_fields(condition, derived, replay, audit)
    _validate_quality_report(
        condition, parsed[f"{condition}_data_quality_report"], audit
    )
    _validate_replay_summary(condition, parsed[f"{condition}_replay_summary"], audit)
    _validate_holdout_summary(condition, parsed[f"{condition}_holdout_summary"], audit)
    return {
        "metrics": list(metrics.values()),
        "derived": derived,
        "replay": replay,
        "holdout_summary": parsed[f"{condition}_holdout_summary"],
    }


def _scope_bounds(
    scope: str, common_horizon: float, endpoint_start: float
) -> tuple[float, float]:
    if scope == "full_interval":
        return 0.0, common_horizon
    if scope == "endpoint_interval":
        return endpoint_start, common_horizon
    raise H2Error(f"unknown H2 scope: {scope}")


def _source_rows(condition: dict[str, Any], specification: dict[str, Any]) -> list[dict[str, Any]]:
    source = specification.get("source")
    if source not in {"metrics", "derived", "replay"}:
        raise H2Error(f"unknown metric source: {source}")
    return condition[source]


def _row_exclusion_reason(
    row: dict[str, Any], specification: dict[str, Any]
) -> str | None:
    exclude_when_true = specification.get("exclude_when_true")
    if exclude_when_true and row.get(exclude_when_true) is True:
        return f"{exclude_when_true}=true"
    require_positive = specification.get("require_positive")
    if require_positive:
        try:
            if _number(row, require_positive) <= 0.0:
                return f"{require_positive}<=0"
        except H2Error:
            return f"{require_positive}=missing_or_non_finite"
    aggregation = specification.get("aggregation")
    if aggregation == "ratio_of_sums":
        try:
            _number(row, specification["numerator"])
            denominator = _number(row, specification["denominator"])
        except (H2Error, KeyError):
            return "ratio_component_missing_or_non_finite"
        if denominator <= 0.0:
            return "denominator<=0"
    elif aggregation == "iteration_seconds_weighted_mean":
        try:
            _number(row, specification["value"])
            weight = _number(row, specification["weight"])
        except (H2Error, KeyError):
            return "arithmetic_component_missing_or_non_finite"
        if weight <= 0.0:
            return "weight<=0"
    else:
        return "unknown_aggregation"
    return None


def _row_value(row: dict[str, Any], specification: dict[str, Any]) -> float:
    if specification["aggregation"] == "ratio_of_sums":
        return _number(row, specification["numerator"]) / _number(
            row, specification["denominator"]
        )
    return _number(row, specification["value"])


def align_metrics(
    conditions: dict[str, dict[str, Any]],
    metric_specs: dict[str, dict[str, Any]],
    grid: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in grid:
        for metric, specification in metric_specs.items():
            row: dict[str, Any] = {
                **target,
                "metric": metric,
                "source": specification["source"],
                "interpolation": "none",
            }
            for condition_name, condition in conditions.items():
                eligible = [
                    candidate
                    for candidate in _source_rows(condition, specification)
                    if _number(candidate, "cumulative_gpu_hours")
                    <= target["target_gpu_hours"]
                    and _row_exclusion_reason(candidate, specification) is None
                ]
                prefix = condition_name
                if not eligible:
                    row[f"{prefix}_selected_iteration"] = None
                    row[f"{prefix}_selected_gpu_hours"] = None
                    row[f"{prefix}_value"] = None
                    row[f"{prefix}_status"] = "no_eligible_completed_row"
                    continue
                selected = max(
                    eligible, key=lambda item: _number(item, "cumulative_gpu_hours")
                )
                row[f"{prefix}_selected_iteration"] = _integer(selected, "iteration")
                row[f"{prefix}_selected_gpu_hours"] = _number(
                    selected, "cumulative_gpu_hours"
                )
                row[f"{prefix}_value"] = _row_value(selected, specification)
                row[f"{prefix}_status"] = "selected"
            output.append(row)
    return output


def _aggregate_condition(
    condition: dict[str, Any],
    specification: dict[str, Any],
    *,
    lower_bound: float,
    upper_bound: float,
    minimum_points: int,
) -> dict[str, Any]:
    scoped = [
        row
        for row in _source_rows(condition, specification)
        if lower_bound < _number(row, "cumulative_gpu_hours") <= upper_bound
    ]
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in scoped:
        reason = _row_exclusion_reason(row, specification)
        if reason is None:
            eligible.append(row)
        else:
            excluded.append(
                {
                    "iteration": _integer(row, "iteration"),
                    "cumulative_gpu_hours": _number(row, "cumulative_gpu_hours"),
                    "reason": reason,
                }
            )
    numerator: float | None = None
    denominator: float | None = None
    value: float | None = None
    status = "available"
    if specification["aggregation"] == "ratio_of_sums":
        numerator = sum(_number(row, specification["numerator"]) for row in eligible)
        denominator = sum(
            _number(row, specification["denominator"]) for row in eligible
        )
        if denominator > 0.0:
            value = numerator / denominator
        else:
            status = "zero_denominator"
    else:
        numerator = sum(
            _number(row, specification["value"])
            * _number(row, specification["weight"])
            for row in eligible
        )
        denominator = sum(_number(row, specification["weight"]) for row in eligible)
        if denominator > 0.0:
            value = numerator / denominator
        else:
            status = "zero_weight"
    if len(eligible) < minimum_points:
        value = None
        status = "insufficient_valid_points"
    coverage = sum(_number(row, "iteration_gpu_hours") for row in eligible)
    return {
        "value": value,
        "status": status,
        "valid_points": len(eligible),
        "coverage_gpu_hours": coverage,
        "aggregation_numerator": numerator,
        "aggregation_denominator": denominator,
        "eligible_iterations": [_integer(row, "iteration") for row in eligible],
        "eligible_completion_gpu_hours": [
            _number(row, "cumulative_gpu_hours") for row in eligible
        ],
        "excluded_rows": excluded,
        "scope_rows": len(scoped),
    }


def _effect_assessment(
    baseline: dict[str, Any],
    adaptive: dict[str, Any],
    specification: dict[str, Any],
    *,
    audit_status: str,
) -> tuple[float | None, float | None, str]:
    if audit_status != "passed":
        return None, None, "not_assessable_input_audit_failed"
    if baseline["value"] is None or adaptive["value"] is None:
        return None, None, "unavailable"
    if baseline["value"] == 0.0:
        return None, None, "unavailable_zero_baseline_value"
    raw = adaptive["value"] / baseline["value"] - 1.0
    adjusted = raw if specification["expected_direction"] == "increase" else -raw
    assessment = (
        "improved"
        if adjusted >= float(specification["practical_threshold"])
        else "not_improved"
    )
    return raw, adjusted, assessment


def calculate_effects(
    conditions: dict[str, dict[str, Any]] | None,
    metric_specs: dict[str, dict[str, Any]],
    *,
    common_horizon: float | None,
    endpoint_start: float | None,
    minimum_points: int,
    audit_status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for metric, specification in metric_specs.items():
        coverage[metric] = {}
        for scope in SCOPES:
            if (
                conditions is None
                or common_horizon is None
                or endpoint_start is None
                or audit_status != "passed"
            ):
                baseline = {
                    "value": None,
                    "valid_points": 0,
                    "coverage_gpu_hours": 0.0,
                    "aggregation_numerator": None,
                    "aggregation_denominator": None,
                    "excluded_rows": [],
                    "status": "input_audit_failed",
                    "eligible_iterations": [],
                    "eligible_completion_gpu_hours": [],
                    "scope_rows": 0,
                }
                adaptive = dict(baseline)
            else:
                lower, upper = _scope_bounds(scope, common_horizon, endpoint_start)
                baseline = _aggregate_condition(
                    conditions["baseline"],
                    specification,
                    lower_bound=lower,
                    upper_bound=upper,
                    minimum_points=minimum_points,
                )
                adaptive = _aggregate_condition(
                    conditions["adaptive"],
                    specification,
                    lower_bound=lower,
                    upper_bound=upper,
                    minimum_points=minimum_points,
                )
            raw, adjusted, assessment = _effect_assessment(
                baseline,
                adaptive,
                specification,
                audit_status=audit_status,
            )
            rows.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "baseline_value": baseline["value"],
                    "adaptive_value": adaptive["value"],
                    "raw_relative_effect": raw,
                    "direction_adjusted_effect": adjusted,
                    "practical_threshold": float(
                        specification["practical_threshold"]
                    ),
                    "valid_points": min(
                        baseline["valid_points"], adaptive["valid_points"]
                    ),
                    "coverage_gpu_hours": min(
                        baseline["coverage_gpu_hours"],
                        adaptive["coverage_gpu_hours"],
                    ),
                    "assessment": assessment,
                    "decision_role": specification["decision_role"],
                    "audit_status": audit_status,
                    "aggregation": specification["aggregation"],
                    "baseline_valid_points": baseline["valid_points"],
                    "adaptive_valid_points": adaptive["valid_points"],
                    "baseline_coverage_gpu_hours": baseline[
                        "coverage_gpu_hours"
                    ],
                    "adaptive_coverage_gpu_hours": adaptive[
                        "coverage_gpu_hours"
                    ],
                    "baseline_aggregation_numerator": baseline[
                        "aggregation_numerator"
                    ],
                    "baseline_aggregation_denominator": baseline[
                        "aggregation_denominator"
                    ],
                    "adaptive_aggregation_numerator": adaptive[
                        "aggregation_numerator"
                    ],
                    "adaptive_aggregation_denominator": adaptive[
                        "aggregation_denominator"
                    ],
                    "baseline_excluded_rows": len(baseline["excluded_rows"]),
                    "adaptive_excluded_rows": len(adaptive["excluded_rows"]),
                }
            )
            coverage[metric][scope] = {
                "baseline": baseline,
                "adaptive": adaptive,
            }
    return rows, coverage


def decision_from_effects(effect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the complete H2 status using only effects.csv-equivalent rows."""
    if not effect_rows or any(row["audit_status"] != "passed" for row in effect_rows):
        return {
            "status": "not_assessable",
            "primary_improved": None,
            "improved_complementary_metrics": [],
            "reason": "input_audit_failed",
        }
    primary = [
        row
        for row in effect_rows
        if row["scope"] == "full_interval" and row["decision_role"] == "primary"
    ]
    if len(primary) != 1 or primary[0]["assessment"] in {
        "unavailable",
        "unavailable_zero_baseline_value",
    }:
        return {
            "status": "not_assessable",
            "primary_improved": None,
            "improved_complementary_metrics": [],
            "reason": "primary_metric_unavailable",
        }
    primary_improved = primary[0]["assessment"] == "improved"
    complementary = [
        row
        for row in effect_rows
        if row["scope"] == "full_interval"
        and row["decision_role"] == "complementary_replay"
    ]
    improved = [
        row["metric"] for row in complementary if row["assessment"] == "improved"
    ]
    if not primary_improved:
        status = "not_supported"
        reason = "primary_improvement_below_threshold"
    elif improved:
        status = "supported"
        reason = "primary_and_complementary_thresholds_met"
    else:
        status = "partially_supported"
        reason = "primary_threshold_met_but_no_complementary_threshold_met"
    return {
        "status": status,
        "primary_improved": primary_improved,
        "improved_complementary_metrics": improved,
        "reason": reason,
    }


def _resource_rows(
    conditions: dict[str, dict[str, Any]] | None,
    *,
    audit_status: str,
    common_horizon: float | None,
    endpoint_start: float | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition_name in ("baseline", "adaptive"):
        for scope in SCOPES:
            if (
                conditions is None
                or audit_status != "passed"
                or common_horizon is None
                or endpoint_start is None
            ):
                output.append(
                    {
                        "condition": condition_name,
                        "scope": scope,
                        "audit_status": audit_status,
                    }
                )
                continue
            lower, upper = _scope_bounds(scope, common_horizon, endpoint_start)
            selected = [
                row
                for row in conditions[condition_name]["metrics"]
                if lower < _number(row, "cumulative_gpu_hours") <= upper
            ]
            seconds = sum(_number(row, "iteration_seconds") for row in selected)
            scheduler = (
                0.0
                if condition_name == "baseline"
                else sum(_number(row, "scheduler_seconds") for row in selected)
                / seconds
            )
            instrumentation = (
                None
                if condition_name == "baseline"
                else sum(
                    _number(row, "instrumentation_seconds") for row in selected
                )
                / seconds
            )
            output.append(
                {
                    "condition": condition_name,
                    "scope": scope,
                    "audit_status": audit_status,
                    "scope_start_gpu_hours_exclusive": lower,
                    "scope_end_gpu_hours_inclusive": upper,
                    "selected_final_iteration": _integer(selected[-1], "iteration"),
                    "selected_final_gpu_hours": _number(
                        selected[-1], "cumulative_gpu_hours"
                    ),
                    "valid_iterations": len(selected),
                    "coverage_gpu_hours": seconds / 3600.0,
                    "positions_generated": sum(
                        _number(row, "positions_generated") for row in selected
                    ),
                    "games_completed": sum(
                        _number(row, "games_completed") for row in selected
                    ),
                    "optimizer_steps": sum(
                        _number(row, "optimizer_steps") for row in selected
                    ),
                    "self_play_fraction": sum(
                        _number(row, "self_play_seconds") for row in selected
                    )
                    / seconds,
                    "training_fraction": sum(
                        _number(row, "training_seconds") for row in selected
                    )
                    / seconds,
                    "scheduler_overhead_fraction": scheduler,
                    "instrumentation_overhead_fraction": instrumentation,
                }
            )
    return output


def _holdout_evidence(parsed: dict[str, Any]) -> dict[str, Any]:
    baseline = parsed.get("baseline_holdout_summary", {})
    adaptive = parsed.get("adaptive_holdout_summary", {})
    return {
        "metric": "approximate_online_train_holdout_total_gap",
        "status": "unavailable_from_summary",
        "baseline_summary_status": baseline.get("status"),
        "adaptive_summary_status": adaptive.get("status"),
        "reason": "the required summary.json inputs certify evaluation status but do not contain checkpoint-level gap values",
        "can_change_h2_decision": False,
    }


def _audit_payload(
    audit: Audit,
    *,
    common_horizon: float | None,
    endpoint_start: float | None,
    grid: list[dict[str, Any]],
    conditions: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    horizon_details: dict[str, Any] = {
        "common_horizon_gpu_hours": common_horizon,
        "endpoint_start_gpu_hours": endpoint_start,
        "grid_targets": grid,
    }
    if conditions is not None and common_horizon is not None:
        selections: dict[str, Any] = {}
        for name, condition in conditions.items():
            inside = [
                row
                for row in condition["metrics"]
                if _number(row, "cumulative_gpu_hours") <= common_horizon
            ]
            beyond = [
                _integer(row, "iteration")
                for row in condition["metrics"]
                if _number(row, "cumulative_gpu_hours") > common_horizon
            ]
            selections[name] = {
                "selected_iteration": _integer(inside[-1], "iteration"),
                "selected_gpu_hours": _number(
                    inside[-1], "cumulative_gpu_hours"
                ),
                "excluded_beyond_horizon_iterations": beyond,
            }
        horizon_details["conditions"] = selections
    return {
        "schema_version": 1,
        "status": "passed" if audit.passed else "failed",
        "checks": audit.checks,
        "failed_checks": audit.failures,
        "common_horizon": horizon_details,
        "failure_policy": "not_assessable",
    }


def compare(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    _require(config.get("config_id") == "h2_v2", "unexpected H2 config_id")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_experiments_path(config["outputs"]["root"])
    )

    audit = Audit()
    paths, input_entries = _input_paths_and_hashes(config, audit)
    parsed = _parse_inputs(paths, audit)
    common_horizon: float | None = None
    endpoint_start: float | None = None
    grid: list[dict[str, Any]] = []
    if "matched_compute" in parsed:
        try:
            common_horizon, endpoint_start, grid = common_horizon_from_matched_compute(
                parsed["matched_compute"]
            )
            audit.record(
                "alignment.dynamic_common_horizon",
                True,
                {
                    "common_horizon_gpu_hours": common_horizon,
                    "endpoint_start_gpu_hours": endpoint_start,
                    "grid_target_count": len(grid),
                },
            )
        except H2Error as exc:
            audit.record("alignment.dynamic_common_horizon", False, str(exc))
    else:
        audit.record("alignment.dynamic_common_horizon", False, "matched compute did not parse")

    baseline = _build_condition("baseline", parsed, audit)
    adaptive = _build_condition("adaptive", parsed, audit)
    conditions = (
        {"baseline": baseline, "adaptive": adaptive}
        if baseline is not None and adaptive is not None
        else None
    )
    if conditions is not None and common_horizon is not None:
        for condition_name, condition in conditions.items():
            inside = [
                row
                for row in condition["metrics"]
                if _number(row, "cumulative_gpu_hours") <= common_horizon
            ]
            beyond = [
                row
                for row in condition["metrics"]
                if _number(row, "cumulative_gpu_hours") > common_horizon
            ]
            audit.record(
                f"alignment.{condition_name}.horizon_selection",
                bool(inside)
                and all(
                    _number(row, "cumulative_gpu_hours") > common_horizon
                    for row in beyond
                ),
                {
                    "selected_iteration": _integer(inside[-1], "iteration"),
                    "selected_gpu_hours": _number(
                        inside[-1], "cumulative_gpu_hours"
                    ),
                    "excluded_iterations": [
                        _integer(row, "iteration") for row in beyond
                    ],
                },
            )

    audit_status = "passed" if audit.passed else "failed"
    metric_specs = config.get("metrics")
    _require(isinstance(metric_specs, dict) and metric_specs, "H2 metric definitions are empty")
    minimum_points = int(
        config["metric_defaults"]["minimum_valid_points_per_condition"]
    )
    if audit_status == "passed":
        _require(conditions is not None, "passed audit lacks condition data")
        aligned = align_metrics(conditions, metric_specs, grid)
    else:
        aligned = []
    effects, metric_coverage = calculate_effects(
        conditions,
        metric_specs,
        common_horizon=common_horizon,
        endpoint_start=endpoint_start,
        minimum_points=minimum_points,
        audit_status=audit_status,
    )
    resources = _resource_rows(
        conditions,
        audit_status=audit_status,
        common_horizon=common_horizon,
        endpoint_start=endpoint_start,
    )
    decision_core = decision_from_effects(effects)
    audit_payload = _audit_payload(
        audit,
        common_horizon=common_horizon,
        endpoint_start=endpoint_start,
        grid=grid,
        conditions=conditions,
    )
    input_manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "h2_protocol": {
            "path": config_path.as_posix(),
            "sha256": _sha256_file(config_path),
        },
        "inputs": input_entries,
    }
    resolved = copy.deepcopy(config)
    resolved["runtime"] = {
        "config_path": config_path.as_posix(),
        "output_root": output_dir.as_posix(),
        "common_horizon_gpu_hours": common_horizon,
        "endpoint_start_gpu_hours": endpoint_start,
        "audit_status": audit_status,
    }
    holdout_evidence = _holdout_evidence(parsed)
    log_lines = [
        "h2_v2 formal comparison",
        f"input audit: {audit_status}",
        f"common horizon GPU-hours: {common_horizon}",
        f"endpoint start GPU-hours: {endpoint_start}",
        "alignment axis: GPU-hours",
        "interpolation: none",
        "extrapolation: none",
    ]

    _atomic_write_yaml(output_dir / "resolved_config.yaml", resolved)
    _atomic_write_json(output_dir / "input_manifest.json", input_manifest)
    _atomic_write_json(output_dir / "input_audit.json", audit_payload)
    _atomic_write_csv(output_dir / "aligned_metrics.csv", ALIGNED_FIELDS, aligned)
    _atomic_write_csv(output_dir / "effects.csv", EFFECT_FIELDS, effects)
    _atomic_write_csv(output_dir / "resource_report.csv", RESOURCE_FIELDS, resources)

    effects_path = output_dir / "effects.csv"
    decision = {
        "schema_version": 1,
        "hypothesis": "H2",
        **decision_core,
        "decision_scope": "full_interval",
        "effects_sha256": _sha256_file(effects_path),
        "recomputation_rule": {
            "input": "effects.csv only",
            "audit_failure": "any audit_status != passed => not_assessable",
            "primary": "the full_interval row with decision_role=primary",
            "supported": "primary assessment is improved and at least one full_interval complementary_replay row is improved",
            "partially_supported": "primary assessment is improved and no full_interval complementary_replay row is improved",
            "not_supported": "primary assessment is not_improved",
            "not_assessable": "input audit failed or primary assessment is unavailable",
        },
        "supplementary_holdout": holdout_evidence,
        "holdout_gap_changed_decision": False,
    }
    _atomic_write_json(output_dir / "decision.json", decision)

    summary = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "status": "completed",
        "input_audit_status": audit_status,
        "common_horizon_gpu_hours": common_horizon,
        "endpoint_start_gpu_hours": endpoint_start,
        "alignment": {
            "axis": "gpu_hours",
            "interpolation": "none",
            "extrapolation": "none",
        },
        "decision": decision_core,
        "metric_coverage": metric_coverage,
        "scope_exclusions": audit_payload["common_horizon"].get(
            "conditions", {}
        ),
        "supplementary_holdout": holdout_evidence,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    log_lines.extend(
        [
            f"effect rows: {len(effects)}",
            f"aligned metric rows: {len(aligned)}",
            f"decision: {decision_core['status']}",
            "status: completed",
        ]
    )
    _atomic_write_text(output_dir / "run.log", log_lines)

    print(f"Input audit: {audit_status}")
    print(f"Common horizon: {common_horizon} GPU-hours")
    if conditions is not None and common_horizon is not None:
        for name, condition in conditions.items():
            selected = [
                row
                for row in condition["metrics"]
                if _number(row, "cumulative_gpu_hours") <= common_horizon
            ][-1]
            print(
                f"{name.title()} horizon iteration: {_integer(selected, 'iteration')} "
                f"({_number(selected, 'cumulative_gpu_hours'):.12f} GPU-hours)"
            )
    print(f"H2 decision: {decision_core['status']}")
    print(f"Outputs: {output_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        compare(parse_args(argv))
    except (H2Error, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
