from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


DERIVED_FIELDS = (
    "fresh_states_per_update",
    "states_per_gpu_hour",
    "games_per_gpu_hour",
    "self_play_fraction",
    "training_fraction",
    "buffer_inflow_fraction",
    "buffer_fraction_consumed",
    "mean_sample_exposure",
    "selected_sample_reuse",
    "evicted_states",
    "turnover_fraction",
    "mean_sample_age",
    "median_sample_age",
    "p90_sample_age",
)

CSV_FIELDS = (
    "iteration",
    "positions_generated",
    "games_completed",
    "optimizer_steps",
    "replay_buffer_size",
    "reconstructed_buffer_size",
    "examples_used",
    "samples_seen",
    "iteration_seconds",
    "self_play_seconds",
    "training_seconds",
    *DERIVED_FIELDS,
)

REQUIRED_INTEGER_FIELDS = (
    "iteration",
    "positions_generated",
    "games_completed",
    "optimizer_steps",
    "replay_buffer_size",
    "available_examples",
    "examples_used",
    "samples_seen",
)

REQUIRED_FINITE_NONNEGATIVE_FIELDS = (
    "iteration_seconds",
    "self_play_seconds",
    "training_seconds",
    "peak_gpu_memory_mb",
)

FORMULAS = {
    "fresh_states_per_update": "positions_generated / optimizer_steps",
    "states_per_gpu_hour": "positions_generated / (iteration_seconds / 3600)",
    "games_per_gpu_hour": "games_completed / (iteration_seconds / 3600)",
    "self_play_fraction": "self_play_seconds / iteration_seconds",
    "training_fraction": "training_seconds / iteration_seconds",
    "buffer_inflow_fraction": "positions_generated / replay_buffer_size",
    "buffer_fraction_consumed": "examples_used / replay_buffer_size",
    "mean_sample_exposure": "samples_seen / replay_buffer_size",
    "selected_sample_reuse": "samples_seen / examples_used",
    "evicted_states": "positions_generated_(t-history_iterations) if t > history_iterations else 0",
    "turnover_fraction": "evicted_states / previous_replay_buffer_size; 0 when evicted_states is 0",
    "mean_sample_age": "sum(positions_generated_i * (t-i)) / replay_buffer_size for retained i",
    "median_sample_age": "weighted empirical 0.50 quantile of retained sample ages",
    "p90_sample_age": "weighted empirical 0.90 quantile of retained sample ages",
}


class DataQualityError(ValueError):
    """Raised when the baseline inputs or derived data fail validation."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive per-iteration baseline efficiency and replay metrics."
    )
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: <metrics parent>/gate2).",
    )
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


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in CSV_FIELDS})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataQualityError(f"resolved configuration not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DataQualityError(f"invalid resolved configuration: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DataQualityError("resolved configuration must contain a mapping")
    return loaded


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataQualityError(f"metrics file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataQualityError(
                    f"invalid JSON on metrics line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise DataQualityError(
                    f"metrics line {line_number} must contain a JSON object"
                )
            records.append(record)
    if not records:
        raise DataQualityError("metrics file contains no records")
    return records


def _configured_counts(config: dict[str, Any]) -> tuple[int, int, str]:
    try:
        expected_iterations = config["self_play"]["iterations"]
        history_iterations = config["replay"]["history_iterations"]
        run_id = config["run"]["id"]
    except (KeyError, TypeError) as exc:
        raise DataQualityError(
            "resolved configuration is missing run.id, self_play.iterations, "
            "or replay.history_iterations"
        ) from exc
    for name, value in (
        ("self_play.iterations", expected_iterations),
        ("replay.history_iterations", history_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DataQualityError(f"{name} must be an integer >= 1")
    if not isinstance(run_id, str) or not run_id.strip():
        raise DataQualityError("run.id must be a non-empty string")
    return expected_iterations, history_iterations, run_id


def _raw_record_errors(
    records: list[dict[str, Any]], expected_iterations: int
) -> dict[int, list[str]]:
    errors: dict[int, list[str]] = {}
    if len(records) != expected_iterations:
        errors.setdefault(0, []).append(
            f"expected {expected_iterations} records, found {len(records)}"
        )

    expected_sequence = list(range(1, expected_iterations + 1))
    observed_sequence = [record.get("iteration") for record in records]
    if observed_sequence != expected_sequence:
        errors.setdefault(0, []).append(
            "iteration sequence must be contiguous, unique, and ordered from 1 "
            f"through {expected_iterations}"
        )

    for row_number, record in enumerate(records, start=1):
        iteration_value = record.get("iteration")
        iteration = (
            iteration_value
            if isinstance(iteration_value, int) and not isinstance(iteration_value, bool)
            else row_number
        )
        row_errors = errors.setdefault(iteration, [])
        for field in REQUIRED_INTEGER_FIELDS:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                row_errors.append(f"{field} must be an integer")
            elif value <= 0:
                row_errors.append(f"{field} must be > 0")
        for field in REQUIRED_FINITE_NONNEGATIVE_FIELDS:
            value = record.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                row_errors.append(f"{field} must be a finite number >= 0")
        iteration_seconds = record.get("iteration_seconds")
        if isinstance(iteration_seconds, (int, float)) and not isinstance(
            iteration_seconds, bool
        ):
            if iteration_seconds <= 0:
                row_errors.append("iteration_seconds must be > 0")
            elif all(
                isinstance(record.get(field), (int, float))
                and not isinstance(record.get(field), bool)
                for field in ("self_play_seconds", "training_seconds")
            ) and (
                record["self_play_seconds"] + record["training_seconds"]
                > iteration_seconds + 1e-9
            ):
                row_errors.append(
                    "self_play_seconds + training_seconds exceeds iteration_seconds"
                )
        replay_size = record.get("replay_buffer_size")
        available = record.get("available_examples")
        used = record.get("examples_used")
        if isinstance(replay_size, int) and isinstance(available, int):
            if replay_size != available:
                row_errors.append("available_examples differs from replay_buffer_size")
        if isinstance(replay_size, int) and isinstance(used, int) and used > replay_size:
            row_errors.append("examples_used exceeds replay_buffer_size")
        if not row_errors:
            errors.pop(iteration, None)
    return errors


def _weighted_quantile(age_weights: list[tuple[int, int]], quantile: float) -> int:
    total_weight = sum(weight for _, weight in age_weights)
    threshold = quantile * total_weight
    cumulative = 0
    for age, weight in sorted(age_weights):
        cumulative += weight
        if cumulative >= threshold:
            return age
    return max(age for age, _ in age_weights)


def _derive_rows(
    records: list[dict[str, Any]], history_iterations: int
) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    derived: list[dict[str, Any]] = []
    quality_errors: dict[int, list[str]] = {}

    for index, record in enumerate(records):
        iteration = int(record["iteration"])
        window_start = max(0, index - history_iterations + 1)
        retained = records[window_start : index + 1]
        reconstructed = sum(int(item["positions_generated"]) for item in retained)
        recorded_buffer = int(record["replay_buffer_size"])
        if reconstructed != recorded_buffer:
            quality_errors.setdefault(iteration, []).append(
                "reconstructed replay buffer mismatch: "
                f"expected {reconstructed}, recorded {recorded_buffer}"
            )

        age_weights = [
            (
                iteration - int(item["iteration"]),
                int(item["positions_generated"]),
            )
            for item in retained
        ]
        mean_age = (
            sum(age * weight for age, weight in age_weights) / reconstructed
        )
        evicted = (
            int(records[index - history_iterations]["positions_generated"])
            if index >= history_iterations
            else 0
        )
        previous_buffer = (
            int(records[index - 1]["replay_buffer_size"]) if index > 0 else 0
        )
        iteration_hours = float(record["iteration_seconds"]) / 3600.0

        row = {
            "iteration": iteration,
            "positions_generated": int(record["positions_generated"]),
            "games_completed": int(record["games_completed"]),
            "optimizer_steps": int(record["optimizer_steps"]),
            "replay_buffer_size": recorded_buffer,
            "reconstructed_buffer_size": reconstructed,
            "examples_used": int(record["examples_used"]),
            "samples_seen": int(record["samples_seen"]),
            "iteration_seconds": float(record["iteration_seconds"]),
            "self_play_seconds": float(record["self_play_seconds"]),
            "training_seconds": float(record["training_seconds"]),
            "fresh_states_per_update": (
                record["positions_generated"] / record["optimizer_steps"]
            ),
            "states_per_gpu_hour": record["positions_generated"] / iteration_hours,
            "games_per_gpu_hour": record["games_completed"] / iteration_hours,
            "self_play_fraction": (
                record["self_play_seconds"] / record["iteration_seconds"]
            ),
            "training_fraction": (
                record["training_seconds"] / record["iteration_seconds"]
            ),
            "buffer_inflow_fraction": record["positions_generated"] / recorded_buffer,
            "buffer_fraction_consumed": record["examples_used"] / recorded_buffer,
            "mean_sample_exposure": record["samples_seen"] / recorded_buffer,
            "selected_sample_reuse": record["samples_seen"] / record["examples_used"],
            "evicted_states": evicted,
            "turnover_fraction": evicted / previous_buffer if evicted else 0.0,
            "mean_sample_age": mean_age,
            "median_sample_age": _weighted_quantile(age_weights, 0.50),
            "p90_sample_age": _weighted_quantile(age_weights, 0.90),
        }
        for field in DERIVED_FIELDS:
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                quality_errors.setdefault(iteration, []).append(
                    f"derived field {field} is missing or non-finite"
                )
        derived.append(row)
    return derived, quality_errors


def _quality_report(
    *,
    metrics_path: Path,
    config_path: Path,
    run_id: str,
    expected_iterations: int,
    history_iterations: int,
    records: list[dict[str, Any]],
    errors: dict[int, list[str]],
) -> dict[str, Any]:
    failed_iterations = sorted(iteration for iteration in errors if iteration != 0)
    global_errors = errors.get(0, [])
    observed_iterations = len(records)
    passed_iterations = max(0, observed_iterations - len(failed_iterations))
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "run_id": run_id,
        "inputs": {
            "metrics_jsonl": metrics_path.as_posix(),
            "metrics_sha256": _sha256(metrics_path),
            "resolved_config_yaml": config_path.as_posix(),
            "resolved_config_sha256": _sha256(config_path),
        },
        "expected_iterations": expected_iterations,
        "observed_iterations": observed_iterations,
        "iterations_passed": passed_iterations,
        "iterations_failed": len(failed_iterations),
        "history_iterations": history_iterations,
        "required_derived_fields": list(DERIVED_FIELDS),
        "checks": {
            "complete_contiguous_iteration_sequence": not any(
                "iteration sequence" in message or "expected" in message
                for message in global_errors
            ),
            "required_raw_fields_present_finite_and_valid": not any(
                not (
                    message.startswith("derived field")
                    or message.startswith("reconstructed replay buffer mismatch")
                )
                for messages in errors.values()
                for message in messages
            ),
            "replay_buffer_exactly_reconstructed_from_positions": not any(
                "reconstructed replay buffer mismatch" in message
                for messages in errors.values()
                for message in messages
            ),
            "derived_metrics_complete_and_finite": not any(
                "derived field" in message
                for messages in errors.values()
                for message in messages
            ),
        },
        "global_errors": global_errors,
        "failed_iterations": [
            {"iteration": iteration, "errors": errors[iteration]}
            for iteration in failed_iterations
        ],
    }


def _resource_summary(
    records: list[dict[str, Any]],
    derived: list[dict[str, Any]],
    run_id: str,
    history_iterations: int,
) -> dict[str, Any]:
    total_seconds = sum(float(record["iteration_seconds"]) for record in records)
    total_self_play_seconds = sum(
        float(record["self_play_seconds"]) for record in records
    )
    total_training_seconds = sum(
        float(record["training_seconds"]) for record in records
    )
    total_positions = sum(int(record["positions_generated"]) for record in records)
    total_games = sum(int(record["games_completed"]) for record in records)
    total_hours = total_seconds / 3600.0
    final = derived[-1]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "iterations": len(records),
        "history_iterations": history_iterations,
        "totals": {
            "positions_generated": total_positions,
            "games_completed": total_games,
            "optimizer_steps": sum(
                int(record["optimizer_steps"]) for record in records
            ),
            "samples_seen": sum(int(record["samples_seen"]) for record in records),
            "iteration_seconds": total_seconds,
            "gpu_hours": total_hours,
            "self_play_seconds": total_self_play_seconds,
            "training_seconds": total_training_seconds,
        },
        "aggregate_rates": {
            "states_per_gpu_hour": total_positions / total_hours,
            "games_per_gpu_hour": total_games / total_hours,
            "self_play_fraction": total_self_play_seconds / total_seconds,
            "training_fraction": total_training_seconds / total_seconds,
            "unattributed_iteration_fraction": (
                (total_seconds - total_self_play_seconds - total_training_seconds)
                / total_seconds
            ),
        },
        "replay": {
            "final_buffer_size": final["replay_buffer_size"],
            "final_mean_sample_age": final["mean_sample_age"],
            "final_median_sample_age": final["median_sample_age"],
            "final_p90_sample_age": final["p90_sample_age"],
            "total_evicted_states": sum(row["evicted_states"] for row in derived),
        },
        "resources": {
            "peak_gpu_memory_mb": max(
                float(record["peak_gpu_memory_mb"])
                for record in records
                if isinstance(record.get("peak_gpu_memory_mb"), (int, float))
                and math.isfinite(float(record["peak_gpu_memory_mb"]))
            ),
        },
        "formulas": FORMULAS,
    }


def derive_baseline_metrics(
    metrics_path: Path,
    resolved_config_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    metrics_path = metrics_path.expanduser().resolve()
    resolved_config_path = resolved_config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    quality_path = output_dir / "data_quality_report.json"

    config = _load_config(resolved_config_path)
    records = _load_metrics(metrics_path)
    expected_iterations, history_iterations, run_id = _configured_counts(config)
    raw_errors = _raw_record_errors(records, expected_iterations)
    if raw_errors:
        report = _quality_report(
            metrics_path=metrics_path,
            config_path=resolved_config_path,
            run_id=run_id,
            expected_iterations=expected_iterations,
            history_iterations=history_iterations,
            records=records,
            errors=raw_errors,
        )
        _atomic_write_json(quality_path, report)
        raise DataQualityError(
            f"raw metric validation failed; see {quality_path}"
        )

    derived, derived_errors = _derive_rows(records, history_iterations)
    report = _quality_report(
        metrics_path=metrics_path,
        config_path=resolved_config_path,
        run_id=run_id,
        expected_iterations=expected_iterations,
        history_iterations=history_iterations,
        records=records,
        errors=derived_errors,
    )
    _atomic_write_json(quality_path, report)
    if derived_errors:
        raise DataQualityError(
            f"derived metric validation failed; see {quality_path}"
        )

    csv_path = output_dir / "derived_metrics.csv"
    summary_path = output_dir / "baseline_resource_summary.json"
    _atomic_write_csv(csv_path, derived)
    _atomic_write_json(
        summary_path,
        _resource_summary(records, derived, run_id, history_iterations),
    )
    return csv_path, summary_path, quality_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.metrics.expanduser().resolve().parent / "gate2"
    try:
        csv_path, summary_path, quality_path = derive_baseline_metrics(
            args.metrics,
            args.resolved_config,
            output_dir,
        )
    except (DataQualityError, OSError, ValueError) as exc:
        print(f"Baseline metric derivation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Derived metrics: {csv_path}")
    print(f"Resource summary: {summary_path}")
    print(f"Data quality: {quality_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
