#!/usr/bin/env python3
"""Derive H2-ready Adaptive metrics without evaluating the H2 decision."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.Adaptive.experiment_runtime import sha256_file  # noqa: E402
from experiments.Adaptive.replay_instrumentation import (  # noqa: E402
    ReplayInstrumentationConfig,
    ReplayInstrumentationError,
    summarize_replay_snapshot,
)
from experiments.scripts.verify_adaptive import (  # noqa: E402
    AdaptiveVerificationError,
    VerifiedRun,
    verify_run,
)


DEFAULT_RUN_DIR = (
    EXPERIMENTS_ROOT / "outputs" / "adaptive_formal_seed1001_4090_v2"
)
DEFAULT_OUTPUT_DIR = (
    EXPERIMENTS_ROOT / "outputs" / "adaptive_seed1001_4090_v2_analysis"
)
DEFAULT_MATCHED_COMPUTE = EXPERIMENTS_ROOT / "configs" / "matched_compute_v1.yaml"

DERIVED_FIELDS = (
    "fresh_states_per_update",
    "states_per_gpu_hour",
    "mean_sample_exposure",
    "selected_sample_reuse",
    "evicted_states",
    "turnover_fraction",
    "mean_sample_age",
    "median_sample_age",
    "p90_sample_age",
)

DERIVED_CSV_FIELDS = (
    "iteration",
    "cumulative_gpu_hours",
    "beyond_h2_common_horizon",
    "positions_generated",
    "games_completed",
    "optimizer_steps",
    "replay_buffer_size",
    "reconstructed_buffer_size",
    "examples_used",
    "samples_seen",
    "iteration_seconds",
    "iteration_gpu_hours",
    *DERIVED_FIELDS,
)

REPLAY_ITERATION_FIELDS = (
    "iteration",
    "beyond_h2_common_horizon",
    "states",
    "unique_canonical_states",
    "unique_canonical_state_ratio",
    "incoming_unique_states",
    "incoming_unique_state_ratio",
    "incoming_ratio_left_censored",
    "duplicate_hash_occurrences",
    "duplicate_hash_groups",
    "duplicate_rate",
    "state_effective_count",
    "metrics_positions_generated",
    "count_matches_metrics",
)

REPLAY_TRAJECTORY_FIELDS = (
    "iteration",
    "beyond_h2_common_horizon",
    "expected_games",
    "recovered_games",
    "empty_games",
    "anomalous_games",
    "min_game_length",
    "max_game_length",
    "mean_game_length",
    "median_game_length",
    "p90_game_length",
    "total_recovered_positions",
    "metrics_games_completed",
    "metrics_min_game_length",
    "metrics_max_game_length",
    "metrics_mean_game_length",
    "games_match_metrics",
    "length_distribution_matches_metrics",
)

FORMULAS = {
    "fresh_states_per_update": "positions_generated / optimizer_steps",
    "states_per_gpu_hour": (
        "positions_generated / (iteration_seconds * allocated_gpu_count / 3600)"
    ),
    "mean_sample_exposure": "samples_seen / replay_buffer_size",
    "selected_sample_reuse": "samples_seen / examples_used",
    "evicted_states": (
        "positions_generated_(t-history_iterations) when t > history_iterations; "
        "otherwise 0"
    ),
    "turnover_fraction": (
        "evicted_states / previous_replay_buffer_size; 0 before first eviction"
    ),
    "mean_sample_age": (
        "sum(positions_generated_i * (current_iteration - i)) / "
        "replay_buffer_size over retained iterations"
    ),
    "median_sample_age": "weighted empirical 0.50 quantile of retained ages",
    "p90_sample_age": "weighted empirical 0.90 quantile of retained ages",
    "unique_canonical_state_ratio": "unique_canonical_states / states",
    "incoming_unique_state_ratio": "incoming_unique_states / states",
    "duplicate_rate": "duplicate_hash_occurrences / states",
}


class AdaptiveMetricDerivationError(ValueError):
    """Raised when verified Adaptive outputs cannot be safely derived."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--matched-compute",
        type=Path,
        default=DEFAULT_MATCHED_COMPUTE,
        help="Protocol containing the Baseline GPU-hour checkpoint targets.",
    )
    return parser.parse_args(argv)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveMetricDerivationError(message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
                writer.writerow({field: row[field] for field in fields})
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveMetricDerivationError(f"invalid {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AdaptiveMetricDerivationError(f"invalid {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must contain a mapping")
    return payload


def _common_horizon_target(matched_compute_path: Path) -> float:
    protocol = _load_yaml(matched_compute_path, "matched-compute protocol")
    try:
        targets = protocol["pairing_and_randomness"]["checkpoint_grid"]["targets"]
        maximum_gpu_hours = protocol["compute_budget"]["common_horizon"][
            "maximum_gpu_hours"
        ]
    except (KeyError, TypeError) as exc:
        raise AdaptiveMetricDerivationError(
            "matched-compute protocol is missing checkpoint targets or the "
            "common-horizon maximum"
        ) from exc
    _require(isinstance(targets, list) and bool(targets), "checkpoint targets are empty")
    gpu_hour_targets: list[float] = []
    for index, target in enumerate(targets):
        _require(isinstance(target, dict), f"checkpoint target {index} is invalid")
        value = target.get("gpu_hours")
        _require(
            _is_finite_number(value) and float(value) >= 0.0,
            f"checkpoint target {index} has invalid gpu_hours",
        )
        gpu_hour_targets.append(float(value))
    _require(
        _is_finite_number(maximum_gpu_hours) and float(maximum_gpu_hours) > 0.0,
        "common-horizon maximum_gpu_hours must be finite and > 0",
    )
    return min(max(gpu_hour_targets), float(maximum_gpu_hours))


def _select_horizon_iteration(
    records: list[dict[str, Any]], common_horizon_gpu_hours: float
) -> int:
    _require(
        _is_finite_number(common_horizon_gpu_hours)
        and common_horizon_gpu_hours > 0.0,
        "common horizon must be finite and > 0 GPU-hours",
    )
    eligible = [
        int(record["iteration"])
        for record in records
        if _is_integer(record.get("iteration"))
        and _is_finite_number(record.get("cumulative_gpu_hours"))
        and float(record["cumulative_gpu_hours"]) <= common_horizon_gpu_hours
    ]
    _require(bool(eligible), "no completed Adaptive iteration is within the horizon")
    return max(eligible)


def _weighted_quantile(age_weights: list[tuple[int, int]], quantile: float) -> int:
    total_weight = sum(weight for _, weight in age_weights)
    _require(total_weight > 0, "cannot calculate sample age from an empty replay")
    threshold = quantile * total_weight
    cumulative = 0
    for age, weight in sorted(age_weights):
        cumulative += weight
        if cumulative >= threshold:
            return age
    return max(age for age, _ in age_weights)


def _validate_metric_fields(records: list[dict[str, Any]]) -> None:
    _require(bool(records), "metrics.jsonl contains no records")
    observed = [record.get("iteration") for record in records]
    _require(
        observed == list(range(1, len(records) + 1)),
        "metrics iterations must be contiguous, unique, and ordered from 1",
    )
    integer_minima = {
        "positions_generated": 1,
        "games_completed": 1,
        "optimizer_steps": 1,
        "replay_buffer_size": 1,
        "available_examples": 1,
        "examples_used": 1,
        "samples_seen": 1,
    }
    for record in records:
        iteration = int(record["iteration"])
        for field, minimum in integer_minima.items():
            value = record.get(field)
            _require(
                _is_integer(value) and int(value) >= minimum,
                f"metrics iteration {iteration} field {field} must be an integer "
                f">= {minimum}",
            )
        for field in ("iteration_seconds", "cumulative_gpu_hours"):
            value = record.get(field)
            _require(
                _is_finite_number(value) and float(value) > 0.0,
                f"metrics iteration {iteration} field {field} must be finite and > 0",
            )
        _require(
            record["available_examples"] == record["replay_buffer_size"],
            f"metrics iteration {iteration} available_examples differs from replay size",
        )
        _require(
            record["examples_used"] <= record["replay_buffer_size"],
            f"metrics iteration {iteration} examples_used exceeds replay size",
        )


def _derive_metric_rows(
    records: list[dict[str, Any]],
    *,
    history_iterations: int,
    allocated_gpu_count: int,
    h2_common_horizon_iteration: int,
) -> list[dict[str, Any]]:
    _validate_metric_fields(records)
    _require(history_iterations >= 1, "history_iterations must be >= 1")
    _require(allocated_gpu_count >= 1, "allocated_gpu_count must be >= 1")
    _require(
        1 <= h2_common_horizon_iteration <= len(records),
        "H2 common-horizon iteration must be within the completed run",
    )
    derived: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        iteration = int(record["iteration"])
        retained_start = max(0, index - history_iterations + 1)
        retained = records[retained_start : index + 1]
        reconstructed_buffer = sum(
            int(item["positions_generated"]) for item in retained
        )
        replay_buffer_size = int(record["replay_buffer_size"])
        _require(
            reconstructed_buffer == replay_buffer_size,
            f"reconstructed replay buffer mismatch at iteration {iteration}: "
            f"reconstructed {reconstructed_buffer}, recorded {replay_buffer_size}",
        )
        age_weights = [
            (
                iteration - int(item["iteration"]),
                int(item["positions_generated"]),
            )
            for item in retained
        ]
        mean_age = (
            sum(age * weight for age, weight in age_weights) / reconstructed_buffer
        )
        evicted_states = (
            int(records[index - history_iterations]["positions_generated"])
            if index >= history_iterations
            else 0
        )
        previous_buffer = (
            int(records[index - 1]["replay_buffer_size"]) if index > 0 else 0
        )
        iteration_gpu_hours = (
            float(record["iteration_seconds"]) * allocated_gpu_count / 3600.0
        )
        row = {
            "iteration": iteration,
            "cumulative_gpu_hours": float(record["cumulative_gpu_hours"]),
            "beyond_h2_common_horizon": (
                iteration > h2_common_horizon_iteration
            ),
            "positions_generated": int(record["positions_generated"]),
            "games_completed": int(record["games_completed"]),
            "optimizer_steps": int(record["optimizer_steps"]),
            "replay_buffer_size": replay_buffer_size,
            "reconstructed_buffer_size": reconstructed_buffer,
            "examples_used": int(record["examples_used"]),
            "samples_seen": int(record["samples_seen"]),
            "iteration_seconds": float(record["iteration_seconds"]),
            "iteration_gpu_hours": iteration_gpu_hours,
            "fresh_states_per_update": (
                int(record["positions_generated"]) / int(record["optimizer_steps"])
            ),
            "states_per_gpu_hour": (
                int(record["positions_generated"]) / iteration_gpu_hours
            ),
            "mean_sample_exposure": (
                int(record["samples_seen"]) / replay_buffer_size
            ),
            "selected_sample_reuse": (
                int(record["samples_seen"]) / int(record["examples_used"])
            ),
            "evicted_states": evicted_states,
            "turnover_fraction": (
                evicted_states / previous_buffer if evicted_states else 0.0
            ),
            "mean_sample_age": mean_age,
            "median_sample_age": _weighted_quantile(age_weights, 0.50),
            "p90_sample_age": _weighted_quantile(age_weights, 0.90),
        }
        for field in DERIVED_FIELDS:
            _require(
                _is_finite_number(row[field]),
                f"derived field {field} is non-finite at iteration {iteration}",
            )
        derived.append(row)
    return derived


def _load_replay_snapshot(path: Path) -> tuple[int, list[Any]]:
    _require(path.is_file(), f"missing final replay snapshot: {path}")
    try:
        with path.open("rb") as source:
            payload = pickle.load(source)
    except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
        raise AdaptiveMetricDerivationError(
            f"cannot load final replay snapshot: {exc}"
        ) from exc
    _require(isinstance(payload, dict), "replay snapshot must contain a mapping")
    replay_iteration = payload.get("iteration")
    buckets = payload.get("examples")
    _require(
        _is_integer(replay_iteration) and int(replay_iteration) >= 1,
        "replay snapshot iteration is invalid",
    )
    _require(
        isinstance(buckets, (list, tuple)),
        "replay snapshot examples must be a sequence",
    )
    return int(replay_iteration), list(buckets)


def _annotate_replay_rows(
    rows: list[dict[str, Any]],
    *,
    h2_common_horizon_iteration: int,
    include_unique_ratio: bool,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        iteration = int(row["iteration"])
        row["beyond_h2_common_horizon"] = (
            iteration > h2_common_horizon_iteration
        )
        if include_unique_ratio:
            states = int(row["states"])
            row["unique_canonical_state_ratio"] = (
                int(row["unique_canonical_states"]) / states if states else 0.0
            )
        annotated.append(row)
    return annotated


def _input_evidence(
    verified: VerifiedRun, matched_compute_path: Path
) -> dict[str, dict[str, Any]]:
    run_dir = verified.run_dir
    latest_path = run_dir / "recovery" / "latest_commit.json"
    latest = _load_json(latest_path, "latest recovery pointer")
    iteration = int(latest["iteration"])
    commit_path = (
        run_dir / "recovery" / f"iteration_{iteration:06d}" / "commit_manifest.json"
    )
    commit = _load_json(commit_path, "recovery commit manifest")
    replay_entry = commit.get("artifacts", {}).get("replay", {})
    replay_path = verified.recovery_artifacts["replay"]
    return {
        "resolved_config": {
            "path": (run_dir / "resolved_config.yaml").as_posix(),
            "sha256": sha256_file(run_dir / "resolved_config.yaml"),
            "verification": "matched input_manifest.resolved_config",
        },
        "input_manifest": {
            "path": (run_dir / "input_manifest.json").as_posix(),
            "sha256": sha256_file(run_dir / "input_manifest.json"),
            "verification": "parsed; source-protocol hashes intentionally not enforced",
        },
        "metrics": {
            "path": (run_dir / "metrics.jsonl").as_posix(),
            "sha256": sha256_file(run_dir / "metrics.jsonl"),
            "verification": "contiguous and final record matched recovery commit",
        },
        "checkpoint_manifest": {
            "path": (run_dir / "checkpoint_manifest.json").as_posix(),
            "sha256": sha256_file(run_dir / "checkpoint_manifest.json"),
            "verification": "matched verified recovery snapshot and checkpoint files",
        },
        "latest_commit": {
            "path": latest_path.as_posix(),
            "sha256": sha256_file(latest_path),
            "commit_manifest_path": commit_path.as_posix(),
            "commit_manifest_sha256": str(latest["commit_manifest_sha256"]),
            "verification": "commit manifest hash matched latest recovery pointer",
        },
        "replay_snapshot": {
            "path": replay_path.as_posix(),
            "sha256": str(replay_entry.get("sha256", "")),
            "verification": "matched recovery commit manifest",
        },
        "matched_compute": {
            "path": matched_compute_path.as_posix(),
            "sha256": sha256_file(matched_compute_path),
            "verification": "parsed as the common-horizon target source",
        },
    }


def _build_replay_summary(
    *,
    raw: dict[str, Any],
    run_id: str,
    input_evidence: dict[str, dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    h2_common_horizon_iteration: int,
    h2_common_horizon_gpu_hours: float,
    output_dir: Path,
) -> dict[str, Any]:
    left_censored = [
        int(row["iteration"])
        for row in iteration_rows
        if row["incoming_ratio_left_censored"]
    ]
    return {
        "schema_version": 1,
        "status": raw["status"],
        "run_id": run_id,
        "inputs": input_evidence,
        "replay": {
            "iteration": raw["replay_iteration"],
            "history_buckets": raw["history_buckets"],
            "recovered_iteration_start": raw["recovered_start_iteration"],
            "recovered_iteration_end": raw["recovered_end_iteration"],
            "total_states": raw["total_states"],
            "empty_buckets": raw["empty_buckets"],
        },
        "h2_scope": {
            "common_horizon_gpu_hours": h2_common_horizon_gpu_hours,
            "common_horizon_iteration": h2_common_horizon_iteration,
            "selection_rule": (
                "max(iteration where cumulative_gpu_hours <= common_horizon_gpu_hours)"
            ),
            "beyond_common_horizon_rule": (
                "iteration > common_horizon_iteration"
            ),
            "left_censored_incoming_ratio_iterations": left_censored,
            "h2_decision_performed": False,
        },
        "state_diversity": {
            "unique_canonical_states": raw["final_unique_states"],
            "final_buffer_unique_ratio": raw["final_unique_state_ratio"],
            "incoming_unique_states_across_retained_window": sum(
                int(row["incoming_unique_states"]) for row in iteration_rows
            ),
            "duplicate_hash_groups": raw["duplicate_hash_groups"],
            "duplicate_hash_occurrences": raw["duplicate_hash_occurrences"],
            "duplicate_rate": raw["duplicate_rate"],
        },
        "trajectories": {
            "recovered_games": sum(
                int(row["recovered_games"]) for row in trajectory_rows
            ),
            "empty_games": raw["empty_games"],
            "anomalous_games": raw["anomalous_games"],
        },
        "anomaly_counts": raw["anomaly_count_by_type"],
        "validations": raw["validations"],
        "definitions": {
            "canonical_state_hash": (
                "SHA256(dtype + shape + contiguous canonical-board bytes)"
            ),
            "unique_canonical_state_ratio": FORMULAS[
                "unique_canonical_state_ratio"
            ],
            "incoming_unique_state_ratio": FORMULAS[
                "incoming_unique_state_ratio"
            ],
            "duplicate_rate": FORMULAS["duplicate_rate"],
            "incoming_novelty_scope": (
                "not seen in earlier buckets retained in the final replay snapshot"
            ),
        },
        "limitations": raw["limitations"],
        "outputs": {
            "replay_iteration_stats": (
                output_dir / "replay_iteration_stats.csv"
            ).as_posix(),
            "replay_trajectory_stats": (
                output_dir / "replay_trajectory_stats.csv"
            ).as_posix(),
            "replay_final_summary": (
                output_dir / "replay_final_summary.json"
            ).as_posix(),
            "raw_states_exported": False,
            "state_hashes_exported": False,
        },
    }


def _quality_report(
    *,
    verified: VerifiedRun,
    input_evidence: dict[str, dict[str, Any]],
    derived_rows: list[dict[str, Any]],
    replay_summary: dict[str, Any],
    replay_iteration_rows: list[dict[str, Any]],
    replay_trajectory_rows: list[dict[str, Any]],
    h2_common_horizon_iteration: int,
    h2_common_horizon_gpu_hours: float,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    completed_iterations = len(verified.metrics)
    history_iterations = int(
        verified.resolved["replay_instrumentation"]["history_iterations"]
    )
    expected_replay_start = completed_iterations - history_iterations + 1
    left_censored = [
        int(row["iteration"])
        for row in replay_iteration_rows
        if row["incoming_ratio_left_censored"]
    ]
    checks = {
        "input_hashes_and_adaptive_run_verified": (
            verified.report.get("status") == "verified"
        ),
        "complete_contiguous_iteration_sequence": (
            [row["iteration"] for row in derived_rows]
            == list(range(1, completed_iterations + 1))
        ),
        "derived_metrics_complete_and_finite": all(
            _is_finite_number(row[field])
            for row in derived_rows
            for field in DERIVED_FIELDS
        ),
        "replay_snapshot_contains_final_150_iterations": (
            history_iterations == 150
            and replay_summary["replay"]["history_buckets"] == 150
            and replay_summary["replay"]["recovered_iteration_start"]
            == expected_replay_start
            and replay_summary["replay"]["recovered_iteration_end"]
            == completed_iterations
        ),
        "replay_counts_match_metrics": bool(
            replay_summary["validations"]["counts_match_metrics"]
        ),
        "replay_trajectories_match_metrics": bool(
            replay_summary["validations"]["trajectories_match_metrics"]
        ),
        "replay_has_no_empty_buckets_or_trajectory_anomalies": (
            bool(replay_summary["validations"]["no_empty_buckets"])
            and bool(replay_summary["validations"]["no_empty_games"])
            and bool(replay_summary["validations"]["no_trajectory_anomalies"])
        ),
        "incoming_ratio_left_censored_only_at_replay_window_start": (
            left_censored == [expected_replay_start]
        ),
        "iterations_after_selected_horizon_marked_beyond_h2_common_horizon": (
            all(
                row["beyond_h2_common_horizon"]
                == (int(row["iteration"]) > h2_common_horizon_iteration)
                for row in (
                    derived_rows + replay_iteration_rows + replay_trajectory_rows
                )
            )
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    findings = [
        {
            "severity": "high",
            "check": name,
            "evidence": "quality check returned false",
            "impact": "affected rows are not safe for downstream H2 analysis",
        }
        for name in failed_checks
    ]
    return {
        "schema_version": 1,
        "status": "passed" if not failed_checks else "failed",
        "run_id": verified.resolved["run"]["id"],
        "dataset_grain": {
            "derived_metrics": "one row per completed training iteration",
            "replay_iteration_stats": "one row per retained replay iteration",
            "replay_trajectory_stats": "one row per retained replay iteration",
        },
        "input_integrity": {
            "inputs": input_evidence,
            "source_protocol_hashes_enforced": False,
            "source_protocol_hash_scope_note": (
                "current downstream protocol hashes are intentionally excluded; "
                "run outputs, checkpoints, recovery artifacts, and direct inputs "
                "remain verified"
            ),
        },
        "row_counts": {
            "completed_iterations": completed_iterations,
            "derived_metric_rows": len(derived_rows),
            "replay_iteration_rows": len(replay_iteration_rows),
            "replay_trajectory_rows": len(replay_trajectory_rows),
        },
        "h2_scope": {
            "common_horizon_gpu_hours": h2_common_horizon_gpu_hours,
            "common_horizon_iteration": h2_common_horizon_iteration,
            "selection_rule": (
                "max(iteration where cumulative_gpu_hours <= common_horizon_gpu_hours)"
            ),
            "left_censored_incoming_ratio_iterations": left_censored,
            "beyond_common_horizon_iterations": [
                row["iteration"]
                for row in derived_rows
                if row["beyond_h2_common_horizon"]
            ],
            "h2_decision_performed": False,
        },
        "checks": checks,
        "findings": findings,
        "formulas": FORMULAS,
        "limitations": replay_summary["limitations"],
        "outputs": {
            label: {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
            }
            for label, path in output_paths.items()
        },
    }


def derive_adaptive_metrics(
    run_dir: Path,
    output_dir: Path,
    *,
    matched_compute_path: Path = DEFAULT_MATCHED_COMPUTE,
) -> tuple[dict[str, Path], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    matched_compute_path = matched_compute_path.expanduser().resolve()
    verified = verify_run(run_dir)
    _require(verified.completed, "Adaptive run is verified but not completed")
    resolved = verified.resolved
    history_iterations = int(
        resolved["replay_instrumentation"]["history_iterations"]
    )
    allocated_gpu_count = int(
        resolved["resource_accounting"]["allocated_gpu_count"]
    )
    h2_common_horizon_gpu_hours = _common_horizon_target(matched_compute_path)
    h2_common_horizon_iteration = _select_horizon_iteration(
        verified.metrics, h2_common_horizon_gpu_hours
    )
    derived_rows = _derive_metric_rows(
        verified.metrics,
        history_iterations=history_iterations,
        allocated_gpu_count=allocated_gpu_count,
        h2_common_horizon_iteration=h2_common_horizon_iteration,
    )
    input_evidence = _input_evidence(verified, matched_compute_path)

    replay_path = verified.recovery_artifacts["replay"]
    gc.collect()
    replay_iteration, replay_buckets = _load_replay_snapshot(replay_path)
    _require(
        replay_iteration == len(verified.metrics),
        "final replay iteration differs from completed metrics",
    )
    _require(
        len(replay_buckets) == history_iterations == 150,
        "final replay snapshot must contain exactly 150 iteration buckets",
    )
    instrumentation_config = ReplayInstrumentationConfig(
        **resolved["replay_instrumentation"]
    )
    metrics_by_iteration = {
        int(record["iteration"]): record for record in verified.metrics
    }
    raw_replay = summarize_replay_snapshot(
        replay_iteration,
        replay_buckets,
        metrics_by_iteration,
        instrumentation_config,
    )
    replay_iteration_rows = _annotate_replay_rows(
        raw_replay["iteration_rows"],
        h2_common_horizon_iteration=h2_common_horizon_iteration,
        include_unique_ratio=True,
    )
    replay_trajectory_rows = _annotate_replay_rows(
        raw_replay["trajectory_rows"],
        h2_common_horizon_iteration=h2_common_horizon_iteration,
        include_unique_ratio=False,
    )
    replay_output_dir = output_dir / "replay"
    replay_summary = _build_replay_summary(
        raw=raw_replay,
        run_id=str(resolved["run"]["id"]),
        input_evidence=input_evidence,
        iteration_rows=replay_iteration_rows,
        trajectory_rows=replay_trajectory_rows,
        h2_common_horizon_iteration=h2_common_horizon_iteration,
        h2_common_horizon_gpu_hours=h2_common_horizon_gpu_hours,
        output_dir=replay_output_dir,
    )

    output_paths = {
        "derived_metrics": output_dir / "derived_metrics.csv",
        "replay_iteration_stats": replay_output_dir
        / "replay_iteration_stats.csv",
        "replay_trajectory_stats": replay_output_dir
        / "replay_trajectory_stats.csv",
        "replay_final_summary": replay_output_dir / "replay_final_summary.json",
    }
    _atomic_write_csv(
        output_paths["derived_metrics"], DERIVED_CSV_FIELDS, derived_rows
    )
    _atomic_write_csv(
        output_paths["replay_iteration_stats"],
        REPLAY_ITERATION_FIELDS,
        replay_iteration_rows,
    )
    _atomic_write_csv(
        output_paths["replay_trajectory_stats"],
        REPLAY_TRAJECTORY_FIELDS,
        replay_trajectory_rows,
    )
    _atomic_write_json(output_paths["replay_final_summary"], replay_summary)

    quality = _quality_report(
        verified=verified,
        input_evidence=input_evidence,
        derived_rows=derived_rows,
        replay_summary=replay_summary,
        replay_iteration_rows=replay_iteration_rows,
        replay_trajectory_rows=replay_trajectory_rows,
        h2_common_horizon_iteration=h2_common_horizon_iteration,
        h2_common_horizon_gpu_hours=h2_common_horizon_gpu_hours,
        output_paths=output_paths,
    )
    quality_path = output_dir / "data_quality_report.json"
    _atomic_write_json(quality_path, quality)
    output_paths["data_quality_report"] = quality_path
    return output_paths, quality


def _write_failure_report(
    output_dir: Path,
    run_dir: Path,
    error: Exception,
    matched_compute_path: Path,
) -> Path:
    path = output_dir.expanduser().resolve() / "data_quality_report.json"
    payload = {
        "schema_version": 1,
        "status": "failed",
        "run_dir": run_dir.expanduser().resolve().as_posix(),
        "h2_scope": {
            "matched_compute": matched_compute_path.expanduser().resolve().as_posix(),
            "h2_decision_performed": False,
        },
        "findings": [
            {
                "severity": "critical",
                "check": "derivation_completed",
                "evidence": str(error),
                "impact": "derived outputs are not ready for downstream H2 analysis",
            }
        ],
    }
    _atomic_write_json(path, payload)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_paths, quality = derive_adaptive_metrics(
            args.run_dir,
            args.output_dir,
            matched_compute_path=args.matched_compute,
        )
    except (
        AdaptiveMetricDerivationError,
        AdaptiveVerificationError,
        ReplayInstrumentationError,
        OSError,
        RuntimeError,
        ValueError,
        MemoryError,
    ) as exc:
        quality_path = _write_failure_report(
            args.output_dir,
            args.run_dir,
            exc,
            args.matched_compute,
        )
        print(f"Adaptive metric derivation failed: {exc}", file=sys.stderr)
        print(f"Data quality: {quality_path}", file=sys.stderr)
        return 2
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    return 0 if quality["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
