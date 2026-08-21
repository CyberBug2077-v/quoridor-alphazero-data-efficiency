#!/usr/bin/env python3
"""Verify an Adaptive run and, for Pilot runs, evaluate the frozen gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.Adaptive.adaptive_scheduler import (  # noqa: E402
    AdaptiveScheduler,
    IterationLengthStats,
    SchedulerConfig,
)
from experiments.Adaptive.experiment_runtime import sha256_file  # noqa: E402
from experiments.Adaptive.replay_instrumentation import (  # noqa: E402
    ReplayInstrumentation,
    ReplayInstrumentationConfig,
)
from experiments.Adaptive.resource_accounting import (  # noqa: E402
    ResourceAccountant,
    ResourceConfig,
)


class AdaptiveVerificationError(RuntimeError):
    """Raised when a persisted Adaptive run fails structural verification."""


@dataclass(frozen=True)
class VerifiedRun:
    run_dir: Path
    resolved: dict[str, Any]
    input_manifest: dict[str, Any]
    metrics: list[dict[str, Any]]
    checkpoint_manifest: dict[str, Any]
    scheduler_state: dict[str, Any]
    tracker_state: dict[str, Any]
    resource_state: dict[str, Any]
    recovery_artifacts: dict[str, Path]
    completed: bool
    resumed: bool
    report: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveVerificationError(message)


def _require_finite(value: Any, location: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"non-finite number at {location}")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite(nested, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite(nested, f"{location}[{index}]")


def _load_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing YAML file: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AdaptiveVerificationError(f"invalid YAML {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"YAML must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveVerificationError(f"invalid JSON {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON must contain an object: {path}")
    _require_finite(payload, path.name)
    return payload


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing metrics file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(bool(lines), "metrics.jsonl is empty")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"blank metrics line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdaptiveVerificationError(
                f"invalid metrics line {line_number}: {exc}"
            ) from exc
        _require(
            isinstance(record, dict),
            f"metrics line {line_number} is not an object",
        )
        _require_finite(record, f"metrics[{line_number}]")
        records.append(record)
    iterations = [record.get("iteration") for record in records]
    _require(
        iterations == list(range(1, len(records) + 1)),
        "metrics iterations are not contiguous from 1",
    )
    return records


def _verify_observation_metrics(
    resolved: dict[str, Any], metrics: list[dict[str, Any]]
) -> None:
    instrumentation_schema = int(
        resolved.get("replay_instrumentation", {}).get("schema_version", 1)
    )
    if instrumentation_schema < 2:
        return
    max_length = int(
        resolved["replay_instrumentation"]["max_valid_game_length"]
    )
    exclusion_reasons = {"empty_game", "malformed_game", "abnormal_length"}
    for iteration, record in enumerate(metrics, 1):
        lengths = record.get("valid_game_lengths")
        exclusions = record.get("excluded_game_count_by_reason")
        _require(isinstance(lengths, list), f"iteration {iteration} lengths are missing")
        _require(
            all(
                isinstance(length, int)
                and not isinstance(length, bool)
                and 1 <= length <= max_length
                for length in lengths
            ),
            f"iteration {iteration} contains an invalid scheduler length",
        )
        _require(
            isinstance(exclusions, dict) and set(exclusions) == exclusion_reasons,
            f"iteration {iteration} exclusion reasons differ from policy",
        )
        _require(
            record.get("games_completed")
            == record.get("valid_length_observations")
            == len(lengths),
            f"iteration {iteration} completed and valid observation counts differ",
        )
        _require(
            record.get("realised_valid_states") == sum(lengths),
            f"iteration {iteration} realised valid states differ from lengths",
        )
        _require(
            record.get("excluded_length_observations") == sum(exclusions.values()),
            f"iteration {iteration} excluded observation count differs from reasons",
        )
        truncated_games = record.get("truncated_games")
        truncated_positions = record.get("truncated_positions")
        _require(
            isinstance(truncated_games, int)
            and not isinstance(truncated_games, bool)
            and 0 <= truncated_games <= len(lengths),
            f"iteration {iteration} truncated game count is invalid",
        )
        _require(
            isinstance(truncated_positions, int)
            and not isinstance(truncated_positions, bool)
            and 0 <= truncated_positions <= sum(lengths),
            f"iteration {iteration} truncated position count is invalid",
        )
        _require(
            (truncated_games == 0) == (truncated_positions == 0),
            f"iteration {iteration} truncated games and positions disagree",
        )


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(
        isinstance(expected, str) and len(expected) == 64,
        f"invalid SHA-256 for {label}",
    )
    actual = sha256_file(path)
    _require(actual == expected.lower(), f"SHA-256 mismatch for {label}: {path}")
    return actual


def _normalise_persisted_path(path_value: str) -> str:
    return path_value.replace("\\", "/").rstrip("/")


def _recorded_source_root(resolved: dict[str, Any]) -> str | None:
    configured_output = resolved.get("run", {}).get("output_dir")
    if not isinstance(configured_output, str):
        return None
    normalised = _normalise_persisted_path(configured_output)
    marker = "/experiments/"
    marker_index = normalised.casefold().find(marker)
    if marker_index <= 0:
        return None
    return normalised[:marker_index]


def _resolve_persisted_path(
    path_value: str,
    *,
    run_dir: Path,
    resolved: dict[str, Any],
) -> Path:
    normalised = _normalise_persisted_path(path_value)
    configured_output = resolved.get("run", {}).get("output_dir")
    recorded_run_dir = (
        _normalise_persisted_path(configured_output)
        if isinstance(configured_output, str)
        else None
    )
    mappings = (
        (recorded_run_dir, run_dir),
        (_recorded_source_root(resolved), SOURCE_ROOT),
    )
    for recorded_root, actual_root in mappings:
        if not recorded_root:
            continue
        normalised_casefold = normalised.casefold()
        recorded_casefold = recorded_root.casefold()
        if normalised_casefold == recorded_casefold:
            return actual_root.resolve()
        if normalised_casefold.startswith(recorded_casefold + "/"):
            relative = normalised[len(recorded_root) + 1 :]
            return actual_root.joinpath(*relative.split("/")).resolve()
    return Path(path_value).expanduser().resolve()


def _verify_manifest_entry(
    entry: Any,
    label: str,
    *,
    run_dir: Path,
    resolved: dict[str, Any],
    verify_sha256: bool = True,
) -> Path:
    _require(isinstance(entry, dict), f"invalid manifest entry for {label}")
    path_value = entry.get("path")
    _require(isinstance(path_value, str), f"missing path for {label}")
    path = _resolve_persisted_path(path_value, run_dir=run_dir, resolved=resolved)
    if verify_sha256:
        _verify_hash(path, entry.get("sha256"), label)
    else:
        _require(path.is_file(), f"missing {label}: {path}")
    return path


def _load_torch(path: Path, label: str) -> Any:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AdaptiveVerificationError(
            f"{label} cannot be loaded on CPU: {path}: {exc}"
        ) from exc
    _require(payload is not None, f"{label} is empty: {path}")
    return payload


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def _verify_inputs(run_dir: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_json(run_dir / "input_manifest.json")
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, dict), "input_manifest.inputs must be an object")
    for label, entry in inputs.items():
        _verify_manifest_entry(
            entry,
            f"input {label}",
            run_dir=run_dir,
            resolved=resolved,
            verify_sha256=False,
        )
    _verify_manifest_entry(
        manifest.get("resolved_config"),
        "resolved_config",
        run_dir=run_dir,
        resolved=resolved,
    )
    _require(resolved.get("mode") == "adaptive", "resolved mode must be adaptive")
    _require(resolved.get("status") == "frozen", "resolved status must be frozen")
    configured_output = _resolve_persisted_path(
        str(resolved.get("run", {}).get("output_dir", "")),
        run_dir=run_dir,
        resolved=resolved,
    )
    _require(
        configured_output == run_dir,
        "resolved run.output_dir differs from the verified directory",
    )
    return manifest


def _scheduler_from_metrics(
    resolved: dict[str, Any], metrics: list[dict[str, Any]]
) -> tuple[AdaptiveScheduler, bool, int]:
    scheduler = AdaptiveScheduler(SchedulerConfig(**resolved["adaptive_scheduler"]))
    direction_correct = True
    eligible_direction_updates = 0
    previous_estimate = float(scheduler.state.current_ema_length)
    first_games = scheduler.next_iteration_games
    for index, record in enumerate(metrics):
        iteration = index + 1
        expected_games = (
            first_games if iteration == 1 else metrics[index - 1]["scheduler_planned_games"]
        )
        _require(
            record.get("games_planned") == expected_games,
            f"iteration {iteration} Nt differs from the previous scheduler decision",
        )
        lengths = record.get("valid_game_lengths")
        exclusions = record.get("excluded_game_count_by_reason")
        _require(isinstance(lengths, list), f"iteration {iteration} lengths are missing")
        _require(
            isinstance(exclusions, dict),
            f"iteration {iteration} exclusion counts are missing",
        )
        decision = scheduler.update(
            IterationLengthStats(
                iteration=iteration,
                valid_game_lengths=tuple(lengths),
                excluded_game_count_by_reason=exclusions,
            )
        )
        comparisons = {
            "scheduler_length_estimate": decision.updated_length_estimate,
            "scheduler_unclipped_games": decision.rounded_next_games,
            "scheduler_planned_games": decision.next_iteration_games,
            "scheduler_clipped": decision.clipped,
        }
        for field_name, expected in comparisons.items():
            actual = record.get(field_name)
            matches = (
                _same_number(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            _require(
                matches,
                f"iteration {iteration} {field_name} differs from Scheduler",
            )

        if decision.rounded_next_games is not None:
            eligible_direction_updates += 1
            if decision.updated_length_estimate < previous_estimate:
                previous_plan = math.ceil(
                    scheduler.config.target_states / previous_estimate
                )
                direction_correct &= decision.rounded_next_games >= previous_plan
            elif decision.updated_length_estimate > previous_estimate:
                previous_plan = math.ceil(
                    scheduler.config.target_states / previous_estimate
                )
                direction_correct &= decision.rounded_next_games <= previous_plan
        previous_estimate = decision.updated_length_estimate
    return scheduler, direction_correct, eligible_direction_updates


def _verify_resource_trajectory(
    resolved: dict[str, Any], metrics: list[dict[str, Any]]
) -> None:
    gpu_count = int(resolved["resource_accounting"]["allocated_gpu_count"])
    cumulative_seconds = 0.0
    previous_gpu_hours = -1.0
    for record in metrics:
        iteration = int(record["iteration"])
        iteration_seconds = float(record["iteration_seconds"])
        evaluation_seconds = float(record.get("evaluation_seconds", 0.0))
        _require(iteration_seconds >= 0.0, f"negative iteration time at {iteration}")
        _require(evaluation_seconds >= 0.0, f"negative evaluation time at {iteration}")
        cumulative_seconds += iteration_seconds
        expected_gpu_hours = cumulative_seconds * gpu_count / 3600.0
        actual_gpu_hours = record.get("cumulative_gpu_hours")
        _require(
            _same_number(actual_gpu_hours, expected_gpu_hours),
            f"cumulative GPU-hours are inconsistent at iteration {iteration}",
        )
        _require(
            float(actual_gpu_hours) >= previous_gpu_hours,
            f"cumulative GPU-hours decrease at iteration {iteration}",
        )
        previous_gpu_hours = float(actual_gpu_hours)


def _verify_checkpoints(
    run_dir: Path,
    resolved: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "checkpoint_manifest.json")
    latest_iteration = len(metrics)
    _require(
        manifest.get("last_committed_iteration") == latest_iteration,
        "checkpoint manifest is not at the latest metrics iteration",
    )
    entries = manifest.get("checkpoints")
    _require(isinstance(entries, list) and bool(entries), "checkpoint list is empty")
    iterations = [entry.get("iteration") for entry in entries]
    _require(iterations == sorted(set(iterations)), "checkpoint iterations are invalid")
    gpu_hours_by_iteration = {0: 0.0} | {
        int(record["iteration"]): float(record["cumulative_gpu_hours"])
        for record in metrics
    }
    milestone_selection: dict[float, int] = {}
    for entry in entries:
        iteration = entry.get("iteration")
        _require(
            isinstance(iteration, int) and not isinstance(iteration, bool),
            "checkpoint iteration must be an integer",
        )
        path = _verify_manifest_entry(
            entry,
            f"checkpoint iteration {iteration}",
            run_dir=run_dir,
            resolved=resolved,
        )
        _load_torch(path, f"checkpoint iteration {iteration}")
        _require(
            iteration in gpu_hours_by_iteration,
            f"checkpoint iteration {iteration} has no matching metrics point",
        )
        _require(
            _same_number(entry.get("actual_gpu_hours"), gpu_hours_by_iteration[iteration]),
            f"checkpoint GPU-hours differ at iteration {iteration}",
        )
        milestones = entry.get("milestones", [])
        _require(isinstance(milestones, list), "checkpoint milestones must be a list")
        for target in milestones:
            value = float(target)
            _require(
                float(entry["actual_gpu_hours"]) <= value,
                f"checkpoint iteration {iteration} is after milestone {value}",
            )
            eligible = [
                candidate_iteration
                for candidate_iteration, hours in gpu_hours_by_iteration.items()
                if hours <= value
            ]
            expected_iteration = max(eligible)
            _require(
                iteration == expected_iteration,
                f"milestone {value} did not select latest checkpoint not after target",
            )
            _require(value not in milestone_selection, f"milestone {value} is duplicated")
            milestone_selection[value] = iteration
    return manifest


def _verify_recovery(
    run_dir: Path,
    resolved: dict[str, Any],
    metrics: list[dict[str, Any]],
    checkpoint_manifest: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pointer = _load_json(run_dir / "recovery" / "latest_commit.json")
    iteration = len(metrics)
    _require(pointer.get("iteration") == iteration, "latest commit iteration mismatch")
    manifest_path = _resolve_persisted_path(
        str(pointer.get("commit_manifest", "")),
        run_dir=run_dir,
        resolved=resolved,
    )
    _verify_hash(
        manifest_path,
        pointer.get("commit_manifest_sha256"),
        "latest commit manifest",
    )
    commit = _load_json(manifest_path)
    _require(commit.get("iteration") == iteration, "commit manifest iteration mismatch")
    artifacts = commit.get("artifacts")
    _require(isinstance(artifacts, dict), "commit artifacts must be an object")
    required = {
        "model",
        "replay",
        "runtime_state",
        "scheduler_state",
        "tracker_state",
        "resource_state",
        "metrics_record",
        "checkpoint_manifest",
    }
    _require(required <= set(artifacts), "resume commit is missing required artifacts")
    paths = {
        label: _verify_manifest_entry(
            entry,
            f"recovery {label}",
            run_dir=run_dir,
            resolved=resolved,
        )
        for label, entry in artifacts.items()
    }
    _load_torch(paths["model"], "recovery model")
    try:
        with paths["replay"].open("rb") as source:
            replay = pickle.load(source)
    except Exception as exc:
        raise AdaptiveVerificationError(f"recovery replay cannot be loaded: {exc}") from exc
    runtime_state = _load_torch(paths["runtime_state"], "runtime state")
    scheduler_state = _load_json(paths["scheduler_state"])
    tracker_state = _load_json(paths["tracker_state"])
    resource_state = _load_json(paths["resource_state"])
    metric_record = _load_json(paths["metrics_record"])
    checkpoint_snapshot = _load_json(paths["checkpoint_manifest"])
    artifact_iterations = {
        replay.get("iteration") if isinstance(replay, dict) else None,
        runtime_state.get("iteration") if isinstance(runtime_state, dict) else None,
        scheduler_state.get("completed_iteration"),
        tracker_state.get("completed_iteration"),
        resource_state.get("completed_iteration"),
        metric_record.get("iteration"),
        checkpoint_snapshot.get("last_committed_iteration"),
        iteration,
    }
    if "evaluation" in paths:
        artifact_iterations.add(_load_json(paths["evaluation"]).get("iteration"))
    _require(
        artifact_iterations == {iteration},
        f"recovery artifacts disagree on iteration: {artifact_iterations}",
    )
    _require(isinstance(replay.get("examples"), list), "replay history is invalid")
    replay_size = sum(len(bucket) for bucket in replay["examples"])
    _require(
        replay_size == metrics[-1].get("replay_buffer_size"),
        "replay size differs from final metrics",
    )
    rng_state = runtime_state.get("rng_state")
    _require(isinstance(rng_state, dict), "runtime RNG state is missing")
    _require(
        {"python", "numpy", "torch_cpu", "torch_cuda"} <= set(rng_state),
        "runtime RNG state is incomplete",
    )
    _require(metric_record == metrics[-1], "committed metric differs from metrics.jsonl")
    _require(
        checkpoint_snapshot == checkpoint_manifest,
        "checkpoint manifest differs from committed snapshot",
    )

    root_scheduler = _load_json(run_dir / "scheduler_state" / "latest.json")
    root_tracker = _load_json(run_dir / "tracker.json")
    root_resource = _load_json(run_dir / "resource_state.json")
    _require(root_scheduler == scheduler_state, "root Scheduler state is stale")
    _require(root_tracker == tracker_state, "root tracker state is stale")
    _require(root_resource == resource_state, "root resource state is stale")

    if int(resolved["replay_instrumentation"].get("schema_version", 1)) >= 2:
        _require(
            tracker_state.get("total_truncated_games")
            == sum(record["truncated_games"] for record in metrics),
            "tracker truncated game total differs from metrics",
        )
        _require(
            tracker_state.get("total_truncated_positions")
            == sum(record["truncated_positions"] for record in metrics),
            "tracker truncated position total differs from metrics",
        )

    scheduler = AdaptiveScheduler.from_state_dict(
        SchedulerConfig(**resolved["adaptive_scheduler"]), scheduler_state
    )
    ReplayInstrumentation.from_state_dict(
        ReplayInstrumentationConfig(**resolved["replay_instrumentation"]),
        tracker_state,
    )
    ResourceAccountant.from_state_dict(
        ResourceConfig(**resolved["resource_accounting"]), resource_state
    )
    _require(
        scheduler.next_iteration_games == metrics[-1]["scheduler_planned_games"],
        "saved Scheduler Nt+1 differs from final decision",
    )
    final_hash_fields = {
        "model": "checkpoint_sha256",
        "replay": "replay_state_sha256",
        "runtime_state": "runtime_state_sha256",
        "scheduler_state": "scheduler_state_sha256",
        "tracker_state": "tracker_state_sha256",
        "resource_state": "resource_state_sha256",
    }
    for artifact, metric_field in final_hash_fields.items():
        expected = metrics[-1].get(metric_field)
        if expected is not None:
            _require(
                sha256_file(paths[artifact]) == expected,
                f"final {artifact} hash differs from metrics",
            )
    return paths, scheduler_state, tracker_state, resource_state


def _verify_shared_parameters(run_dir: Path, resolved: dict[str, Any]) -> None:
    sources = resolved.get("protocol_sources", {})
    baseline_path = _resolve_persisted_path(
        str(sources.get("baseline_config", "")),
        run_dir=run_dir,
        resolved=resolved,
    )
    baseline = _load_yaml(baseline_path)
    for section in ("model", "training", "replay"):
        _require(
            resolved.get(section) == baseline.get(section),
            f"Adaptive {section} differs from Baseline",
        )
    for field_name, baseline_value in baseline.get("self_play", {}).items():
        if field_name in {"games_per_iteration", "iterations"}:
            continue
        _require(
            resolved["self_play"].get(field_name) == baseline_value,
            f"Adaptive self_play.{field_name} differs from Baseline",
        )
    for field_name, baseline_value in baseline.get("evaluation", {}).items():
        if field_name in {"enabled", "evaluate_every_iterations"}:
            continue
        _require(
            resolved["evaluation"].get(field_name) == baseline_value,
            f"Adaptive evaluation.{field_name} differs from Baseline",
        )


def _verify_evaluations(
    run_dir: Path,
    resolved: dict[str, Any],
    metrics: list[dict[str, Any]],
    checkpoint_manifest: dict[str, Any],
) -> list[int]:
    checkpoints = {
        int(entry["iteration"]): entry for entry in checkpoint_manifest["checkpoints"]
    }
    verified: list[int] = []
    for record in metrics:
        iteration = int(record["iteration"])
        evaluation_seconds = float(record.get("evaluation_seconds", 0.0))
        if evaluation_seconds == 0.0:
            _require(
                record.get("evaluation_training_state_preserved") is None,
                f"iteration {iteration} has an unexpected evaluation state flag",
            )
            continue
        evaluation = _load_json(
            run_dir / "evaluations" / f"evaluation_checkpoint_{iteration}.json"
        )
        _require(
            record.get("evaluation_training_state_preserved") is True,
            f"iteration {iteration} did not preserve training state",
        )
        _require(
            evaluation.get("training_state_preserved") is True,
            f"evaluation {iteration} lacks training-state preservation evidence",
        )
        entry = checkpoints.get(iteration)
        _require(entry is not None, f"evaluated checkpoint {iteration} was not archived")
        _require(
            _resolve_persisted_path(
                str(evaluation.get("checkpoint_path", "")),
                run_dir=run_dir,
                resolved=resolved,
            )
            == _resolve_persisted_path(
                str(entry["path"]),
                run_dir=run_dir,
                resolved=resolved,
            ),
            f"evaluation {iteration} references the wrong checkpoint",
        )
        _require(
            evaluation.get("checkpoint_sha256") == entry.get("sha256"),
            f"evaluation {iteration} checkpoint hash mismatch",
        )
        verified.append(iteration)
    return verified


def verify_run(run_dir: Path | str) -> VerifiedRun:
    path = Path(run_dir).expanduser().resolve()
    _require(path.is_dir(), f"run directory not found: {path}")
    resolved = _load_yaml(path / "resolved_config.yaml")
    input_manifest = _verify_inputs(path, resolved)
    metrics = _load_metrics(path / "metrics.jsonl")
    _verify_observation_metrics(resolved, metrics)
    scheduler, direction_correct, eligible_direction_updates = (
        _scheduler_from_metrics(resolved, metrics)
    )
    _verify_resource_trajectory(resolved, metrics)
    checkpoint_manifest = _verify_checkpoints(path, resolved, metrics)
    recovery_paths, scheduler_state, tracker_state, resource_state = _verify_recovery(
        path, resolved, metrics, checkpoint_manifest
    )
    _require(
        scheduler.state_dict() == scheduler_state,
        "replayed Scheduler trajectory differs from saved Scheduler state",
    )
    _verify_shared_parameters(path, resolved)
    evaluations = _verify_evaluations(path, resolved, metrics, checkpoint_manifest)

    summary_path = path / "summary.json"
    completed = summary_path.is_file()
    if completed:
        summary = _load_json(summary_path)
        _require(summary.get("status") == "completed", "summary status is not completed")
        _require(
            summary.get("final_iteration") == len(metrics),
            "summary final iteration differs from metrics",
        )
    resumed = "resume validation started" in (path / "run.log").read_text(
        encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "status": "verified",
        "run_id": resolved["run"]["id"],
        "config_id": resolved["config_id"],
        "run_dir": path.as_posix(),
        "completed_iterations": len(metrics),
        "completed": completed,
        "resumed": resumed,
        "checks": {
            "iteration_continuity": True,
            "finite_metrics": True,
            "checkpoint_hashes_and_loadability": True,
            "scheduler_plan_continuity": True,
            "cumulative_gpu_hours_monotonic": True,
            "recovery_iteration_consistency": True,
            "resume_state_complete": True,
            "shared_baseline_parameters_equal": True,
            "evaluation_training_state_preserved": True,
        },
        "scheduler_direction_correct": direction_correct,
        "eligible_direction_updates": eligible_direction_updates,
        "evaluated_iterations": evaluations,
        "latest_recovery_artifacts": {
            label: artifact.as_posix() for label, artifact in recovery_paths.items()
        },
    }
    return VerifiedRun(
        path,
        resolved,
        input_manifest,
        metrics,
        checkpoint_manifest,
        scheduler_state,
        tracker_state,
        resource_state,
        recovery_paths,
        completed,
        resumed,
        report,
    )


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _semantic_equal(left[key], right[key]) for key in left
        )
    sequence_types = (list, tuple, deque)
    if isinstance(left, sequence_types) and isinstance(right, sequence_types):
        return len(left) == len(right) and all(
            _semantic_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _replay_history(run: VerifiedRun) -> list[Any]:
    try:
        with run.recovery_artifacts["replay"].open("rb") as source:
            payload = pickle.load(source)
    except Exception as exc:
        raise AdaptiveVerificationError(
            f"could not reload replay for equivalence comparison: {exc}"
        ) from exc
    history = payload.get("examples") if isinstance(payload, dict) else None
    _require(isinstance(history, list), "equivalence replay history is invalid")
    return history


def _resume_equivalence(
    run: VerifiedRun,
    reference_run_dir: Path | str | None,
) -> dict[str, Any]:
    if reference_run_dir is None:
        return {
            "passed": False,
            "reason": "resume reference run was not provided",
            "compared_fields": [],
            "mismatches": [],
        }
    reference = verify_run(reference_run_dir)
    if reference.run_dir == run.run_dir:
        return {
            "passed": False,
            "reason": "resume reference must be a distinct run directory",
            "compared_fields": [],
            "mismatches": [],
        }
    same_protocol_sections = (
        "model",
        "adaptive_scheduler",
        "self_play",
        "training",
        "replay",
    )
    mismatches: list[str] = []
    for section in same_protocol_sections:
        if reference.resolved.get(section) != run.resolved.get(section):
            mismatches.append(f"resolved.{section}")
    if reference.resolved["run"].get("seed") != run.resolved["run"].get("seed"):
        mismatches.append("resolved.run.seed")
    if len(reference.metrics) != len(run.metrics):
        mismatches.append("completed_iterations")
    fields = (
        "games_planned",
        "games_completed",
        "positions_generated",
        "valid_game_lengths",
        "truncated_games",
        "truncated_positions",
        "realised_valid_states",
        "excluded_length_observations",
        "scheduler_length_estimate",
        "scheduler_unclipped_games",
        "scheduler_planned_games",
        "scheduler_clipped",
        "runtime_state_sha256",
        "scheduler_state_sha256",
        "tracker_state_sha256",
        "checkpoint_sha256",
    )
    for left, right in zip(run.metrics, reference.metrics):
        iteration = left.get("iteration")
        for field_name in fields:
            left_value = left.get(field_name)
            right_value = right.get(field_name)
            matches = (
                _same_number(left_value, right_value)
                if isinstance(left_value, float) and isinstance(right_value, float)
                else left_value == right_value
            )
            if not matches:
                mismatches.append(f"metrics[{iteration}].{field_name}")
    if not _semantic_equal(_replay_history(run), _replay_history(reference)):
        mismatches.append("replay_history_membership_by_iteration")
    if run.resumed == reference.resumed:
        mismatches.append("exactly_one_run_must_use_resume")
    return {
        "passed": not mismatches,
        "reason": None if not mismatches else "resume comparison differs",
        "reference_run_dir": reference.run_dir.as_posix(),
        "compared_fields": [*fields, "replay_history_membership_by_iteration"],
        "mismatches": mismatches,
    }


def _source_protocol(run: VerifiedRun) -> dict[str, Any]:
    source_path = _resolve_persisted_path(
        str(run.resolved.get("protocol_sources", {}).get("adaptive_config", "")),
        run_dir=run.run_dir,
        resolved=run.resolved,
    )
    return _load_yaml(source_path)


def _pilot_gate_output_path(run: VerifiedRun) -> Path:
    source_path = _resolve_persisted_path(
        str(run.resolved.get("protocol_sources", {}).get("adaptive_config", "")),
        run_dir=run.run_dir,
        resolved=run.resolved,
    )
    source = _load_yaml(source_path)
    outputs = source.get("outputs", {})
    configured = outputs.get("pilot_gate_summary") if isinstance(outputs, dict) else None
    if not isinstance(configured, str) or not configured:
        return run.run_dir / "pilot_gate_summary.json"
    requested = Path(configured).expanduser()
    if requested.is_absolute():
        output_path = _resolve_persisted_path(
            configured,
            run_dir=run.run_dir,
            resolved=run.resolved,
        )
    else:
        local_candidate = (source_path.parent / requested).resolve()
        experiments_candidate = (EXPERIMENTS_ROOT / requested).resolve()
        output_path = (
            local_candidate if local_candidate.parent.exists() else experiments_candidate
        )
    _require(
        output_path.parent == run.run_dir,
        "configured Pilot gate summary must be inside the verified run directory",
    )
    return output_path


def _pilot_gate(
    run: VerifiedRun,
    resume_reference_run_dir: Path | str | None,
) -> dict[str, Any]:
    source = _source_protocol(run)
    gate = run.resolved.get("pilot_gate") or source.get("pilot_gate")
    _require(isinstance(gate, dict) and gate, "Pilot gate configuration is missing")
    scheduler_source = source.get("adaptive_scheduler", {})
    baseline_games = int(scheduler_source.get("baseline_games", 75))
    target_states_block = scheduler_source.get("target_states")
    target_states = (
        int(target_states_block["value"])
        if isinstance(target_states_block, dict)
        else int(run.resolved["adaptive_scheduler"]["target_states"])
    )
    eligible = [
        record
        for record in run.metrics
        if record.get("scheduler_unclipped_games") is not None
    ]
    clipping_fraction = (
        sum(bool(record.get("scheduler_clipped")) for record in eligible)
        / len(eligible)
        if eligible
        else None
    )
    target_error_records = [
        record
        for record in run.metrics[1:]
        if isinstance(record.get("realised_valid_states"), (int, float))
    ]
    mean_target_error = (
        sum(
            abs(float(record["realised_valid_states"]) - target_states)
            / target_states
            for record in target_error_records
        )
        / len(target_error_records)
        if target_error_records
        else None
    )
    overhead_fractions = [
        float(record["scheduler_seconds"]) / float(record["iteration_seconds"])
        for record in run.metrics
        if float(record["iteration_seconds"]) > 0.0
    ]
    maximum_overhead_fraction = max(overhead_fractions) if overhead_fractions else None
    executed_plan_changed = any(
        int(record["games_planned"]) != baseline_games for record in run.metrics
    )
    resume_equivalence = _resume_equivalence(run, resume_reference_run_dir)

    checks = {
        "completed_run": run.completed,
        "resume_equivalence": bool(resume_equivalence["passed"]),
        "finite_metrics": True,
        "scheduler_direction_correct": bool(
            run.report["scheduler_direction_correct"]
            and run.report["eligible_direction_updates"] > 0
        ),
        "fresh_state_inflow_change": executed_plan_changed,
        "clipping_fraction_within_limit": (
            clipping_fraction is not None
            and clipping_fraction <= float(gate["maximum_clipping_fraction"])
        ),
        "target_error_within_limit": (
            mean_target_error is not None
            and mean_target_error <= float(gate["maximum_target_error"])
        ),
        "scheduler_overhead_within_limit": (
            maximum_overhead_fraction is not None
            and maximum_overhead_fraction
            <= float(gate["maximum_scheduler_overhead_fraction"])
        ),
    }
    required_checks = {
        "completed_run": bool(gate.get("require_completed_run", True)),
        "resume_equivalence": bool(gate.get("require_resume_equivalence", True)),
        "finite_metrics": bool(gate.get("require_finite_metrics", True)),
        "scheduler_direction_correct": bool(
            gate.get("require_scheduler_direction_correct", True)
        ),
        "fresh_state_inflow_change": bool(
            gate.get("require_fresh_state_inflow_change", True)
        ),
        "clipping_fraction_within_limit": True,
        "target_error_within_limit": True,
        "scheduler_overhead_within_limit": True,
    }
    failures = [
        name
        for name, required in required_checks.items()
        if required and not checks[name]
    ]
    metrics_path = run.run_dir / "metrics.jsonl"
    checkpoint_manifest_path = run.run_dir / "checkpoint_manifest.json"
    latest_commit_path = run.run_dir / "recovery" / "latest_commit.json"
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "protocol_id": run.resolved["config_id"],
        "run_id": run.resolved["run"]["id"],
        "generated_at_utc": _utc_now(),
        "completed_iterations": len(run.metrics),
        "verification": run.report,
        "checks": checks,
        "failures": failures,
        "pilot_metrics": {
            "eligible_scheduler_updates": len(eligible),
            "clipped_scheduler_updates": sum(
                bool(record.get("scheduler_clipped")) for record in eligible
            ),
            "clipping_fraction": clipping_fraction,
            "maximum_clipping_fraction": gate["maximum_clipping_fraction"],
            "target_states": target_states,
            "target_error_iterations": [
                record["iteration"] for record in target_error_records
            ],
            "mean_target_error": mean_target_error,
            "maximum_target_error": gate["maximum_target_error"],
            "maximum_observed_scheduler_overhead_fraction": (
                maximum_overhead_fraction
            ),
            "maximum_scheduler_overhead_fraction": gate[
                "maximum_scheduler_overhead_fraction"
            ],
            "baseline_games": baseline_games,
            "executed_plan_changed_from_baseline": executed_plan_changed,
        },
        "resume_equivalence": resume_equivalence,
        "inputs": {
            "resolved_config": {
                "path": (run.run_dir / "resolved_config.yaml").as_posix(),
                "sha256": sha256_file(run.run_dir / "resolved_config.yaml"),
            },
            "metrics": {
                "path": metrics_path.as_posix(),
                "sha256": sha256_file(metrics_path),
            },
            "checkpoint_manifest": {
                "path": checkpoint_manifest_path.as_posix(),
                "sha256": sha256_file(checkpoint_manifest_path),
            },
            "latest_commit": {
                "path": latest_commit_path.as_posix(),
                "sha256": sha256_file(latest_commit_path),
            },
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(
                payload,
                output,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def verify_adaptive(
    run_dir: Path | str,
    *,
    resume_reference_run_dir: Path | str | None = None,
) -> dict[str, Any]:
    run = verify_run(run_dir)
    if str(run.resolved.get("config_id", "")).startswith("adaptive_pilot_"):
        summary = _pilot_gate(run, resume_reference_run_dir)
        output_path = _pilot_gate_output_path(run)
        _atomic_write_json(output_path, summary)
        summary["output_path"] = output_path.as_posix()
        summary["output_sha256"] = sha256_file(output_path)
        return summary
    return {
        **run.report,
        "generated_at_utc": _utc_now(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-reference-run-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_adaptive(
            args.run_dir,
            resume_reference_run_dir=args.resume_reference_run_dir,
        )
    except (AdaptiveVerificationError, OSError, RuntimeError, ValueError) as exc:
        print(f"Adaptive verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result.get("output_path"):
        print(f"Pilot gate summary: {result['output_path']}")
    return 0 if result.get("status") in {"verified", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
