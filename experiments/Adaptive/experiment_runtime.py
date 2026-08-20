"""Adaptive experiment orchestration and iteration-boundary recovery.

The runtime resolves the frozen protocol, adapts the inherited Baseline Coach,
coordinates the three Adaptive components, and owns durable run artifacts.  It
does not duplicate scheduler, replay-metric, or GPU-hour formulae.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from .adaptive_scheduler import (
        AdaptiveScheduler,
        IterationLengthStats,
        SchedulerConfig,
    )
    from .replay_instrumentation import (
        ReplayInstrumentation,
        ReplayInstrumentationConfig,
    )
    from .resource_accounting import ResourceAccountant, ResourceConfig
except ImportError:  # Direct script execution from experiments/Adaptive.
    from adaptive_scheduler import (  # type: ignore[no-redef]
        AdaptiveScheduler,
        IterationLengthStats,
        SchedulerConfig,
    )
    from replay_instrumentation import (  # type: ignore[no-redef]
        ReplayInstrumentation,
        ReplayInstrumentationConfig,
    )
    from resource_accounting import (  # type: ignore[no-redef]
        ResourceAccountant,
        ResourceConfig,
    )


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
BASELINE_ROOT = SOURCE_ROOT / "baseline"
_MODES = frozenset({"dry-run", "fresh", "resume"})
_SHA256_LENGTH = 64


class ExperimentRuntimeError(RuntimeError):
    """Raised when an Adaptive run cannot preserve its protocol boundary."""


class ProtocolValidationError(ExperimentRuntimeError):
    """Raised when a configuration, hash, or frozen protocol is invalid."""


class ResumeStateError(ExperimentRuntimeError):
    """Raised when no complete and internally consistent resume boundary exists."""


class _AdaptiveStop(Exception):
    """Internal control flow raised after a terminal iteration is committed."""


@dataclass(frozen=True)
class RuntimeRequest:
    mode: str
    config_path: Path | None
    run_dir: Path


@dataclass(frozen=True)
class ResolvedRuntime:
    config: dict[str, Any]
    input_manifest: dict[str, Any]
    run_dir: Path


@dataclass
class BaselineRuntime:
    game: Any
    network: Any
    coach: Any
    train_args: Any
    initial_weights_loaded: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentRuntimeError(f"configuration not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentRuntimeError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentRuntimeError(f"configuration must contain a mapping: {path}")
    return payload


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    return value


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _atomic_write_json(path: Path, payload: Any) -> Path:
    plain = _plain(payload)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(
                plain,
                output,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(path, write)


def _atomic_write_yaml(path: Path, payload: Any) -> Path:
    plain = _plain(payload)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(plain, output, sort_keys=False, allow_unicode=True)
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(path, write)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    plain_records = [_plain(record) for record in records]

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record in plain_records:
                output.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(path, write)


def _atomic_write_pickle(path: Path, payload: Any) -> Path:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as output:
            pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(path, write)


def _atomic_write_torch(path: Path, payload: Any) -> Path:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(path, write)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentRuntimeError(f"required artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentRuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentRuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ExperimentRuntimeError(
                f"blank metrics line {line_number} in {path}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentRuntimeError(
                f"invalid metrics JSON line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ExperimentRuntimeError(
                f"metrics line {line_number} must contain an object"
            )
        records.append(record)
    return records


def _resolve_reference(
    requested: str | Path,
    *,
    config_path: Path,
    experiments_root: Path,
) -> Path:
    path = Path(requested).expanduser()
    if path.is_absolute():
        return path.resolve()
    local_candidate = (config_path.parent / path).resolve()
    experiment_candidate = (experiments_root / path).resolve()
    if local_candidate.exists():
        return local_candidate
    if experiment_candidate.exists():
        return experiment_candidate
    return experiment_candidate


def _resolve_baseline_reference(requested: str | Path, baseline_path: Path) -> Path:
    path = Path(requested).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        (baseline_path.parent / path).resolve(),
        (baseline_path.parent.parent / path).resolve(),
        (SOURCE_ROOT / path).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def configured_run_dir(config_path: Path | str) -> Path:
    """Return the output directory declared by an Adaptive protocol."""

    path = Path(config_path).expanduser().resolve()
    try:
        protocol = _load_yaml(path)
        run = _require_mapping(protocol, "run", "protocol")
    except (ExperimentRuntimeError, OSError, ValueError, KeyError) as exc:
        raise ProtocolValidationError(str(exc)) from exc
    outputs = protocol.get("outputs", {})
    requested = run.get("output_dir")
    if requested is None and isinstance(outputs, dict):
        requested = outputs.get("root")
    if not isinstance(requested, str) or not requested:
        raise ProtocolValidationError("Adaptive protocol output directory is missing")
    return _resolve_reference(
        requested,
        config_path=path,
        experiments_root=EXPERIMENTS_ROOT,
    )


def _expected_sha(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ExperimentRuntimeError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _verified_input(
    label: str,
    path: Path,
    expected_sha256: object,
) -> dict[str, Any]:
    expected = _expected_sha(expected_sha256, label=f"{label}.sha256")
    if not path.is_file():
        raise ExperimentRuntimeError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ExperimentRuntimeError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual} path={path}"
        )
    return {"path": path.as_posix(), "sha256": actual}


def _git_metadata(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _require_mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ExperimentRuntimeError(f"{context}.{key} must be a mapping")
    return value


def _scheduler_config(protocol: dict[str, Any]) -> SchedulerConfig:
    scheduler = _require_mapping(protocol, "adaptive_scheduler", "protocol")
    target = scheduler.get("target_states")
    if isinstance(target, dict):
        target = target.get("value")
    estimator = _require_mapping(scheduler, "length_estimator", "adaptive_scheduler")
    initialization = scheduler.get("initialization") or scheduler.get("warm_start")
    if not isinstance(initialization, dict):
        raise ExperimentRuntimeError("adaptive_scheduler initialization is missing")
    bounds = _require_mapping(scheduler, "bounds", "adaptive_scheduler")
    return SchedulerConfig(
        target_states=target,
        alpha=estimator.get("alpha"),
        minimum_observations=estimator.get("minimum_observations"),
        initial_length=initialization.get(
            "initial_length_value", initialization.get("initial_length")
        ),
        first_iteration_games=initialization.get("first_iteration_games"),
        min_games=bounds.get("min_games"),
        max_games=bounds.get("max_games"),
        max_valid_game_length=150,
        rounding=scheduler.get("rounding"),
        schema_version=1,
    )


def _validate_protocol_status(protocol: dict[str, Any]) -> None:
    config_id = protocol.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ExperimentRuntimeError("Adaptive protocol config_id is missing")
    status = protocol.get("status")
    is_formal = config_id.startswith("adaptive_formal_")
    if status != "frozen":
        if is_formal:
            raise ExperimentRuntimeError(
                "Formal Adaptive protocol is not runnable: status must be frozen "
                "after the Pilot gate and experiment commit are recorded"
            )
        raise ExperimentRuntimeError(
            f"Adaptive protocol {config_id} is not frozen: status={status!r}"
        )
    if is_formal:
        freeze_gate = _require_mapping(protocol, "freeze_gate", "protocol")
        if freeze_gate.get("current_state") not in {"passed", "frozen"}:
            raise ExperimentRuntimeError("Formal Pilot freeze gate has not passed")
        run = _require_mapping(protocol, "run", "protocol")
        commit = run.get("experiment_code_commit")
        if not isinstance(commit, str) or not commit:
            raise ExperimentRuntimeError("Formal experiment_code_commit is missing")


def _normalise_evaluation(
    baseline: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    evaluation = copy.deepcopy(baseline["evaluation"])
    adaptive_evaluation = protocol.get("evaluation", {})
    if "online" in adaptive_evaluation:
        online = adaptive_evaluation["online"]
    else:
        online = adaptive_evaluation
    if isinstance(online, dict):
        cadence = online.get("evaluate_every_iterations")
        if cadence is not None:
            evaluation["evaluate_every_iterations"] = cadence
    evaluation["enabled"] = bool(evaluation.get("evaluate_every_iterations"))
    return evaluation


def _normalise_checkpoint(protocol: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    checkpoint = copy.deepcopy(protocol.get("checkpoint", {}))
    checkpoint["directory"] = (run_dir / "checkpoints").as_posix()
    checkpoint.setdefault("save_every_iterations", 1)
    checkpoint.setdefault("save_replay_state", True)
    checkpoint.setdefault("save_scheduler_state", True)
    checkpoint.setdefault("save_instrumentation_state", True)
    checkpoint.setdefault("save_rng_state", True)
    checkpoint.setdefault("compute_sha256", True)
    checkpoint.setdefault("gpu_hour_analysis_milestones", [])
    return checkpoint


def _verify_formal_gate(
    protocol: dict[str, Any],
    *,
    config_path: Path,
    experiments_root: Path,
    inputs: dict[str, Any],
    source_root: Path,
) -> None:
    config_id = str(protocol["config_id"])
    if not config_id.startswith("adaptive_formal_"):
        return
    gate = protocol["freeze_gate"]
    gate_summary = gate.get("pilot_gate_summary", {})
    path = _resolve_reference(
        gate_summary.get("path", ""),
        config_path=config_path,
        experiments_root=experiments_root,
    )
    inputs["pilot_gate_summary"] = _verified_input(
        "pilot_gate_summary", path, gate_summary.get("sha256")
    )
    summary = _read_json(path)
    if summary.get("status") != gate_summary.get("required_status", "passed"):
        raise ExperimentRuntimeError("Formal Pilot gate summary is not passed")
    current_git = _git_metadata(source_root)
    expected_commit = protocol["run"]["experiment_code_commit"]
    if current_git["commit"] != expected_commit:
        raise ExperimentRuntimeError(
            "current experiment commit differs from Formal protocol"
        )
    if protocol["run"].get("require_clean_worktree") and current_git["dirty"]:
        raise ExperimentRuntimeError("Formal run requires a clean worktree")


def _verify_formal_inputs(
    protocol: dict[str, Any],
    *,
    config_path: Path,
    experiments_root: Path,
    inputs: dict[str, Any],
) -> None:
    if not str(protocol["config_id"]).startswith("adaptive_formal_"):
        return
    formal_inputs = _require_mapping(protocol, "inputs", "protocol")
    for key in ("accepted_pilot_config", "shared_holdout_contract"):
        reference = formal_inputs.get(key)
        if not isinstance(reference, dict):
            raise ExperimentRuntimeError(f"Formal input {key} is missing")
        path = _resolve_reference(
            reference.get("path", ""),
            config_path=config_path,
            experiments_root=experiments_root,
        )
        inputs[key] = _verified_input(key, path, reference.get("sha256"))

    evaluation = protocol.get("evaluation", {})
    matched_evaluation = (
        evaluation.get("matched_compute", {})
        if isinstance(evaluation, dict)
        else {}
    )
    protocols = matched_evaluation.get("protocols", [])
    if not isinstance(protocols, list):
        raise ExperimentRuntimeError("Formal evaluation protocols must be a list")
    for index, reference in enumerate(protocols):
        if not isinstance(reference, dict):
            raise ExperimentRuntimeError("Formal evaluation protocol entry is invalid")
        label = f"evaluation_protocol_{index + 1}"
        path = _resolve_reference(
            reference.get("path", ""),
            config_path=config_path,
            experiments_root=experiments_root,
        )
        inputs[label] = _verified_input(label, path, reference.get("sha256"))


def _validate_matched_contract(
    matched: dict[str, Any],
    *,
    matched_path: Path,
    baseline: dict[str, Any],
    baseline_path: Path,
    adaptive_initialization: dict[str, Any],
    adaptive_checkpoint_path: Path,
    inputs: dict[str, Any],
    experiments_root: Path,
) -> None:
    if matched.get("config_id") != "matched_compute_v1":
        raise ExperimentRuntimeError("matched-compute config_id must be matched_compute_v1")
    if matched.get("status") != "frozen":
        raise ExperimentRuntimeError("matched-compute contract must be frozen")

    matched_provenance = _require_mapping(matched, "provenance", "matched_compute")
    if matched_provenance.get("baseline_config_sha256") != inputs["baseline_config"][
        "sha256"
    ]:
        raise ExperimentRuntimeError(
            "matched-compute and Adaptive protocols reference different Baseline configs"
        )

    common = _require_mapping(matched, "common_initialization", "matched_compute")
    baseline_initialization = _require_mapping(
        baseline, "initialization", "baseline_config"
    )
    flag_fields = (
        "mode",
        "load_weights",
        "load_replay",
        "load_optimizer_state",
        "load_tracker_state",
        "start_iteration",
    )
    for field_name in flag_fields:
        values = {
            common.get(field_name),
            baseline_initialization.get(field_name),
            adaptive_initialization.get(field_name),
        }
        if len(values) != 1:
            raise ExperimentRuntimeError(
                f"common initialization disagrees on {field_name}"
            )

    common_sha = _expected_sha(
        common.get("checkpoint_sha256"),
        label="matched_compute.common_initialization.checkpoint_sha256",
    )
    baseline_sha = _expected_sha(
        baseline_initialization.get("expected_sha256"),
        label="baseline.initialization.expected_sha256",
    )
    if {common_sha, baseline_sha, inputs["pretrained_checkpoint"]["sha256"]} != {
        common_sha
    }:
        raise ExperimentRuntimeError(
            "Baseline, matched-compute, and Adaptive checkpoint hashes differ"
        )

    common_checkpoint_path = _resolve_reference(
        common.get("checkpoint_path", ""),
        config_path=matched_path,
        experiments_root=experiments_root,
    )
    baseline_checkpoint_path = _resolve_baseline_reference(
        baseline_initialization.get("checkpoint_path", ""), baseline_path
    )
    if {
        common_checkpoint_path,
        baseline_checkpoint_path,
        adaptive_checkpoint_path,
    } != {adaptive_checkpoint_path}:
        raise ExperimentRuntimeError(
            "Baseline, matched-compute, and Adaptive checkpoint paths differ"
        )


def resolve_adaptive_protocol(
    config_path: Path | str,
    run_dir: Path | str,
    *,
    experiments_root: Path = EXPERIMENTS_ROOT,
    source_root: Path = SOURCE_ROOT,
) -> ResolvedRuntime:
    config_path = Path(config_path).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    protocol = _load_yaml(config_path)
    _validate_protocol_status(protocol)

    provenance = _require_mapping(protocol, "provenance", "protocol")
    baseline_path = _resolve_reference(
        provenance.get("baseline_config", ""),
        config_path=config_path,
        experiments_root=experiments_root,
    )
    inputs: dict[str, Any] = {
        "adaptive_config": {
            "path": config_path.as_posix(),
            "sha256": sha256_file(config_path),
        },
        "baseline_config": _verified_input(
            "baseline_config",
            baseline_path,
            provenance.get("baseline_config_sha256"),
        ),
    }
    baseline = _load_yaml(baseline_path)

    matched_reference = None
    expected_matched_sha = None
    shared = protocol.get("shared_algorithm_parameters", {})
    if isinstance(shared, dict) and isinstance(shared.get("equality_contract"), dict):
        matched_reference = shared["equality_contract"].get("path")
        expected_matched_sha = shared["equality_contract"].get("sha256")
    formal_inputs = protocol.get("inputs", {})
    if isinstance(formal_inputs, dict) and isinstance(
        formal_inputs.get("matched_compute"), dict
    ):
        matched_reference = formal_inputs["matched_compute"].get("path")
        expected_matched_sha = formal_inputs["matched_compute"].get("sha256")
    if matched_reference is None:
        raise ExperimentRuntimeError("matched-compute contract reference is missing")
    matched_path = _resolve_reference(
        matched_reference,
        config_path=config_path,
        experiments_root=experiments_root,
    )
    inputs["matched_compute"] = _verified_input(
        "matched_compute", matched_path, expected_matched_sha
    )
    matched = _load_yaml(matched_path)

    initialization = _require_mapping(protocol, "initialization", "protocol")
    expected_initialization = {
        "mode": "pretrained_checkpoint",
        "load_weights": True,
        "load_replay": False,
        "load_optimizer_state": False,
        "load_tracker_state": False,
        "start_iteration": 1,
    }
    for field_name, expected_value in expected_initialization.items():
        if initialization.get(field_name) != expected_value:
            raise ExperimentRuntimeError(
                f"initialization.{field_name} must be {expected_value!r}"
            )
    checkpoint_path = _resolve_reference(
        initialization.get("checkpoint_path", ""),
        config_path=config_path,
        experiments_root=experiments_root,
    )
    checkpoint_sha = initialization.get("checkpoint_sha256")
    inputs["pretrained_checkpoint"] = _verified_input(
        "pretrained_checkpoint", checkpoint_path, checkpoint_sha
    )
    _validate_matched_contract(
        matched,
        matched_path=matched_path,
        baseline=baseline,
        baseline_path=baseline_path,
        adaptive_initialization=initialization,
        adaptive_checkpoint_path=checkpoint_path,
        inputs=inputs,
        experiments_root=experiments_root,
    )

    scheduler_config = _scheduler_config(protocol)
    adaptive_run = _require_mapping(protocol, "run", "protocol")
    budget = _require_mapping(protocol, "budget", "protocol")
    declared_output = _resolve_reference(
        adaptive_run.get("output_dir", protocol.get("outputs", {}).get("root", "")),
        config_path=config_path,
        experiments_root=experiments_root,
    )
    if declared_output != run_dir:
        raise ExperimentRuntimeError(
            f"RuntimeRequest.run_dir differs from protocol output_dir: "
            f"{run_dir} != {declared_output}"
        )

    for section in ("model", "training", "replay"):
        if section in protocol and protocol[section] != baseline[section]:
            raise ExperimentRuntimeError(
                f"Adaptive protocol may not override inherited {section} parameters"
            )
    adaptive_self_play = protocol.get("self_play", {})
    if not isinstance(adaptive_self_play, dict):
        raise ExperimentRuntimeError("protocol.self_play must be a mapping")
    allowed_self_play_fields = {
        "games_per_iteration_source",
        "iterations_safety_limit",
    }
    for field_name, value in adaptive_self_play.items():
        if field_name in allowed_self_play_fields:
            continue
        if field_name not in baseline["self_play"] or value != baseline["self_play"][field_name]:
            raise ExperimentRuntimeError(
                f"Adaptive protocol may not change self_play.{field_name}"
            )
    inherit_exceptions = shared.get("inherit_all_except", []) if isinstance(shared, dict) else []
    allowed_inherit_exceptions = {
        "run",
        "budget",
        "self_play.games_per_iteration",
        "self_play.iterations",
        "checkpoint.directory",
        "checkpoint.save_every_iterations",
        "evaluation.evaluate_every_iterations",
        "adaptive_scheduler",
    }
    unexpected_exceptions = sorted(
        set(inherit_exceptions) - allowed_inherit_exceptions
    )
    if unexpected_exceptions:
        raise ExperimentRuntimeError(
            "shared algorithm inheritance contains forbidden exception(s): "
            + ", ".join(unexpected_exceptions)
        )

    resolved_self_play = copy.deepcopy(baseline["self_play"])
    resolved_self_play["games_per_iteration"] = scheduler_config.first_iteration_games
    resolved_self_play["iterations"] = budget.get("max_iterations")
    resolved_initialization = {
        "mode": "pretrained_checkpoint",
        "checkpoint_path": checkpoint_path.as_posix(),
        "expected_sha256": inputs["pretrained_checkpoint"]["sha256"],
        "load_weights": True,
        "load_replay": False,
        "load_optimizer_state": False,
        "load_tracker_state": False,
        "start_iteration": 1,
    }
    resolved = {
        "schema_version": 1,
        "mode": "adaptive",
        "config_id": protocol["config_id"],
        "status": protocol["status"],
        "run": copy.deepcopy(adaptive_run),
        "initialization": resolved_initialization,
        "model": copy.deepcopy(baseline["model"]),
        "budget": copy.deepcopy(budget),
        "self_play": resolved_self_play,
        "training": copy.deepcopy(baseline["training"]),
        "replay": copy.deepcopy(baseline["replay"]),
        "checkpoint": _normalise_checkpoint(protocol, run_dir),
        "instrumentation": copy.deepcopy(protocol.get("instrumentation", {})),
        "metrics": copy.deepcopy(protocol.get("metrics", {})),
        "pilot_gate": copy.deepcopy(protocol.get("pilot_gate", {})),
        "evaluation": _normalise_evaluation(baseline, protocol),
        "logging": {
            "metrics_file": "metrics.jsonl",
            "metadata_file": "run_metadata.json",
            "summary_file": "summary.json",
            "tracker_file": "tracker.json",
            "checkpoint_manifest_file": "checkpoint_manifest.json",
        },
        "adaptive_scheduler": asdict(scheduler_config),
        "replay_instrumentation": asdict(
            ReplayInstrumentationConfig(
                board_size=baseline["model"]["board_size"],
                history_iterations=baseline["replay"]["history_iterations"],
                max_valid_game_length=baseline["self_play"]["max_game_length"],
            )
        ),
        "resource_accounting": asdict(
            ResourceConfig(
                allocated_gpu_count=budget.get("allocated_gpu_count", 1),
                max_gpu_hours=budget.get("max_gpu_hours"),
                max_iterations=budget.get("max_iterations"),
                max_wall_clock_hours=budget.get("max_wall_clock_hours"),
                evaluation_time_included=budget.get(
                    "evaluation_time_included", False
                ),
                instrumentation_overhead_included=budget.get(
                    "instrumentation_overhead_included", True
                ),
                crossing_policy=budget.get(
                    "crossing_policy", "keep_crossing_iteration"
                ),
            )
        ),
        "protocol_sources": {
            "adaptive_config": config_path.as_posix(),
            "baseline_config": baseline_path.as_posix(),
            "matched_compute": matched_path.as_posix(),
        },
        "protocol_requirements": {
            "require_clean_worktree": bool(
                provenance.get("require_clean_worktree", False)
            ),
            "baseline_git_commit": provenance.get("baseline_git_commit"),
        },
        "matched_compute": {
            "allowed_condition_differences": copy.deepcopy(
                matched.get("allowed_condition_differences", [])
            ),
            "compute_budget": copy.deepcopy(matched.get("compute_budget", {})),
            "checkpoint_alignment": copy.deepcopy(
                matched.get("checkpoint_alignment", {})
            ),
        },
    }
    resolved["run"]["output_dir"] = run_dir.as_posix()

    target_block = protocol["adaptive_scheduler"].get("target_states")
    if isinstance(target_block, dict) and target_block.get("calibration_source"):
        calibration_path = _resolve_reference(
            target_block["calibration_source"],
            config_path=config_path,
            experiments_root=experiments_root,
        )
        inputs["scheduler_calibration_metrics"] = _verified_input(
            "scheduler_calibration_metrics",
            calibration_path,
            target_block.get("calibration_source_sha256"),
        )

    _verify_formal_gate(
        protocol,
        config_path=config_path,
        experiments_root=experiments_root,
        inputs=inputs,
        source_root=source_root,
    )
    _verify_formal_inputs(
        protocol,
        config_path=config_path,
        experiments_root=experiments_root,
        inputs=inputs,
    )
    input_manifest = {
        "schema_version": 1,
        "config_id": protocol["config_id"],
        "generated_at_utc": _utc_now(),
        "hash_algorithm": "sha256",
        "inputs": inputs,
    }
    return ResolvedRuntime(resolved, input_manifest, run_dir)


def _ensure_fresh_output(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ExperimentRuntimeError(
            f"fresh Adaptive run directory must not exist or must be empty: {run_dir}"
        )


def _set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _initialise_device(resolved: dict[str, Any]) -> None:
    run = resolved["run"]
    if run.get("device") != "cuda":
        return
    if not torch.cuda.is_available():
        raise ExperimentRuntimeError("Adaptive CUDA run requested but CUDA is unavailable")
    gpu_index = int(run.get("gpu_index", 0))
    if gpu_index >= torch.cuda.device_count():
        raise ExperimentRuntimeError(f"invalid Adaptive GPU index: {gpu_index}")
    torch.cuda.set_device(gpu_index)


def build_baseline_runtime(resolved_config: dict[str, Any]) -> BaselineRuntime:
    """Create the inherited QuoridorGame, NNetWrapper, Coach, and Coach args."""

    alphazero_root = BASELINE_ROOT / "external" / "alphazero"
    pathfinder_root = alphazero_root / "quoridor" / "pathFinder-module"
    baseline_scripts = BASELINE_ROOT / "scripts"
    for path in (alphazero_root, pathfinder_root, baseline_scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from Coach import Coach
    from quoridor.QuoridorGame import QuoridorGame
    from quoridor.pytorch import NNet as nnet_module
    from quoridor.pytorch.NNet import NNetWrapper
    from runtime.config import map_baseline_to_train_args, map_model_to_nn_args
    from utils import dotdict

    mapped_source = copy.deepcopy(resolved_config)
    checkpoint_dir = Path(resolved_config["checkpoint"]["directory"])
    mapped_source["checkpoint"]["_directory_path"] = checkpoint_dir
    nn_args = map_model_to_nn_args(mapped_source)
    train_mapping = map_baseline_to_train_args(mapped_source)
    train_mapping["numIters"] = int(resolved_config["budget"]["max_iterations"]) + 1
    train_mapping["numEps"] = int(
        resolved_config["adaptive_scheduler"]["first_iteration_games"]
    )
    train_mapping["checkpoint"] = checkpoint_dir.as_posix()
    train_mapping["save_every_n_iterations"] = train_mapping["numIters"] + 1
    nnet_module.args.update(copy.deepcopy(nn_args))
    game = QuoridorGame(resolved_config["model"]["board_size"])
    network = NNetWrapper(game, custom_args=dotdict(copy.deepcopy(nn_args)))
    train_args = dotdict(train_mapping)
    coach = Coach(game, network, train_args)
    coach.save_every_n_iterations = train_mapping["save_every_n_iterations"]
    runtime = BaselineRuntime(game, network, coach, train_args)
    _load_initial_weights(runtime, resolved_config)
    return runtime


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ExperimentRuntimeError("RNG state missing: " + ", ".join(missing))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ExperimentRuntimeError("saved CUDA RNG state cannot be restored")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _capture_optimizer_state(network: Any) -> Any:
    optimizer = getattr(network, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        return optimizer.state_dict()
    method = getattr(network, "optimizer_state_dict", None)
    return method() if callable(method) else None


def _restore_optimizer_state(network: Any, optimizer_state: Any) -> None:
    if optimizer_state is None:
        return
    optimizer = getattr(network, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(optimizer_state)
        return
    method = getattr(network, "load_optimizer_state_dict", None)
    if callable(method):
        method(optimizer_state)
        return
    raise ExperimentRuntimeError(
        "recovery contains optimizer state but Baseline runtime cannot restore it"
    )


def _save_network_checkpoint(network: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = path.name + ".tmp"
    temporary = path.with_name(temporary_name)
    try:
        if hasattr(network, "save_checkpoint"):
            network.save_checkpoint(path.parent.as_posix(), temporary_name)
        elif hasattr(network, "state_dict"):
            torch.save({"state_dict": network.state_dict()}, temporary)
        else:
            raise TypeError("network must implement save_checkpoint or state_dict")
        with temporary.open("rb+") as output:
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _load_network_checkpoint(network: Any, path: Path) -> None:
    if hasattr(network, "load_checkpoint"):
        network.load_checkpoint(path.parent.as_posix(), path.name)
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    network.load_state_dict(state)


def _load_initial_weights(runtime: BaselineRuntime, resolved: dict[str, Any]) -> None:
    path = Path(resolved["initialization"]["checkpoint_path"])
    expected = resolved["initialization"]["expected_sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise ExperimentRuntimeError("pretrained checkpoint changed after resolution")
    _load_network_checkpoint(runtime.network, path)
    runtime.initial_weights_loaded = True


def _append_run_log(run_dir: Path, message: str) -> None:
    path = run_dir / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"{_utc_now()} {message}\n")
        output.flush()
        os.fsync(output.fileno())


def _write_initial_outputs(resolution: ResolvedRuntime) -> None:
    run_dir = resolution.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        "checkpoints",
        "recovery",
        "scheduler_state",
        "evaluations",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    resolved_path = _atomic_write_yaml(
        run_dir / "resolved_config.yaml", resolution.config
    )
    manifest = copy.deepcopy(resolution.input_manifest)
    manifest["resolved_config"] = {
        "path": resolved_path.as_posix(),
        "sha256": sha256_file(resolved_path),
    }
    _atomic_write_json(run_dir / "input_manifest.json", manifest)
    metadata = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "run_id": resolution.config["run"]["id"],
        "config_id": resolution.config["config_id"],
        "mode": "fresh",
        "git": _git_metadata(SOURCE_ROOT),
        "input_manifest_sha256": sha256_file(run_dir / "input_manifest.json"),
        "starting_iteration": 1,
        "initial_replay_size": 0,
    }
    _atomic_write_json(run_dir / "run_metadata.json", metadata)
    _atomic_write_json(
        run_dir / "checkpoint_manifest.json",
        {
            "schema_version": 1,
            "last_committed_iteration": 0,
            "checkpoints": [
                {
                    "iteration": 0,
                    "path": resolution.config["initialization"]["checkpoint_path"],
                    "sha256": resolution.config["initialization"][
                        "expected_sha256"
                    ],
                    "actual_gpu_hours": 0.0,
                    "is_final": False,
                    "is_milestone": True,
                    "milestones": [0.0],
                }
            ],
        },
    )
    _atomic_write_jsonl(run_dir / "metrics.jsonl", [])
    _append_run_log(run_dir, "fresh Adaptive run initialized")


def _default_evaluation_runner(
    resolved: dict[str, Any],
    runtime: BaselineRuntime,
    checkpoint_path: Path,
    output_path: Path,
) -> float:
    baseline_scripts = BASELINE_ROOT / "scripts"
    if str(baseline_scripts) not in sys.path:
        sys.path.insert(0, str(baseline_scripts))
    from evaluate import evaluate_checkpoint

    started = time.perf_counter()
    evaluate_checkpoint(
        resolved,
        runtime.game,
        runtime.network,
        checkpoint=checkpoint_path,
        output_path=output_path,
    )
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ExperimentRuntimeError("evaluation timer returned an invalid duration")
    return elapsed


def _required_metric_fields(resolved: dict[str, Any]) -> list[str]:
    metrics = resolved.get("metrics", {})
    fields = metrics.get("required_fields", []) if isinstance(metrics, dict) else []
    return [str(field) for field in fields]


def _archive_checkpoint(
    recovery_model: Path,
    archive_path: Path,
) -> str:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    try:
        shutil.copyfile(recovery_model, temporary)
        with temporary.open("rb+") as output:
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, archive_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(archive_path)


def _upsert_archived_checkpoint(
    manifest: dict[str, Any],
    *,
    source_model: Path,
    archive_dir: Path,
    iteration: int,
    actual_gpu_hours: float,
    is_final: bool,
    milestones: list[float],
) -> Path:
    entries = manifest.get("checkpoints")
    if not isinstance(entries, list):
        raise ExperimentRuntimeError("checkpoint manifest entries are invalid")
    archive_path = archive_dir / f"checkpoint_{iteration}.pth.tar"
    archive_sha = _archive_checkpoint(source_model, archive_path)
    existing = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("iteration") == iteration
        ),
        None,
    )
    if existing is None:
        existing = {
            "iteration": iteration,
            "path": archive_path.as_posix(),
            "sha256": archive_sha,
            "actual_gpu_hours": actual_gpu_hours,
            "is_final": is_final,
            "is_milestone": bool(milestones),
            "milestones": sorted(set(milestones)),
        }
        entries.append(existing)
    else:
        if not math.isclose(
            float(existing["actual_gpu_hours"]),
            actual_gpu_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExperimentRuntimeError(
                f"checkpoint iteration {iteration} has conflicting GPU-hours"
            )
        existing.update(
            {
                "path": archive_path.as_posix(),
                "sha256": archive_sha,
                "is_final": bool(existing.get("is_final")) or is_final,
                "is_milestone": bool(existing.get("is_milestone"))
                or bool(milestones),
                "milestones": sorted(
                    set(float(value) for value in existing.get("milestones", []))
                    | set(milestones)
                ),
            }
        )
    entries.sort(key=lambda entry: int(entry["iteration"]))
    return archive_path


def _prune_recovery_directories(recovery_root: Path, keep: Path) -> None:
    for child in recovery_root.iterdir():
        if not child.is_dir() or child == keep:
            continue
        if child.name.startswith("iteration_") or child.name.endswith(".pending"):
            shutil.rmtree(child)


def commit_iteration(
    *,
    resolution: ResolvedRuntime,
    runtime: BaselineRuntime,
    scheduler: AdaptiveScheduler,
    instrumentation: ReplayInstrumentation,
    accountant: ResourceAccountant,
    iteration: int,
    games_planned: int,
    coach_metrics: dict[str, Any],
    replay_stats: Any,
    scheduler_decision: Any,
    instrumentation_seconds: float,
    observer_seconds: float,
    evaluation_runner: Callable[
        [dict[str, Any], BaselineRuntime, Path, Path], float
    ]
    | None = None,
) -> tuple[dict[str, Any], Any]:
    """Durably commit one fully trained iteration and return its budget decision."""

    run_dir = resolution.run_dir
    recovery_root = run_dir / "recovery"
    final_dir = recovery_root / f"iteration_{iteration:06d}"
    if final_dir.exists():
        raise ExperimentRuntimeError(
            f"recovery iteration already exists and cannot be reused: {final_dir}"
        )
    pending_dir = recovery_root / f".iteration_{iteration:06d}.{os.getpid()}.pending"
    if pending_dir.exists():
        raise ExperimentRuntimeError(f"pending recovery path already exists: {pending_dir}")
    pending_dir.mkdir(parents=True, exist_ok=False)

    previous_gpu_hours = accountant.state.cumulative_gpu_hours
    checkpoint_started = time.perf_counter()
    pending_model = pending_dir / "model.pth.tar"
    pending_replay = pending_dir / "replay.examples"
    pending_runtime_state = pending_dir / "runtime_state.pt"
    try:
        _save_network_checkpoint(runtime.network, pending_model)
        _atomic_write_pickle(
            pending_replay,
            {
                "schema_version": 1,
                "iteration": iteration,
                "examples": runtime.coach.trainExamplesHistory,
            },
        )
        checkpoint_before_evaluation = time.perf_counter() - checkpoint_started
        if (
            not math.isfinite(checkpoint_before_evaluation)
            or checkpoint_before_evaluation < 0.0
        ):
            raise ExperimentRuntimeError("checkpoint timer returned an invalid duration")

        evaluation_seconds = 0.0
        evaluation_output_path: Path | None = None
        evaluation_state_preserved: bool | None = None
        evaluation = resolution.config["evaluation"]
        cadence = evaluation.get("evaluate_every_iterations")
        if evaluation.get("enabled") and cadence and iteration % int(cadence) == 0:
            evaluation_output_path = (
                run_dir
                / "evaluations"
                / f"evaluation_checkpoint_{iteration}.json"
            )
            runner = evaluation_runner or _default_evaluation_runner
            training_rng_state = _capture_rng_state()
            optimizer_state_before_evaluation = copy.deepcopy(
                _capture_optimizer_state(runtime.network)
            )
            try:
                evaluation_seconds = float(
                    runner(
                        resolution.config,
                        runtime,
                        pending_model,
                        evaluation_output_path,
                    )
                )
            finally:
                _load_network_checkpoint(runtime.network, pending_model)
                _restore_optimizer_state(
                    runtime.network, optimizer_state_before_evaluation
                )
                _restore_rng_state(training_rng_state)
            if not math.isfinite(evaluation_seconds) or evaluation_seconds < 0.0:
                raise ExperimentRuntimeError(
                    "evaluation runner returned an invalid duration"
                )
            if not evaluation_output_path.is_file():
                raise ExperimentRuntimeError(
                    "evaluation runner did not write its audit record"
                )
            evaluation_state_preserved = True

        _atomic_write_torch(
            pending_runtime_state,
            {
                "schema_version": 1,
                "iteration": iteration,
                "rng_state": _capture_rng_state(),
                "optimizer_state": _capture_optimizer_state(runtime.network),
            },
        )
        scheduler_state_path = pending_dir / "scheduler_state.json"
        tracker_state_path = pending_dir / "tracker_state.json"
        resource_state_path = pending_dir / "resource_state.json"
        _atomic_write_json(scheduler_state_path, scheduler.state_dict())
        _atomic_write_json(tracker_state_path, instrumentation.state_dict())

        model_sha = sha256_file(pending_model)
        replay_sha = sha256_file(pending_replay)
        runtime_state_sha = sha256_file(pending_runtime_state)
        scheduler_state_sha = sha256_file(scheduler_state_path)
        tracker_state_sha = sha256_file(tracker_state_path)
        runtime_checkpoint_seconds = (
            time.perf_counter() - checkpoint_started - evaluation_seconds
        )
        if (
            not math.isfinite(runtime_checkpoint_seconds)
            or runtime_checkpoint_seconds < 0.0
        ):
            raise ExperimentRuntimeError("checkpoint timer returned an invalid duration")

        coach_iteration_seconds = float(coach_metrics["iteration_seconds"])
        coach_self_play_seconds = float(coach_metrics["self_play_seconds"])
        coach_training_seconds = float(coach_metrics["training_seconds"])
        coach_checkpoint_overhead = max(
            0.0,
            coach_iteration_seconds
            - coach_self_play_seconds
            - coach_training_seconds,
        )
        checkpoint_seconds = runtime_checkpoint_seconds + coach_checkpoint_overhead
        exclusive_self_play_seconds = max(
            0.0, coach_self_play_seconds - observer_seconds
        )
        post_coach_instrumentation_seconds = max(
            0.0, instrumentation_seconds - observer_seconds
        )
        wall_clock_seconds = (
            coach_iteration_seconds
            + post_coach_instrumentation_seconds
            + runtime_checkpoint_seconds
            + evaluation_seconds
        )
        resource_record, budget_decision = accountant.record_iteration(
            iteration,
            self_play_seconds=exclusive_self_play_seconds,
            training_seconds=coach_training_seconds,
            instrumentation_seconds=float(instrumentation_seconds),
            checkpoint_seconds=checkpoint_seconds,
            evaluation_seconds=evaluation_seconds,
            wall_clock_seconds=wall_clock_seconds,
        )

        _atomic_write_json(resource_state_path, accountant.state_dict())
        resource_state_sha = sha256_file(resource_state_path)
        final_model = final_dir / pending_model.name
        final_replay = final_dir / pending_replay.name
        final_runtime_state = final_dir / pending_runtime_state.name

        metric = copy.deepcopy(coach_metrics)
        metric.update(
            {
                "schema_version": 1,
                "iteration": iteration,
                "games_planned": games_planned,
                "games_completed": replay_stats.games_completed,
                "positions_generated": replay_stats.positions_generated,
                "mean_game_length": replay_stats.mean_game_length,
                "min_game_length": replay_stats.min_game_length,
                "max_game_length": replay_stats.max_game_length,
                "valid_length_observations": replay_stats.valid_game_count,
                "valid_game_lengths": list(replay_stats.valid_game_lengths),
                "realised_valid_states": replay_stats.realised_valid_states,
                "excluded_length_observations": sum(
                    replay_stats.excluded_game_count_by_reason.values()
                ),
                "excluded_game_count_by_reason": (
                    replay_stats.excluded_game_count_by_reason
                ),
                "anomaly_count_by_type": replay_stats.anomaly_count_by_type,
                "scheduler_length_estimate": (
                    scheduler_decision.updated_length_estimate
                ),
                "scheduler_unclipped_games": (
                    scheduler_decision.rounded_next_games
                ),
                "scheduler_planned_games": (
                    scheduler_decision.next_iteration_games
                ),
                "scheduler_clipped": scheduler_decision.clipped,
                "scheduler_seconds": scheduler_decision.scheduler_seconds,
                "self_play_seconds": resource_record.self_play_seconds,
                "training_seconds": resource_record.training_seconds,
                "instrumentation_seconds": resource_record.instrumentation_seconds,
                "checkpoint_seconds": resource_record.checkpoint_seconds,
                "evaluation_seconds": resource_record.evaluation_seconds,
                "evaluation_training_state_preserved": (
                    evaluation_state_preserved
                ),
                "iteration_seconds": resource_record.iteration_seconds,
                "cumulative_training_seconds": (
                    resource_record.cumulative_training_seconds
                ),
                "cumulative_gpu_hours": resource_record.cumulative_gpu_hours,
                "self_play_time_fraction": resource_record.self_play_time_fraction,
                "checkpoint_path": final_model.as_posix(),
                "checkpoint_sha256": model_sha,
                "replay_state_sha256": replay_sha,
                "runtime_state_sha256": runtime_state_sha,
                "scheduler_state_sha256": scheduler_state_sha,
                "tracker_state_sha256": tracker_state_sha,
                "resource_state_sha256": resource_state_sha,
                "budget_continue_run": budget_decision.continue_run,
                "budget_stop_reason": budget_decision.stop_reason,
                "budget_crossing_iteration": (
                    budget_decision.crossing_iteration
                ),
                "budget_overshoot_gpu_hours": (
                    budget_decision.overshoot_gpu_hours
                ),
            }
        )
        required_fields = _required_metric_fields(resolution.config)
        missing_metrics = sorted(field for field in required_fields if field not in metric)
        if missing_metrics:
            raise ExperimentRuntimeError(
                "metrics record is missing required field(s): "
                + ", ".join(missing_metrics)
            )
        try:
            json.dumps(_plain(metric), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ExperimentRuntimeError(
                f"metrics iteration {iteration} is not finite JSON"
            ) from exc
        _atomic_write_json(pending_dir / "metrics_record.json", metric)

        current_gpu_hours = resource_record.cumulative_gpu_hours
        crossed_milestones = [
            float(value)
            for value in resolution.config["checkpoint"].get(
                "gpu_hour_analysis_milestones", []
            )
            if previous_gpu_hours < float(value) <= current_gpu_hours
        ]
        current_milestones = [
            value for value in crossed_milestones if current_gpu_hours <= value
        ]
        previous_milestones = [
            value for value in crossed_milestones if value not in current_milestones
        ]
        checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
        checkpoint_manifest = _read_json(checkpoint_manifest_path)
        if checkpoint_manifest.get("last_committed_iteration") != iteration - 1:
            raise ExperimentRuntimeError(
                "checkpoint manifest is not at the previous commit boundary"
            )

        if previous_milestones:
            previous_iteration = iteration - 1
            if previous_iteration == 0:
                initial_entry = next(
                    entry
                    for entry in checkpoint_manifest["checkpoints"]
                    if entry.get("iteration") == 0
                )
                initial_entry["is_milestone"] = True
                initial_entry["milestones"] = sorted(
                    set(float(value) for value in initial_entry.get("milestones", []))
                    | set(previous_milestones)
                )
            else:
                previous_model = (
                    recovery_root
                    / f"iteration_{previous_iteration:06d}"
                    / "model.pth.tar"
                )
                _upsert_archived_checkpoint(
                    checkpoint_manifest,
                    source_model=previous_model,
                    archive_dir=run_dir / "checkpoints",
                    iteration=previous_iteration,
                    actual_gpu_hours=previous_gpu_hours,
                    is_final=False,
                    milestones=previous_milestones,
                )

        cadence = int(resolution.config["checkpoint"]["save_every_iterations"])
        is_final = not budget_decision.continue_run
        current_archive_path: Path | None = None
        if (
            current_milestones
            or iteration % cadence == 0
            or is_final
            or evaluation_output_path is not None
        ):
            current_archive_path = _upsert_archived_checkpoint(
                checkpoint_manifest,
                source_model=pending_model,
                archive_dir=run_dir / "checkpoints",
                iteration=iteration,
                actual_gpu_hours=current_gpu_hours,
                is_final=is_final,
                milestones=current_milestones,
            )
        if evaluation_output_path is not None:
            assert current_archive_path is not None
            evaluation_record = _read_json(evaluation_output_path)
            evaluation_record.update(
                {
                    "iteration": iteration,
                    "checkpoint_path": current_archive_path.as_posix(),
                    "checkpoint_sha256": sha256_file(current_archive_path),
                    "training_state_preserved": True,
                }
            )
            _atomic_write_json(evaluation_output_path, evaluation_record)
        checkpoint_manifest["last_committed_iteration"] = iteration
        pending_checkpoint_manifest = pending_dir / "checkpoint_manifest.json"
        _atomic_write_json(pending_checkpoint_manifest, checkpoint_manifest)

        component_manifest = {
            "schema_version": 1,
            "iteration": iteration,
            "artifacts": {
                "model": {"path": final_model.as_posix(), "sha256": model_sha},
                "replay": {"path": final_replay.as_posix(), "sha256": replay_sha},
                "runtime_state": {
                    "path": final_runtime_state.as_posix(),
                    "sha256": runtime_state_sha,
                },
                "scheduler_state": {
                    "path": (final_dir / scheduler_state_path.name).as_posix(),
                    "sha256": scheduler_state_sha,
                },
                "tracker_state": {
                    "path": (final_dir / tracker_state_path.name).as_posix(),
                    "sha256": tracker_state_sha,
                },
                "resource_state": {
                    "path": (final_dir / resource_state_path.name).as_posix(),
                    "sha256": resource_state_sha,
                },
                "metrics_record": {
                    "path": (final_dir / "metrics_record.json").as_posix(),
                    "sha256": sha256_file(pending_dir / "metrics_record.json"),
                },
                "checkpoint_manifest": {
                    "path": (final_dir / pending_checkpoint_manifest.name).as_posix(),
                    "sha256": sha256_file(pending_checkpoint_manifest),
                },
            },
        }
        if evaluation_output_path is not None:
            component_manifest["artifacts"]["evaluation"] = {
                "path": evaluation_output_path.as_posix(),
                "sha256": sha256_file(evaluation_output_path),
            }
        _atomic_write_json(pending_dir / "commit_manifest.json", component_manifest)
        os.replace(pending_dir, final_dir)
        _atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)

        metrics_path = run_dir / "metrics.jsonl"
        existing_metrics = _read_metrics(metrics_path)
        expected_existing = list(range(1, iteration))
        if [record.get("iteration") for record in existing_metrics] != expected_existing:
            raise ExperimentRuntimeError(
                "metrics do not end at the previous committed iteration"
            )
        _atomic_write_jsonl(metrics_path, [*existing_metrics, metric])
        _atomic_write_json(
            run_dir / "tracker.json", instrumentation.state_dict()
        )
        _atomic_write_json(
            run_dir / "resource_state.json", accountant.state_dict()
        )
        _atomic_write_json(
            run_dir / "scheduler_state" / "latest.json", scheduler.state_dict()
        )

        committed_manifest = final_dir / "commit_manifest.json"
        latest_pointer = {
            "schema_version": 1,
            "iteration": iteration,
            "commit_manifest": committed_manifest.as_posix(),
            "commit_manifest_sha256": sha256_file(committed_manifest),
            "committed_at_utc": _utc_now(),
        }
        _atomic_write_json(recovery_root / "latest_commit.json", latest_pointer)
        _prune_recovery_directories(recovery_root, final_dir)
        return metric, budget_decision
    except Exception:
        if pending_dir.exists():
            shutil.rmtree(pending_dir)
        raise


def _verify_manifest_artifact(entry: object, *, label: str) -> Path:
    if not isinstance(entry, dict):
        raise ExperimentRuntimeError(f"commit manifest {label} is invalid")
    path_value = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(path_value, str):
        raise ExperimentRuntimeError(f"commit manifest {label}.path is invalid")
    path = Path(path_value).expanduser().resolve()
    _verified_input(label, path, expected)
    return path


def _load_saved_resolution(
    run_dir: Path,
    config_path: Path | None,
) -> ResolvedRuntime:
    run_dir = run_dir.expanduser().resolve()
    resolved = _load_yaml(run_dir / "resolved_config.yaml")
    if resolved.get("mode") != "adaptive" or resolved.get("status") != "frozen":
        raise ExperimentRuntimeError("saved resolved config is not a frozen Adaptive run")
    if Path(resolved["run"]["output_dir"]).resolve() != run_dir:
        raise ExperimentRuntimeError("saved run.output_dir differs from RuntimeRequest")
    manifest = _read_json(run_dir / "input_manifest.json")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ExperimentRuntimeError("input manifest inputs are invalid")
    for label, entry in inputs.items():
        _verify_manifest_artifact(entry, label=str(label))
    _verify_manifest_artifact(
        manifest.get("resolved_config"), label="resolved_config"
    )
    if config_path is not None:
        config_path = config_path.expanduser().resolve()
        adaptive_entry = inputs.get("adaptive_config", {})
        if config_path != Path(adaptive_entry.get("path", "")).resolve():
            raise ExperimentRuntimeError(
                "resume config_path differs from the original Adaptive config"
            )
    return ResolvedRuntime(resolved, manifest, run_dir)


def _load_latest_commit(
    resolution: ResolvedRuntime,
    runtime: BaselineRuntime,
) -> tuple[
    AdaptiveScheduler,
    ReplayInstrumentation,
    ResourceAccountant,
    list[dict[str, Any]],
]:
    run_dir = resolution.run_dir
    pointer = _read_json(run_dir / "recovery" / "latest_commit.json")
    iteration = pointer.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ExperimentRuntimeError("latest recovery iteration is invalid")
    manifest_path = Path(pointer.get("commit_manifest", "")).resolve()
    _verified_input(
        "commit_manifest", manifest_path, pointer.get("commit_manifest_sha256")
    )
    commit_manifest = _read_json(manifest_path)
    if commit_manifest.get("iteration") != iteration:
        raise ExperimentRuntimeError("commit manifest iteration mismatch")
    artifacts = commit_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ExperimentRuntimeError("commit manifest artifacts are invalid")
    paths = {
        label: _verify_manifest_artifact(entry, label=label)
        for label, entry in artifacts.items()
    }

    replay_payload = pickle.loads(paths["replay"].read_bytes())
    runtime_state = torch.load(
        paths["runtime_state"], map_location="cpu", weights_only=False
    )
    scheduler_state = _read_json(paths["scheduler_state"])
    tracker_state = _read_json(paths["tracker_state"])
    resource_state = _read_json(paths["resource_state"])
    committed_metric = _read_json(paths["metrics_record"])
    checkpoint_manifest = _read_json(paths["checkpoint_manifest"])
    artifact_iterations = {
        replay_payload.get("iteration") if isinstance(replay_payload, dict) else None,
        runtime_state.get("iteration") if isinstance(runtime_state, dict) else None,
        scheduler_state.get("completed_iteration"),
        tracker_state.get("completed_iteration"),
        resource_state.get("completed_iteration"),
        committed_metric.get("iteration"),
        checkpoint_manifest.get("last_committed_iteration"),
        iteration,
    }
    if "evaluation" in paths:
        artifact_iterations.add(_read_json(paths["evaluation"]).get("iteration"))
    if artifact_iterations != {iteration}:
        raise ExperimentRuntimeError(
            f"resume artifacts do not share one iteration: {artifact_iterations}"
        )

    metrics = _read_metrics(run_dir / "metrics.jsonl")
    committed_metrics = [
        record for record in metrics if int(record.get("iteration", -1)) <= iteration
    ]
    if [record.get("iteration") for record in committed_metrics] != list(
        range(1, iteration + 1)
    ):
        raise ExperimentRuntimeError("committed metrics sequence is incomplete")
    if committed_metrics[-1] != committed_metric:
        raise ExperimentRuntimeError("latest metrics record differs from commit bundle")
    if len(metrics) != len(committed_metrics):
        _atomic_write_jsonl(run_dir / "metrics.jsonl", committed_metrics)

    _atomic_write_json(run_dir / "checkpoint_manifest.json", checkpoint_manifest)

    _load_network_checkpoint(runtime.network, paths["model"])
    _restore_optimizer_state(runtime.network, runtime_state.get("optimizer_state"))
    replay_history = replay_payload.get("examples")
    if not isinstance(replay_history, list):
        raise ExperimentRuntimeError("recovery replay history is invalid")
    runtime.coach.trainExamplesHistory = replay_history
    runtime.coach.current_iteration = iteration
    runtime.coach.skipFirstSelfPlay = False

    scheduler = AdaptiveScheduler.from_state_dict(
        SchedulerConfig(**resolution.config["adaptive_scheduler"]),
        scheduler_state,
    )
    instrumentation = ReplayInstrumentation.from_state_dict(
        ReplayInstrumentationConfig(**resolution.config["replay_instrumentation"]),
        tracker_state,
    )
    accountant = ResourceAccountant.from_state_dict(
        ResourceConfig(**resolution.config["resource_accounting"]),
        resource_state,
    )
    if accountant.state.last_budget_decision is not None and not (
        accountant.state.last_budget_decision.continue_run
    ):
        raise ExperimentRuntimeError(
            f"run already reached terminal budget boundary: "
            f"{accountant.state.last_budget_decision.stop_reason}"
        )
    _atomic_write_json(run_dir / "tracker.json", instrumentation.state_dict())
    _atomic_write_json(run_dir / "resource_state.json", accountant.state_dict())
    _atomic_write_json(
        run_dir / "scheduler_state" / "latest.json", scheduler.state_dict()
    )
    _prune_recovery_directories(run_dir / "recovery", manifest_path.parent)
    _restore_rng_state(runtime_state["rng_state"])
    runtime.train_args.numEps = scheduler.next_iteration_games
    return scheduler, instrumentation, accountant, committed_metrics


def _write_summary(
    resolution: ResolvedRuntime,
    *,
    mode: str,
    status: str,
    stop_reason: str | None,
) -> Path:
    metrics = _read_metrics(resolution.run_dir / "metrics.jsonl")
    final = metrics[-1] if metrics else None
    return _atomic_write_json(
        resolution.run_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": resolution.config["run"]["id"],
            "config_id": resolution.config["config_id"],
            "mode": mode,
            "status": status,
            "stop_reason": stop_reason,
            "completed_iterations": [record["iteration"] for record in metrics],
            "final_iteration": final["iteration"] if final else 0,
            "final_cumulative_gpu_hours": (
                final["cumulative_gpu_hours"] if final else 0.0
            ),
            "final_checkpoint": final["checkpoint_path"] if final else None,
            "updated_at_utc": _utc_now(),
            "resume_boundary": "latest committed complete iteration only",
        },
    )


def _run_training(
    resolution: ResolvedRuntime,
    runtime: BaselineRuntime,
    *,
    mode: str,
    evaluation_runner: Callable[
        [dict[str, Any], BaselineRuntime, Path, Path], float
    ]
    | None,
) -> dict[str, Any]:
    resolved = resolution.config
    if mode == "fresh":
        scheduler = AdaptiveScheduler(
            SchedulerConfig(**resolved["adaptive_scheduler"])
        )
        instrumentation = ReplayInstrumentation(
            ReplayInstrumentationConfig(**resolved["replay_instrumentation"])
        )
        accountant = ResourceAccountant(
            ResourceConfig(**resolved["resource_accounting"])
        )
        runtime.coach.trainExamplesHistory = []
        runtime.coach.current_iteration = 0
        _atomic_write_json(
            resolution.run_dir / "tracker.json", instrumentation.state_dict()
        )
        _atomic_write_json(
            resolution.run_dir / "resource_state.json", accountant.state_dict()
        )
        _atomic_write_json(
            resolution.run_dir / "scheduler_state" / "latest.json",
            scheduler.state_dict(),
        )
    else:
        try:
            scheduler, instrumentation, accountant, _ = _load_latest_commit(
                resolution, runtime
            )
        except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
            raise ResumeStateError(str(exc)) from exc

    runtime.train_args.numIters = int(resolved["budget"]["max_iterations"]) + 1
    runtime.train_args.numEps = scheduler.next_iteration_games
    runtime.coach.save_every_n_iterations = runtime.train_args.numIters + 1
    runtime.coach.iteration_callback = None
    runtime.coach.saveTrainExamples = lambda _iteration: None

    observer_seconds = {"value": 0.0}
    original_execute_episode = runtime.coach.executeEpisode

    def observed_execute_episode():
        episode_examples = original_execute_episode()
        started = time.perf_counter()
        instrumentation.observe_episode(episode_examples)
        observer_seconds["value"] += time.perf_counter() - started
        return episode_examples

    runtime.coach.executeEpisode = observed_execute_episode
    first_iteration = scheduler.state.completed_iteration + 1
    instrumentation.begin_iteration(first_iteration, scheduler.next_iteration_games)
    accountant.begin_iteration(first_iteration)
    terminal_decision = {"value": None}

    def on_iteration(coach_metrics: dict[str, Any]) -> None:
        iteration = int(coach_metrics["iteration"])
        games_planned = int(runtime.train_args.numEps)
        instrumentation_started = time.perf_counter()
        replay_stats = instrumentation.finalize_iteration(coach_metrics)
        scheduler_decision = scheduler.update(
            IterationLengthStats(
                iteration=iteration,
                valid_game_lengths=replay_stats.valid_game_lengths,
                excluded_game_count_by_reason=(
                    replay_stats.excluded_game_count_by_reason
                ),
            )
        )
        component_seconds = time.perf_counter() - instrumentation_started
        total_instrumentation_seconds = observer_seconds["value"] + component_seconds
        _, budget_decision = commit_iteration(
            resolution=resolution,
            runtime=runtime,
            scheduler=scheduler,
            instrumentation=instrumentation,
            accountant=accountant,
            iteration=iteration,
            games_planned=games_planned,
            coach_metrics=coach_metrics,
            replay_stats=replay_stats,
            scheduler_decision=scheduler_decision,
            instrumentation_seconds=total_instrumentation_seconds,
            observer_seconds=observer_seconds["value"],
            evaluation_runner=evaluation_runner,
        )
        _append_run_log(
            resolution.run_dir,
            f"committed iteration={iteration} gpu_hours="
            f"{accountant.state.cumulative_gpu_hours:.12f} "
            f"continue={budget_decision.continue_run}",
        )
        if not budget_decision.continue_run:
            terminal_decision["value"] = budget_decision
            raise _AdaptiveStop

        next_iteration = iteration + 1
        runtime.train_args.numEps = scheduler.next_iteration_games
        observer_seconds["value"] = 0.0
        instrumentation.begin_iteration(
            next_iteration, scheduler.next_iteration_games
        )
        accountant.begin_iteration(next_iteration)

    runtime.coach.iteration_callback = on_iteration
    try:
        runtime.coach.learn()
    except _AdaptiveStop:
        pass
    except Exception:
        instrumentation.abort_iteration()
        accountant.abort_iteration()
        raise
    finally:
        runtime.coach.executeEpisode = original_execute_episode

    decision = terminal_decision["value"]
    if decision is None:
        raise ExperimentRuntimeError(
            "Coach returned without reaching a complete budget boundary"
        )
    summary_path = _write_summary(
        resolution,
        mode=mode,
        status="completed",
        stop_reason=decision.stop_reason,
    )
    _append_run_log(
        resolution.run_dir,
        f"Adaptive run completed stop_reason={decision.stop_reason}",
    )
    return {
        "mode": mode,
        "status": "completed",
        "stop_reason": decision.stop_reason,
        "run_dir": resolution.run_dir,
        "summary_path": summary_path,
    }


def run_experiment(
    request: RuntimeRequest,
    *,
    runtime_builder: Callable[[dict[str, Any]], BaselineRuntime] = build_baseline_runtime,
    evaluation_runner: Callable[
        [dict[str, Any], BaselineRuntime, Path, Path], float
    ]
    | None = None,
) -> dict[str, Any]:
    if request.mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")
    run_dir = Path(request.run_dir).expanduser().resolve()

    if request.mode in {"dry-run", "fresh"}:
        if request.config_path is None:
            raise ValueError("config_path is required for dry-run and fresh modes")
        try:
            resolution = resolve_adaptive_protocol(request.config_path, run_dir)
        except (ExperimentRuntimeError, OSError, ValueError, KeyError) as exc:
            raise ProtocolValidationError(str(exc)) from exc
        if request.mode == "dry-run":
            return {
                "mode": "dry-run",
                "status": "validated",
                "resolved_config": resolution.config,
                "input_manifest": resolution.input_manifest,
                "files_written": False,
            }
        _ensure_fresh_output(run_dir)
        git = _git_metadata(SOURCE_ROOT)
        if (
            resolution.config["protocol_requirements"]["require_clean_worktree"]
            and git["dirty"]
        ):
            raise ProtocolValidationError(
                "fresh Adaptive run requires a clean worktree"
            )
        _initialise_device(resolution.config)
        _set_seed(
            int(resolution.config["run"]["seed"]),
            bool(resolution.config["run"].get("deterministic", False)),
        )
        runtime = runtime_builder(resolution.config)
        if not runtime.initial_weights_loaded:
            _load_initial_weights(runtime, resolution.config)
        _write_initial_outputs(resolution)
        return _run_training(
            resolution,
            runtime,
            mode="fresh",
            evaluation_runner=evaluation_runner,
        )

    try:
        resolution = _load_saved_resolution(run_dir, request.config_path)
    except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
        raise ResumeStateError(str(exc)) from exc
    _initialise_device(resolution.config)
    runtime = runtime_builder(resolution.config)
    _append_run_log(run_dir, "resume validation started")
    return _run_training(
        resolution,
        runtime,
        mode="resume",
        evaluation_runner=evaluation_runner,
    )


def parse_args(argv: list[str] | None = None) -> RuntimeRequest:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(_MODES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode in {"dry-run", "fresh"} and args.config is None:
        parser.error("dry-run and fresh require --config")
    return RuntimeRequest(args.mode, args.config, args.run_dir)


def main(argv: list[str] | None = None) -> int:
    request = parse_args(argv)
    try:
        result = run_experiment(request)
    except (ExperimentRuntimeError, OSError, ValueError, pickle.PickleError) as exc:
        print(f"Adaptive runtime error: {exc}", file=sys.stderr)
        return 2
    if request.mode == "dry-run":
        print(yaml.safe_dump(result["resolved_config"], sort_keys=False))
        print("Adaptive dry-run validation passed; no files were written.")
    else:
        print(f"Adaptive run status: {result['status']}")
        print(f"Summary: {result['summary_path']}")
    return 0


__all__ = [
    "BaselineRuntime",
    "ExperimentRuntimeError",
    "ProtocolValidationError",
    "ResolvedRuntime",
    "ResumeStateError",
    "RuntimeRequest",
    "build_baseline_runtime",
    "commit_iteration",
    "configured_run_dir",
    "resolve_adaptive_protocol",
    "run_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
