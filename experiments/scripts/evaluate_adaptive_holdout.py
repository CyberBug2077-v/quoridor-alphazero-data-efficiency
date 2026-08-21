#!/usr/bin/env python3
"""Evaluate matched-compute Adaptive checkpoints on fixed_holdout_v1."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
BASELINE_ANALYSIS_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for import_root in (SOURCE_ROOT, BASELINE_ANALYSIS_SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# The loss definition and the whole-trajectory bootstrap are authoritative here.
from evaluate_holdout import (  # noqa: E402
    _read_training_metrics,
    _training_columns,
    cluster_bootstrap_intervals,
    loss_arrays,
    trajectory_rows,
)
from holdout_common import (  # noqa: E402
    HoldoutError,
    build_network,
    load_npz,
    load_protocol,
    sha256_file,
)
from verify_holdout import verify_holdout  # noqa: E402


DEFAULT_CONFIG = EXPERIMENTS_ROOT / "configs" / "adaptive_holdout_v2.yaml"

CHECKPOINT_FIELDS = (
    "target_baseline_iteration",
    "target_gpu_hours",
    "selected_adaptive_iteration",
    "selected_adaptive_gpu_hours",
    "states",
    "trajectories",
    "holdout_policy_loss",
    "holdout_policy_loss_ci_low",
    "holdout_policy_loss_ci_high",
    "holdout_value_loss",
    "holdout_value_loss_ci_low",
    "holdout_value_loss_ci_high",
    "holdout_total_loss",
    "holdout_total_loss_ci_low",
    "holdout_total_loss_ci_high",
    "logged_train_policy_loss",
    "logged_train_value_loss",
    "logged_train_total_loss",
    "approx_policy_gap",
    "approx_value_gap",
    "approx_total_gap",
    "checkpoint_sha256",
    "dataset_content_sha256",
)

TRAJECTORY_FIELDS = (
    "target_baseline_iteration",
    "target_gpu_hours",
    "selected_adaptive_iteration",
    "selected_adaptive_gpu_hours",
    "game_id",
    "states",
    "policy_loss",
    "value_loss",
    "total_loss",
)


class AdaptiveHoldoutError(HoldoutError):
    """Raised when Adaptive holdout inputs violate the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveHoldoutError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AdaptiveHoldoutError(f"invalid {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must contain a mapping")
    return payload


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveHoldoutError(f"invalid {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


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
                writer.writerow({field: row[field] for field in fields})
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


def _matched_targets(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        raw_targets = protocol["pairing_and_randomness"]["checkpoint_grid"]["targets"]
    except (KeyError, TypeError) as exc:
        raise AdaptiveHoldoutError(
            "matched-compute protocol lacks pairing_and_randomness.checkpoint_grid.targets"
        ) from exc
    _require(isinstance(raw_targets, list) and raw_targets, "matched targets are empty")
    targets: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_targets):
        _require(isinstance(raw, dict), f"matched target {index} is not a mapping")
        iteration = raw.get("baseline_checkpoint_iteration")
        gpu_hours = raw.get("gpu_hours")
        _require(_is_int(iteration) and iteration >= 0, f"matched target {index} has invalid iteration")
        _require(_finite_number(gpu_hours) and float(gpu_hours) >= 0.0, f"matched target {index} has invalid GPU-hours")
        targets.append(
            {
                "target_baseline_iteration": int(iteration),
                "target_gpu_hours": float(gpu_hours),
            }
        )
    _require(
        all(
            targets[index]["target_gpu_hours"]
            < targets[index + 1]["target_gpu_hours"]
            for index in range(len(targets) - 1)
        ),
        "matched GPU-hour targets are not strictly increasing",
    )
    return targets


def _checkpoint_registry(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = payload.get("checkpoints")
    _require(isinstance(raw_entries, list) and raw_entries, "checkpoint registry is empty")
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(raw_entries):
        _require(isinstance(raw, dict), f"checkpoint registry entry {index} is invalid")
        iteration = raw.get("iteration")
        gpu_hours = raw.get("actual_gpu_hours")
        digest = raw.get("sha256")
        path = raw.get("path")
        _require(_is_int(iteration) and iteration >= 0, f"checkpoint entry {index} has invalid iteration")
        _require(iteration not in seen, f"duplicate checkpoint iteration {iteration}")
        _require(_finite_number(gpu_hours) and float(gpu_hours) >= 0.0, f"checkpoint {iteration} has invalid GPU-hours")
        _require(isinstance(digest, str) and len(digest) == 64, f"checkpoint {iteration} has invalid SHA-256")
        _require(isinstance(path, str) and path, f"checkpoint {iteration} has invalid path")
        seen.add(iteration)
        entries.append(
            {
                "iteration": int(iteration),
                "actual_gpu_hours": float(gpu_hours),
                "sha256": digest.lower(),
                "path": path,
            }
        )
    entries.sort(key=lambda item: item["iteration"])
    _require(
        all(
            entries[index]["actual_gpu_hours"]
            < entries[index + 1]["actual_gpu_hours"]
            for index in range(len(entries) - 1)
        ),
        "checkpoint GPU-hours are not strictly increasing by iteration",
    )
    return entries


def select_checkpoints(
    targets: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Map each Baseline target to the latest completed Adaptive checkpoint."""
    selected: list[dict[str, Any]] = []
    for target in targets:
        eligible = [
            entry
            for entry in entries
            if entry["actual_gpu_hours"] <= target["target_gpu_hours"]
        ]
        _require(
            bool(eligible),
            f"no Adaptive checkpoint is available by {target['target_gpu_hours']} GPU-hours",
        )
        checkpoint = max(eligible, key=lambda item: item["iteration"])
        selected.append({**target, **checkpoint})
    return selected


def _resolve_checkpoint_path(
    entry: dict[str, Any], run_dir: Path, checkpoint_zero_path: Path
) -> Path:
    if entry["iteration"] == 0:
        return checkpoint_zero_path
    recorded = Path(entry["path"])
    candidates = [recorded, run_dir / "checkpoints" / recorded.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise AdaptiveHoldoutError(
        f"checkpoint {entry['iteration']} cannot be resolved below {run_dir}"
    )


def _validate_training_alignment(
    selected: list[dict[str, Any]], training: dict[int, dict[str, Any]]
) -> None:
    for entry in {item["iteration"]: item for item in selected}.values():
        if entry["iteration"] == 0:
            continue
        row = training.get(entry["iteration"])
        _require(row is not None, f"training metrics lack iteration {entry['iteration']}")
        gpu_hours = row.get("cumulative_gpu_hours")
        _require(
            _finite_number(gpu_hours)
            and float(gpu_hours) == entry["actual_gpu_hours"],
            f"checkpoint and metrics GPU-hours differ at iteration {entry['iteration']}",
        )


def _validate_model_identity(
    holdout_protocol: dict[str, Any], adaptive_resolved: dict[str, Any]
) -> None:
    adaptive_model = adaptive_resolved.get("model")
    _require(isinstance(adaptive_model, dict), "Adaptive resolved config lacks model")
    for key in (
        "board_size",
        "num_channels",
        "num_res_blocks",
        "attn_depth",
        "num_heads",
        "se_enabled",
    ):
        _require(
            adaptive_model.get(key) == holdout_protocol["model"].get(key),
            f"Adaptive model.{key} differs from the holdout source protocol",
        )


def _assert_all_finite(rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float):
                _require(math.isfinite(value), f"row {row_index} field {key} is non-finite")


def _resolved_protocol(
    config: dict[str, Any],
    *,
    config_path: Path,
    matched_compute_path: Path,
    run_dir: Path,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["runtime"] = {
        "config_path": config_path.as_posix(),
        "matched_compute_path": matched_compute_path.as_posix(),
        "adaptive_run_root": run_dir.as_posix(),
        "holdout_dataset_path": dataset_path.as_posix(),
        "holdout_manifest_path": manifest_path.as_posix(),
        "output_root": output_dir.as_posix(),
        "device": device,
    }
    return resolved


def evaluate(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path, "Adaptive holdout protocol")
    _require(config.get("config_id") == "adaptive_holdout_v2", "unexpected config_id")

    matched_compute_path = (
        args.matched_compute.expanduser().resolve()
        if args.matched_compute is not None
        else _resolve_experiments_path(config["checkpoint_alignment"]["contract"])
    )
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else _resolve_experiments_path(config["evaluated_condition"]["run_root"])
    )
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else _resolve_experiments_path(config["holdout"]["dataset_path"])
    )
    manifest_path = (
        args.holdout_manifest.expanduser().resolve()
        if args.holdout_manifest is not None
        else _resolve_experiments_path(config["holdout"]["manifest_path"])
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_experiments_path(config["outputs"]["root"])
    )
    source_protocol_path = _resolve_experiments_path(config["source_protocol"]["config"])
    checkpoint_zero_path = _resolve_experiments_path(
        config["checkpoint_alignment"]["checkpoint_0"]["path"]
    )
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    adaptive_resolved_path = run_dir / "resolved_config.yaml"

    _require(sha256_file(config_path) != "", "Adaptive holdout protocol hash failed")
    _require(
        sha256_file(source_protocol_path)
        == config["source_protocol"]["expected_sha256"],
        "source holdout protocol SHA-256 differs from adaptive_holdout_v2.yaml",
    )
    _require(
        sha256_file(matched_compute_path)
        == config["checkpoint_alignment"]["contract_sha256"],
        "matched-compute protocol SHA-256 differs from adaptive_holdout_v2.yaml",
    )
    _require(
        manifest_path.parent == dataset_path.parent,
        "holdout dataset and manifest must belong to the same frozen directory",
    )
    _require(
        sha256_file(manifest_path) == config["holdout"]["manifest_sha256"],
        "holdout manifest SHA-256 differs from adaptive_holdout_v2.yaml",
    )
    _require(
        sha256_file(dataset_path) == config["holdout"]["expected_file_sha256"],
        "holdout states.npz SHA-256 differs from adaptive_holdout_v2.yaml",
    )

    holdout_protocol = load_protocol(source_protocol_path)
    verification = verify_holdout(
        source_protocol_path, dataset_path.parent, dataset_path=dataset_path
    )
    _require(verification["status"] == "passed", "holdout verification did not pass")
    _require(
        verification["games"] == int(config["holdout"]["expected_games"]),
        "holdout game count differs from adaptive_holdout_v2.yaml",
    )
    _require(
        verification["states"] == int(config["holdout"]["expected_states"]),
        "holdout state count differs from adaptive_holdout_v2.yaml",
    )
    _require(
        verification["dataset_content_sha256"]
        == config["holdout"]["expected_content_sha256"],
        "holdout logical content SHA-256 differs from adaptive_holdout_v2.yaml",
    )

    matched_compute = _load_yaml(matched_compute_path, "matched-compute protocol")
    targets = _matched_targets(matched_compute)
    _require(
        len(targets) == int(config["checkpoint_alignment"]["expected_target_count"]),
        "matched-compute target count differs from adaptive_holdout_v2.yaml",
    )
    checkpoint_manifest = _load_json(checkpoint_manifest_path, "checkpoint manifest")
    registry = _checkpoint_registry(checkpoint_manifest)
    selected = select_checkpoints(targets, registry)
    _require(selected[0]["iteration"] == 0, "the zero-GPU-hour target did not select checkpoint 0")

    expected_checkpoint_zero_sha = config["checkpoint_alignment"]["checkpoint_0"][
        "sha256"
    ]
    _require(
        selected[0]["sha256"] == expected_checkpoint_zero_sha,
        "checkpoint manifest iteration 0 SHA-256 differs from the protocol",
    )
    _require(
        verification["source_checkpoint_sha256"] == expected_checkpoint_zero_sha,
        "checkpoint 0 differs from the checkpoint used to generate the holdout",
    )

    adaptive_resolved = _load_yaml(adaptive_resolved_path, "Adaptive resolved config")
    _require(
        adaptive_resolved.get("run", {}).get("id")
        == config["evaluated_condition"]["run_id"],
        "Adaptive resolved run id differs from adaptive_holdout_v2.yaml",
    )
    _validate_model_identity(holdout_protocol, adaptive_resolved)
    training = _read_training_metrics(metrics_path)
    _validate_training_alignment(selected, training)

    selected_paths: dict[int, Path] = {}
    selected_entries = {item["iteration"]: item for item in selected}
    for iteration, entry in selected_entries.items():
        checkpoint_path = _resolve_checkpoint_path(entry, run_dir, checkpoint_zero_path)
        _require(
            sha256_file(checkpoint_path) == entry["sha256"],
            f"checkpoint {iteration} SHA-256 differs from checkpoint_manifest.json",
        )
        selected_paths[iteration] = checkpoint_path

    arrays = load_npz(dataset_path)
    inference_protocol = copy.deepcopy(holdout_protocol)
    inference_protocol["evaluation"]["batch_size"] = int(
        config["evaluation"]["batch_size"]
    )
    evaluation_cache: dict[int, dict[str, Any]] = {}
    log_lines = [
        "adaptive_holdout_v2 evaluation",
        f"holdout verification: passed ({verification['states']} states, {verification['games']} trajectories)",
        f"matched targets: {len(targets)}",
        f"unique selected checkpoints: {len(selected_entries)}",
    ]
    for iteration in sorted(selected_entries):
        checkpoint_path = selected_paths[iteration]
        _, network = build_network(inference_protocol, checkpoint_path, device=args.device)
        policy_losses, value_losses, total_losses = loss_arrays(
            network,
            arrays["boards"],
            arrays["policies"],
            arrays["values"],
            arrays["valids"],
            batch_size=int(config["evaluation"]["batch_size"]),
            device=args.device,
        )
        game_rows = trajectory_rows(
            iteration,
            arrays["game_ids"],
            policy_losses,
            value_losses,
            total_losses,
        )
        _require(
            len(game_rows) == int(config["holdout"]["expected_games"]),
            f"checkpoint {iteration} did not produce 200 trajectory rows",
        )
        intervals = cluster_bootstrap_intervals(
            game_rows,
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + iteration,
            confidence_level=float(config["evaluation"]["confidence_level"]),
        )
        means = {
            "policy_loss": float(policy_losses.mean()),
            "value_loss": float(value_losses.mean()),
            "total_loss": float(total_losses.mean()),
        }
        training_columns = _training_columns(iteration, training, means)
        logged_policy = training_columns["logged_train_policy_loss"]
        logged_value = training_columns["logged_train_value_loss"]
        evaluation_cache[iteration] = {
            "means": means,
            "intervals": intervals,
            "trajectory_rows": game_rows,
            "logged_train_policy_loss": logged_policy,
            "logged_train_value_loss": logged_value,
            "logged_train_total_loss": (
                None
                if logged_policy is None or logged_value is None
                else float(logged_policy + logged_value)
            ),
            "approx_policy_gap": training_columns["approx_policy_gap"],
            "approx_value_gap": training_columns["approx_value_gap"],
            "approx_total_gap": (
                None
                if training_columns["approx_policy_gap"] is None
                or training_columns["approx_value_gap"] is None
                else float(
                    training_columns["approx_policy_gap"]
                    + training_columns["approx_value_gap"]
                )
            ),
        }
        log_lines.append(
            f"checkpoint {iteration}: policy={means['policy_loss']:.12g}, value={means['value_loss']:.12g}"
        )
        del network, policy_losses, value_losses, total_losses
        if args.device == "cuda":
            torch.cuda.empty_cache()

    checkpoint_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    for mapping in selected:
        cached = evaluation_cache[mapping["iteration"]]
        means = cached["means"]
        intervals = cached["intervals"]
        checkpoint_rows.append(
            {
                "target_baseline_iteration": mapping["target_baseline_iteration"],
                "target_gpu_hours": mapping["target_gpu_hours"],
                "selected_adaptive_iteration": mapping["iteration"],
                "selected_adaptive_gpu_hours": mapping["actual_gpu_hours"],
                "states": int(arrays["boards"].shape[0]),
                "trajectories": int(verification["games"]),
                "holdout_policy_loss": means["policy_loss"],
                "holdout_policy_loss_ci_low": intervals["policy_loss"][0],
                "holdout_policy_loss_ci_high": intervals["policy_loss"][1],
                "holdout_value_loss": means["value_loss"],
                "holdout_value_loss_ci_low": intervals["value_loss"][0],
                "holdout_value_loss_ci_high": intervals["value_loss"][1],
                "holdout_total_loss": means["total_loss"],
                "holdout_total_loss_ci_low": intervals["total_loss"][0],
                "holdout_total_loss_ci_high": intervals["total_loss"][1],
                "logged_train_policy_loss": cached["logged_train_policy_loss"],
                "logged_train_value_loss": cached["logged_train_value_loss"],
                "logged_train_total_loss": cached["logged_train_total_loss"],
                "approx_policy_gap": cached["approx_policy_gap"],
                "approx_value_gap": cached["approx_value_gap"],
                "approx_total_gap": cached["approx_total_gap"],
                "checkpoint_sha256": mapping["sha256"],
                "dataset_content_sha256": verification["dataset_content_sha256"],
            }
        )
        for game_row in cached["trajectory_rows"]:
            all_trajectory_rows.append(
                {
                    "target_baseline_iteration": mapping["target_baseline_iteration"],
                    "target_gpu_hours": mapping["target_gpu_hours"],
                    "selected_adaptive_iteration": mapping["iteration"],
                    "selected_adaptive_gpu_hours": mapping["actual_gpu_hours"],
                    "game_id": game_row["game_id"],
                    "states": game_row["states"],
                    "policy_loss": game_row["policy_loss"],
                    "value_loss": game_row["value_loss"],
                    "total_loss": game_row["total_loss"],
                }
            )
    _assert_all_finite(checkpoint_rows)
    _assert_all_finite(all_trajectory_rows)
    _require(len(checkpoint_rows) == 12, "evaluation did not produce 12 target rows")
    _require(
        len(all_trajectory_rows) == 12 * int(verification["games"]),
        "trajectory output does not contain 200 rows per target",
    )

    selected_registry = [
        {
            "target_baseline_iteration": item["target_baseline_iteration"],
            "target_gpu_hours": item["target_gpu_hours"],
            "selected_adaptive_iteration": item["iteration"],
            "selected_adaptive_gpu_hours": item["actual_gpu_hours"],
            "checkpoint_path": selected_paths[item["iteration"]].as_posix(),
            "checkpoint_sha256": item["sha256"],
            "selection_rule": "latest completed checkpoint with actual_gpu_hours <= target_gpu_hours",
        }
        for item in selected
    ]
    input_manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "inputs": {
            "adaptive_holdout_protocol": {
                "path": config_path.as_posix(),
                "sha256": sha256_file(config_path),
            },
            "matched_compute_protocol": {
                "path": matched_compute_path.as_posix(),
                "sha256": sha256_file(matched_compute_path),
            },
            "source_holdout_protocol": {
                "path": source_protocol_path.as_posix(),
                "sha256": sha256_file(source_protocol_path),
            },
            "holdout_dataset": {
                "path": dataset_path.as_posix(),
                "sha256": sha256_file(dataset_path),
                "content_sha256": verification["dataset_content_sha256"],
            },
            "holdout_manifest": {
                "path": manifest_path.as_posix(),
                "sha256": sha256_file(manifest_path),
            },
            "adaptive_resolved_config": {
                "path": adaptive_resolved_path.as_posix(),
                "sha256": sha256_file(adaptive_resolved_path),
            },
            "adaptive_checkpoint_manifest": {
                "path": checkpoint_manifest_path.as_posix(),
                "sha256": sha256_file(checkpoint_manifest_path),
            },
            "adaptive_training_metrics": {
                "path": metrics_path.as_posix(),
                "sha256": sha256_file(metrics_path),
            },
            "selected_checkpoints": selected_registry,
        },
    }
    evaluation_manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "status": "completed",
        "selection_rule": "latest completed Adaptive checkpoint not after each Baseline GPU-hour target",
        "parameter_interpolation": False,
        "target_count": len(selected),
        "unique_checkpoint_count": len(selected_entries),
        "checkpoint_0_identity_verified": True,
        "checkpoint_sha256_verified": True,
        "holdout_verification": verification,
        "selected_checkpoints": selected_registry,
    }
    summary = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "status": "completed",
        "acceptance": {
            "target_count": len(checkpoint_rows),
            "all_selected_checkpoints_not_after_target": all(
                row["selected_adaptive_gpu_hours"] <= row["target_gpu_hours"]
                for row in checkpoint_rows
            ),
            "states_per_target": int(arrays["boards"].shape[0]),
            "trajectories_per_target": int(verification["games"]),
            "checkpoint_0_identity_verified": True,
            "all_metrics_finite": True,
            "deterministic_bootstrap": True,
        },
        "target_count": len(checkpoint_rows),
        "unique_checkpoint_count": len(selected_entries),
        "bootstrap": {
            "implementation": "baseline/analysis/scripts/evaluate_holdout.py:cluster_bootstrap_intervals",
            "resamples": int(config["evaluation"]["bootstrap_resamples"]),
            "seed": int(config["evaluation"]["bootstrap_seed"]),
            "seed_per_checkpoint": "seed + selected_adaptive_iteration",
            "confidence_level": float(config["evaluation"]["confidence_level"]),
            "cluster": "complete holdout game trajectory",
        },
        "loss_implementation": "baseline/analysis/scripts/evaluate_holdout.py:loss_arrays",
        "structured_outputs_are_deterministic": True,
    }
    resolved = _resolved_protocol(
        config,
        config_path=config_path,
        matched_compute_path=matched_compute_path,
        run_dir=run_dir,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        device=args.device,
    )
    log_lines.extend(
        [
            f"target rows: {len(checkpoint_rows)}",
            f"trajectory rows: {len(all_trajectory_rows)}",
            "checkpoint 0 identity: passed",
            "non-finite metrics: 0",
            "status: completed",
        ]
    )

    _atomic_write_yaml(output_dir / "resolved_config.yaml", resolved)
    _atomic_write_json(output_dir / "input_manifest.json", input_manifest)
    _atomic_write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    _atomic_write_csv(output_dir / "checkpoint_metrics.csv", CHECKPOINT_FIELDS, checkpoint_rows)
    _atomic_write_csv(
        output_dir / "trajectory_checkpoint_metrics.csv",
        TRAJECTORY_FIELDS,
        all_trajectory_rows,
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_text(output_dir / "evaluation.log", log_lines)

    print(f"Targets evaluated: {len(checkpoint_rows)}")
    print(f"Unique checkpoints inferred: {len(selected_entries)}")
    print(f"States per target: {arrays['boards'].shape[0]}")
    print(f"Trajectories per target: {verification['games']}")
    print(f"Outputs: {output_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matched-compute", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        evaluate(parse_args(argv))
    except (AdaptiveHoldoutError, HoldoutError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
