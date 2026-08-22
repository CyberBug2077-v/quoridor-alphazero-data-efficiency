#!/usr/bin/env python3
"""Run and summarize the matched Adaptive fixed basket and joint Elo fit."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
BASELINE_ANALYSIS_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for import_root in (SOURCE_ROOT, BASELINE_ANALYSIS_SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# These functions are the authoritative Baseline Arena, seed, persistence, score,
# bootstrap, and provisional-Elo implementations.  This script only adapts the
# target checkpoint registry and adds the requested joint fit.
from evaluate_fixed_basket import (  # noqa: E402
    EvaluationLogger,
    FixedBasketError,
    check_js_determinism,
    evaluate_matchups,
    load_protocol,
    stable_seed,
)
from summarize_fixed_basket import (  # noqa: E402
    _load_games,
    summarize_results,
)


DEFAULT_CONFIG = EXPERIMENTS_ROOT / "configs" / "adaptive_fixed_basket_v2.yaml"
DEFAULT_OUTPUT = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "adaptive_seed1001_4090_v2_analysis"
    / "fixed_basket_v2"
)

JOINT_ELO_FIELDS = (
    "target_baseline_iteration",
    "target_gpu_hours",
    "selected_adaptive_iteration",
    "selected_adaptive_gpu_hours",
    "baseline_participant",
    "adaptive_participant",
    "baseline_joint_elo",
    "baseline_joint_elo_ci95_low",
    "baseline_joint_elo_ci95_high",
    "adaptive_joint_elo",
    "adaptive_joint_elo_ci95_low",
    "adaptive_joint_elo_ci95_high",
    "adaptive_minus_baseline_elo",
    "adaptive_minus_baseline_elo_ci95_low",
    "adaptive_minus_baseline_elo_ci95_high",
    "fit_id",
    "fit_scope",
    "fit_method",
    "bootstrap_iterations",
    "bootstrap_seed",
)


class JointBasketError(FixedBasketError):
    """Raised when a joint fixed-basket protocol invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointBasketError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), f"{label} must contain a mapping")
    return loaded


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JointBasketError(f"invalid {label}: {exc}") from exc
    _require(isinstance(loaded, dict), f"{label} must contain an object")
    return loaded


def _load_csv(path: Path, label: str) -> list[dict[str, str]]:
    _require(path.is_file(), f"{label} not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        encoded = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path, rows: list[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_experiments_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (EXPERIMENTS_ROOT / path).resolve()


def _expected_hash(path: Path, expected: Any, label: str) -> str:
    _require(path.is_file(), f"{label} not found: {path}")
    actual = sha256_file(path)
    if expected is not None:
        _require(
            isinstance(expected, str) and actual == expected.lower(),
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}",
        )
    return actual


def matched_targets(matched_compute: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen Baseline target grid from matched_compute_v1."""
    try:
        raw = matched_compute["pairing_and_randomness"]["checkpoint_grid"]["targets"]
    except (KeyError, TypeError) as exc:
        raise JointBasketError("matched-compute target grid is missing") from exc
    _require(isinstance(raw, list) and raw, "matched-compute target grid is empty")
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        _require(isinstance(item, dict), f"matched target {index} is invalid")
        iteration = item.get("baseline_checkpoint_iteration")
        gpu_hours = item.get("gpu_hours")
        _require(_is_int(iteration), f"matched target {index} iteration is invalid")
        _require(
            _finite_number(gpu_hours) and float(gpu_hours) >= 0.0,
            f"matched target {iteration} GPU-hours are invalid",
        )
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
        "matched GPU-hour targets must be strictly increasing",
    )
    return targets


def checkpoint_registry(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("checkpoints")
    _require(isinstance(raw, list) and raw, "Adaptive checkpoint registry is empty")
    registry: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        _require(isinstance(item, dict), f"checkpoint registry entry {index} is invalid")
        iteration = item.get("iteration")
        gpu_hours = item.get("actual_gpu_hours")
        digest = item.get("sha256")
        recorded_path = item.get("path")
        _require(_is_int(iteration) and iteration >= 0, f"checkpoint {index} iteration is invalid")
        _require(iteration not in seen, f"duplicate Adaptive checkpoint iteration {iteration}")
        _require(
            _finite_number(gpu_hours) and float(gpu_hours) >= 0.0,
            f"checkpoint {iteration} GPU-hours are invalid",
        )
        _require(
            isinstance(digest, str) and len(digest) == 64,
            f"checkpoint {iteration} SHA-256 is invalid",
        )
        _require(isinstance(recorded_path, str) and recorded_path, f"checkpoint {iteration} path is invalid")
        seen.add(int(iteration))
        registry.append(
            {
                "iteration": int(iteration),
                "actual_gpu_hours": float(gpu_hours),
                "sha256": digest.lower(),
                "recorded_path": recorded_path,
            }
        )
    registry.sort(key=lambda item: item["iteration"])
    _require(
        all(
            registry[index]["actual_gpu_hours"]
            < registry[index + 1]["actual_gpu_hours"]
            for index in range(len(registry) - 1)
        ),
        "Adaptive checkpoint GPU-hours must increase with iteration",
    )
    return registry


def select_adaptive_checkpoints(
    targets: list[dict[str, Any]], registry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select the latest completed checkpoint not after each GPU-hour target."""
    selected: list[dict[str, Any]] = []
    for target in targets:
        eligible = [
            item
            for item in registry
            if item["actual_gpu_hours"] <= target["target_gpu_hours"]
        ]
        _require(
            bool(eligible),
            f"no Adaptive checkpoint is available by {target['target_gpu_hours']} GPU-hours",
        )
        checkpoint = max(eligible, key=lambda item: item["iteration"])
        selected.append({**target, **checkpoint})
    return selected


def _resolve_checkpoint_path(
    item: dict[str, Any], run_dir: Path, checkpoint_zero: Path
) -> Path:
    if item["iteration"] == 0:
        candidates = [checkpoint_zero, Path(item["recorded_path"])]
    else:
        recorded = Path(item["recorded_path"])
        candidates = [recorded, run_dir / "checkpoints" / recorded.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise JointBasketError(
        f"Adaptive checkpoint {item['iteration']} cannot be resolved below {run_dir}"
    )


@dataclass(frozen=True)
class BasketContext:
    config_path: Path
    config: dict[str, Any]
    matched_path: Path
    matched: dict[str, Any]
    base_protocol_path: Path
    base_protocol: dict[str, Any]
    run_dir: Path
    checkpoint_manifest_path: Path
    checkpoint_manifest: dict[str, Any]
    baseline_games_path: Path
    baseline_manifest_path: Path
    baseline_checkpoint_summary_path: Path
    output_dir: Path
    selected: list[dict[str, Any]]

    @property
    def games_path(self) -> Path:
        return self.output_dir / "games.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "evaluation_manifest.json"


def resolve_context(args: argparse.Namespace) -> BasketContext:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path, "Adaptive fixed-basket protocol")
    _require(config.get("config_id") == "adaptive_fixed_basket_v2", "config_id must be adaptive_fixed_basket_v2")

    alignment = config.get("checkpoint_alignment")
    adaptive = config.get("adaptive_run")
    base = config.get("base_protocol")
    joint_inputs = config.get("joint_elo", {}).get("inputs")
    _require(isinstance(alignment, dict), "checkpoint_alignment is missing")
    _require(isinstance(adaptive, dict), "adaptive_run is missing")
    _require(isinstance(base, dict), "base_protocol is missing")
    _require(isinstance(joint_inputs, dict), "joint_elo.inputs is missing")

    matched_path = (
        args.matched_compute.expanduser().resolve()
        if args.matched_compute is not None
        else _resolve_experiments_path(alignment["contract"])
    )
    base_protocol_path = _resolve_experiments_path(base["config"])
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else _resolve_experiments_path(adaptive["root"])
    )
    checkpoint_manifest_path = (
        (run_dir / "checkpoint_manifest.json").resolve()
        if args.run_dir is not None
        else _resolve_experiments_path(adaptive["checkpoint_manifest"])
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_experiments_path(config["outputs"]["root"])
    )
    baseline_games_path = _resolve_experiments_path(joint_inputs["baseline_games"]["path"])
    baseline_manifest_path = _resolve_experiments_path(
        joint_inputs["baseline_evaluation_manifest"]["path"]
    )
    baseline_checkpoint_summary_path = _resolve_experiments_path(
        joint_inputs["baseline_checkpoint_summary"]["path"]
    )

    _expected_hash(base_protocol_path, base.get("expected_sha256"), "Baseline fixed-basket protocol")
    _expected_hash(matched_path, alignment.get("expected_sha256"), "matched-compute protocol")
    _expected_hash(
        checkpoint_manifest_path,
        adaptive.get("checkpoint_manifest_expected_sha256"),
        "Adaptive checkpoint manifest",
    )
    _expected_hash(
        baseline_games_path,
        joint_inputs["baseline_games"].get("sha256"),
        "Baseline fixed-basket games",
    )
    _expected_hash(
        baseline_manifest_path,
        joint_inputs["baseline_evaluation_manifest"].get("sha256"),
        "Baseline fixed-basket evaluation manifest",
    )
    _expected_hash(
        baseline_checkpoint_summary_path,
        joint_inputs["baseline_checkpoint_summary"].get("sha256"),
        "Baseline fixed-basket checkpoint summary",
    )

    matched = _load_yaml(matched_path, "matched-compute protocol")
    checkpoint_manifest = _load_json(checkpoint_manifest_path, "Adaptive checkpoint manifest")
    base_protocol = load_protocol(base_protocol_path)
    targets = matched_targets(matched)
    required_count = int(alignment.get("required_target_count", 12))
    _require(len(targets) == required_count, f"expected {required_count} matched targets, found {len(targets)}")
    _require(
        [item["target_baseline_iteration"] for item in targets]
        == list(base_protocol["checkpoints"]),
        "matched target iterations differ from fixed_basket_v1",
    )
    selected = select_adaptive_checkpoints(targets, checkpoint_registry(checkpoint_manifest))
    checkpoint_zero = _resolve_experiments_path(adaptive["checkpoint_0"]["path"])
    for item in selected:
        path = _resolve_checkpoint_path(item, run_dir, checkpoint_zero)
        actual = _expected_hash(path, item["sha256"], f"Adaptive checkpoint {item['iteration']}")
        item["path"] = path.as_posix()
        item["sha256"] = actual
        _require(
            item["actual_gpu_hours"] <= item["target_gpu_hours"],
            f"Adaptive checkpoint {item['iteration']} exceeds target {item['target_baseline_iteration']}",
        )
    initial = next(item for item in selected if item["target_baseline_iteration"] == 0)
    _require(initial["iteration"] == 0, "target 0 did not select Adaptive checkpoint 0")
    _require(
        initial["sha256"] == adaptive["checkpoint_0"]["sha256"],
        "checkpoint 0 is not the shared pretrained checkpoint",
    )

    return BasketContext(
        config_path=config_path,
        config=config,
        matched_path=matched_path,
        matched=matched,
        base_protocol_path=base_protocol_path,
        base_protocol=base_protocol,
        run_dir=run_dir,
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_manifest=checkpoint_manifest,
        baseline_games_path=baseline_games_path,
        baseline_manifest_path=baseline_manifest_path,
        baseline_checkpoint_summary_path=baseline_checkpoint_summary_path,
        output_dir=output_dir,
        selected=selected,
    )


def evaluator_entries(context: BasketContext) -> list[dict[str, Any]]:
    """Expose Adaptive files under Baseline target ids to preserve paired seeds."""
    return [
        {
            "iteration": item["target_baseline_iteration"],
            "path": item["path"],
            "filename": Path(item["path"]).name,
            "size_bytes": Path(item["path"]).stat().st_size,
            "sha256": item["sha256"],
            "resolution_source": "latest_adaptive_checkpoint_not_after_target_gpu_hours",
            "target_baseline_iteration": item["target_baseline_iteration"],
            "target_gpu_hours": item["target_gpu_hours"],
            "adaptive_iteration": item["iteration"],
            "adaptive_gpu_hours": item["actual_gpu_hours"],
        }
        for item in context.selected
    ]


def build_tasks(context: BasketContext) -> list[dict[str, Any]]:
    protocol = context.base_protocol
    tasks: list[dict[str, Any]] = []
    games_per_side = int(protocol["games_per_opponent"]) // 2
    for selected in context.selected:
        target = int(selected["target_baseline_iteration"])
        for opponent in protocol["opponents"]:
            opponent_id = str(opponent["id"])
            for game_index in range(int(protocol["games_per_opponent"])):
                model_color = "white" if game_index < games_per_side else "black"
                game_seed = stable_seed(
                    protocol["protocol_id"],
                    target,
                    opponent_id,
                    game_index,
                    base_seed=int(protocol["base_seed"]),
                )
                tasks.append(
                    {
                        "game_key": f"{target}|{opponent_id}|{game_index}",
                        "target_baseline_iteration": target,
                        "target_gpu_hours": selected["target_gpu_hours"],
                        "selected_adaptive_iteration": selected["iteration"],
                        "selected_adaptive_gpu_hours": selected["actual_gpu_hours"],
                        "checkpoint_path": selected["path"],
                        "checkpoint_sha256": selected["sha256"],
                        "opponent": opponent_id,
                        "game_index": game_index,
                        "model_color": model_color,
                        "game_seed": game_seed,
                    }
                )
    return tasks


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_preparation(
    context: BasketContext,
    *,
    run_js_check: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entries = evaluator_entries(context)
    tasks = build_tasks(context)
    expected_games = (
        len(context.base_protocol["checkpoints"])
        * len(context.base_protocol["opponents"])
        * int(context.base_protocol["games_per_opponent"])
    )
    _require(len(tasks) == expected_games == 2400, "prepared task manifest is not exactly 2,400 games")
    _require(len({task["game_key"] for task in tasks}) == len(tasks), "prepared task keys are not unique")

    js_report = (
        check_js_determinism(context.base_protocol)
        if run_js_check
        else {"status": "not_run_in_this_mode"}
    )
    if run_js_check:
        _require(js_report.get("status") == "passed", "seeded JS determinism check failed")

    resolved_config = copy.deepcopy(context.config)
    resolved_config["runtime_resolution"] = {
        "config_path": context.config_path.as_posix(),
        "matched_compute_path": context.matched_path.as_posix(),
        "base_protocol_path": context.base_protocol_path.as_posix(),
        "adaptive_run_dir": context.run_dir.as_posix(),
        "checkpoint_manifest_path": context.checkpoint_manifest_path.as_posix(),
        "output_dir": context.output_dir.as_posix(),
        "selection_rule": "latest Adaptive checkpoint with actual_gpu_hours <= target_gpu_hours",
        "selected_checkpoints": [
            {
                "target_baseline_iteration": item["target_baseline_iteration"],
                "target_gpu_hours": item["target_gpu_hours"],
                "selected_adaptive_iteration": item["iteration"],
                "selected_adaptive_gpu_hours": item["actual_gpu_hours"],
                "checkpoint_path": item["path"],
                "checkpoint_sha256": item["sha256"],
            }
            for item in context.selected
        ],
    }

    implementation_paths = {
        "evaluate_joint_basket": Path(__file__).resolve(),
        "verify_joint_basket": Path(__file__).resolve().with_name("verify_joint_basket.py"),
        "joint_elo": Path(__file__).resolve().with_name("joint_elo.py"),
        "evaluate_fixed_basket": BASELINE_ANALYSIS_SCRIPTS / "evaluate_fixed_basket.py",
        "summarize_fixed_basket": BASELINE_ANALYSIS_SCRIPTS / "summarize_fixed_basket.py",
        "arena": SOURCE_ROOT / "baseline" / "arena" / "arena.py",
        "bot_alphazero": SOURCE_ROOT / "baseline" / "arena" / "bot_alphazero.py",
        "bot_greedy": SOURCE_ROOT / "baseline" / "arena" / "bot_greedy.py",
        "bot_js_mcts": SOURCE_ROOT / "baseline" / "arena" / "bot_js_mcts.py",
        "bot_random": SOURCE_ROOT / "baseline" / "arena" / "bot_random.py",
        "bot_random_greedy": SOURCE_ROOT / "baseline" / "arena" / "bot_random_greedy.py",
        "seeded_js_bridge": SOURCE_ROOT / "baseline" / "analysis" / "js" / "seeded_bot.js",
        "js_mcts_entry": SOURCE_ROOT / "baseline" / "external" / "js-mcts" / "bot.js",
        "js_mcts_ai": SOURCE_ROOT / "baseline" / "external" / "js-mcts" / "src" / "js" / "ai.js",
    }
    input_manifest = {
        "schema_version": 2,
        "config_id": context.config["config_id"],
        "inputs": {
            "adaptive_protocol": _input_entry(context.config_path),
            "matched_compute": _input_entry(context.matched_path),
            "base_protocol": _input_entry(context.base_protocol_path),
            "adaptive_checkpoint_manifest": _input_entry(context.checkpoint_manifest_path),
            "baseline_games": _input_entry(context.baseline_games_path),
            "baseline_evaluation_manifest": _input_entry(context.baseline_manifest_path),
            "baseline_checkpoint_summary": _input_entry(context.baseline_checkpoint_summary_path),
            "implementations": {
                name: _input_entry(path) for name, path in implementation_paths.items()
            },
            "selected_checkpoint_files": [
                {
                    "target_baseline_iteration": entry["iteration"],
                    "selected_adaptive_iteration": entry["adaptive_iteration"],
                    **_input_entry(Path(entry["path"])),
                }
                for entry in entries
            ],
        },
    }

    manifest = {
        "schema_version": 2,
        "config_id": context.config["config_id"],
        "protocol_id": context.base_protocol["protocol_id"],
        "evaluation_mode": "formal",
        "status": "prepared",
        "selection_axis": "gpu_hours",
        "selection_rule": "latest_completed_checkpoint_not_after_target",
        "parameter_interpolation": False,
        "protocol_config": context.base_protocol_path.as_posix(),
        "protocol_config_sha256": sha256_file(context.base_protocol_path),
        "adaptive_checkpoint_manifest": context.checkpoint_manifest_path.as_posix(),
        "adaptive_checkpoint_manifest_sha256": sha256_file(context.checkpoint_manifest_path),
        "checkpoints": entries,
        "selected_checkpoints": list(context.base_protocol["checkpoints"]),
        "selected_checkpoint_registry": resolved_config["runtime_resolution"]["selected_checkpoints"],
        "expected_protocol_games": expected_games,
        "expected_evaluation_games": expected_games,
        "games_recorded": 0,
        "full_protocol_completed": False,
        "game_seed_derivation": "fixed_basket_v1 stable_seed using target Baseline iteration",
        "side_schedule": {
            "model_white_game_indices": [0, 24],
            "model_black_game_indices": [25, 49],
            "games_per_colour": 25,
        },
        "max_turns": int(context.base_protocol["max_turns"]),
        "js_determinism": js_report,
        "js_determinism_status": js_report.get("status"),
        "tasks": tasks,
        "task_manifest_sha256": _canonical_sha256(tasks),
        "outputs": {
            "resolved_config": (context.output_dir / "resolved_config.yaml").as_posix(),
            "input_manifest": (context.output_dir / "input_manifest.json").as_posix(),
            "evaluation_manifest": context.manifest_path.as_posix(),
            "games": context.games_path.as_posix(),
            "checkpoint_summary": (context.output_dir / "checkpoint_summary.csv").as_posix(),
            "opponent_summary": (context.output_dir / "opponent_summary.csv").as_posix(),
            "elo_summary": (context.output_dir / "elo_summary.csv").as_posix(),
            "joint_elo_summary": (context.output_dir / "joint_elo_summary.csv").as_posix(),
            "summary": (context.output_dir / "summary.json").as_posix(),
            "evaluation_log": (context.output_dir / "evaluation.log").as_posix(),
        },
        "persistence": {
            "games": "append one JSON record, flush, and fsync after every completed game",
            "resume_key": ["target_baseline_iteration", "opponent", "game_index"],
        },
    }
    return resolved_config, input_manifest, manifest


def write_preparation(
    context: BasketContext,
    resolved_config: dict[str, Any],
    input_manifest: dict[str, Any],
    evaluation_manifest: dict[str, Any],
) -> None:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_yaml(context.output_dir / "resolved_config.yaml", resolved_config)
    _atomic_write_json(context.output_dir / "input_manifest.json", input_manifest)
    _atomic_write_json(context.manifest_path, evaluation_manifest)


def _validate_resume_manifest(
    context: BasketContext, candidate: dict[str, Any]
) -> dict[str, Any]:
    existing = _load_json(context.manifest_path, "existing evaluation manifest")
    for field in (
        "config_id",
        "protocol_id",
        "selected_checkpoints",
        "selected_checkpoint_registry",
        "expected_evaluation_games",
        "task_manifest_sha256",
    ):
        _require(existing.get(field) == candidate.get(field), f"resume manifest {field} mismatch")
    return existing


def _score(record: dict[str, Any]) -> float:
    result = record.get("model_result")
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    if result == "loss":
        return 0.0
    raise JointBasketError(f"invalid model_result: {result!r}")


def validate_condition_games(
    context: BasketContext,
    records: list[dict[str, Any]],
    condition: str,
) -> list[dict[str, Any]]:
    """Validate one complete basket and convert it to joint-model records."""
    _require(condition in {"baseline", "adaptive"}, f"unknown condition {condition}")
    protocol = context.base_protocol
    targets = [int(item["target_baseline_iteration"]) for item in context.selected]
    target_set = set(targets)
    opponent_ids = [str(item["id"]) for item in protocol["opponents"]]
    opponent_set = set(opponent_ids)
    games_per_opponent = int(protocol["games_per_opponent"])
    games_per_side = games_per_opponent // 2
    expected_total = len(targets) * len(opponent_ids) * games_per_opponent
    _require(
        len(records) == expected_total,
        f"{condition} basket must contain {expected_total} games, found {len(records)}",
    )

    if condition == "adaptive":
        manifest = _load_json(context.manifest_path, "Adaptive evaluation manifest")
    else:
        manifest = _load_json(context.baseline_manifest_path, "Baseline evaluation manifest")
    manifest_entries = {
        int(item["iteration"]): item for item in manifest.get("checkpoints", [])
    }

    seen: set[tuple[int, str, int]] = set()
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    joint_records: list[dict[str, Any]] = []
    for record_number, record in enumerate(records, start=1):
        try:
            target = int(record["checkpoint"])
            opponent = str(record["opponent"])
            game_index = int(record["game_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JointBasketError(
                f"{condition} game {record_number} has an invalid stable key"
            ) from exc
        key = (target, opponent, game_index)
        _require(key not in seen, f"duplicate {condition} game key {key}")
        seen.add(key)
        _require(target in target_set, f"{condition} game targets unconfigured checkpoint {target}")
        _require(opponent in opponent_set, f"{condition} game targets unconfigured opponent {opponent}")
        _require(0 <= game_index < games_per_opponent, f"{condition} game index is outside range: {key}")
        expected_color = "white" if game_index < games_per_side else "black"
        expected_seed = stable_seed(
            protocol["protocol_id"],
            target,
            opponent,
            game_index,
            base_seed=int(protocol["base_seed"]),
        )
        _require(record.get("protocol_id") == protocol["protocol_id"], f"{condition} protocol mismatch at {key}")
        _require(record.get("game_seed") == expected_seed, f"{condition} stable seed mismatch at {key}")
        _require(record.get("model_color") == expected_color, f"{condition} colour mismatch at {key}")
        _require(record.get("max_turns") == protocol["max_turns"], f"{condition} max_turns mismatch at {key}")
        _require(record.get("fault") is None, f"{condition} game fault at {key}")
        _require(record.get("termination") not in {"invalid_move", "bot_error"}, f"{condition} invalid termination at {key}")
        entry = manifest_entries.get(target)
        _require(entry is not None, f"{condition} manifest lacks target {target}")
        _require(record.get("checkpoint_path") == entry.get("path"), f"{condition} checkpoint path mismatch at {key}")
        _require(record.get("checkpoint_sha256") == entry.get("sha256"), f"{condition} checkpoint SHA mismatch at {key}")
        score = _score(record)
        groups[(target, opponent)].append(record)
        joint_records.append(
            {
                "condition": condition,
                "target": target,
                "opponent": opponent,
                "model_color": expected_color,
                "game_index": game_index,
                "bot1": f"{condition}_checkpoint_{target}",
                "bot2": opponent,
                "score_bot1": score,
            }
        )

    _require(len(seen) == expected_total, f"{condition} game keys are incomplete")
    for target in targets:
        for opponent in opponent_ids:
            group = groups[(target, opponent)]
            _require(len(group) == games_per_opponent, f"{condition} target {target} vs {opponent} has {len(group)} games")
            white = sum(item.get("model_color") == "white" for item in group)
            black = sum(item.get("model_color") == "black" for item in group)
            _require(
                white == games_per_side and black == games_per_side,
                f"{condition} target {target} vs {opponent} colour split is {white}/{black}",
            )
    return joint_records


# The pure NumPy module is authoritative for fitting.  Keeping this binding here
# lets the Arena-facing script remain import-compatible while statistical tests do
# not need Torch or pyquoridor.
from joint_elo import (  # noqa: E402
    aggregate_joint_records as aggregate_joint_records,
    bootstrap_joint_elo as bootstrap_joint_elo,
    fit_joint_elo as fit_joint_elo,
)


def run_joint_elo(
    context: BasketContext,
    adaptive_games: list[dict[str, Any]],
    *,
    logger: EvaluationLogger,
) -> dict[str, Any]:
    baseline_games = _load_games(context.baseline_games_path)
    baseline_records = validate_condition_games(context, baseline_games, "baseline")
    adaptive_records = validate_condition_games(context, adaptive_games, "adaptive")
    records = baseline_records + adaptive_records

    targets = [int(item["target_baseline_iteration"]) for item in context.selected]
    anchors = [str(value) for value in context.config["joint_elo"]["shared_anchors"]]
    participants = (
        [f"baseline_checkpoint_{target}" for target in targets]
        + [f"adaptive_checkpoint_{target}" for target in targets]
        + anchors
    )
    _require(len(participants) == len(set(participants)), "joint Elo participant ids are not unique")
    strata = aggregate_joint_records(records, by_bootstrap_stratum=True)
    expected_strata = 2 * len(targets) * len(anchors) * 2
    _require(
        len(strata) == expected_strata == 192,
        f"joint Elo requires 192 strata, found {len(strata)}",
    )
    fit_config = context.config["joint_elo"]["fit"]
    fitted = fit_joint_elo(participants, anchors, strata, fit_config)
    uncertainty = context.config["joint_elo"]["uncertainty"]
    resamples = int(uncertainty["resamples"])
    bootstrap_seed = int(uncertainty["random_seed"])
    logger.write(
        "Joint Elo full fit completed: "
        f"participants={len(participants)} games={len(records)} "
        f"iterations={fitted['iterations']}"
    )
    bootstrap = bootstrap_joint_elo(
        participants,
        anchors,
        strata,
        fit_config,
        resamples=resamples,
        seed=bootstrap_seed,
        initial_theta=fitted["theta"],
        progress=logger.write,
    )

    fit_identity = {
        "method": fit_config["method"],
        "fit_scope": context.config["joint_elo"]["fit_scope"],
        "participants": participants,
        "anchors": anchors,
        "baseline_games_sha256": sha256_file(context.baseline_games_path),
        "adaptive_games_sha256": sha256_file(context.games_path),
        "fit_config": fit_config,
        "bootstrap": {
            "strata": uncertainty["strata"],
            "resamples": resamples,
            "seed": bootstrap_seed,
        },
    }
    fit_id = _canonical_sha256(fit_identity)
    participant_index = {name: index for index, name in enumerate(participants)}
    ratings = fitted["ratings"]
    rating_quantiles = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    rows: list[dict[str, Any]] = []
    for selected in context.selected:
        target = int(selected["target_baseline_iteration"])
        baseline_name = f"baseline_checkpoint_{target}"
        adaptive_name = f"adaptive_checkpoint_{target}"
        baseline_index = participant_index[baseline_name]
        adaptive_index = participant_index[adaptive_name]
        difference_samples = bootstrap[:, adaptive_index] - bootstrap[:, baseline_index]
        difference_ci = np.quantile(difference_samples, [0.025, 0.975])
        rows.append(
            {
                "target_baseline_iteration": target,
                "target_gpu_hours": selected["target_gpu_hours"],
                "selected_adaptive_iteration": selected["iteration"],
                "selected_adaptive_gpu_hours": selected["actual_gpu_hours"],
                "baseline_participant": baseline_name,
                "adaptive_participant": adaptive_name,
                "baseline_joint_elo": ratings[baseline_name],
                "baseline_joint_elo_ci95_low": float(rating_quantiles[0, baseline_index]),
                "baseline_joint_elo_ci95_high": float(rating_quantiles[1, baseline_index]),
                "adaptive_joint_elo": ratings[adaptive_name],
                "adaptive_joint_elo_ci95_low": float(rating_quantiles[0, adaptive_index]),
                "adaptive_joint_elo_ci95_high": float(rating_quantiles[1, adaptive_index]),
                "adaptive_minus_baseline_elo": ratings[adaptive_name] - ratings[baseline_name],
                "adaptive_minus_baseline_elo_ci95_low": float(difference_ci[0]),
                "adaptive_minus_baseline_elo_ci95_high": float(difference_ci[1]),
                "fit_id": fit_id,
                "fit_scope": context.config["joint_elo"]["fit_scope"],
                "fit_method": fit_config["method"],
                "bootstrap_iterations": resamples,
                "bootstrap_seed": bootstrap_seed,
            }
        )
    output_path = context.output_dir / "joint_elo_summary.csv"
    _atomic_write_csv(output_path, rows, JOINT_ELO_FIELDS)
    anchor_summary = {}
    for anchor in anchors:
        index = participant_index[anchor]
        anchor_summary[anchor] = {
            "elo": ratings[anchor],
            "ci95": [
                float(rating_quantiles[0, index]),
                float(rating_quantiles[1, index]),
            ],
        }
    return {
        "status": "completed",
        "single_joint_fit": True,
        "fit_id": fit_id,
        "fit_scope": context.config["joint_elo"]["fit_scope"],
        "fit_method": fit_config["method"],
        "location_constraint": context.config["joint_elo"]["location_constraint"],
        "participants": len(participants),
        "shared_anchors": anchor_summary,
        "baseline_games": len(baseline_records),
        "adaptive_games": len(adaptive_records),
        "total_games": len(records),
        "strata": len(strata),
        "stratum_definition": list(uncertainty["strata"]),
        "games_per_stratum": 25,
        "bootstrap_iterations": resamples,
        "bootstrap_seed": bootstrap_seed,
        "optimizer_iterations": fitted["iterations"],
        "penalized_log_likelihood": fitted["penalized_log_likelihood"],
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
    }


def _augment_metric_csv(
    path: Path,
    selected_by_target: dict[int, dict[str, Any]],
) -> None:
    rows = _load_csv(path, path.name)
    if not rows:
        return
    original_fields = list(rows[0])
    added_fields = [
        "target_baseline_iteration",
        "target_gpu_hours",
        "selected_adaptive_iteration",
        "selected_adaptive_gpu_hours",
    ]
    for row in rows:
        target = int(row["checkpoint"])
        selected = selected_by_target[target]
        row.update(
            {
                "target_baseline_iteration": target,
                "target_gpu_hours": selected["target_gpu_hours"],
                "selected_adaptive_iteration": selected["iteration"],
                "selected_adaptive_gpu_hours": selected["actual_gpu_hours"],
            }
        )
    _atomic_write_csv(path, rows, original_fields + added_fields)


def _rewrite_provisional_elo(path: Path, summary: dict[str, Any]) -> None:
    rows = _load_csv(path, "provisional Adaptive Elo summary")
    fields = list(rows[0]) if rows else [
        "participant",
        "participant_type",
        "elo",
        "status",
        "fit_scope",
        "random_seed",
    ]
    renamed_ratings: dict[str, Any] = {}
    for row in rows:
        participant = row["participant"]
        if participant.startswith("checkpoint_"):
            participant = f"adaptive_{participant}"
            row["participant"] = participant
            row["participant_type"] = "adaptive_checkpoint"
        row["fit_scope"] = "Adaptive checkpoints and fixed opponents; provisional only"
        renamed_ratings[participant] = float(row["elo"])
    _atomic_write_csv(path, rows, fields)
    summary["elo"]["fit_scope"] = "Adaptive checkpoints and fixed opponents; provisional only"
    summary["elo"]["ratings"] = dict(sorted(renamed_ratings.items()))
    summary["elo"]["comparable_across_separate_fits"] = False


def summarize_complete_run(
    context: BasketContext,
    *,
    logger: EvaluationLogger,
) -> dict[str, Any]:
    manifest = _load_json(context.manifest_path, "Adaptive evaluation manifest")
    games = _load_games(context.games_path)
    base_summary = summarize_results(
        context.base_protocol,
        manifest,
        games,
        context.output_dir,
        games_path=context.games_path,
        manifest_path=context.manifest_path,
        mode="formal",
    )
    _require(base_summary.get("status") == "completed", "Baseline fixed-basket summarizer rejected Adaptive games")
    selected_by_target = {
        int(item["target_baseline_iteration"]): item for item in context.selected
    }
    _augment_metric_csv(context.output_dir / "checkpoint_summary.csv", selected_by_target)
    _augment_metric_csv(context.output_dir / "opponent_summary.csv", selected_by_target)
    _rewrite_provisional_elo(context.output_dir / "elo_summary.csv", base_summary)
    joint_summary = run_joint_elo(context, games, logger=logger)

    base_summary["schema_version"] = 2
    base_summary["config_id"] = context.config["config_id"]
    base_summary["status"] = "completed"
    base_summary["selected_checkpoint_registry"] = [
        {
            "target_baseline_iteration": item["target_baseline_iteration"],
            "target_gpu_hours": item["target_gpu_hours"],
            "selected_adaptive_iteration": item["iteration"],
            "selected_adaptive_gpu_hours": item["actual_gpu_hours"],
            "checkpoint_sha256": item["sha256"],
        }
        for item in context.selected
    ]
    base_summary["joint_elo"] = joint_summary
    base_summary["outputs"].update(
        {
            "joint_elo_summary": joint_summary["output"],
            "summary": (context.output_dir / "summary.json").as_posix(),
        }
    )
    base_summary["output_sha256"] = {
        "checkpoint_summary": sha256_file(context.output_dir / "checkpoint_summary.csv"),
        "opponent_summary": sha256_file(context.output_dir / "opponent_summary.csv"),
        "elo_summary": sha256_file(context.output_dir / "elo_summary.csv"),
        "joint_elo_summary": sha256_file(context.output_dir / "joint_elo_summary.csv"),
    }
    _atomic_write_json(context.output_dir / "summary.json", base_summary)

    final_manifest = _load_json(context.manifest_path, "Adaptive evaluation manifest")
    final_manifest["status"] = "completed"
    final_manifest["full_protocol_completed"] = True
    final_manifest["games_recorded"] = len(games)
    final_manifest["games_sha256"] = sha256_file(context.games_path)
    final_manifest["summary"] = base_summary
    final_manifest["outputs"].update(base_summary["outputs"])
    _atomic_write_json(context.manifest_path, final_manifest)
    logger.write(
        "Adaptive fixed-basket summary completed: "
        f"games={len(games)} joint_fit_id={joint_summary['fit_id']}"
    )
    return base_summary


def _board_size(context: BasketContext) -> int:
    resolved_path = context.run_dir / "resolved_config.yaml"
    resolved = _load_yaml(resolved_path, "Adaptive resolved training config")
    try:
        board_size = int(resolved["model"]["board_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JointBasketError("Adaptive resolved config lacks model.board_size") from exc
    _require(board_size > 0, "Adaptive board size must be positive")
    return board_size


def execute(args: argparse.Namespace, logger: EvaluationLogger) -> None:
    context = resolve_context(args)
    if args.summarize_only:
        _require(context.games_path.is_file(), "--summarize-only requires games.jsonl")
        _require(context.manifest_path.is_file(), "--summarize-only requires evaluation_manifest.json")
        summarize_complete_run(context, logger=logger)
        return

    if context.games_path.exists() and not args.resume:
        raise JointBasketError(
            "games.jsonl already exists; pass --resume to continue by stable game key"
        )
    if args.resume and not context.games_path.is_file():
        raise JointBasketError("--resume requires an existing games.jsonl")

    resolved_config, input_manifest, candidate_manifest = build_preparation(
        context, run_js_check=True
    )
    if args.resume:
        manifest = _validate_resume_manifest(context, candidate_manifest)
        existing_inputs = _load_json(
            context.output_dir / "input_manifest.json",
            "existing input manifest",
        )
        _require(
            existing_inputs == input_manifest,
            "resume input manifest differs from the prepared run",
        )
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["js_determinism"] = candidate_manifest["js_determinism"]
        manifest["js_determinism_status"] = "passed"
        _atomic_write_json(context.manifest_path, manifest)
    else:
        write_preparation(context, resolved_config, input_manifest, candidate_manifest)
        manifest = candidate_manifest
    logger.write(
        "Adaptive fixed-basket prepared: "
        f"targets={len(context.selected)} tasks={len(candidate_manifest['tasks'])} "
        "JS_determinism=passed"
    )
    if args.prepare_only:
        logger.write("Evaluation status: prepared; no formal games started")
        return

    manifest["status"] = "running"
    manifest.pop("failure", None)
    _atomic_write_json(context.manifest_path, manifest)

    def log_completed(record: dict[str, Any]) -> None:
        logger.write(
            "Game completed and fsynced: "
            f"target={record['checkpoint']} opponent={record['opponent']} "
            f"game_index={record['game_index']} result={record['model_result']}"
        )

    new_games = evaluate_matchups(
        context.base_protocol,
        evaluator_entries(context),
        context.games_path,
        board_size=_board_size(context),
        on_game_completed=log_completed,
        on_event=logger.write,
    )
    games = _load_games(context.games_path)
    expected = int(manifest["expected_evaluation_games"])
    _require(
        len(games) == expected,
        f"formal evaluation is incomplete: expected {expected} games, found {len(games)}",
    )
    manifest = _load_json(context.manifest_path, "Adaptive evaluation manifest")
    manifest["status"] = "games_completed"
    manifest["games_recorded"] = len(games)
    manifest["new_games_completed"] = new_games
    manifest["games_sha256"] = sha256_file(context.games_path)
    _atomic_write_json(context.manifest_path, manifest)
    logger.write(f"Formal games completed: new={new_games} total={len(games)}")
    summarize_complete_run(context, logger=logger)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matched-compute", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs/JS and write all 2,400 tasks without playing games.",
    )
    modes.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted formal run by stable game key.",
    )
    modes.add_argument(
        "--summarize-only",
        action="store_true",
        help="Validate and summarize an already complete games.jsonl.",
    )
    modes.add_argument(
        "--verify-only",
        action="store_true",
        help="Run the read-only fixed_basket_v2 acceptance verifier.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        try:
            from verify_joint_basket import verify_joint_basket

            report = verify_joint_basket(
                config_path=args.config,
                output_dir=args.output_dir,
                matched_compute_path=args.matched_compute,
                run_dir=args.run_dir,
            )
            print(
                "Joint fixed-basket verification passed: "
                f"games={report['games']['unique_game_keys']} "
                f"fit_id={report['joint_elo']['fit_id']}"
            )
            return 0
        except (JointBasketError, FixedBasketError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    log_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT
    )
    with EvaluationLogger(log_dir / "evaluation.log") as logger:
        try:
            execute(args, logger)
            return 0
        except (JointBasketError, FixedBasketError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            logger.write(f"Adaptive joint fixed-basket failed: {exc}")
            manifest_path = log_dir / "evaluation_manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = _load_json(manifest_path, "evaluation manifest")
                    manifest["status"] = "failed"
                    manifest["failure"] = str(exc)
                    _atomic_write_json(manifest_path, manifest)
                except Exception:
                    pass
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
