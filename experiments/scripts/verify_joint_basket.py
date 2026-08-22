#!/usr/bin/env python3
"""Read-only acceptance verifier for adaptive fixed_basket_v2 outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
SCRIPTS_ROOT = EXPERIMENTS_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_joint_basket import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    JointBasketError,
    _canonical_sha256,
    _load_csv,
    _load_json,
    _load_yaml,
    _require,
    aggregate_joint_records,
    build_tasks,
    fit_joint_elo,
    resolve_context,
    sha256_file,
    validate_condition_games,
)
from summarize_fixed_basket import _load_games  # noqa: E402


REQUIRED_OUTPUTS = (
    "resolved_config.yaml",
    "input_manifest.json",
    "evaluation_manifest.json",
    "games.jsonl",
    "checkpoint_summary.csv",
    "opponent_summary.csv",
    "elo_summary.csv",
    "joint_elo_summary.csv",
    "summary.json",
    "evaluation.log",
)


def _verify_file_entry(entry: Any, label: str) -> None:
    _require(isinstance(entry, dict), f"input manifest {label} is invalid")
    path_value = entry.get("path")
    expected = entry.get("sha256")
    _require(isinstance(path_value, str) and path_value, f"input manifest {label} path is invalid")
    _require(isinstance(expected, str) and len(expected) == 64, f"input manifest {label} SHA is invalid")
    path = Path(path_value)
    _require(path.is_file(), f"input manifest {label} file is missing: {path}")
    _require(sha256_file(path) == expected, f"input manifest {label} SHA mismatch")
    size = entry.get("size_bytes")
    _require(isinstance(size, int) and size == path.stat().st_size, f"input manifest {label} size mismatch")


def _verify_input_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("config_id") == "adaptive_fixed_basket_v2", "input manifest config_id mismatch")
    inputs = payload.get("inputs")
    _require(isinstance(inputs, dict), "input manifest lacks inputs")
    for name in (
        "adaptive_protocol",
        "matched_compute",
        "base_protocol",
        "adaptive_checkpoint_manifest",
        "baseline_games",
        "baseline_evaluation_manifest",
        "baseline_checkpoint_summary",
    ):
        _verify_file_entry(inputs.get(name), name)
    implementations = inputs.get("implementations")
    _require(isinstance(implementations, dict) and implementations, "input manifest lacks implementation hashes")
    for name, entry in implementations.items():
        _verify_file_entry(entry, f"implementations.{name}")
    checkpoints = inputs.get("selected_checkpoint_files")
    _require(isinstance(checkpoints, list) and len(checkpoints) == 12, "input manifest must list 12 selected checkpoints")
    for index, entry in enumerate(checkpoints):
        _verify_file_entry(entry, f"selected_checkpoint_files[{index}]")


def _float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise JointBasketError(f"invalid numeric field {field} in joint_elo_summary.csv") from exc
    _require(math.isfinite(value), f"non-finite {field} in joint_elo_summary.csv")
    return value


def _verify_summary_csvs(output_dir: Path) -> None:
    checkpoint_rows = _load_csv(output_dir / "checkpoint_summary.csv", "checkpoint summary")
    opponent_rows = _load_csv(output_dir / "opponent_summary.csv", "opponent summary")
    elo_rows = _load_csv(output_dir / "elo_summary.csv", "provisional Elo summary")
    _require(len(checkpoint_rows) == 12, f"checkpoint_summary.csv must have 12 rows, found {len(checkpoint_rows)}")
    _require(len(opponent_rows) == 48, f"opponent_summary.csv must have 48 rows, found {len(opponent_rows)}")
    _require(len(elo_rows) == 16, f"elo_summary.csv must have 16 rows, found {len(elo_rows)}")
    for row in checkpoint_rows:
        _require(int(row["total_games"]) == 200, "each checkpoint summary must contain 200 games")
        _require(int(row["invalid_moves"]) == 0, "checkpoint summary contains invalid moves")
        _require(int(row["bot_errors"]) == 0, "checkpoint summary contains bot errors")
        for key, value in row.items():
            if key in {"checkpoint"}:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            _require(math.isfinite(number), f"checkpoint_summary.csv contains non-finite {key}")
    for row in opponent_rows:
        _require(int(row["games"]) == 50, "each checkpoint-opponent summary must contain 50 games")
        _require(int(row["model_white_games"]) == 25, "opponent summary white split is not 25")
        _require(int(row["model_black_games"]) == 25, "opponent summary black split is not 25")
        _require(int(row["faults"]) == 0, "opponent summary contains game faults")
    _require(
        sum(row["participant_type"] == "adaptive_checkpoint" for row in elo_rows) == 12,
        "provisional Elo summary does not contain 12 Adaptive checkpoints",
    )


def _verify_joint_fit(
    context: Any,
    adaptive_games: list[dict[str, Any]],
    rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    _require(len(rows) == 12, f"joint_elo_summary.csv must have 12 rows, found {len(rows)}")
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
    strata = aggregate_joint_records(records, by_bootstrap_stratum=True)
    _require(len(strata) == 192, f"joint Elo source must contain 192 strata, found {len(strata)}")
    fitted = fit_joint_elo(
        participants,
        anchors,
        strata,
        context.config["joint_elo"]["fit"],
    )
    ratings = fitted["ratings"]
    fit_ids = {row.get("fit_id") for row in rows}
    fit_scopes = {row.get("fit_scope") for row in rows}
    methods = {row.get("fit_method") for row in rows}
    _require(len(fit_ids) == 1 and None not in fit_ids, "joint Elo rows do not share one fit_id")
    _require(fit_scopes == {context.config["joint_elo"]["fit_scope"]}, "joint Elo fit_scope mismatch")
    _require(methods == {context.config["joint_elo"]["fit"]["method"]}, "joint Elo fit_method mismatch")
    rows_by_target = {int(row["target_baseline_iteration"]): row for row in rows}
    _require(set(rows_by_target) == set(targets), "joint Elo target coverage is incomplete")
    selected_by_target = {
        int(item["target_baseline_iteration"]): item for item in context.selected
    }
    for target in targets:
        row = rows_by_target[target]
        selected = selected_by_target[target]
        baseline_name = f"baseline_checkpoint_{target}"
        adaptive_name = f"adaptive_checkpoint_{target}"
        _require(row["baseline_participant"] == baseline_name, f"baseline participant mismatch at target {target}")
        _require(row["adaptive_participant"] == adaptive_name, f"Adaptive participant mismatch at target {target}")
        _require(int(row["selected_adaptive_iteration"]) == selected["iteration"], f"selected Adaptive iteration mismatch at target {target}")
        _require(_float(row, "selected_adaptive_gpu_hours") <= _float(row, "target_gpu_hours"), f"selected Adaptive checkpoint exceeds target {target}")
        baseline_elo = _float(row, "baseline_joint_elo")
        adaptive_elo = _float(row, "adaptive_joint_elo")
        effect = _float(row, "adaptive_minus_baseline_elo")
        _require(math.isclose(baseline_elo, ratings[baseline_name], abs_tol=1.0e-8), f"baseline joint Elo is not reproduced by the single fit at target {target}")
        _require(math.isclose(adaptive_elo, ratings[adaptive_name], abs_tol=1.0e-8), f"Adaptive joint Elo is not reproduced by the single fit at target {target}")
        _require(math.isclose(effect, adaptive_elo - baseline_elo, abs_tol=1.0e-8), f"joint Elo difference is not reproducible at target {target}")
        for prefix in ("baseline_joint_elo", "adaptive_joint_elo", "adaptive_minus_baseline_elo"):
            low = _float(row, f"{prefix}_ci95_low")
            high = _float(row, f"{prefix}_ci95_high")
            _require(low <= high, f"joint Elo CI is reversed for {prefix} at target {target}")
        _require(int(row["bootstrap_iterations"]) == 10000, "joint Elo bootstrap count is not 10,000")
        _require(int(row["bootstrap_seed"]) == int(context.config["joint_elo"]["uncertainty"]["random_seed"]), "joint Elo bootstrap seed mismatch")

    joint_summary = summary.get("joint_elo")
    _require(isinstance(joint_summary, dict), "summary.json lacks joint_elo")
    fit_id = next(iter(fit_ids))
    _require(joint_summary.get("single_joint_fit") is True, "summary does not attest one joint fit")
    _require(joint_summary.get("fit_id") == fit_id, "summary and joint Elo CSV fit_id differ")
    _require(joint_summary.get("total_games") == 4800, "joint Elo must fit all 4,800 games")
    _require(joint_summary.get("baseline_games") == 2400, "joint Elo baseline game count mismatch")
    _require(joint_summary.get("adaptive_games") == 2400, "joint Elo Adaptive game count mismatch")
    _require(joint_summary.get("strata") == 192, "joint Elo stratum count mismatch")
    anchors_summary = joint_summary.get("shared_anchors")
    _require(isinstance(anchors_summary, dict) and set(anchors_summary) == set(anchors), "joint Elo shared anchors mismatch")
    _require(
        math.isclose(sum(float(anchors_summary[name]["elo"]) for name in anchors), 0.0, abs_tol=1.0e-8),
        "joint Elo shared-anchor mean is not zero",
    )
    return {
        "fit_id": fit_id,
        "single_joint_fit_recomputed": True,
        "total_games": len(records),
        "strata": len(strata),
    }


def verify_joint_basket(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    matched_compute_path: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    args = SimpleNamespace(
        config=Path(config_path),
        output_dir=Path(output_dir) if output_dir is not None else None,
        matched_compute=Path(matched_compute_path) if matched_compute_path is not None else None,
        run_dir=Path(run_dir) if run_dir is not None else None,
    )
    context = resolve_context(args)
    for name in REQUIRED_OUTPUTS:
        _require((context.output_dir / name).is_file(), f"required output is missing: {name}")
    resolved = _load_yaml(context.output_dir / "resolved_config.yaml", "resolved config")
    _require(resolved.get("config_id") == "adaptive_fixed_basket_v2", "resolved config_id mismatch")
    input_manifest = _load_json(context.output_dir / "input_manifest.json", "input manifest")
    _verify_input_manifest(input_manifest)
    manifest = _load_json(context.manifest_path, "evaluation manifest")
    _require(manifest.get("status") == "completed", "evaluation manifest is not completed")
    _require(manifest.get("full_protocol_completed") is True, "evaluation manifest is not a full protocol run")
    _require(manifest.get("expected_evaluation_games") == 2400, "evaluation manifest expected game count mismatch")
    _require(manifest.get("games_recorded") == 2400, "evaluation manifest recorded game count mismatch")
    expected_tasks = build_tasks(context)
    _require(len(manifest.get("tasks", [])) == 2400, "evaluation manifest task count is not 2,400")
    _require(manifest.get("tasks") == expected_tasks, "evaluation manifest tasks differ from the frozen task grid")
    _require(manifest.get("task_manifest_sha256") == _canonical_sha256(expected_tasks), "evaluation task manifest SHA mismatch")
    _require(manifest.get("js_determinism_status") == "passed", "seeded JS determinism did not pass")
    _require(manifest.get("games_sha256") == sha256_file(context.games_path), "games.jsonl SHA differs from evaluation manifest")

    adaptive_games = _load_games(context.games_path)
    adaptive_records = validate_condition_games(context, adaptive_games, "adaptive")
    _require(len(adaptive_records) == 2400, "Adaptive basket does not contain 2,400 valid games")
    _verify_summary_csvs(context.output_dir)
    summary = _load_json(context.output_dir / "summary.json", "summary")
    _require(summary.get("status") == "completed", "summary status is not completed")
    quality = summary.get("data_quality")
    _require(isinstance(quality, dict), "summary lacks data_quality")
    _require(quality.get("unique_game_keys") == 2400, "summary unique game-key count mismatch")
    _require(quality.get("invalid_records") == 0, "summary reports invalid records")
    _require(quality.get("faults") == 0, "summary reports game faults")
    joint_rows = _load_csv(context.output_dir / "joint_elo_summary.csv", "joint Elo summary")
    joint_report = _verify_joint_fit(context, adaptive_games, joint_rows, summary)
    return {
        "status": "passed",
        "config_id": context.config["config_id"],
        "games": {
            "records": len(adaptive_games),
            "unique_game_keys": len(adaptive_records),
            "checkpoint_opponent_games": 50,
            "games_per_colour": 25,
            "invalid_moves": 0,
            "bot_errors": 0,
            "faults": 0,
        },
        "joint_elo": joint_report,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matched-compute", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_joint_basket(
            config_path=args.config,
            output_dir=args.output_dir,
            matched_compute_path=args.matched_compute,
            run_dir=args.run_dir,
        )
    except (JointBasketError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

