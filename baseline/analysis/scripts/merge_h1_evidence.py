from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H1_CONFIG = BASELINE_ROOT / "analysis" / "configs" / "h1_v1.yaml"

FIXED_OPPONENT_SCORE_COLUMNS = (
    "heuristic_20_score",
    "heuristic_200_score",
    "greedy_random_50_score",
    "random_score",
)

ALIGNED_FIELDS = (
    "checkpoint",
    "block_start_iteration",
    "block_end_iteration",
    "block_iterations",
    "is_short_block",
    "training_gpu_hours",
    "positions_generated",
    "games_completed",
    "optimizer_steps",
    "fresh_states_per_update",
    "states_per_gpu_hour",
    "games_per_gpu_hour",
    "self_play_fraction",
    "training_fraction",
    "buffer_inflow_fraction",
    "buffer_fraction_consumed",
    "mean_sample_exposure",
    "selected_sample_reuse",
    "mean_sample_age",
    "p90_sample_age",
    "turnover_fraction",
    "turnover_observable",
    "incoming_unique_state_ratio",
    "duplicate_rate",
    "state_effective_ratio",
    "replay_observable",
    "holdout_policy_loss",
    "holdout_value_loss",
    "holdout_total_loss",
    "logged_train_policy_loss",
    "logged_train_value_loss",
    "approx_policy_gap",
    "approx_value_gap",
    "approx_total_gap",
    "fixed_basket_macro_score",
    "fixed_basket_score_ci95_low",
    "fixed_basket_score_ci95_high",
    "is_plateau_start",
    "is_at_or_after_plateau",
    "is_pretrained",
    "is_best_observed",
    "is_final",
    "running_best_score",
    "drawdown_from_running_best",
    "is_max_drawdown_peak",
    "is_max_drawdown_trough",
)

EFFECT_FIELDS = (
    "evidence_stage",
    "metric",
    "analysis_mode",
    "start_checkpoint",
    "end_checkpoint",
    "valid_checkpoints",
    "start_value",
    "end_value",
    "absolute_change",
    "relative_change",
    "slope_per_20_iterations",
    "slope_per_gpu_hour",
    "expected_direction",
    "actual_direction",
    "direction_consistent",
    "availability_status",
    "limitation",
)

EFFECT_SPECS = (
    ("supply", "fresh_states_per_update", "decrease"),
    ("supply", "buffer_inflow_fraction", "decrease"),
    ("supply", "states_per_gpu_hour", "not_increase"),
    ("replay", "mean_sample_exposure", "increase"),
    ("replay", "mean_sample_age", "increase"),
    ("replay", "p90_sample_age", "increase"),
    ("replay", "turnover_fraction", "decrease"),
    ("replay", "duplicate_rate", "increase"),
    ("replay", "incoming_unique_state_ratio", "decrease"),
    ("replay", "state_effective_ratio", "decrease"),
    ("generalisation", "approx_policy_gap", "increase"),
    ("generalisation", "approx_value_gap", "increase"),
    ("generalisation", "approx_total_gap", "increase"),
)


class H1InputError(ValueError):
    """Raised when an H1 source cannot be used at its registered grain."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and merge the registered baseline H1 evidence package."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_H1_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gate-summary", type=Path)
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise H1InputError(f"required YAML file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise H1InputError(f"cannot read YAML file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise H1InputError(f"YAML root must be a mapping: {path}")
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise H1InputError(f"required JSON file not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H1InputError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise H1InputError(f"JSON root must be an object: {path}")
    return loaded


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise H1InputError(f"required CSV file not found: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise H1InputError(f"CSV has no header: {path}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise H1InputError(f"cannot read CSV file {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise H1InputError(f"required JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise H1InputError(
                        f"invalid JSONL line {line_number} in {path}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise H1InputError(
                        f"JSONL line {line_number} in {path} is not an object"
                    )
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise H1InputError(f"cannot read JSONL file {path}: {exc}") from exc
    return rows


def _optional_float(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise H1InputError(f"{label} must be numeric or null")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise H1InputError(f"{label} must be numeric or null") from exc
    if not math.isfinite(converted):
        raise H1InputError(f"{label} must be finite")
    return converted


def _required_float(value: Any, label: str) -> float:
    converted = _optional_float(value, label)
    if converted is None:
        raise H1InputError(f"{label} must not be null")
    return converted


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise H1InputError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise H1InputError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise H1InputError(f"{label} must be an integer")
    return converted


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise H1InputError(f"{label} must be boolean")


def _resolve_path(value: str, baseline_root: Path = BASELINE_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = baseline_root / path
    return path.resolve()


def _input_paths(config_path: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise H1InputError("h1_v1 inputs must be a mapping")
    required = {
        "baseline_gate2_config",
        "plateau_result",
        "plateau_windows",
        "training_metrics",
        "derived_metrics",
        "data_quality_report",
        "replay_iteration_metrics",
        "replay_summary",
        "fixed_basket_checkpoint_summary",
        "fixed_basket_opponent_summary",
        "holdout_checkpoint_metrics",
        "holdout_trajectory_checkpoint_metrics",
        "holdout_manifest",
        "holdout_summary",
    }
    missing = required - inputs.keys()
    if missing:
        raise H1InputError(
            f"h1_v1 inputs missing registrations: {sorted(missing)}"
        )
    resolved = {name: _resolve_path(str(value)) for name, value in inputs.items()}
    resolved["h1_config"] = config_path.expanduser().resolve()
    return resolved


def hash_input_files(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    hashed: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        entry: dict[str, Any] = {"path": path.as_posix()}
        if not path.is_file():
            entry.update({"status": "missing", "size_bytes": None, "sha256": None})
        else:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry.update(
                {
                    "status": "available",
                    "size_bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        hashed[name] = entry
    return hashed


def validate_upstream_statuses(
    *,
    input_hashes: Mapping[str, Mapping[str, Any]],
    plateau: Mapping[str, Any],
    plateau_windows: Sequence[Mapping[str, Any]],
    data_quality: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    fixed_checkpoint_rows: Sequence[Mapping[str, Any]],
    fixed_opponent_rows: Sequence[Mapping[str, Any]],
    holdout_checkpoint_rows: Sequence[Mapping[str, Any]],
    holdout_trajectory_rows: Sequence[Mapping[str, Any]],
    holdout_manifest: Mapping[str, Any],
    holdout_summary: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: str) -> None:
        checks[name] = {
            "status": "passed" if passed else "failed",
            "detail": detail,
        }

    missing = [name for name, entry in input_hashes.items() if entry["status"] != "available"]
    add("input_files", not missing, "all registered inputs available" if not missing else f"missing: {missing}")
    add(
        "plateau",
        plateau.get("status") == "completed" and len(plateau_windows) == 9,
        f"status={plateau.get('status')}, windows={len(plateau_windows)}",
    )
    quality_checks = data_quality.get("checks")
    add(
        "baseline_metrics",
        data_quality.get("status") == "passed"
        and isinstance(quality_checks, dict)
        and all(value is True for value in quality_checks.values()),
        f"status={data_quality.get('status')}",
    )
    replay_validations = replay_summary.get("validations")
    add(
        "replay",
        replay_summary.get("status") == "completed"
        and isinstance(replay_validations, dict)
        and all(value is True for value in replay_validations.values()),
        f"status={replay_summary.get('status')}",
    )
    fixed_ok = (
        len(fixed_checkpoint_rows) == 12
        and len(fixed_opponent_rows) == 48
        and all(_required_int(row.get("faults"), "opponent faults") == 0 for row in fixed_opponent_rows)
    )
    add(
        "fixed_basket",
        fixed_ok,
        f"checkpoint_rows={len(fixed_checkpoint_rows)}, opponent_rows={len(fixed_opponent_rows)}",
    )
    holdout_dataset = holdout_summary.get("dataset")
    holdout_ok = (
        holdout_manifest.get("status") == "completed"
        and holdout_summary.get("status") == "completed"
        and isinstance(holdout_dataset, dict)
        and holdout_dataset.get("hash_verification_status") == "passed"
        and len(holdout_checkpoint_rows) == 12
        and len(holdout_trajectory_rows) == 2400
    )
    add(
        "holdout",
        holdout_ok,
        (
            f"manifest={holdout_manifest.get('status')}, "
            f"summary={holdout_summary.get('status')}, "
            f"checkpoint_rows={len(holdout_checkpoint_rows)}, "
            f"trajectory_rows={len(holdout_trajectory_rows)}"
        ),
    )
    return {
        "status": (
            "passed"
            if all(check["status"] == "passed" for check in checks.values())
            else "failed"
        ),
        "checks": checks,
    }


def build_input_manifest(
    input_hashes: Mapping[str, Mapping[str, Any]],
    upstream_statuses: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_id": "h1_v1",
        "status": upstream_statuses["status"],
        "inputs": dict(input_hashes),
        "upstream_statuses": dict(upstream_statuses),
    }


def resolve_protocol(
    h1_config: Mapping[str, Any],
    gate_config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(h1_config))
    plateau = gate_config.get("plateau")
    if not isinstance(plateau, dict):
        raise H1InputError("baseline_gate2 plateau section is missing")
    resolved["resolved_plateau"] = copy.deepcopy(plateau)
    resolved["resolved_inputs"] = {
        name: path.as_posix() for name, path in input_paths.items()
    }
    return resolved


def build_checkpoint_blocks(checkpoints: Sequence[int]) -> list[dict[str, Any]]:
    values = [_required_int(value, "checkpoint") for value in checkpoints]
    if not values or values[0] != 0:
        raise H1InputError("checkpoint grid must start at 0")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise H1InputError("checkpoint grid must be strictly increasing")
    blocks = []
    for index, checkpoint in enumerate(values):
        if index == 0:
            start = None
            end = None
            iterations = 0
        else:
            start = values[index - 1] + 1
            end = checkpoint
            iterations = end - start + 1
        blocks.append(
            {
                "checkpoint": checkpoint,
                "block_start_iteration": start,
                "block_end_iteration": end,
                "block_iterations": iterations,
                "is_short_block": iterations > 0 and iterations != 20,
            }
        )
    return blocks


def aggregate_ratio_of_sums(
    rows: Sequence[Mapping[str, Any]],
    numerator: str,
    denominator: str,
    *,
    scale: float = 1.0,
) -> float | None:
    if not rows:
        return None
    numerator_sum = sum(
        _required_float(row.get(numerator), numerator) for row in rows
    )
    denominator_sum = sum(
        _required_float(row.get(denominator), denominator) for row in rows
    )
    if denominator_sum == 0.0:
        return None
    return numerator_sum / denominator_sum * scale


def aggregate_iteration_means(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, float | None]:
    aggregated: dict[str, float | None] = {}
    for field in fields:
        values = [_optional_float(row.get(field), field) for row in rows]
        present = [value for value in values if value is not None]
        aggregated[field] = sum(present) / len(present) if present else None
    return aggregated


def aggregate_iteration_endpoints(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, float | None]:
    if not rows:
        return {field: None for field in fields}
    last = max(rows, key=lambda row: _required_int(row.get("iteration"), "iteration"))
    return {field: _optional_float(last.get(field), field) for field in fields}


def _index_iterations(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        iteration = _required_int(row.get("iteration"), f"{label} iteration")
        if iteration in indexed:
            raise H1InputError(f"duplicate {label} iteration {iteration}")
        indexed[iteration] = row
    return indexed


def aggregate_baseline_metrics(
    raw_metrics: Sequence[Mapping[str, Any]],
    derived_metrics: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_by_iteration = _index_iterations(raw_metrics, "raw metric")
    derived_by_iteration = _index_iterations(derived_metrics, "derived metric")
    expected_iterations = set(range(1, 211))
    if set(raw_by_iteration) != expected_iterations:
        raise H1InputError("raw metrics must cover iterations 1..210 exactly")
    if set(derived_by_iteration) != expected_iterations:
        raise H1InputError("derived metrics must cover iterations 1..210 exactly")

    output = []
    for block in blocks:
        row = dict(block)
        start = block["block_start_iteration"]
        end = block["block_end_iteration"]
        if start is None or end is None:
            iterations: list[int] = []
        else:
            iterations = list(range(start, end + 1))
        raw = [raw_by_iteration[value] for value in iterations]
        derived = [derived_by_iteration[value] for value in iterations]
        for value in iterations:
            if _required_int(
                derived_by_iteration[value].get("positions_generated"),
                "derived positions_generated",
            ) != _required_int(
                raw_by_iteration[value].get("positions_generated"),
                "raw positions_generated",
            ):
                raise H1InputError(
                    f"raw and derived positions differ at iteration {value}"
                )
        row.update(
            {
                "positions_generated": (
                    sum(_required_int(item.get("positions_generated"), "positions_generated") for item in raw)
                    if raw
                    else None
                ),
                "games_completed": (
                    sum(_required_int(item.get("games_completed"), "games_completed") for item in raw)
                    if raw
                    else None
                ),
                "optimizer_steps": (
                    sum(_required_int(item.get("optimizer_steps"), "optimizer_steps") for item in raw)
                    if raw
                    else None
                ),
                "fresh_states_per_update": aggregate_ratio_of_sums(
                    raw, "positions_generated", "optimizer_steps"
                ),
                "states_per_gpu_hour": aggregate_ratio_of_sums(
                    raw, "positions_generated", "iteration_seconds", scale=3600.0
                ),
                "games_per_gpu_hour": aggregate_ratio_of_sums(
                    raw, "games_completed", "iteration_seconds", scale=3600.0
                ),
                "self_play_fraction": aggregate_ratio_of_sums(
                    raw, "self_play_seconds", "iteration_seconds"
                ),
                "training_fraction": aggregate_ratio_of_sums(
                    raw, "training_seconds", "iteration_seconds"
                ),
                "buffer_inflow_fraction": aggregate_ratio_of_sums(
                    raw, "positions_generated", "replay_buffer_size"
                ),
                "buffer_fraction_consumed": aggregate_ratio_of_sums(
                    raw, "examples_used", "replay_buffer_size"
                ),
                "mean_sample_exposure": aggregate_ratio_of_sums(
                    raw, "samples_seen", "replay_buffer_size"
                ),
                "selected_sample_reuse": aggregate_ratio_of_sums(
                    raw, "samples_seen", "examples_used"
                ),
            }
        )
        row.update(
            aggregate_iteration_means(
                derived,
                ("mean_sample_age", "p90_sample_age"),
            )
        )
        turnover_rows = [
            item
            for item in derived
            if _required_int(item.get("iteration"), "iteration") >= 151
        ]
        row["turnover_fraction"] = aggregate_iteration_means(
            turnover_rows, ("turnover_fraction",)
        )["turnover_fraction"]
        row["turnover_observable"] = bool(turnover_rows)
        endpoint = aggregate_iteration_endpoints(
            raw, ("cumulative_gpu_hours", "replay_buffer_size")
        )
        row["training_gpu_hours"] = endpoint["cumulative_gpu_hours"]
        row["replay_buffer_size"] = endpoint["replay_buffer_size"]
        output.append(row)
    return output


def _aggregate_replay_metrics(
    replay_metrics: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replay_by_iteration = _index_iterations(replay_metrics, "replay metric")
    if set(replay_by_iteration) != set(range(61, 211)):
        raise H1InputError("replay metrics must cover retained iterations 61..210")
    output = []
    for block in blocks:
        start = block["block_start_iteration"]
        end = block["block_end_iteration"]
        eligible = (
            []
            if start is None or end is None
            else [
                replay_by_iteration[value]
                for value in range(max(start, 61), end + 1)
                if value in replay_by_iteration
            ]
        )
        incoming = [
            row
            for row in eligible
            if not _as_bool(
                row.get("incoming_ratio_left_censored"),
                "incoming_ratio_left_censored",
            )
        ]
        output.append(
            {
                "checkpoint": block["checkpoint"],
                "incoming_unique_state_ratio": aggregate_ratio_of_sums(
                    incoming, "incoming_unique_states", "states"
                ),
                "duplicate_rate": aggregate_ratio_of_sums(
                    eligible, "duplicate_hash_occurrences", "states"
                ),
                "state_effective_ratio": aggregate_ratio_of_sums(
                    eligible, "state_effective_count", "states"
                ),
                "replay_observable": bool(eligible),
            }
        )
    return output


def apply_observability_rules(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the registered null and left-truncation rules without imputation."""
    output: list[dict[str, Any]] = []
    gap_fields = ("approx_policy_gap", "approx_value_gap", "approx_total_gap")
    replay_fields = (
        "mean_sample_exposure",
        "selected_sample_reuse",
        "mean_sample_age",
        "p90_sample_age",
        "incoming_unique_state_ratio",
        "duplicate_rate",
        "state_effective_ratio",
    )
    for source in rows:
        row = dict(source)
        checkpoint = _required_int(row.get("checkpoint"), "checkpoint")
        if checkpoint == 0:
            for field in gap_fields:
                row[field] = None
        if checkpoint < 61:
            for field in replay_fields:
                row[field] = None
            row["replay_observable"] = False
        if checkpoint < 151:
            row["turnover_fraction"] = None
            row["turnover_observable"] = False
        output.append(row)
    return output


def _index_checkpoints(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        checkpoint = _required_int(row.get("checkpoint"), f"{label} checkpoint")
        if checkpoint in indexed:
            raise H1InputError(f"duplicate {label} checkpoint {checkpoint}")
        indexed[checkpoint] = row
    return indexed


def _fixed_basket_values(row: Mapping[str, Any]) -> dict[str, float]:
    opponent_scores = [
        _required_float(row.get(field), field)
        for field in FIXED_OPPONENT_SCORE_COLUMNS
    ]
    macro_score = sum(opponent_scores) / len(opponent_scores)
    registered_score = _required_float(row.get("score_rate"), "score_rate")
    if not math.isclose(macro_score, registered_score, rel_tol=0.0, abs_tol=1e-12):
        raise H1InputError(
            "fixed-basket score_rate does not equal the four-opponent macro score "
            f"at checkpoint {row.get('checkpoint')}"
        )
    return {
        "fixed_basket_macro_score": macro_score,
        "fixed_basket_score_ci95_low": _required_float(
            row.get("score_rate_ci95_low"), "score_rate_ci95_low"
        ),
        "fixed_basket_score_ci95_high": _required_float(
            row.get("score_rate_ci95_high"), "score_rate_ci95_high"
        ),
    }


def merge_checkpoint_sources(
    baseline_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    fixed_basket_rows: Sequence[Mapping[str, Any]],
    plateau_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline = _index_checkpoints(baseline_rows, "baseline")
    replay = _index_checkpoints(replay_rows, "replay")
    holdout = _index_checkpoints(holdout_rows, "holdout")
    fixed = _index_checkpoints(fixed_basket_rows, "fixed-basket")
    expected = set(baseline)
    for label, indexed in (
        ("replay", replay),
        ("holdout", holdout),
        ("fixed-basket", fixed),
    ):
        if set(indexed) != expected:
            raise H1InputError(
                f"{label} checkpoint set does not match the registered grid"
            )

    detected = plateau_result.get("plateau_detected") is True
    plateau_checkpoint = plateau_result.get("plateau_iteration") if detected else None
    if plateau_checkpoint is not None:
        plateau_checkpoint = _required_int(plateau_checkpoint, "plateau_iteration")
        if plateau_checkpoint not in expected:
            raise H1InputError("plateau_iteration is not a registered checkpoint")

    merged: list[dict[str, Any]] = []
    for checkpoint in sorted(expected):
        row = dict(baseline[checkpoint])
        row.update(
            {
                key: value
                for key, value in replay[checkpoint].items()
                if key != "checkpoint"
            }
        )
        holdout_row = holdout[checkpoint]
        for field in (
            "holdout_policy_loss",
            "holdout_value_loss",
            "holdout_total_loss",
            "logged_train_policy_loss",
            "logged_train_value_loss",
            "approx_policy_gap",
            "approx_value_gap",
        ):
            row[field] = _optional_float(holdout_row.get(field), field)
        if row["approx_policy_gap"] is None or row["approx_value_gap"] is None:
            row["approx_total_gap"] = None
        else:
            row["approx_total_gap"] = (
                row["approx_policy_gap"] + row["approx_value_gap"]
            )
        row.update(_fixed_basket_values(fixed[checkpoint]))
        row["is_plateau_start"] = checkpoint == plateau_checkpoint
        row["is_at_or_after_plateau"] = (
            plateau_checkpoint is not None and checkpoint >= plateau_checkpoint
        )
        merged.append(row)
    return apply_observability_rules(merged)


def validate_aligned_table(
    rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]
) -> None:
    expected = [_required_int(value, "checkpoint") for value in checkpoints]
    observed = [_required_int(row.get("checkpoint"), "checkpoint") for row in rows]
    if observed != expected:
        raise H1InputError(
            f"aligned checkpoint table must contain exactly {expected}; got {observed}"
        )
    if len(rows) != 12:
        raise H1InputError(f"aligned checkpoint table must have 12 rows; got {len(rows)}")
    for row in rows:
        checkpoint = _required_int(row.get("checkpoint"), "checkpoint")
        if checkpoint == 0 and any(
            row.get(field) is not None
            for field in ("approx_policy_gap", "approx_value_gap", "approx_total_gap")
        ):
            raise H1InputError("checkpoint 0 train-hold-out gaps must be null")
        if checkpoint < 61 and any(
            row.get(field) is not None
            for field in (
                "mean_sample_exposure",
                "mean_sample_age",
                "p90_sample_age",
                "incoming_unique_state_ratio",
                "duplicate_rate",
            )
        ):
            raise H1InputError("replay metrics before iteration 61 must be null")
        if checkpoint < 151 and row.get("turnover_fraction") is not None:
            raise H1InputError("turnover before iteration 151 must be unobservable")
    final = rows[-1]
    if (
        _required_int(final.get("block_start_iteration"), "block_start_iteration")
        != 201
        or _required_int(final.get("block_end_iteration"), "block_end_iteration")
        != 210
        or _required_int(final.get("block_iterations"), "block_iterations") != 10
        or final.get("is_short_block") is not True
    ):
        raise H1InputError("checkpoint 210 must be registered as the 201..210 short block")


def mark_pretrained_checkpoint(
    rows: Sequence[Mapping[str, Any]], checkpoint: int = 0
) -> list[dict[str, Any]]:
    return [
        {**row, "is_pretrained": _required_int(row.get("checkpoint"), "checkpoint") == checkpoint}
        for row in rows
    ]


def find_best_observed_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    score_field: str = "fixed_basket_macro_score",
) -> int | None:
    candidates = [
        (
            _required_float(row.get(score_field), score_field),
            _required_int(row.get("checkpoint"), "checkpoint"),
        )
        for row in rows
        if row.get(score_field) not in (None, "")
    ]
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    return min(checkpoint for score, checkpoint in candidates if score == best_score)


def mark_final_checkpoint(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    final = max(
        (_required_int(row.get("checkpoint"), "checkpoint") for row in rows),
        default=None,
    )
    return [
        {**row, "is_final": final is not None and _required_int(row.get("checkpoint"), "checkpoint") == final}
        for row in rows
    ]


def calculate_running_best(
    rows: Sequence[Mapping[str, Any]],
    score_field: str = "fixed_basket_macro_score",
) -> list[dict[str, Any]]:
    running: float | None = None
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        score = _optional_float(row.get(score_field), score_field)
        if score is not None:
            running = score if running is None else max(running, score)
        row["running_best_score"] = running
        output.append(row)
    return output


def calculate_drawdowns(
    rows: Sequence[Mapping[str, Any]],
    score_field: str = "fixed_basket_macro_score",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        score = _optional_float(row.get(score_field), score_field)
        running = _optional_float(row.get("running_best_score"), "running_best_score")
        row["drawdown_from_running_best"] = (
            None if score is None or running is None else running - score
        )
        output.append(row)
    return output


def find_maximum_drawdown(
    rows: Sequence[Mapping[str, Any]],
    score_field: str = "fixed_basket_macro_score",
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("drawdown_from_running_best") not in (None, "")
    ]
    if not candidates:
        return None
    maximum = max(
        _required_float(row.get("drawdown_from_running_best"), "drawdown")
        for row in candidates
    )
    trough = min(
        (
            row
            for row in candidates
            if math.isclose(
                _required_float(row.get("drawdown_from_running_best"), "drawdown"),
                maximum,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ),
        key=lambda row: _required_int(row.get("checkpoint"), "checkpoint"),
    )
    trough_checkpoint = _required_int(trough.get("checkpoint"), "checkpoint")
    peak_score = _required_float(trough.get("running_best_score"), "running_best_score")
    peak_checkpoint = min(
        _required_int(row.get("checkpoint"), "checkpoint")
        for row in rows
        if _required_int(row.get("checkpoint"), "checkpoint") <= trough_checkpoint
        and row.get(score_field) not in (None, "")
        and math.isclose(
            _required_float(row.get(score_field), score_field),
            peak_score,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    return {
        "maximum_drawdown": maximum,
        "peak_checkpoint": peak_checkpoint,
        "trough_checkpoint": trough_checkpoint,
    }


def _annotate_strength(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = mark_pretrained_checkpoint(rows)
    best_checkpoint = find_best_observed_checkpoint(output)
    output = [
        {
            **row,
            "is_best_observed": (
                best_checkpoint is not None
                and _required_int(row.get("checkpoint"), "checkpoint") == best_checkpoint
            ),
        }
        for row in output
    ]
    output = mark_final_checkpoint(output)
    output = calculate_drawdowns(calculate_running_best(output))
    maximum = find_maximum_drawdown(output)
    for row in output:
        checkpoint = _required_int(row.get("checkpoint"), "checkpoint")
        row["is_max_drawdown_peak"] = (
            maximum is not None and checkpoint == maximum["peak_checkpoint"]
        )
        row["is_max_drawdown_trough"] = (
            maximum is not None and checkpoint == maximum["trough_checkpoint"]
        )
    return output


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0.0:
        return None
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    ) / denominator


def select_pre_plateau_range(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    plateau_result: Mapping[str, Any],
    minimum_points: int = 4,
) -> dict[str, Any]:
    plateau_detected = plateau_result.get("plateau_detected") is True
    plateau_checkpoint = (
        _required_int(plateau_result.get("plateau_iteration"), "plateau_iteration")
        if plateau_detected
        else None
    )
    eligible = [
        row
        for row in rows
        if row.get(metric) not in (None, "")
        and (
            plateau_checkpoint is None
            or _required_int(row.get("checkpoint"), "checkpoint") <= plateau_checkpoint
        )
    ]
    mode = "pre_plateau" if plateau_detected else "descriptive_full_run"
    limitation = ""
    if not plateau_detected:
        limitation = (
            "Plateau was not reproduced; the full-run effect is descriptive only."
        )
    if len(eligible) < minimum_points:
        shortfall = (
            f"Only {len(eligible)} valid checkpoints are available; "
            f"at least {minimum_points} are required."
        )
        limitation = f"{limitation} {shortfall}".strip()
    return {
        "metric": metric,
        "analysis_mode": mode,
        "rows": eligible,
        "availability_status": (
            "available" if len(eligible) >= minimum_points else "unavailable"
        ),
        "limitation": limitation,
    }


def calculate_metric_effect(
    selected: Mapping[str, Any], metric: str
) -> dict[str, Any]:
    points = list(selected.get("rows", []))
    if not points:
        return {
            "start_checkpoint": None,
            "end_checkpoint": None,
            "valid_checkpoints": len(points),
            "start_value": None,
            "end_value": None,
            "absolute_change": None,
            "relative_change": None,
            "slope_per_20_iterations": None,
            "slope_per_gpu_hour": None,
        }
    start = points[0]
    end = points[-1]
    start_value = _required_float(start.get(metric), metric)
    end_value = _required_float(end.get(metric), metric)
    if selected.get("availability_status") == "available":
        checkpoints = [
            float(_required_int(row.get("checkpoint"), "checkpoint")) for row in points
        ]
        values = [_required_float(row.get(metric), metric) for row in points]
        iteration_slope = _ols_slope(checkpoints, values)
        gpu_points = [
            (row.get("training_gpu_hours"), row.get(metric))
            for row in points
            if row.get("training_gpu_hours") not in (None, "")
        ]
        gpu_slope = _ols_slope(
            [_required_float(value[0], "training_gpu_hours") for value in gpu_points],
            [_required_float(value[1], metric) for value in gpu_points],
        )
    else:
        iteration_slope = None
        gpu_slope = None
    return {
        "start_checkpoint": _required_int(start.get("checkpoint"), "checkpoint"),
        "end_checkpoint": _required_int(end.get("checkpoint"), "checkpoint"),
        "valid_checkpoints": len(points),
        "start_value": start_value,
        "end_value": end_value,
        "absolute_change": end_value - start_value,
        "relative_change": (
            None if start_value == 0.0 else (end_value - start_value) / abs(start_value)
        ),
        "slope_per_20_iterations": (
            None if iteration_slope is None else iteration_slope * 20.0
        ),
        "slope_per_gpu_hour": gpu_slope,
    }


def classify_metric_direction(
    change: float | None, expected_direction: str
) -> dict[str, Any]:
    if change is None:
        return {"actual_direction": None, "direction_consistent": None}
    tolerance = 1e-12
    if change > tolerance:
        actual = "increase"
    elif change < -tolerance:
        actual = "decrease"
    else:
        actual = "flat"
    if expected_direction == "increase":
        consistent = actual == "increase"
    elif expected_direction == "decrease":
        consistent = actual == "decrease"
    elif expected_direction == "not_increase":
        consistent = actual in {"decrease", "flat"}
    else:
        raise H1InputError(f"unsupported expected direction: {expected_direction}")
    return {"actual_direction": actual, "direction_consistent": consistent}


def _metric_limitation(metric: str) -> str:
    if metric.startswith("approx_"):
        return "This is an approximate online train–hold-out gap, not an exact paired loss gap."
    if metric == "turnover_fraction":
        return "Turnover is interpretable only from iteration 151 after the buffer fills."
    if metric in {
        "mean_sample_exposure",
        "mean_sample_age",
        "p90_sample_age",
        "incoming_unique_state_ratio",
        "duplicate_rate",
        "state_effective_ratio",
    }:
        return "Replay evidence is left-truncated at retained iteration 61."
    return ""


def build_effect_rows(
    rows: Sequence[Mapping[str, Any]],
    plateau_result: Mapping[str, Any],
    minimum_points: int = 4,
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for stage, metric, expected in EFFECT_SPECS:
        selected = select_pre_plateau_range(
            rows, metric, plateau_result, minimum_points=minimum_points
        )
        effect = calculate_metric_effect(selected, metric)
        direction = classify_metric_direction(
            (
                effect["absolute_change"]
                if selected["availability_status"] == "available"
                else None
            ),
            expected,
        )
        limitations = [selected.get("limitation", ""), _metric_limitation(metric)]
        if any(
            _required_int(row.get("checkpoint"), "checkpoint") == 210
            for row in selected.get("rows", [])
        ):
            limitations.append("Checkpoint 210 represents a 10-iteration block.")
        effects.append(
            {
                "evidence_stage": stage,
                "metric": metric,
                "analysis_mode": selected["analysis_mode"],
                **effect,
                "expected_direction": expected,
                **direction,
                "availability_status": selected["availability_status"],
                "limitation": " ".join(value for value in limitations if value),
            }
        )
    return effects


def _effect_by_metric(
    effects: Sequence[Mapping[str, Any]], metric: str
) -> Mapping[str, Any] | None:
    matches = [row for row in effects if row.get("metric") == metric]
    if len(matches) > 1:
        raise H1InputError(f"duplicate effect row for {metric}")
    return matches[0] if matches else None


def _metric_evidence(
    effects: Sequence[Mapping[str, Any]], metric: str
) -> bool | None:
    row = _effect_by_metric(effects, metric)
    if row is None or row.get("availability_status") != "available":
        return None
    value = row.get("direction_consistent")
    if value is None or isinstance(value, bool):
        return value
    return _as_bool(value, f"{metric} direction_consistent")


def _stage_result(
    stage: str,
    status: str,
    metrics: Mapping[str, bool | None],
    rationale: str,
) -> dict[str, Any]:
    if status not in {"consistent", "mixed", "inconsistent", "unavailable"}:
        raise H1InputError(f"invalid stage status: {status}")
    return {
        "stage": stage,
        "status": status,
        "metric_consistency": dict(metrics),
        "rationale": rationale,
    }


def judge_fresh_state_supply(
    effects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = {
        metric: _metric_evidence(effects, metric)
        for metric in (
            "fresh_states_per_update",
            "buffer_inflow_fraction",
            "states_per_gpu_hour",
        )
    }
    core = metrics["fresh_states_per_update"]
    supplements = [
        value for key, value in metrics.items() if key != "fresh_states_per_update" and value is not None
    ]
    if core is None:
        status = "unavailable"
        rationale = "The core fresh-states-per-update effect has fewer than four valid checkpoints."
    elif core is False:
        status = "inconsistent"
        rationale = "The core fresh-states-per-update metric moves opposite to H1."
    elif any(value is False for value in supplements):
        status = "mixed"
        rationale = "The core supply metric is consistent, but a supporting efficiency metric is not."
    else:
        status = "consistent"
        rationale = "The core supply metric decreases without contradictory available supplements."
    return _stage_result("fresh_state_supply", status, metrics, rationale)


def judge_replay_pressure(
    effects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core_names = (
        "mean_sample_exposure",
        "mean_sample_age",
        "p90_sample_age",
    )
    supplemental_names = (
        "turnover_fraction",
        "duplicate_rate",
        "incoming_unique_state_ratio",
        "state_effective_ratio",
    )
    metrics = {
        metric: _metric_evidence(effects, metric)
        for metric in core_names + supplemental_names
    }
    core = [metrics[name] for name in core_names if metrics[name] is not None]
    supplements = [
        metrics[name] for name in supplemental_names if metrics[name] is not None
    ]
    if not core:
        status = "unavailable"
        rationale = "No replay-pressure core metric has four valid checkpoints."
    elif all(value is True for value in core) and not any(
        value is False for value in supplements
    ):
        status = "consistent"
        rationale = "Exposure and available sample-age evidence rise as expected."
    elif all(value is False for value in core):
        status = "inconsistent"
        rationale = "All available replay-pressure core metrics move opposite to H1."
    else:
        status = "mixed"
        rationale = "Available replay-pressure and supplementary metrics do not agree."
    return _stage_result("replay_pressure", status, metrics, rationale)


def judge_generalisation(
    effects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = ("approx_policy_gap", "approx_value_gap", "approx_total_gap")
    metrics = {metric: _metric_evidence(effects, metric) for metric in names}
    primary = [metrics[name] for name in names[:2] if metrics[name] is not None]
    if not primary:
        status = "unavailable"
        rationale = "The approximate online train–hold-out gaps have fewer than four valid checkpoints."
    elif all(value is True for value in primary):
        status = "consistent"
        rationale = "The available approximate online train–hold-out gaps widen."
    elif all(value is False for value in primary):
        status = "inconsistent"
        rationale = "The available approximate online train–hold-out gaps narrow."
    else:
        status = "mixed"
        rationale = "Policy and value approximate online train–hold-out gaps disagree."
    return _stage_result("generalisation", status, metrics, rationale)


def judge_temporal_order(
    effects: Sequence[Mapping[str, Any]],
    plateau_result: Mapping[str, Any],
) -> dict[str, Any]:
    if plateau_result.get("plateau_detected") is not True:
        return _stage_result(
            "temporal_order",
            "unavailable",
            {},
            "Temporal order cannot be judged because the plateau was not reproduced.",
        )
    plateau_checkpoint = _required_int(
        plateau_result.get("plateau_iteration"), "plateau_iteration"
    )
    signal_rows = [
        row
        for row in effects
        if row.get("availability_status") == "available"
        and row.get("direction_consistent") is True
        and row.get("end_checkpoint") not in (None, "")
    ]
    checkpoints = {
        str(row.get("metric")): _required_int(
            row.get("end_checkpoint"), "end_checkpoint"
        )
        for row in signal_rows
    }
    if not checkpoints:
        status = "unavailable"
        rationale = "No diagnostic trend has four valid points on which to locate an H1-consistent signal."
    elif all(value <= plateau_checkpoint for value in checkpoints.values()):
        status = "consistent"
        relation = "before" if any(
            value < plateau_checkpoint for value in checkpoints.values()
        ) else "at"
        rationale = f"Available diagnostic signals appear {relation} or no later than the plateau start."
    elif all(value > plateau_checkpoint for value in checkpoints.values()):
        status = "inconsistent"
        rationale = "Available diagnostic signals appear only after the plateau start."
    else:
        status = "mixed"
        rationale = "Available diagnostic signals occur on both sides of the plateau start."
    return _stage_result("temporal_order", status, checkpoints, rationale)


def _status_value(value: str | Mapping[str, Any]) -> str:
    if isinstance(value, str):
        return value
    status = value.get("status")
    if not isinstance(status, str):
        raise H1InputError("stage result has no status")
    return status


def judge_h1(
    *,
    input_status: str,
    plateau_detected: bool,
    supply: str | Mapping[str, Any],
    replay: str | Mapping[str, Any],
    generalisation: str | Mapping[str, Any],
    temporal_order: str | Mapping[str, Any],
) -> str:
    if input_status != "passed" or not plateau_detected:
        return "not_assessable"
    statuses = {
        "supply": _status_value(supply),
        "replay": _status_value(replay),
        "generalisation": _status_value(generalisation),
        "temporal_order": _status_value(temporal_order),
    }
    if statuses["supply"] == "inconsistent":
        return "not_supported"
    if all(value == "consistent" for value in statuses.values()):
        return "supported_with_limitations"
    if sum(value == "inconsistent" for value in statuses.values()) >= 2:
        return "not_supported"
    if statuses["supply"] == "consistent" and any(
        statuses[name] in {"mixed", "unavailable"}
        for name in ("replay", "generalisation", "temporal_order")
    ):
        return "partially_supported"
    if statuses["supply"] == "unavailable":
        return "not_assessable"
    return "partially_supported"


def build_h1_decision(
    *,
    input_status: str,
    plateau_result: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    maximum_drawdown: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plateau_detected = plateau_result.get("plateau_detected") is True
    status = judge_h1(
        input_status=input_status,
        plateau_detected=plateau_detected,
        supply=stages["fresh_state_supply"],
        replay=stages["replay_pressure"],
        generalisation=stages["generalisation"],
        temporal_order=stages["temporal_order"],
    )
    return {
        "schema_version": 1,
        "analysis_id": "h1_v1",
        "hypothesis": "H1",
        "status": status,
        "analysis_mode": (
            "pre_plateau"
            if plateau_detected
            else "descriptive_full_run"
        ),
        "input_status": input_status,
        "plateau": {
            "status": plateau_result.get("status"),
            "detected": plateau_detected,
            "start_checkpoint": plateau_result.get("plateau_iteration"),
        },
        "evidence_stages": dict(stages),
        "maximum_drawdown": (
            dict(maximum_drawdown) if maximum_drawdown is not None else None
        ),
        "limitations": [
            "Hold-out gaps are approximate online train–hold-out gaps.",
            "Replay evidence is left-truncated at iteration 61.",
            "Turnover is observable only from iteration 151.",
        ],
    }


def build_h1_summary(
    decision: Mapping[str, Any], effects: Sequence[Mapping[str, Any]]
) -> str:
    plateau = decision["plateau"]
    stages = decision["evidence_stages"]
    available = sum(
        row.get("availability_status") == "available" for row in effects
    )
    lines = [
        "# H1 evidence summary",
        "",
        f"- Final status: `{decision['status']}`",
        f"- Analysis mode: `{decision['analysis_mode']}`",
        f"- Input audit: `{decision['input_status']}`",
        f"- Plateau reproduced: `{str(plateau['detected']).lower()}`",
        f"- Plateau start checkpoint: `{plateau['start_checkpoint']}`",
        f"- Available metric effects: `{available}/{len(effects)}`",
        "",
        "## Evidence stages",
        "",
    ]
    for name in (
        "fresh_state_supply",
        "replay_pressure",
        "generalisation",
        "temporal_order",
    ):
        stage = stages[name]
        lines.append(
            f"- **{name.replace('_', ' ').title()}**: `{stage['status']}` — {stage['rationale']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Generalisation evidence is an approximate online train–hold-out gap, not an exact paired loss gap.",
            "- Replay evidence is left-truncated at iteration 61; turnover becomes observable at iteration 151.",
            "- Metrics with fewer than four valid checkpoints are unavailable and are not force-classified.",
        ]
    )
    if decision["analysis_mode"] == "descriptive_full_run":
        lines.append(
            "- Because the plateau was not reproduced, full-run effects are descriptive only and do not support an H1 temporal claim."
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _output_paths(
    config: Mapping[str, Any], output_dir: Path | None
) -> dict[str, Path]:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise H1InputError("h1_v1 outputs must be a mapping")
    names = (
        "resolved_protocol",
        "input_manifest",
        "aligned_checkpoint_metrics",
        "h1_effects",
        "h1_decision",
        "h1_summary",
    )
    missing = [name for name in names if name not in outputs]
    if missing:
        raise H1InputError(f"h1_v1 outputs missing registrations: {missing}")
    if output_dir is None:
        return {name: _resolve_path(str(outputs[name])) for name in names}
    root = output_dir.expanduser().resolve()
    return {name: root / Path(str(outputs[name])).name for name in names}


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return hash_input_files(paths)


def _build_gate_summary(
    *,
    upstream_statuses: Mapping[str, Any],
    plateau_result: Mapping[str, Any],
    decision: Mapping[str, Any],
    package_root: Path,
    artifact_hashes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_id": "baseline_gate2",
        "status": (
            "completed" if upstream_statuses.get("status") == "passed" else "failed"
        ),
        "upstream_status": upstream_statuses.get("status"),
        "upstream_checks": upstream_statuses.get("checks", {}),
        "plateau": {
            "status": plateau_result.get("status"),
            "detected": plateau_result.get("plateau_detected") is True,
            "start_checkpoint": plateau_result.get("plateau_iteration"),
        },
        "h1": {
            "status": decision.get("status"),
            "analysis_mode": decision.get("analysis_mode"),
        },
        "h1_package": package_root.as_posix(),
        "artifacts": dict(artifact_hashes),
    }


def _failed_stages(reason: str) -> dict[str, dict[str, Any]]:
    return {
        name: _stage_result(name, "unavailable", {}, reason)
        for name in (
            "fresh_state_supply",
            "replay_pressure",
            "generalisation",
            "temporal_order",
        )
    }


def _gate_summary_path(
    gate_config: Mapping[str, Any], override: Path | None
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    gate_outputs = gate_config.get("outputs")
    if not isinstance(gate_outputs, dict) or "summary" not in gate_outputs:
        raise H1InputError("baseline_gate2 output summary is not registered")
    return _resolve_path(str(gate_outputs["summary"]))


def _write_failed_package(
    *,
    reason: str,
    output_paths: Mapping[str, Path],
    input_hashes: Mapping[str, Mapping[str, Any]],
    plateau_result: Mapping[str, Any],
    gate_config: Mapping[str, Any] | None,
    gate_summary_override: Path | None,
) -> None:
    upstream = {
        "status": "failed",
        "checks": {
            "merge_input_audit": {"status": "failed", "detail": reason}
        },
    }
    _write_json(
        output_paths["input_manifest"],
        build_input_manifest(input_hashes, upstream),
    )
    _write_csv(output_paths["aligned_checkpoint_metrics"], [], ALIGNED_FIELDS)
    _write_csv(output_paths["h1_effects"], [], EFFECT_FIELDS)
    stages = _failed_stages(reason)
    decision = build_h1_decision(
        input_status="failed",
        plateau_result=plateau_result,
        stages=stages,
        maximum_drawdown=None,
    )
    _write_json(output_paths["h1_decision"], decision)
    output_paths["h1_summary"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["h1_summary"].write_text(
        build_h1_summary(decision, []), encoding="utf-8"
    )
    if gate_config is not None:
        gate_summary = _build_gate_summary(
            upstream_statuses=upstream,
            plateau_result=plateau_result,
            decision=decision,
            package_root=output_paths["h1_decision"].parent,
            artifact_hashes=_artifact_hashes(output_paths),
        )
        _write_json(
            _gate_summary_path(gate_config, gate_summary_override), gate_summary
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    h1_config: dict[str, Any] | None = None
    gate_config: dict[str, Any] | None = None
    paths: dict[str, Path] = {}
    output_paths: dict[str, Path] = {}
    input_hashes: dict[str, dict[str, Any]] = {}
    plateau: dict[str, Any] = {
        "status": "unavailable",
        "plateau_detected": False,
        "plateau_iteration": None,
    }
    try:
        config_path = args.config.expanduser().resolve()
        h1_config = _load_yaml(config_path)
        paths = _input_paths(config_path, h1_config)
        output_paths = _output_paths(h1_config, args.output_dir)
        input_hashes = hash_input_files(paths)
        if input_hashes["baseline_gate2_config"]["status"] == "available":
            gate_config = _load_yaml(paths["baseline_gate2_config"])
            _write_yaml(
                output_paths["resolved_protocol"],
                resolve_protocol(h1_config, gate_config, paths),
            )
        missing = [
            name for name, entry in input_hashes.items() if entry["status"] != "available"
        ]
        if missing:
            raise H1InputError(f"registered inputs are missing: {missing}")

        plateau = _load_json(paths["plateau_result"])
        plateau_windows = _load_csv(paths["plateau_windows"])
        raw_metrics = _load_jsonl(paths["training_metrics"])
        derived_metrics = _load_csv(paths["derived_metrics"])
        data_quality = _load_json(paths["data_quality_report"])
        replay_metrics = _load_csv(paths["replay_iteration_metrics"])
        replay_summary = _load_json(paths["replay_summary"])
        fixed_checkpoints = _load_csv(paths["fixed_basket_checkpoint_summary"])
        fixed_opponents = _load_csv(paths["fixed_basket_opponent_summary"])
        holdout_checkpoints = _load_csv(paths["holdout_checkpoint_metrics"])
        holdout_trajectory = _load_csv(
            paths["holdout_trajectory_checkpoint_metrics"]
        )
        holdout_manifest = _load_json(paths["holdout_manifest"])
        holdout_summary = _load_json(paths["holdout_summary"])

        upstream = validate_upstream_statuses(
            input_hashes=input_hashes,
            plateau=plateau,
            plateau_windows=plateau_windows,
            data_quality=data_quality,
            replay_summary=replay_summary,
            fixed_checkpoint_rows=fixed_checkpoints,
            fixed_opponent_rows=fixed_opponents,
            holdout_checkpoint_rows=holdout_checkpoints,
            holdout_trajectory_rows=holdout_trajectory,
            holdout_manifest=holdout_manifest,
            holdout_summary=holdout_summary,
        )
        manifest = build_input_manifest(input_hashes, upstream)
        resolved_protocol = resolve_protocol(h1_config, gate_config, paths)
        _write_json(output_paths["input_manifest"], manifest)
        _write_yaml(output_paths["resolved_protocol"], resolved_protocol)

        if upstream["status"] != "passed":
            _write_failed_package(
                reason="An upstream audit check failed.",
                output_paths=output_paths,
                input_hashes=input_hashes,
                plateau_result=plateau,
                gate_config=gate_config,
                gate_summary_override=args.gate_summary,
            )
            return 2

        checkpoint_grid = h1_config.get("checkpoint_grid", {}).get("iterations")
        if not isinstance(checkpoint_grid, list):
            raise H1InputError("h1_v1 checkpoint grid is missing")
        blocks = build_checkpoint_blocks(checkpoint_grid)
        baseline_rows = aggregate_baseline_metrics(
            raw_metrics, derived_metrics, blocks
        )
        replay_rows = _aggregate_replay_metrics(replay_metrics, blocks)
        aligned = merge_checkpoint_sources(
            baseline_rows,
            replay_rows,
            holdout_checkpoints,
            fixed_checkpoints,
            plateau,
        )
        validate_aligned_table(aligned, checkpoint_grid)
        aligned = _annotate_strength(aligned)
        minimum_points = _required_int(
            h1_config.get("trend_analysis", {}).get("minimum_valid_checkpoints"),
            "minimum_valid_checkpoints",
        )
        effects = build_effect_rows(
            aligned, plateau, minimum_points=minimum_points
        )
        stages = {
            "fresh_state_supply": judge_fresh_state_supply(effects),
            "replay_pressure": judge_replay_pressure(effects),
            "generalisation": judge_generalisation(effects),
        }
        stages["temporal_order"] = judge_temporal_order(effects, plateau)
        maximum_drawdown = find_maximum_drawdown(aligned)
        decision = build_h1_decision(
            input_status=upstream["status"],
            plateau_result=plateau,
            stages=stages,
            maximum_drawdown=maximum_drawdown,
        )

        _write_csv(output_paths["aligned_checkpoint_metrics"], aligned, ALIGNED_FIELDS)
        _write_csv(output_paths["h1_effects"], effects, EFFECT_FIELDS)
        _write_json(output_paths["h1_decision"], decision)
        output_paths["h1_summary"].write_text(
            build_h1_summary(decision, effects), encoding="utf-8"
        )

        artifact_hashes = _artifact_hashes(output_paths)
        gate_summary_path = _gate_summary_path(gate_config, args.gate_summary)
        gate_summary = _build_gate_summary(
            upstream_statuses=upstream,
            plateau_result=plateau,
            decision=decision,
            package_root=output_paths["h1_decision"].parent,
            artifact_hashes=artifact_hashes,
        )
        _write_json(gate_summary_path, gate_summary)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "h1_status": decision["status"],
                    "output_dir": output_paths["h1_decision"].parent.as_posix(),
                    "gate_summary": gate_summary_path.as_posix(),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except H1InputError as exc:
        if output_paths:
            _write_failed_package(
                reason=str(exc),
                output_paths=output_paths,
                input_hashes=input_hashes,
                plateau_result=plateau,
                gate_config=gate_config,
                gate_summary_override=args.gate_summary,
            )
        print(f"H1 input audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
