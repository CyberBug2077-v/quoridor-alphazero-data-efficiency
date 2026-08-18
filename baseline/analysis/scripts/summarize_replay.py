from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = BASELINE_ROOT / "outputs" / "baseline_reproduction_seed1001_4090"
DEFAULT_REPLAY = DEFAULT_RUN_DIR / "checkpoints" / "latest.examples"
DEFAULT_METRICS = DEFAULT_RUN_DIR / "metrics.jsonl"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "resolved_config.yaml"
DEFAULT_OUTPUT_DIR = (
    BASELINE_ROOT / "outputs" / "baseline_seed1001_4090_analysis" / "replay"
)

ITERATION_FIELDS = (
    "iteration",
    "states",
    "unique_canonical_states",
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

TRAJECTORY_FIELDS = (
    "iteration",
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

LIMITATIONS = (
    "The initial states from rounds 1–60 have been pruned by the rolling window.",
    "Early buffer-level diversity cannot be recovered.",
    "It is not possible to recover how many times each state was actually sampled during training.",
)


class ReplayAnalysisError(ValueError):
    """Raised when replay inputs cannot be safely summarized."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the final rolling replay without exporting states."
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-iteration", type=int, default=210)
    parser.add_argument("--expected-history-buckets", type=int, default=150)
    parser.add_argument("--expected-total-states", type=int, default=284234)
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
            json.dump(
                payload,
                destination,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            destination.write("\n")
        temporary.replace(path)
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
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayAnalysisError(f"resolved configuration not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ReplayAnalysisError(f"invalid resolved configuration: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ReplayAnalysisError("resolved configuration must contain a mapping")
    return loaded


def _load_metrics(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ReplayAnalysisError(f"metrics file not found: {path}")
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayAnalysisError(
                    f"invalid JSON on metrics line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ReplayAnalysisError(
                    f"metrics line {line_number} must contain an object"
                )
            iteration = record.get("iteration")
            if (
                isinstance(iteration, bool)
                or not isinstance(iteration, int)
                or iteration in records
            ):
                raise ReplayAnalysisError(
                    f"invalid or duplicate iteration on metrics line {line_number}"
                )
            records[iteration] = record
    return records


def _load_replay(path: Path) -> tuple[int, list[Any]]:
    if not path.is_file():
        raise ReplayAnalysisError(f"replay snapshot not found: {path}")
    try:
        with path.open("rb") as source:
            loaded = pickle.load(source)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise ReplayAnalysisError(f"cannot load replay snapshot: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ReplayAnalysisError("replay snapshot must contain a mapping")
    iteration = loaded.get("iteration")
    buckets = loaded.get("examples")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise ReplayAnalysisError("replay snapshot has an invalid iteration")
    if not isinstance(buckets, (list, tuple)):
        raise ReplayAnalysisError("replay snapshot examples must be a sequence")
    return iteration, list(buckets)


def canonical_state_hash(board: Any) -> bytes:
    array = np.asarray(board)
    if array.dtype.hasobject:
        raise ReplayAnalysisError("canonical board has object dtype")
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("utf-8"))
    digest.update(",".join(str(dimension) for dimension in canonical.shape).encode("utf-8"))
    digest.update(canonical.tobytes(order="C"))
    return digest.digest()


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise ReplayAnalysisError("cannot calculate a quantile of an empty list")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _finish_game(
    lengths: list[int],
    observed_length: int,
    declared_length: int | None,
    anomalous: bool,
) -> int:
    lengths.append(observed_length)
    if declared_length != observed_length:
        anomalous = True
    return int(anomalous)


def _analyze_bucket(
    bucket: Any,
    *,
    expected_shape: tuple[int, int, int],
) -> tuple[Counter[bytes], list[int], Counter[str], int]:
    state_counts: Counter[bytes] = Counter()
    anomaly_types: Counter[str] = Counter()
    game_lengths: list[int] = []
    anomalous_games = 0
    active = False
    observed_length = 0
    declared_length: int | None = None
    previous_step = 0
    current_game_anomalous = False

    try:
        samples = iter(bucket)
    except TypeError as exc:
        raise ReplayAnalysisError("replay history bucket is not iterable") from exc

    for sample in samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 6:
            anomaly_types["malformed_sample"] += 1
            if active:
                current_game_anomalous = True
            continue

        board, step_value, game_length_value = sample[0], sample[4], sample[5]
        try:
            array = np.asarray(board)
            if tuple(array.shape) != expected_shape:
                anomaly_types["unexpected_board_shape"] += 1
            state_counts[canonical_state_hash(array)] += 1
        except (ReplayAnalysisError, TypeError, ValueError):
            anomaly_types["invalid_canonical_board"] += 1

        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, numbers.Integral)
            or isinstance(game_length_value, bool)
            or not isinstance(game_length_value, numbers.Integral)
        ):
            anomaly_types["invalid_step_metadata"] += 1
            if active:
                current_game_anomalous = True
            continue
        step = int(step_value)
        game_length = int(game_length_value)
        if step < 1 or game_length < 1 or step > game_length:
            anomaly_types["invalid_step_range"] += 1
            if active:
                current_game_anomalous = True
            continue

        if step == 1:
            if active:
                anomaly_types["premature_step_reset"] += 1
                anomalous_games += _finish_game(
                    game_lengths,
                    observed_length,
                    declared_length,
                    True,
                )
            active = True
            observed_length = 0
            declared_length = game_length
            previous_step = 0
            current_game_anomalous = False
        elif not active:
            anomaly_types["missing_step_reset"] += 1
            active = True
            observed_length = 0
            declared_length = game_length
            previous_step = 0
            current_game_anomalous = True

        if step != previous_step + 1:
            anomaly_types["nonconsecutive_step"] += 1
            current_game_anomalous = True
        if declared_length != game_length:
            anomaly_types["inconsistent_declared_game_length"] += 1
            current_game_anomalous = True
        observed_length += 1
        previous_step = step

        if step == game_length:
            anomalous_games += _finish_game(
                game_lengths,
                observed_length,
                declared_length,
                current_game_anomalous,
            )
            active = False
            observed_length = 0
            declared_length = None
            previous_step = 0
            current_game_anomalous = False

    if active:
        anomaly_types["incomplete_final_game"] += 1
        anomalous_games += _finish_game(
            game_lengths,
            observed_length,
            declared_length,
            True,
        )
    return state_counts, game_lengths, anomaly_types, anomalous_games


def summarize_replay(
    replay_path: Path,
    metrics_path: Path,
    resolved_config_path: Path,
    output_dir: Path,
    *,
    expected_iteration: int = 210,
    expected_history_buckets: int = 150,
    expected_total_states: int = 284234,
) -> dict[str, Any]:
    replay_path = replay_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    resolved_config_path = resolved_config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    config = _load_yaml(resolved_config_path)
    metrics = _load_metrics(metrics_path)
    try:
        configured_history = int(config["replay"]["history_iterations"])
        games_per_iteration = int(config["self_play"]["games_per_iteration"])
        board_size = int(config["model"]["board_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayAnalysisError(
            "resolved configuration lacks replay history, games per iteration, "
            "or board size"
        ) from exc
    if configured_history != expected_history_buckets:
        raise ReplayAnalysisError(
            f"configured replay history is {configured_history}, expected "
            f"{expected_history_buckets}"
        )

    replay_sha256 = _sha256(replay_path)
    iteration, buckets = _load_replay(replay_path)
    history_buckets = len(buckets)
    recovered_start = iteration - history_buckets + 1
    recovered_end = iteration
    expected_shape = (4, 2 * board_size - 1, 2 * board_size - 1)

    iteration_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    global_counts: Counter[bytes] = Counter()
    seen_hashes: set[bytes] = set()
    all_anomalies: Counter[str] = Counter()
    empty_buckets = 0
    empty_games_total = 0
    anomalous_games_total = 0
    total_states = 0
    count_matches_metrics = True
    trajectory_matches_metrics = True

    for bucket_index, bucket in enumerate(buckets):
        bucket_iteration = recovered_start + bucket_index
        state_counts, game_lengths, anomalies, anomalous_games = _analyze_bucket(
            bucket,
            expected_shape=expected_shape,
        )
        bucket_states = sum(state_counts.values())
        if bucket_states == 0:
            empty_buckets += 1
        unique_states = len(state_counts)
        incoming_unique = len(set(state_counts) - seen_hashes)
        duplicate_occurrences = bucket_states - unique_states
        duplicate_groups = sum(1 for count in state_counts.values() if count > 1)
        squared_frequency_sum = sum(count * count for count in state_counts.values())
        effective_count = (
            bucket_states * bucket_states / squared_frequency_sum
            if squared_frequency_sum
            else 0.0
        )
        metric = metrics.get(bucket_iteration, {})
        metric_positions = metric.get("positions_generated")
        bucket_count_matches = metric_positions == bucket_states
        count_matches_metrics = count_matches_metrics and bucket_count_matches

        iteration_rows.append(
            {
                "iteration": bucket_iteration,
                "states": bucket_states,
                "unique_canonical_states": unique_states,
                "incoming_unique_states": incoming_unique,
                "incoming_unique_state_ratio": (
                    incoming_unique / bucket_states if bucket_states else 0.0
                ),
                "incoming_ratio_left_censored": bucket_index == 0,
                "duplicate_hash_occurrences": duplicate_occurrences,
                "duplicate_hash_groups": duplicate_groups,
                "duplicate_rate": (
                    duplicate_occurrences / bucket_states if bucket_states else 0.0
                ),
                "state_effective_count": effective_count,
                "metrics_positions_generated": metric_positions,
                "count_matches_metrics": bucket_count_matches,
            }
        )

        recovered_games = len(game_lengths)
        empty_games = max(0, games_per_iteration - recovered_games)
        empty_games_total += empty_games
        anomalous_games_total += anomalous_games
        metric_games = metric.get("games_completed")
        games_match = metric_games == recovered_games
        recovered_mean = statistics.fmean(game_lengths) if game_lengths else 0.0
        recovered_min = min(game_lengths) if game_lengths else 0
        recovered_max = max(game_lengths) if game_lengths else 0
        length_match = (
            metric.get("min_game_length") == recovered_min
            and metric.get("max_game_length") == recovered_max
            and isinstance(metric.get("mean_game_length"), (int, float))
            and math.isclose(
                float(metric["mean_game_length"]),
                recovered_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        trajectory_matches_metrics = (
            trajectory_matches_metrics and games_match and length_match
        )
        trajectory_rows.append(
            {
                "iteration": bucket_iteration,
                "expected_games": games_per_iteration,
                "recovered_games": recovered_games,
                "empty_games": empty_games,
                "anomalous_games": anomalous_games,
                "min_game_length": recovered_min,
                "max_game_length": recovered_max,
                "mean_game_length": recovered_mean,
                "median_game_length": (
                    statistics.median(game_lengths) if game_lengths else 0.0
                ),
                "p90_game_length": (
                    _nearest_rank(game_lengths, 0.90) if game_lengths else 0
                ),
                "total_recovered_positions": sum(game_lengths),
                "metrics_games_completed": metric_games,
                "metrics_min_game_length": metric.get("min_game_length"),
                "metrics_max_game_length": metric.get("max_game_length"),
                "metrics_mean_game_length": metric.get("mean_game_length"),
                "games_match_metrics": games_match,
                "length_distribution_matches_metrics": length_match,
            }
        )

        total_states += bucket_states
        global_counts.update(state_counts)
        seen_hashes.update(state_counts)
        all_anomalies.update(anomalies)

    final_unique_states = len(global_counts)
    duplicate_hash_groups = sum(1 for count in global_counts.values() if count > 1)
    duplicate_hash_occurrences = total_states - final_unique_states
    validations = {
        "iteration_is_expected": iteration == expected_iteration,
        "history_bucket_count_is_expected": history_buckets
        == expected_history_buckets,
        "recovered_range_is_expected": (
            recovered_start == expected_iteration - expected_history_buckets + 1
            and recovered_end == expected_iteration
        ),
        "total_state_count_is_expected": total_states == expected_total_states,
        "no_empty_buckets": empty_buckets == 0,
        "counts_match_metrics": count_matches_metrics,
        "trajectories_match_metrics": trajectory_matches_metrics,
        "no_empty_games": empty_games_total == 0,
        "no_trajectory_anomalies": (
            anomalous_games_total == 0 and sum(all_anomalies.values()) == 0
        ),
    }
    status = "completed" if all(validations.values()) else "failed"

    iteration_path = output_dir / "replay_iteration_stats.csv"
    summary_path = output_dir / "replay_final_summary.json"
    trajectory_path = output_dir / "trajectory_stats.csv"
    summary = {
        "schema_version": 1,
        "status": status,
        "inputs": {
            "replay_snapshot": replay_path.as_posix(),
            "replay_snapshot_sha256": replay_sha256,
            "metrics_jsonl": metrics_path.as_posix(),
            "metrics_jsonl_sha256": _sha256(metrics_path),
            "resolved_config_yaml": resolved_config_path.as_posix(),
            "resolved_config_yaml_sha256": _sha256(resolved_config_path),
        },
        "replay": {
            "iteration": iteration,
            "history_buckets": history_buckets,
            "recovered_iteration_start": recovered_start,
            "recovered_iteration_end": recovered_end,
            "total_states": total_states,
            "empty_buckets": empty_buckets,
        },
        "state_diversity": {
            "unique_canonical_states": final_unique_states,
            "final_buffer_unique_ratio": (
                final_unique_states / total_states if total_states else 0.0
            ),
            "duplicate_hash_groups": duplicate_hash_groups,
            "duplicate_hash_occurrences": duplicate_hash_occurrences,
            "duplicate_rate": (
                duplicate_hash_occurrences / total_states if total_states else 0.0
            ),
        },
        "trajectories": {
            "recovered_games": sum(
                int(row["recovered_games"]) for row in trajectory_rows
            ),
            "empty_games": empty_games_total,
            "anomalous_games": anomalous_games_total,
            "step_boundary_method": (
                "start a new game whenever step resets to 1; require consecutive "
                "steps through the declared game length"
            ),
        },
        "anomaly_counts": dict(sorted(all_anomalies.items())),
        "validations": validations,
        "definitions": {
            "state_hash": "SHA256(dtype + shape + contiguous canonical-board bytes)",
            "incoming_unique_state_ratio": (
                "unique hashes in the iteration not seen in earlier retained "
                "buckets / states in the iteration"
            ),
            "duplicate_rate": (
                "(states in the iteration - unique hashes in the iteration) / "
                "states in the iteration"
            ),
            "state_effective_count": "N^2 / sum_h frequency(h)^2",
            "final_buffer_unique_ratio": (
                "unique hashes across retained buckets / total retained states"
            ),
        },
        "limitations": list(LIMITATIONS),
        "outputs": {
            "replay_iteration_stats": iteration_path.as_posix(),
            "replay_final_summary": summary_path.as_posix(),
            "trajectory_stats": trajectory_path.as_posix(),
            "raw_states_exported": False,
            "state_hashes_exported": False,
        },
    }

    _atomic_write_csv(iteration_path, ITERATION_FIELDS, iteration_rows)
    _atomic_write_csv(trajectory_path, TRAJECTORY_FIELDS, trajectory_rows)
    _atomic_write_json(summary_path, summary)
    return summary


def _print_acceptance(summary: dict[str, Any]) -> None:
    replay = summary["replay"]
    validations = summary["validations"]
    print(f"Replay iteration: {replay['iteration']}")
    print(f"History buckets: {replay['history_buckets']}")
    print(
        "Recovered range: "
        f"{replay['recovered_iteration_start']}-{replay['recovered_iteration_end']}"
    )
    print(f"Total states: {replay['total_states']}")
    print(f"Empty buckets: {replay['empty_buckets']}")
    print(
        "Count matches metrics: "
        f"{'yes' if validations['counts_match_metrics'] else 'no'}"
    )
    print(f"Output status: {summary['status']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_replay(
            args.replay,
            args.metrics,
            args.resolved_config,
            args.output_dir,
            expected_iteration=args.expected_iteration,
            expected_history_buckets=args.expected_history_buckets,
            expected_total_states=args.expected_total_states,
        )
    except (ReplayAnalysisError, OSError, ValueError, MemoryError) as exc:
        print(f"Replay analysis failed: {exc}", file=sys.stderr)
        return 2
    _print_acceptance(summary)
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
