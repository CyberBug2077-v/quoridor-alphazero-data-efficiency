"""Build the actual-resource and workload tables for Appendix A.3.

Run from any working directory with::

    python plots/scripts/support/appendix_a3.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PLOTS_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PLOTS_ROOT.parent

DEFAULT_BASELINE_RUN = (
    REPOSITORY_ROOT
    / "baseline"
    / "outputs"
    / "baseline_reproduction_seed1001_4090"
)
DEFAULT_ADAPTIVE_RUN = (
    REPOSITORY_ROOT
    / "experiments"
    / "outputs"
    / "adaptive_formal_seed1001_4090_v2"
)
DEFAULT_OUTPUT = PLOTS_ROOT / "tables" / "appendix_a3.json"


class AppendixDataError(ValueError):
    """Raised when a required Appendix A.3 input is absent or invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the Appendix A.3 resource and workload tables."
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_RUN / "summary.json",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=DEFAULT_BASELINE_RUN / "metrics.jsonl",
    )
    parser.add_argument(
        "--adaptive-summary",
        type=Path,
        default=DEFAULT_ADAPTIVE_RUN / "summary.json",
    )
    parser.add_argument(
        "--adaptive-metrics",
        type=Path,
        default=DEFAULT_ADAPTIVE_RUN / "metrics.jsonl",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AppendixDataError(f"Input not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AppendixDataError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AppendixDataError(f"Expected a JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise AppendixDataError(f"Input not found: {path}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AppendixDataError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise AppendixDataError(
                f"Expected a JSON object at {path}:{line_number}"
            )
        records.append(record)

    if not records:
        raise AppendixDataError(f"No metric records found in {path}")
    return records


def require_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise AppendixDataError(f"{context}.{key} is missing")
    return mapping[key]


def numeric_sum(records: Sequence[Mapping[str, Any]], field: str) -> float:
    try:
        return sum(float(record[field]) for record in records)
    except (KeyError, TypeError, ValueError) as exc:
        raise AppendixDataError(f"Metric field {field!r} is missing or invalid") from exc


def integer_sum(records: Sequence[Mapping[str, Any]], field: str) -> int:
    try:
        return sum(int(record[field]) for record in records)
    except (KeyError, TypeError, ValueError) as exc:
        raise AppendixDataError(f"Metric field {field!r} is missing or invalid") from exc


def completed_iteration_count(summary: Mapping[str, Any]) -> int:
    completed = require_value(summary, "completed_iterations", "summary")
    if not isinstance(completed, list):
        raise AppendixDataError("summary.completed_iterations must be a list")
    return len(completed)


def aggregate_run(
    summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    iteration_seconds = numeric_sum(records, "iteration_seconds")
    self_play_seconds = numeric_sum(records, "self_play_seconds")
    training_seconds = numeric_sum(records, "training_seconds")
    other_seconds = iteration_seconds - self_play_seconds - training_seconds

    final_record = records[-1]
    summary_gpu_hours = summary.get("final_cumulative_gpu_hours")
    gpu_hours = float(
        summary_gpu_hours
        if summary_gpu_hours is not None
        else require_value(final_record, "cumulative_gpu_hours", "final metric")
    )
    games = integer_sum(records, "games_completed")
    positions = integer_sum(records, "positions_generated")
    optimiser_steps = integer_sum(records, "optimizer_steps")

    try:
        peak_gpu_memory = max(float(row["peak_gpu_memory_mb"]) for row in records)
        final_replay_occupancy = int(final_record["replay_buffer_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AppendixDataError(
            "Peak GPU memory or final replay occupancy is missing or invalid"
        ) from exc

    return {
        "resources": {
            "run_status": str(require_value(summary, "status", "summary")).capitalize(),
            "completed_iterations": completed_iteration_count(summary),
            "gpu_hours": round(gpu_hours, 3),
            "self_play_hours": round(self_play_seconds / 3600.0, 3),
            "training_hours": round(training_seconds / 3600.0, 3),
            "other_iteration_overhead_hours": round(other_seconds / 3600.0, 3),
            "self_play_fraction_percent": round(
                100.0 * self_play_seconds / iteration_seconds, 2
            ),
            "training_fraction_percent": round(
                100.0 * training_seconds / iteration_seconds, 2
            ),
            "peak_gpu_memory_mib": round(peak_gpu_memory, 1),
        },
        "workload": {
            "self_play_games": games,
            "generated_positions": positions,
            "optimiser_steps": optimiser_steps,
            "games_per_gpu_hour": round(games / gpu_hours, 2),
            "positions_per_gpu_hour": round(positions / gpu_hours, 2),
            "final_replay_occupancy": final_replay_occupancy,
        },
    }


def paired_rows(
    baseline: Mapping[str, Any], adaptive: Mapping[str, Any], section: str
) -> dict[str, dict[str, Any]]:
    baseline_section = baseline[section]
    adaptive_section = adaptive[section]
    return {
        field: {
            "baseline": baseline_section[field],
            "adaptive": adaptive_section[field],
        }
        for field in baseline_section
    }


def build_summary(
    baseline_summary: Mapping[str, Any],
    baseline_metrics: Sequence[Mapping[str, Any]],
    adaptive_summary: Mapping[str, Any],
    adaptive_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = aggregate_run(baseline_summary, baseline_metrics)
    adaptive = aggregate_run(adaptive_summary, adaptive_metrics)
    return {
        "actual_training_resources": paired_rows(
            baseline, adaptive, "resources"
        ),
        "completed_workload_and_throughput": paired_rows(
            baseline, adaptive, "workload"
        ),
    }


def write_summary(summary: Mapping[str, Any], output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    args = parse_args()
    summary = build_summary(
        load_json(args.baseline_summary.expanduser().resolve()),
        load_jsonl(args.baseline_metrics.expanduser().resolve()),
        load_json(args.adaptive_summary.expanduser().resolve()),
        load_jsonl(args.adaptive_metrics.expanduser().resolve()),
    )
    output = write_summary(summary, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
