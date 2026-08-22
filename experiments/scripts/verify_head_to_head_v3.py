#!/usr/bin/env python3
"""Read-only acceptance verifier for head_to_head_v3 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
SCRIPTS_ROOT = EXPERIMENTS_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import evaluate_head_to_head as v2  # noqa: E402
from evaluate_head_to_head_v3 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    HeadToHeadError,
    build_tasks,
    expected_temperature_history,
    resolve_context,
)
from head_to_head_stats_v3 import (  # noqa: E402
    HeadToHeadStatsV3Error,
    seed_pair_bootstrap,
    trajectory_diversity,
    trajectory_sha256,
)


REQUIRED_OUTPUTS = (
    "resolved_config.yaml",
    "input_manifest.json",
    "evaluation_manifest.json",
    "attempts.jsonl",
    "games.jsonl",
    "checkpoint_pairs.csv",
    "checkpoint_summary.csv",
    "summary.json",
    "evaluation.log",
)


def _require(condition: bool, message: str) -> None:
    v2._require(condition, message)


def _verify_file_entry(entry: Any, label: str) -> None:
    _require(isinstance(entry, dict), f"input manifest {label} is invalid")
    path_value = entry.get("path")
    expected = entry.get("sha256")
    size = entry.get("size_bytes")
    _require(isinstance(path_value, str) and path_value, f"input manifest {label} path is invalid")
    _require(isinstance(expected, str) and len(expected) == 64, f"input manifest {label} SHA is invalid")
    path = Path(path_value)
    _require(path.is_file(), f"input manifest {label} file is missing: {path}")
    _require(v2.sha256_file(path) == expected, f"input manifest {label} SHA mismatch")
    _require(isinstance(size, int) and path.stat().st_size == size, f"input manifest {label} size mismatch")


def _verify_input_manifest(payload: dict[str, Any]) -> set[str]:
    _require(payload.get("config_id") == "head_to_head_v3", "input manifest config_id mismatch")
    inputs = payload.get("inputs")
    _require(isinstance(inputs, dict), "input manifest lacks inputs")
    for name in (
        "protocol",
        "matched_compute",
        "baseline_checkpoint_manifest",
        "adaptive_checkpoint_manifest",
        "baseline_resolved_config",
        "adaptive_resolved_config",
        "selected_baseline_checkpoint",
        "selected_adaptive_checkpoint",
    ):
        _verify_file_entry(inputs.get(name), name)
    implementations = inputs.get("implementations")
    _require(isinstance(implementations, dict) and implementations, "input manifest lacks current implementations")
    for name, entry in implementations.items():
        _verify_file_entry(entry, f"implementations.{name}")
    revisions = payload.get("implementation_revisions")
    _require(isinstance(revisions, list) and revisions, "input manifest lacks implementation revisions")
    digests: set[str] = set()
    for index, revision in enumerate(revisions):
        _require(isinstance(revision, dict), f"implementation revision {index} is invalid")
        digest = revision.get("sha256")
        revision_inputs = revision.get("implementations")
        _require(isinstance(digest, str) and len(digest) == 64, f"implementation revision {index} SHA is invalid")
        _require(isinstance(revision_inputs, dict), f"implementation revision {index} files are invalid")
        _require(v2._canonical_sha256(revision_inputs) == digest, f"implementation revision {index} digest mismatch")
        _require(digest not in digests, f"duplicate implementation revision {digest}")
        digests.add(str(digest))
    _require(v2._canonical_sha256(implementations) in digests, "current implementation set is not recorded as a revision")
    return digests


def _load_csv(path: Path, label: str) -> list[dict[str, str]]:
    _require(path.is_file(), f"{label} not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HeadToHeadError(f"invalid numeric value for {label}") from exc
    _require(math.isfinite(result), f"non-finite numeric value for {label}")
    return result


def verify_head_to_head_v3(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    matched_compute_path: Path | None = None,
    baseline_run_dir: Path | None = None,
    adaptive_run_dir: Path | None = None,
    baseline_checkpoint_manifest: Path | None = None,
    adaptive_checkpoint_manifest: Path | None = None,
) -> dict[str, Any]:
    args = SimpleNamespace(
        config=Path(config_path),
        output_dir=Path(output_dir) if output_dir is not None else None,
        matched_compute=Path(matched_compute_path) if matched_compute_path is not None else None,
        baseline_run_dir=Path(baseline_run_dir) if baseline_run_dir is not None else None,
        adaptive_run_dir=Path(adaptive_run_dir) if adaptive_run_dir is not None else None,
        baseline_checkpoint_manifest=(Path(baseline_checkpoint_manifest) if baseline_checkpoint_manifest is not None else None),
        adaptive_checkpoint_manifest=(Path(adaptive_checkpoint_manifest) if adaptive_checkpoint_manifest is not None else None),
    )
    context = resolve_context(args)
    for name in REQUIRED_OUTPUTS:
        _require((context.output_dir / name).is_file(), f"required output is missing: {name}")
    resolved = v2._load_yaml(context.output_dir / "resolved_config.yaml", "resolved config")
    _require(resolved.get("config_id") == "head_to_head_v3", "resolved config_id mismatch")
    runtime = resolved.get("runtime_resolution")
    _require(isinstance(runtime, dict), "resolved config lacks runtime_resolution")
    _require(runtime.get("best_checkpoint_selection_used") is False, "resolved config used best-checkpoint selection")
    _require(_float(runtime.get("common_horizon_gpu_hours"), "resolved common horizon") == context.common_horizon, "resolved common horizon mismatch")

    input_manifest = v2._load_json(context.output_dir / "input_manifest.json", "input manifest")
    implementation_digests = _verify_input_manifest(input_manifest)
    manifest = v2._load_json(context.manifest_path, "evaluation manifest")
    _require(manifest.get("status") == "completed", "evaluation manifest is not completed")
    _require(manifest.get("best_checkpoint_selection_used") is False, "evaluation manifest used best-checkpoint selection")
    _require(manifest.get("parameter_interpolation") is False, "evaluation manifest used parameter interpolation")
    _require(manifest.get("common_horizon_gpu_hours") == context.common_horizon, "evaluation common horizon mismatch")
    _require(manifest.get("baseline_checkpoint") == v2._checkpoint_public(context.baseline), "Baseline selected checkpoint mismatch")
    _require(manifest.get("adaptive_checkpoint") == v2._checkpoint_public(context.adaptive), "Adaptive selected checkpoint mismatch")
    _require(context.baseline["actual_gpu_hours"] <= context.common_horizon, "Baseline checkpoint exceeds common horizon")
    _require(context.adaptive["actual_gpu_hours"] <= context.common_horizon, "Adaptive checkpoint exceeds common horizon")

    tasks = build_tasks(context)
    _require(manifest.get("tasks") == tasks, "evaluation tasks differ from the frozen task grid")
    _require(manifest.get("task_manifest_sha256") == v2._canonical_sha256(tasks), "task manifest SHA mismatch")
    attempts_by_key, games_by_key = v2.load_and_validate_state(
        context, tasks, recover=False
    )
    _require(len(games_by_key) == 100, f"expected 100 technically valid games, found {len(games_by_key)}")
    _require(len(set(games_by_key)) == 100, "technically valid game keys are not unique")
    _require(not manifest.get("unresolved_technical_failure_keys"), "unresolved technical failures remain")
    for values in attempts_by_key.values():
        for record in values:
            _require(record.get("implementation_set_sha256") in implementation_digests, "attempt references an unknown implementation revision")
    games = [games_by_key[key] for key in sorted(games_by_key)]
    _require(sum(record["adaptive_color"] == "white" for record in games) == 50, "Adaptive white game count is not 50")
    _require(sum(record["adaptive_color"] == "black" for record in games) == 50, "Adaptive black game count is not 50")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in games:
        _require(record.get("config_id") == "head_to_head_v3", "game config_id mismatch")
        by_seed[int(record["seed_pair_index"])].append(record)
        for condition in ("baseline", "adaptive"):
            expected_history = expected_temperature_history(
                int(record[f"{condition}_moves"]), context.config["model_protocol"]
            )
            _require(record[f"{condition}_temperature_history"] == expected_history, f"{condition} temperature history mismatch")
        _require(record.get("max_turns") == 150, "game max_turns mismatch")
        _require(record.get("trajectory_sha256") == trajectory_sha256(record), "game trajectory SHA mismatch")
    _require(set(by_seed) == set(range(50)), "seed-pair coverage is incomplete")
    for seed_pair_index, pair in by_seed.items():
        _require(len(pair) == 2, f"seed pair {seed_pair_index} does not have two games")
        _require(len({record["game_seed"] for record in pair}) == 1, f"seed pair {seed_pair_index} does not share one seed")
        _require({record["adaptive_color"] for record in pair} == {"white", "black"}, f"seed pair {seed_pair_index} does not swap colours")

    bootstrap_config = context.config["analysis"]["bootstrap"]
    recomputed = seed_pair_bootstrap(
        games,
        resamples=int(bootstrap_config["resamples"]),
        seed=int(bootstrap_config["seed"]),
        expected_pairs=int(bootstrap_config["preserve_seed_pairs"]),
    )
    diversity_config = context.config["analysis"]["trajectory_diversity"]
    diversity = trajectory_diversity(
        games,
        minimum_unique_per_colour=int(
            diversity_config["minimum_unique_trajectories_per_colour"]
        ),
    )
    _require(diversity["acceptance_passed"] is True, "trajectory-diversity acceptance failed")
    summary = v2._load_json(context.output_dir / "summary.json", "summary")
    _require(summary.get("status") == "completed", "summary status is not completed")
    _require(summary.get("trajectory_diversity") == diversity, "summary trajectory diversity is not reproducible")
    score = summary.get("adaptive_score")
    _require(isinstance(score, dict), "summary lacks adaptive_score")
    for summary_field, recomputed_field in (
        ("score_rate", "score_rate"),
        ("ci95_low", "ci95_low"),
        ("ci95_high", "ci95_high"),
    ):
        _require(
            math.isclose(
                _float(score.get(summary_field), f"summary {summary_field}"),
                float(recomputed[recomputed_field]),
                abs_tol=1.0e-12,
            ),
            f"summary {summary_field} is not reproducible",
        )
    bootstrap_summary = score.get("bootstrap")
    _require(isinstance(bootstrap_summary, dict), "summary lacks bootstrap metadata")
    _require(bootstrap_summary.get("resampling_unit") == "seed_pair", "summary CI is not seed-pair bootstrapped")
    expected_support = float(recomputed["ci95_low"]) > 0.5
    _require(summary.get("h3_head_to_head_support", {}).get("supported") is expected_support, "H3 support flag is not reproducible")

    pair_rows = _load_csv(context.output_dir / "checkpoint_pairs.csv", "checkpoint pairs")
    _require(len(pair_rows) == 50, "checkpoint_pairs.csv must have 50 rows")
    _require(all(row.get("pair_complete") == "True" for row in pair_rows), "checkpoint_pairs.csv contains incomplete pairs")
    summary_rows = _load_csv(context.output_dir / "checkpoint_summary.csv", "checkpoint summary")
    _require(len(summary_rows) == 1, "checkpoint_summary.csv must have one row")
    _require(summary_rows[0].get("bootstrap_strata") == "seed_pair", "checkpoint summary bootstrap unit mismatch")
    _require(summary_rows[0].get("trajectory_diversity_passed") == "True", "checkpoint summary trajectory diversity failed")
    _require(int(summary_rows[0]["technically_valid_games"]) == 100, "checkpoint summary game count mismatch")
    return {
        "status": "passed",
        "config_id": context.config["config_id"],
        "checkpoints": {
            "common_horizon_gpu_hours": context.common_horizon,
            "baseline_iteration": context.baseline["iteration"],
            "baseline_gpu_hours": context.baseline["actual_gpu_hours"],
            "adaptive_iteration": context.adaptive["iteration"],
            "adaptive_gpu_hours": context.adaptive["actual_gpu_hours"],
        },
        "games": {
            "technically_valid": len(games),
            "unique_game_keys": len(games_by_key),
            "seed_pairs": len(by_seed),
            "adaptive_white": 50,
            "adaptive_black": 50,
        },
        "trajectory_diversity": diversity,
        "adaptive_score": {**recomputed, "h3_support": expected_support},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matched-compute", type=Path)
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--adaptive-run-dir", type=Path)
    parser.add_argument("--baseline-checkpoint-manifest", type=Path)
    parser.add_argument("--adaptive-checkpoint-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_head_to_head_v3(
            config_path=args.config,
            output_dir=args.output_dir,
            matched_compute_path=args.matched_compute,
            baseline_run_dir=args.baseline_run_dir,
            adaptive_run_dir=args.adaptive_run_dir,
            baseline_checkpoint_manifest=args.baseline_checkpoint_manifest,
            adaptive_checkpoint_manifest=args.adaptive_checkpoint_manifest,
        )
    except (HeadToHeadError, HeadToHeadStatsV3Error, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
