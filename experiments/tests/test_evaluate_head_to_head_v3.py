from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = SOURCE_ROOT / "experiments"
SCRIPTS = EXPERIMENTS / "scripts"
BASELINE_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for root in (SCRIPTS, BASELINE_SCRIPTS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_head_to_head_v3 as head
from head_to_head_stats_v3 import seed_pair_bootstrap, trajectory_diversity


def _default_args(output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=EXPERIMENTS / "configs" / "head_to_head_v3.yaml",
        matched_compute=None,
        baseline_run_dir=None,
        adaptive_run_dir=None,
        baseline_checkpoint_manifest=None,
        adaptive_checkpoint_manifest=None,
        output_dir=output_dir,
    )


def test_v3_protocol_freezes_schedule_pairing_and_diversity() -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "head_to_head_v3.yaml").read_text(
            encoding="utf-8"
        )
    )
    schedule = config["model_protocol"]["temperature_schedule"]
    bootstrap = config["analysis"]["bootstrap"]
    diversity = config["analysis"]["trajectory_diversity"]

    assert config["config_id"] == "head_to_head_v3"
    assert config["outputs"]["root"].endswith("/head_to_head_v3")
    assert schedule == {
        "unit": "own_model_turns",
        "early_temperature": 0.18,
        "early_moves_per_player": 5,
        "later_temperature": 0.0,
    }
    assert config["model_protocol"]["identical_for_both_conditions"] is True
    assert bootstrap["method"] == "nonparametric_seed_pair_bootstrap"
    assert bootstrap["resampling_unit"] == "seed_pair"
    assert bootstrap["preserve_seed_pairs"] == 50
    assert diversity["minimum_unique_trajectories_per_colour"] == 2


def test_dynamic_checkpoints_and_paired_seed_tasks(tmp_path: Path) -> None:
    context = head.resolve_context(_default_args(tmp_path))
    tasks = head.build_tasks(context)

    assert context.common_horizon == 20.004163361943395
    assert context.baseline["iteration"] == 210
    assert context.adaptive["iteration"] == 179
    assert len(tasks) == len({task["stable_game_key"] for task in tasks}) == 100
    for seed_pair_index in range(50):
        pair = [task for task in tasks if task["seed_pair_index"] == seed_pair_index]
        assert len(pair) == 2
        assert len({task["game_seed"] for task in pair}) == 1
        assert {task["adaptive_color"] for task in pair} == {"white", "black"}


def test_temperature_history_counts_own_model_turns() -> None:
    model = {
        "temperature_schedule": {
            "early_temperature": 0.18,
            "early_moves_per_player": 5,
            "later_temperature": 0.0,
        }
    }

    assert head.expected_temperature_history(3, model) == [0.18] * 3
    assert head.expected_temperature_history(5, model) == [0.18] * 5
    assert head.expected_temperature_history(8, model) == [0.18] * 5 + [0.0] * 3


def _paired_records(*, diverse: bool = True) -> list[dict]:
    records: list[dict] = []
    for pair_index in range(50):
        for colour in ("white", "black"):
            if colour == "white":
                result = "win" if pair_index < 30 else "draw"
            else:
                result = "draw" if pair_index < 25 else "loss"
            variant = pair_index % 3 if diverse else 0
            records.append(
                {
                    "stable_game_key": f"{pair_index}-{colour}",
                    "seed_pair_index": pair_index,
                    "game_seed": pair_index + 1000,
                    "adaptive_color": colour,
                    "adaptive_result": result,
                    "moves": [
                        {
                            "player": colour,
                            "type": "pawn",
                            "row": variant,
                            "col": 4,
                        }
                    ],
                }
            )
    return records


def test_seed_pair_bootstrap_is_deterministic_and_uses_50_units() -> None:
    records = _paired_records()

    first = seed_pair_bootstrap(records, resamples=10000, seed=92001)
    second = seed_pair_bootstrap(list(reversed(records)), resamples=10000, seed=92001)

    assert first == second
    assert first["seed_pairs"] == 50
    assert first["games"] == 100
    assert first["resampling_unit"] == "seed_pair"
    raw_score = sum(
        1.0 if row["adaptive_result"] == "win" else 0.5 if row["adaptive_result"] == "draw" else 0.0
        for row in records
    ) / 100
    assert first["score_rate"] == pytest.approx(raw_score)
    assert first["ci95_low"] <= first["score_rate"] <= first["ci95_high"]


def test_single_trajectory_per_colour_cannot_pass_acceptance() -> None:
    failed = trajectory_diversity(
        _paired_records(diverse=False), minimum_unique_per_colour=2
    )
    passed = trajectory_diversity(
        _paired_records(diverse=True), minimum_unique_per_colour=2
    )

    assert failed["acceptance_passed"] is False
    assert failed["segments"]["white"]["unique_trajectories"] == 1
    assert failed["segments"]["black"]["unique_trajectories"] == 1
    assert passed["acceptance_passed"] is True
    assert passed["segments"]["white"]["unique_trajectories"] == 3
    assert passed["segments"]["black"]["unique_trajectories"] == 3
