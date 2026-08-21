from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_adaptive_metrics.py"
SPEC = importlib.util.spec_from_file_location("derive_adaptive_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
derive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(derive)


def metric(
    iteration: int,
    positions: int,
    replay_size: int,
    *,
    seconds: float = 100.0,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "positions_generated": positions,
        "games_completed": 5,
        "optimizer_steps": 2,
        "replay_buffer_size": replay_size,
        "available_examples": replay_size,
        "examples_used": min(replay_size, 40),
        "samples_seen": min(replay_size, 40) * 2,
        "iteration_seconds": seconds,
        "cumulative_gpu_hours": iteration * seconds / 3600.0,
    }


def test_derives_h2_metrics_age_turnover_and_horizon_flag() -> None:
    records = [
        metric(1, 10, 10),
        metric(2, 20, 30),
        metric(3, 30, 50),
        metric(4, 40, 70),
    ]

    rows = derive._derive_metric_rows(
        records,
        history_iterations=2,
        allocated_gpu_count=1,
        h2_common_horizon_iteration=3,
    )

    assert rows[0]["fresh_states_per_update"] == 5.0
    assert rows[0]["states_per_gpu_hour"] == pytest.approx(360.0)
    assert rows[1]["mean_sample_age"] == pytest.approx(10 / 30)
    assert rows[1]["median_sample_age"] == 0
    assert rows[1]["p90_sample_age"] == 1
    assert rows[2]["evicted_states"] == 10
    assert rows[2]["turnover_fraction"] == pytest.approx(10 / 30)
    assert rows[2]["beyond_h2_common_horizon"] is False
    assert rows[3]["beyond_h2_common_horizon"] is True


def test_annotates_unique_ratio_and_h2_scope_without_changing_left_censor() -> None:
    rows = [
        {
            "iteration": 66,
            "states": 4,
            "unique_canonical_states": 3,
            "incoming_ratio_left_censored": True,
        },
        {
            "iteration": 181,
            "states": 5,
            "unique_canonical_states": 2,
            "incoming_ratio_left_censored": False,
        },
    ]

    annotated = derive._annotate_replay_rows(
        rows,
        h2_common_horizon_iteration=180,
        include_unique_ratio=True,
    )

    assert annotated[0]["unique_canonical_state_ratio"] == pytest.approx(0.75)
    assert annotated[0]["incoming_ratio_left_censored"] is True
    assert annotated[0]["beyond_h2_common_horizon"] is False
    assert annotated[1]["beyond_h2_common_horizon"] is True
