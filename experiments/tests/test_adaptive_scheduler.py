from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from experiments.Adaptive.adaptive_scheduler import (
    AdaptiveScheduler,
    IterationLengthStats,
    SchedulerConfig,
)


def default_config(**overrides) -> SchedulerConfig:
    values = {
        "target_states": 2516,
        "alpha": 0.25,
        "minimum_observations": 20,
        "initial_length": 33.5363333333333,
        "first_iteration_games": 75,
        "min_games": 50,
        "max_games": 150,
    }
    values.update(overrides)
    return SchedulerConfig(**values)


def stats(
    iteration: int,
    lengths: tuple[int, ...],
    excluded: dict[str, int] | None = None,
) -> IterationLengthStats:
    return IterationLengthStats(iteration, lengths, excluded or {})


def test_initial_plan_is_frozen_first_iteration_value() -> None:
    scheduler = AdaptiveScheduler(default_config())

    assert scheduler.next_iteration_games == 75
    assert scheduler.state.completed_iteration == 0
    assert scheduler.state.current_ema_length == pytest.approx(33.5363333333333)


def test_twenty_observations_apply_ema_and_normal_plan() -> None:
    scheduler = AdaptiveScheduler(default_config())

    decision = scheduler.update(stats(1, (30,) * 20))

    expected_ema = 0.25 * 30.0 + 0.75 * 33.5363333333333
    assert decision.observed_mean_length == 30.0
    assert decision.updated_length_estimate == pytest.approx(expected_ema)
    assert decision.raw_next_games == pytest.approx(2516 / expected_ema)
    assert decision.rounded_next_games == 78
    assert decision.next_iteration_games == 78
    assert decision.update_applied is True
    assert decision.clipped is False
    assert decision.clipping_bound is None
    assert math.isfinite(decision.scheduler_seconds)
    assert decision.scheduler_seconds >= 0.0


@pytest.mark.parametrize("count", [19, 0])
def test_insufficient_observations_keep_ema_and_plan(count: int) -> None:
    scheduler = AdaptiveScheduler(default_config())
    previous_ema = scheduler.state.current_ema_length

    decision = scheduler.update(
        stats(1, (30,) * count, {"incomplete_game": 2})
    )

    assert decision.observed_mean_length == (30.0 if count else None)
    assert decision.updated_length_estimate == previous_ema
    assert decision.next_iteration_games == 75
    assert decision.raw_next_games is None
    assert decision.rounded_next_games is None
    assert decision.update_applied is False
    assert decision.skipped_reason == "insufficient_valid_observations"
    assert scheduler.state.completed_iteration == 1
    assert scheduler.state.per_iteration_valid_game_count_history == [count]
    assert scheduler.state.excluded_game_count_by_reason == {"incomplete_game": 2}


def test_rounding_is_ceil_not_round() -> None:
    scheduler = AdaptiveScheduler(
        default_config(
            target_states=101,
            alpha=1.0,
            initial_length=2.0,
            first_iteration_games=50,
        )
    )

    decision = scheduler.update(stats(1, (2,) * 20))

    assert decision.raw_next_games == 50.5
    assert decision.rounded_next_games == 51
    assert decision.next_iteration_games == 51


@pytest.mark.parametrize(
    ("length", "expected_games", "expected_bound"),
    [(150, 50, "min_games"), (1, 150, "max_games")],
)
def test_plan_clips_after_ceil(
    length: int, expected_games: int, expected_bound: str
) -> None:
    scheduler = AdaptiveScheduler(default_config(alpha=1.0))

    decision = scheduler.update(stats(1, (length,) * 20))

    assert decision.next_iteration_games == expected_games
    assert decision.clipped is True
    assert decision.clipping_bound == expected_bound
    assert scheduler.state.last_clipping_decision == expected_bound


@pytest.mark.parametrize("iteration", [1, 3])
def test_duplicate_or_skipped_iteration_fails_without_state_change(
    iteration: int,
) -> None:
    scheduler = AdaptiveScheduler(default_config())
    scheduler.update(stats(1, (30,) * 20))
    before = scheduler.state_dict()

    with pytest.raises(ValueError, match="stats.iteration must be 2"):
        scheduler.update(stats(iteration, (30,) * 20))

    assert scheduler.state_dict() == before


@pytest.mark.parametrize("length", [0, 151, True, 1.5])
def test_invalid_valid_game_length_fails_without_state_change(length) -> None:
    scheduler = AdaptiveScheduler(default_config())
    before = scheduler.state_dict()

    with pytest.raises(ValueError, match="valid_game_lengths"):
        scheduler.update(stats(1, (length,)))

    assert scheduler.state_dict() == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_states": 0},
        {"alpha": 0.0},
        {"alpha": 1.1},
        {"minimum_observations": 0},
        {"initial_length": float("nan")},
        {"initial_length": float("inf")},
        {"initial_length": 0.0},
        {"min_games": 0},
        {"min_games": 100, "max_games": 50},
        {"first_iteration_games": 49},
        {"first_iteration_games": 151},
        {"max_valid_game_length": 0},
        {"rounding": "round"},
        {"schema_version": 2},
    ],
)
def test_invalid_config_fails(overrides: dict) -> None:
    with pytest.raises(ValueError):
        AdaptiveScheduler(default_config(**overrides))


def test_scheduler_does_not_consume_global_rng() -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    scheduler = AdaptiveScheduler(default_config())
    scheduler.update(stats(1, (30,) * 20))

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_same_inputs_produce_equal_decisions_and_state() -> None:
    first = AdaptiveScheduler(default_config())
    second = AdaptiveScheduler(default_config())
    sequence = [
        stats(1, (30,) * 20),
        stats(2, (25,) * 20, {"incomplete_game": 1}),
        stats(3, (20,) * 19),
    ]

    first_decisions = [first.update(item) for item in sequence]
    second_decisions = [second.update(item) for item in sequence]

    assert first_decisions == second_decisions
    assert first.state_dict() == second.state_dict()
