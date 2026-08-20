from __future__ import annotations

import json

import pytest

from experiments.Adaptive.adaptive_scheduler import (
    AdaptiveScheduler,
    IterationLengthStats,
    SchedulerConfig,
)


def config() -> SchedulerConfig:
    return SchedulerConfig(
        target_states=2516,
        alpha=0.25,
        minimum_observations=20,
        initial_length=33.5363333333333,
        first_iteration_games=75,
        min_games=50,
        max_games=150,
    )


def sequence() -> list[IterationLengthStats]:
    return [
        IterationLengthStats(1, (30,) * 20, {}),
        IterationLengthStats(2, (25,) * 20, {"malformed_game": 1}),
        IterationLengthStats(3, (20,) * 19, {"empty_game": 1}),
        IterationLengthStats(4, (35,) * 20, {}),
    ]


def test_state_dict_round_trip_is_json_serializable() -> None:
    scheduler = AdaptiveScheduler(config())
    scheduler.update(sequence()[0])
    payload = scheduler.state_dict()

    encoded = json.dumps(payload)
    restored = AdaptiveScheduler.from_state_dict(config(), json.loads(encoded))

    assert restored.state_dict() == payload


def test_restore_uses_saved_next_plan_without_recomputation() -> None:
    scheduler = AdaptiveScheduler(config())
    scheduler.update(sequence()[0])
    payload = scheduler.state_dict()
    payload["next_iteration_games"] = 79

    restored = AdaptiveScheduler.from_state_dict(config(), payload)

    assert restored.next_iteration_games == 79
    assert restored.state.current_ema_length == payload["current_ema_length"]


def test_interrupted_resume_matches_uninterrupted_four_iterations() -> None:
    uninterrupted = AdaptiveScheduler(config())
    uninterrupted_decisions = [
        uninterrupted.update(item) for item in sequence()
    ]

    interrupted = AdaptiveScheduler(config())
    resumed_decisions = [interrupted.update(item) for item in sequence()[:2]]
    saved = interrupted.state_dict()
    restored = AdaptiveScheduler.from_state_dict(config(), saved)
    resumed_decisions.extend(restored.update(item) for item in sequence()[2:])

    assert resumed_decisions == uninterrupted_decisions
    assert restored.state_dict() == uninterrupted.state_dict()
    assert restored.state.completed_iteration == 4


def test_restore_continues_with_next_iteration_only() -> None:
    scheduler = AdaptiveScheduler(config())
    scheduler.update(sequence()[0])
    restored = AdaptiveScheduler.from_state_dict(config(), scheduler.state_dict())

    with pytest.raises(ValueError, match="stats.iteration must be 2"):
        restored.update(sequence()[0])

    restored.update(sequence()[1])
    assert restored.state.completed_iteration == 2


def test_schema_version_mismatch_fails() -> None:
    payload = AdaptiveScheduler(config()).state_dict()
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        AdaptiveScheduler.from_state_dict(config(), payload)


def test_missing_state_field_fails() -> None:
    payload = AdaptiveScheduler(config()).state_dict()
    del payload["next_iteration_games"]

    with pytest.raises(ValueError, match="missing required field"):
        AdaptiveScheduler.from_state_dict(config(), payload)


def test_history_length_mismatch_fails() -> None:
    scheduler = AdaptiveScheduler(config())
    scheduler.update(sequence()[0])
    payload = scheduler.state_dict()
    payload["per_iteration_valid_game_count_history"] = []

    with pytest.raises(ValueError, match="history length"):
        AdaptiveScheduler.from_state_dict(config(), payload)
