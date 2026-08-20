from __future__ import annotations

import copy
import json

import pytest

from experiments.Adaptive.resource_accounting import (
    ResourceAccountant,
    ResourceAccountingError,
    ResourceConfig,
)


def config(**overrides) -> ResourceConfig:
    values = {
        "allocated_gpu_count": 1,
        "max_gpu_hours": 6.0,
        "max_iterations": 90,
        "max_wall_clock_hours": 8.0,
    }
    values.update(overrides)
    return ResourceConfig(**values)


def record(
    accountant: ResourceAccountant,
    iteration: int,
    *,
    self_play: float = 100.0,
    training: float = 200.0,
    instrumentation: float = 10.0,
    checkpoint: float = 5.0,
    evaluation: float = 50.0,
    wall: float = 400.0,
):
    return accountant.record_iteration(
        iteration,
        self_play_seconds=self_play,
        training_seconds=training,
        instrumentation_seconds=instrumentation,
        checkpoint_seconds=checkpoint,
        evaluation_seconds=evaluation,
        wall_clock_seconds=wall,
    )


def test_iteration_record_uses_frozen_training_time_boundary() -> None:
    accountant = ResourceAccountant(config())

    resource_record, decision = record(accountant, 1)

    assert resource_record.iteration_seconds == 315.0
    assert resource_record.cumulative_training_seconds == 315.0
    assert resource_record.cumulative_gpu_hours == pytest.approx(315.0 / 3600.0)
    assert resource_record.self_play_time_fraction == pytest.approx(100.0 / 315.0)
    assert resource_record.evaluation_seconds == 50.0
    assert accountant.state.cumulative_evaluation_seconds == 50.0
    assert decision.continue_run is True


def test_gpu_hours_scale_with_allocated_gpu_count() -> None:
    accountant = ResourceAccountant(config(allocated_gpu_count=2))

    resource_record, _ = record(
        accountant,
        1,
        self_play=900.0,
        training=900.0,
        instrumentation=0.0,
        checkpoint=0.0,
        evaluation=1000.0,
        wall=2000.0,
    )

    assert resource_record.iteration_seconds == 1800.0
    assert resource_record.cumulative_gpu_hours == 1.0


def test_evaluation_is_recorded_but_excluded_from_training_gpu_hours() -> None:
    accountant = ResourceAccountant(config())

    resource_record, _ = record(
        accountant,
        1,
        self_play=50.0,
        training=40.0,
        instrumentation=5.0,
        checkpoint=5.0,
        evaluation=3600.0,
        wall=3700.0,
    )

    assert resource_record.iteration_seconds == 100.0
    assert resource_record.cumulative_gpu_hours == pytest.approx(100.0 / 3600.0)
    assert accountant.state.cumulative_evaluation_seconds == 3600.0


@pytest.mark.parametrize(
    ("training_seconds", "expected_overshoot"),
    [(3600.0, 0.0), (3780.0, 0.05)],
)
def test_gpu_hour_crossing_iteration_is_kept_and_stops_run(
    training_seconds: float,
    expected_overshoot: float,
) -> None:
    accountant = ResourceAccountant(config(max_gpu_hours=1.0))

    resource_record, decision = record(
        accountant,
        1,
        self_play=0.0,
        training=training_seconds,
        instrumentation=0.0,
        checkpoint=0.0,
        evaluation=0.0,
        wall=training_seconds,
    )

    assert resource_record.iteration == 1
    assert decision.continue_run is False
    assert decision.stop_reason == "gpu_hour_budget_reached"
    assert decision.crossing_iteration is True
    assert decision.overshoot_gpu_hours == pytest.approx(expected_overshoot)
    assert accountant.state.budget_crossing_iteration == 1
    with pytest.raises(ResourceAccountingError, match="already stopped"):
        accountant.begin_iteration(2)


def test_iteration_limit_stops_only_after_limit_iteration_is_committed() -> None:
    accountant = ResourceAccountant(config(max_iterations=2))

    _, first_decision = record(accountant, 1)
    _, second_decision = record(accountant, 2)

    assert first_decision.continue_run is True
    assert second_decision.continue_run is False
    assert second_decision.stop_reason == "max_iterations_reached"
    assert accountant.state.completed_iteration == 2


def test_gpu_budget_has_priority_over_iteration_limit() -> None:
    accountant = ResourceAccountant(
        config(max_gpu_hours=1.0, max_iterations=1)
    )

    _, decision = record(
        accountant,
        1,
        self_play=0.0,
        training=3600.0,
        instrumentation=0.0,
        checkpoint=0.0,
        evaluation=0.0,
        wall=3600.0,
    )

    assert decision.stop_reason == "gpu_hour_budget_reached"


def test_wall_clock_projection_stops_at_safe_iteration_boundary() -> None:
    accountant = ResourceAccountant(
        config(max_wall_clock_hours=10.0 / 3600.0)
    )

    resource_record, decision = record(
        accountant,
        1,
        self_play=1.0,
        training=0.0,
        instrumentation=0.0,
        checkpoint=0.0,
        evaluation=0.0,
        wall=6.0,
    )

    assert resource_record.iteration == 1
    assert accountant.state.cumulative_wall_clock_seconds == 6.0
    assert decision.continue_run is False
    assert decision.stop_reason == "wall_clock_safety_limit"


def test_wall_clock_projection_allows_an_exact_next_boundary() -> None:
    accountant = ResourceAccountant(
        config(max_wall_clock_hours=10.0 / 3600.0)
    )

    _, decision = record(
        accountant,
        1,
        self_play=1.0,
        training=0.0,
        instrumentation=0.0,
        checkpoint=0.0,
        evaluation=0.0,
        wall=5.0,
    )

    assert decision.continue_run is True


def test_monotonic_timer_supplies_wall_clock_duration() -> None:
    clock_values = iter((10.0, 16.25))
    accountant = ResourceAccountant(config(), clock=lambda: next(clock_values))
    accountant.begin_iteration(1)

    accountant.record_iteration(
        1,
        self_play_seconds=1.0,
        training_seconds=1.0,
        instrumentation_seconds=1.0,
        checkpoint_seconds=1.0,
        evaluation_seconds=1.0,
    )

    assert accountant.state.cumulative_wall_clock_seconds == 6.25
    assert accountant.state.last_iteration_wall_clock_seconds == 6.25


def test_failed_record_is_atomic_and_active_iteration_can_be_aborted() -> None:
    accountant = ResourceAccountant(config(), clock=lambda: 10.0)
    accountant.begin_iteration(1)
    before = copy.deepcopy(accountant.state)

    with pytest.raises(ValueError, match="training_seconds"):
        accountant.record_iteration(
            1,
            self_play_seconds=1.0,
            training_seconds=-1.0,
            instrumentation_seconds=0.0,
            checkpoint_seconds=0.0,
            evaluation_seconds=0.0,
            wall_clock_seconds=1.0,
        )

    assert accountant.state == before
    with pytest.raises(ResourceAccountingError, match="iteration boundary"):
        accountant.state_dict()
    accountant.abort_iteration()
    assert accountant.state_dict()["completed_iteration"] == 0


def test_state_dict_round_trip_and_instance_load() -> None:
    accountant = ResourceAccountant(config())
    record(accountant, 1)
    payload = json.loads(json.dumps(accountant.state_dict()))

    restored = ResourceAccountant.from_state_dict(config(), payload)
    loaded = ResourceAccountant(config())
    loaded.load_state_dict(payload)

    assert restored.state_dict() == payload
    assert loaded.state_dict() == payload


def test_resume_matches_uninterrupted_accounting() -> None:
    uninterrupted = ResourceAccountant(config())
    uninterrupted_outputs = [record(uninterrupted, iteration) for iteration in range(1, 5)]

    resumed = ResourceAccountant(config())
    resumed_outputs = [record(resumed, iteration) for iteration in range(1, 3)]
    payload = json.loads(json.dumps(resumed.state_dict()))
    resumed = ResourceAccountant.from_state_dict(config(), payload)
    resumed_outputs.extend(record(resumed, iteration) for iteration in range(3, 5))

    assert resumed_outputs == uninterrupted_outputs
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_load_state_is_rejected_during_active_iteration() -> None:
    source = ResourceAccountant(config())
    payload = source.state_dict()
    target = ResourceAccountant(config(), clock=lambda: 1.0)
    target.begin_iteration(1)

    with pytest.raises(ResourceAccountingError, match="while an iteration is active"):
        target.load_state_dict(payload)


def test_resume_rejects_inconsistent_gpu_hour_total() -> None:
    accountant = ResourceAccountant(config())
    record(accountant, 1)
    payload = accountant.state_dict()
    payload["cumulative_gpu_hours"] += 1.0

    with pytest.raises(ValueError, match="cumulative_gpu_hours"):
        ResourceAccountant.from_state_dict(config(), payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"allocated_gpu_count": 0},
        {"max_gpu_hours": 0.0},
        {"max_gpu_hours": float("nan")},
        {"max_iterations": 0},
        {"max_wall_clock_hours": 0.0},
        {"evaluation_time_included": True},
        {"instrumentation_overhead_included": False},
        {"crossing_policy": "drop_crossing_iteration"},
        {"schema_version": 2},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ResourceAccountant(config(**overrides))


def test_long_frozen_crossing_policy_spelling_is_accepted() -> None:
    accountant = ResourceAccountant(
        config(
            crossing_policy=(
                "keep_crossing_iteration_and_do_not_start_another"
            )
        )
    )
    assert accountant.state.completed_iteration == 0
