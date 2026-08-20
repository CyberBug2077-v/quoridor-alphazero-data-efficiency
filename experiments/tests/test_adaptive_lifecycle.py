from __future__ import annotations

import json

import numpy as np

from experiments.Adaptive.adaptive_scheduler import (
    AdaptiveScheduler,
    IterationLengthStats,
    SchedulerConfig,
)
from experiments.Adaptive.replay_instrumentation import (
    ReplayInstrumentation,
    ReplayInstrumentationConfig,
)


def scheduler_config() -> SchedulerConfig:
    return SchedulerConfig(
        target_states=4,
        alpha=0.5,
        minimum_observations=1,
        initial_length=2.0,
        first_iteration_games=2,
        min_games=1,
        max_games=3,
    )


def instrumentation_config() -> ReplayInstrumentationConfig:
    return ReplayInstrumentationConfig(board_size=9, history_iterations=4)


def episode(length: int, value: float) -> list[tuple]:
    return [
        (
            np.full((4, 17, 17), step, dtype=np.int8),
            None,
            value,
            None,
            step,
            length,
        )
        for step in range(1, length + 1)
    ]


def run_iteration(
    scheduler: AdaptiveScheduler,
    instrumentation: ReplayInstrumentation,
    iteration: int,
):
    games_planned = scheduler.next_iteration_games
    instrumentation.begin_iteration(iteration, games_planned)
    episodes = [episode(2, 1.0)]
    episodes.extend(episode(1, 0.0) for _ in range(games_planned - 1))

    observations = []
    for item in episodes:
        original_identity = id(item)
        observation = instrumentation.observe_episode(item)
        assert id(item) == original_identity
        observations.append(observation)

    lengths = [len(item) for item in episodes]
    replay_stats = instrumentation.finalize_iteration(
        {
            "games_completed": len(episodes),
            "positions_generated": sum(lengths),
            "min_game_length": min(lengths),
            "max_game_length": max(lengths),
            "mean_game_length": sum(lengths) / len(lengths),
        }
    )
    length_stats = IterationLengthStats(
        iteration=replay_stats.iteration,
        valid_game_lengths=replay_stats.valid_game_lengths,
        excluded_game_count_by_reason=(
            replay_stats.excluded_game_count_by_reason
        ),
    )
    decision = scheduler.update(length_stats)
    return observations, replay_stats, decision


def test_online_lifecycle_updates_scheduler_and_saves_boundary_state() -> None:
    scheduler = AdaptiveScheduler(scheduler_config())
    instrumentation = ReplayInstrumentation(instrumentation_config())

    _, replay_stats, decision = run_iteration(scheduler, instrumentation, 1)

    assert replay_stats.valid_game_lengths == (2,)
    assert replay_stats.excluded_game_count_by_reason["incomplete_game"] == 1
    assert decision.completed_iteration == 1
    assert decision.next_iteration_games == 2
    assert instrumentation.state_dict()["completed_iteration"] == 1
    assert scheduler.state_dict()["completed_iteration"] == 1


def test_four_iteration_resume_matches_uninterrupted_execution() -> None:
    uninterrupted_scheduler = AdaptiveScheduler(scheduler_config())
    uninterrupted_instrumentation = ReplayInstrumentation(instrumentation_config())
    uninterrupted_results = [
        run_iteration(
            uninterrupted_scheduler,
            uninterrupted_instrumentation,
            iteration,
        )
        for iteration in range(1, 5)
    ]

    resumed_scheduler = AdaptiveScheduler(scheduler_config())
    resumed_instrumentation = ReplayInstrumentation(instrumentation_config())
    resumed_results = [
        run_iteration(resumed_scheduler, resumed_instrumentation, iteration)
        for iteration in range(1, 3)
    ]
    saved_scheduler = json.loads(json.dumps(resumed_scheduler.state_dict()))
    saved_instrumentation = json.loads(
        json.dumps(resumed_instrumentation.state_dict())
    )
    resumed_scheduler = AdaptiveScheduler.from_state_dict(
        scheduler_config(), saved_scheduler
    )
    resumed_instrumentation = ReplayInstrumentation.from_state_dict(
        instrumentation_config(), saved_instrumentation
    )
    resumed_results.extend(
        run_iteration(resumed_scheduler, resumed_instrumentation, iteration)
        for iteration in range(3, 5)
    )

    assert [result[1] for result in resumed_results] == [
        result[1] for result in uninterrupted_results
    ]
    assert [result[2] for result in resumed_results] == [
        result[2] for result in uninterrupted_results
    ]
    assert resumed_instrumentation.state_dict() == (
        uninterrupted_instrumentation.state_dict()
    )
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler.state_dict()
