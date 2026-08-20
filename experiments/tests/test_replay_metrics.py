from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch

from experiments.Adaptive.replay_instrumentation import (
    ReplayInstrumentation,
    ReplayInstrumentationConfig,
    ReplayInstrumentationError,
    analyze_replay_bucket,
    canonical_state_hash,
    summarize_replay_snapshot,
)


def config(**overrides) -> ReplayInstrumentationConfig:
    values = {"board_size": 9, "history_iterations": 150}
    values.update(overrides)
    return ReplayInstrumentationConfig(**values)


def board(value: int = 0, *, dtype=np.int8) -> np.ndarray:
    return np.full((4, 17, 17), value, dtype=dtype)


def episode(
    length: int,
    *,
    values: tuple[float, ...] | None = None,
    boards: tuple[np.ndarray, ...] | None = None,
) -> list[tuple[object, object, float, object, int, int]]:
    value_targets = values or (1.0,) * length
    canonical_boards = boards or tuple(board(step) for step in range(length))
    return [
        (canonical_boards[index], None, value_targets[index], None, step, length)
        for index, step in enumerate(range(1, length + 1))
    ]


def coach_metrics(episodes: list[list[object]]) -> dict[str, object]:
    lengths = [len(item) for item in episodes if item]
    return {
        "games_completed": len(lengths),
        "positions_generated": sum(lengths),
        "min_game_length": min(lengths) if lengths else None,
        "max_game_length": max(lengths) if lengths else None,
        "mean_game_length": sum(lengths) / len(lengths) if lengths else None,
    }


def finalize_episodes(
    episodes: list[list[object]],
) -> tuple[ReplayInstrumentation, object]:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, len(episodes))
    for item in episodes:
        instrumentation.observe_episode(item)
    return instrumentation, instrumentation.finalize_iteration(coach_metrics(episodes))


def test_valid_empty_and_incomplete_classifications() -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 3)

    valid = instrumentation.observe_episode(episode(2))
    empty = instrumentation.observe_episode([])
    incomplete = instrumentation.observe_episode(
        episode(2, values=(0.0, 0.0))
    )

    assert valid.classification == "valid"
    assert valid.valid_for_scheduler is True
    assert empty.classification == "empty"
    assert empty.game_length is None
    assert incomplete.classification == "incomplete"
    assert incomplete.valid_for_scheduler is False


def test_mixed_terminal_values_are_abnormal() -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 1)

    observation = instrumentation.observe_episode(
        episode(2, values=(0.0, 1.0))
    )

    assert observation.classification == "abnormal"
    assert "mixed_terminal_values" in observation.anomaly_types


@pytest.mark.parametrize(
    ("mutator", "anomaly"),
    [
        (lambda samples: [samples[0][:5]], "malformed_sample"),
        (
            lambda samples: [
                (*samples[0][:4], 2, samples[0][5]),
                samples[1],
            ],
            "nonconsecutive_step",
        ),
        (
            lambda samples: [
                samples[0],
                (*samples[1][:4], 3, samples[1][5]),
            ],
            "invalid_step_range",
        ),
        (
            lambda samples: [
                (*samples[0][:5], 3),
                (*samples[1][:5], 3),
            ],
            "declared_length_mismatch",
        ),
    ],
)
def test_structural_anomalies_are_abnormal(mutator, anomaly: str) -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 1)

    observation = instrumentation.observe_episode(mutator(episode(2)))

    assert observation.classification == "abnormal"
    assert anomaly in observation.anomaly_types


def test_episode_over_length_limit_is_abnormal() -> None:
    instrumentation = ReplayInstrumentation(config(max_valid_game_length=3))
    instrumentation.begin_iteration(1, 1)

    observation = instrumentation.observe_episode(episode(4))

    assert observation.classification == "abnormal"
    assert observation.anomaly_types == ("game_length_exceeds_limit",)


def test_finalize_separates_scheduler_lengths_from_baseline_counts() -> None:
    episodes = [episode(2), [], episode(3, values=(0.0, 0.0, 0.0))]
    instrumentation, stats = finalize_episodes(episodes)

    assert stats.valid_game_lengths == (2,)
    assert stats.valid_game_count == 1
    assert stats.realised_valid_states == 2
    assert stats.games_attempted == 3
    assert stats.games_completed == 2
    assert stats.positions_generated == 5
    assert stats.min_game_length == 2
    assert stats.max_game_length == 3
    assert stats.mean_game_length == 2.5
    assert stats.excluded_game_count_by_reason == {
        "empty_game": 1,
        "abnormal_game": 0,
        "incomplete_game": 1,
    }
    assert instrumentation.state.completed_iteration == 1


def test_all_nonempty_abnormal_episodes_count_toward_baseline_positions() -> None:
    malformed = [episode(2)[0][:5]]
    episodes = [episode(2), malformed]
    _, stats = finalize_episodes(episodes)

    assert stats.valid_game_lengths == (2,)
    assert stats.games_completed == 2
    assert stats.positions_generated == 3


def test_finalize_rejects_coach_mismatch_atomically() -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 1)
    item = episode(2)
    instrumentation.observe_episode(item)
    before = copy.deepcopy(instrumentation.state)

    metrics = coach_metrics([item])
    metrics["positions_generated"] = 3
    with pytest.raises(ReplayInstrumentationError, match="positions_generated"):
        instrumentation.finalize_iteration(metrics)

    assert instrumentation.state == before
    with pytest.raises(ReplayInstrumentationError, match="iteration boundary"):
        instrumentation.state_dict()


def test_finalize_rejects_incomplete_attempt_loop_atomically() -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 2)
    item = episode(2)
    instrumentation.observe_episode(item)
    before = copy.deepcopy(instrumentation.state)

    with pytest.raises(ReplayInstrumentationError, match="planned 2"):
        instrumentation.finalize_iteration(coach_metrics([item]))

    assert instrumentation.state == before


def test_abort_discards_active_iteration_without_advancing_state() -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 1)
    instrumentation.observe_episode(episode(2))

    instrumentation.abort_iteration()

    assert instrumentation.state.completed_iteration == 0
    assert instrumentation.state.total_games_attempted == 0
    instrumentation.begin_iteration(1, 1)


def test_passive_observer_returns_original_object_unchanged() -> None:
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 1)
    original_episode = episode(2)
    original_samples = tuple(original_episode)

    def observed_execute_episode():
        returned_episode = original_episode
        instrumentation.observe_episode(returned_episode)
        return returned_episode

    returned = observed_execute_episode()

    assert returned is original_episode
    assert tuple(returned) == original_samples


def test_observer_does_not_consume_global_rng() -> None:
    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    instrumentation = ReplayInstrumentation(config())
    instrumentation.begin_iteration(1, 1)

    instrumentation.observe_episode(episode(2))

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_state_round_trip_and_resume_do_not_duplicate_counts() -> None:
    first_episode = episode(2)
    instrumentation, _ = finalize_episodes([first_episode])
    payload = json.loads(json.dumps(instrumentation.state_dict()))

    restored = ReplayInstrumentation.from_state_dict(config(), payload)
    second_episode = episode(3)
    restored.begin_iteration(2, 1)
    restored.observe_episode(second_episode)
    restored.finalize_iteration(coach_metrics([second_episode]))

    assert restored.state.completed_iteration == 2
    assert restored.state.total_games_attempted == 2
    assert restored.state.total_games_completed == 2
    assert restored.state.total_valid_games == 2
    assert restored.state.total_positions_generated == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"board_size": 1},
        {"history_iterations": 0},
        {"max_valid_game_length": 0},
        {"state_hash_algorithm": "md5"},
        {"schema_version": 2},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ReplayInstrumentation(config(**overrides))


def test_canonical_hash_identity_includes_dtype_and_shape() -> None:
    source = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    assert canonical_state_hash(source) == canonical_state_hash(source.copy())
    assert canonical_state_hash(source) != canonical_state_hash(source.astype(np.int32))
    assert canonical_state_hash(source) != canonical_state_hash(source.reshape(3, 2, 4))


def test_noncontiguous_array_hash_matches_contiguous_copy() -> None:
    source = np.arange(24, dtype=np.int16).reshape(2, 3, 4).transpose(0, 2, 1)
    assert source.flags.c_contiguous is False

    assert canonical_state_hash(source) == canonical_state_hash(
        np.ascontiguousarray(source)
    )


def test_object_dtype_is_rejected() -> None:
    with pytest.raises(ReplayInstrumentationError, match="object dtype"):
        canonical_state_hash(np.array([[object()]], dtype=object))


def test_bucket_diversity_duplicate_and_trajectory_metrics() -> None:
    first = board(1)
    second = board(2)
    bucket = episode(2, boards=(first, first)) + episode(1, boards=(second,))

    bucket_stats, trajectory, seen = analyze_replay_bucket(
        bucket,
        iteration=5,
        expected_board_shape=(4, 17, 17),
        seen_hashes=set(),
        left_censored=True,
        expected_games=2,
    )

    assert bucket_stats.states == 3
    assert bucket_stats.unique_canonical_states == 2
    assert bucket_stats.incoming_unique_states == 2
    assert bucket_stats.incoming_unique_state_ratio == pytest.approx(2 / 3)
    assert bucket_stats.incoming_ratio_left_censored is True
    assert bucket_stats.duplicate_hash_occurrences == 1
    assert bucket_stats.duplicate_hash_groups == 1
    assert bucket_stats.duplicate_rate == pytest.approx(1 / 3)
    assert bucket_stats.state_effective_count == pytest.approx(9 / 5)
    assert len(seen) == 2
    assert trajectory.recovered_games == 2
    assert trajectory.min_game_length == 1
    assert trajectory.max_game_length == 2
    assert trajectory.total_recovered_positions == 3


def test_bucket_incoming_unique_excludes_previously_seen_hashes() -> None:
    existing = board(1)
    new = board(2)
    seen = {canonical_state_hash(existing)}
    bucket = episode(2, boards=(existing, new))

    bucket_stats, _, _ = analyze_replay_bucket(
        bucket,
        iteration=2,
        expected_board_shape=(4, 17, 17),
        seen_hashes=seen,
        left_censored=False,
    )

    assert bucket_stats.incoming_unique_states == 1
    assert bucket_stats.incoming_ratio_left_censored is False


def test_trajectory_uses_nearest_rank_p90() -> None:
    bucket = []
    for length in range(1, 11):
        bucket.extend(episode(length))

    _, trajectory, _ = analyze_replay_bucket(
        bucket,
        iteration=1,
        expected_board_shape=(4, 17, 17),
        seen_hashes=set(),
        left_censored=True,
        expected_games=10,
    )

    assert trajectory.recovered_games == 10
    assert trajectory.p90_game_length == 9


def metric_for_bucket(bucket: list[object], *, games_planned: int = 1) -> dict:
    lengths: list[int] = []
    current = 0
    for sample in bucket:
        current += 1
        if sample[4] == sample[5]:
            lengths.append(current)
            current = 0
    return {
        "games_planned": games_planned,
        "games_completed": len(lengths),
        "positions_generated": len(bucket),
        "min_game_length": min(lengths) if lengths else None,
        "max_game_length": max(lengths) if lengths else None,
        "mean_game_length": sum(lengths) / len(lengths) if lengths else None,
    }


def test_snapshot_uses_adaptive_expected_games_and_recovered_window() -> None:
    bucket_three = episode(2)
    bucket_four = episode(1)
    metrics = {
        3: metric_for_bucket(bucket_three, games_planned=1),
        4: metric_for_bucket(bucket_four, games_planned=3),
    }

    summary = summarize_replay_snapshot(
        4,
        [bucket_three, bucket_four],
        metrics,
        config(history_iterations=2),
    )

    assert summary["recovered_start_iteration"] == 3
    assert summary["recovered_end_iteration"] == 4
    assert summary["history_buckets"] == 2
    assert summary["total_states"] == 3
    assert summary["iteration_rows"][0]["incoming_ratio_left_censored"] is True
    assert summary["iteration_rows"][1]["incoming_ratio_left_censored"] is False
    assert summary["trajectory_rows"][1]["expected_games"] == 3
    assert summary["trajectory_rows"][1]["empty_games"] == 2


def test_snapshot_marks_count_mismatch_as_failed() -> None:
    bucket = episode(2)
    metrics = {1: metric_for_bucket(bucket)}
    metrics[1]["positions_generated"] = 3

    summary = summarize_replay_snapshot(1, [bucket], metrics, config())

    assert summary["status"] == "failed"
    assert summary["validations"]["counts_match_metrics"] is False


def test_snapshot_rejects_missing_metrics_iteration() -> None:
    with pytest.raises(ReplayInstrumentationError, match="missing metrics"):
        summarize_replay_snapshot(
            2,
            [episode(1), episode(1)],
            {2: metric_for_bucket(episode(1))},
            config(),
        )
