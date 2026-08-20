"""Passive replay instrumentation for Adaptive self-play experiments.

The online observer validates episode metadata and accumulates only the values
needed by the scheduler and training metrics.  Full canonical-state hashing and
rolling-replay analysis are exposed separately for offline use.
"""

from __future__ import annotations

import copy
import json
import math
import numbers
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from baseline.analysis.scripts.summarize_replay import (
    ReplayAnalysisError,
    canonical_state_hash as _baseline_canonical_state_hash,
)


class ReplayInstrumentationError(RuntimeError):
    """Raised when replay observations cannot be safely committed or analysed."""


@dataclass(frozen=True)
class ReplayInstrumentationConfig:
    board_size: int
    history_iterations: int
    max_valid_game_length: int = 150
    state_hash_algorithm: str = "sha256"
    schema_version: int = 1


@dataclass(frozen=True)
class EpisodeObservation:
    classification: str
    game_length: int | None
    positions_generated: int
    valid_for_scheduler: bool
    anomaly_types: tuple[str, ...]


@dataclass(frozen=True)
class IterationReplayStats:
    iteration: int
    games_planned: int
    games_attempted: int
    games_completed: int
    valid_game_count: int
    valid_game_lengths: tuple[int, ...]
    mean_valid_game_length: float | None
    realised_valid_states: int
    excluded_game_count_by_reason: dict[str, int]
    anomaly_count_by_type: dict[str, int]
    positions_generated: int
    min_game_length: int | None
    max_game_length: int | None
    mean_game_length: float | None


@dataclass
class ReplayInstrumentationState:
    schema_version: int
    completed_iteration: int
    total_games_attempted: int
    total_games_completed: int
    total_valid_games: int
    total_positions_generated: int
    cumulative_excluded_game_count_by_reason: dict[str, int]
    cumulative_anomaly_count_by_type: dict[str, int]


@dataclass(frozen=True)
class ReplayBucketStats:
    iteration: int
    states: int
    unique_canonical_states: int
    incoming_unique_states: int
    incoming_unique_state_ratio: float
    incoming_ratio_left_censored: bool
    duplicate_hash_occurrences: int
    duplicate_hash_groups: int
    duplicate_rate: float
    state_effective_count: float


@dataclass(frozen=True)
class ReplayTrajectoryStats:
    iteration: int
    expected_games: int
    recovered_games: int
    empty_games: int
    anomalous_games: int
    min_game_length: int | None
    max_game_length: int | None
    mean_game_length: float | None
    median_game_length: float | None
    p90_game_length: int | None
    total_recovered_positions: int


_EXCLUSION_REASONS = ("empty_game", "abnormal_game", "incomplete_game")
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "completed_iteration",
        "total_games_attempted",
        "total_games_completed",
        "total_valid_games",
        "total_positions_generated",
        "cumulative_excluded_game_count_by_reason",
        "cumulative_anomaly_count_by_type",
    }
)
_LIMITATIONS = (
    "The final snapshot contains only iterations in the rolling replay window.",
    "Buffer-level diversity before recovered_start_iteration cannot be recovered.",
    "The snapshot cannot recover how often each state was sampled during training.",
    "Incoming uniqueness in the first retained bucket is left-censored.",
)


@dataclass
class _ActiveIteration:
    iteration: int
    games_planned: int
    games_attempted: int = 0
    games_completed: int = 0
    positions_generated: int = 0
    valid_game_lengths: list[int] = field(default_factory=list)
    all_nonempty_game_lengths: list[int] = field(default_factory=list)
    excluded_game_count_by_reason: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in _EXCLUSION_REASONS}
    )
    anomaly_count_by_type: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class _BucketAnalysis:
    bucket_stats: ReplayBucketStats
    trajectory_stats: ReplayTrajectoryStats
    state_counts: Counter[bytes]
    anomaly_counts: Counter[str]


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_nonnegative_counts(counts: object, *, field_name: str) -> None:
    if not isinstance(counts, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    for name, count in counts.items():
        if not isinstance(name, str):
            raise ValueError(f"{field_name} keys must be strings")
        if not _is_integer(count) or count < 0:
            raise ValueError(f"{field_name}[{name!r}] must be a non-negative integer")


def _validate_config(config: ReplayInstrumentationConfig) -> None:
    if not isinstance(config, ReplayInstrumentationConfig):
        raise ValueError("config must be a ReplayInstrumentationConfig")
    if not _is_integer(config.board_size) or config.board_size < 2:
        raise ValueError("board_size must be an integer >= 2")
    if not _is_integer(config.history_iterations) or config.history_iterations < 1:
        raise ValueError("history_iterations must be an integer >= 1")
    if (
        not _is_integer(config.max_valid_game_length)
        or config.max_valid_game_length < 1
    ):
        raise ValueError("max_valid_game_length must be an integer >= 1")
    if config.state_hash_algorithm != "sha256":
        raise ValueError("state_hash_algorithm must be 'sha256'")
    if not _is_integer(config.schema_version) or config.schema_version != 1:
        raise ValueError("ReplayInstrumentationConfig schema_version must be 1")


def _fresh_state() -> ReplayInstrumentationState:
    return ReplayInstrumentationState(
        schema_version=1,
        completed_iteration=0,
        total_games_attempted=0,
        total_games_completed=0,
        total_valid_games=0,
        total_positions_generated=0,
        cumulative_excluded_game_count_by_reason={
            reason: 0 for reason in _EXCLUSION_REASONS
        },
        cumulative_anomaly_count_by_type={},
    )


def _copy_state(state: ReplayInstrumentationState) -> ReplayInstrumentationState:
    return ReplayInstrumentationState(
        schema_version=int(state.schema_version),
        completed_iteration=int(state.completed_iteration),
        total_games_attempted=int(state.total_games_attempted),
        total_games_completed=int(state.total_games_completed),
        total_valid_games=int(state.total_valid_games),
        total_positions_generated=int(state.total_positions_generated),
        cumulative_excluded_game_count_by_reason=dict(
            state.cumulative_excluded_game_count_by_reason
        ),
        cumulative_anomaly_count_by_type=dict(
            state.cumulative_anomaly_count_by_type
        ),
    )


def _validate_state(
    config: ReplayInstrumentationConfig,
    state: ReplayInstrumentationState,
) -> None:
    if not isinstance(state, ReplayInstrumentationState):
        raise ValueError("state must be a ReplayInstrumentationState")
    if (
        not _is_integer(state.schema_version)
        or state.schema_version != 1
        or state.schema_version != config.schema_version
    ):
        raise ValueError("ReplayInstrumentationState schema_version must be 1")
    count_fields = (
        "completed_iteration",
        "total_games_attempted",
        "total_games_completed",
        "total_valid_games",
        "total_positions_generated",
    )
    for field_name in count_fields:
        value = getattr(state, field_name)
        if not _is_integer(value) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    _validate_nonnegative_counts(
        state.cumulative_excluded_game_count_by_reason,
        field_name="cumulative_excluded_game_count_by_reason",
    )
    missing_reasons = set(_EXCLUSION_REASONS) - set(
        state.cumulative_excluded_game_count_by_reason
    )
    if missing_reasons:
        raise ValueError(
            "cumulative excluded counts are missing: "
            + ", ".join(sorted(missing_reasons))
        )
    _validate_nonnegative_counts(
        state.cumulative_anomaly_count_by_type,
        field_name="cumulative_anomaly_count_by_type",
    )
    if state.total_valid_games > state.total_games_completed:
        raise ValueError("total_valid_games cannot exceed total_games_completed")
    if state.total_games_completed > state.total_games_attempted:
        raise ValueError("total_games_completed cannot exceed total_games_attempted")


def _episode_is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


class ReplayInstrumentation:
    """Passive, iteration-scoped observer for Adaptive self-play."""

    def __init__(
        self,
        config: ReplayInstrumentationConfig,
        state: ReplayInstrumentationState | None = None,
    ) -> None:
        _validate_config(config)
        candidate = _fresh_state() if state is None else state
        _validate_state(config, candidate)
        self.config = config
        self.state = _copy_state(candidate)
        self._active: _ActiveIteration | None = None

    @property
    def expected_board_shape(self) -> tuple[int, int, int]:
        side = 2 * int(self.config.board_size) - 1
        return (4, side, side)

    def begin_iteration(self, iteration: int, games_planned: int) -> None:
        if self._active is not None:
            raise ValueError("an iteration is already active")
        expected_iteration = self.state.completed_iteration + 1
        if not _is_integer(iteration) or iteration != expected_iteration:
            raise ValueError(
                f"iteration must be {expected_iteration}, got {iteration!r}"
            )
        if not _is_integer(games_planned) or games_planned < 1:
            raise ValueError("games_planned must be an integer >= 1")
        self._active = _ActiveIteration(
            iteration=int(iteration),
            games_planned=int(games_planned),
        )

    def observe_episode(
        self,
        episode_examples: Sequence[Any],
    ) -> EpisodeObservation:
        active = self._require_active()
        active.games_attempted += 1

        if not _episode_is_sequence(episode_examples):
            return self._record_abnormal_episode(
                active,
                game_length=None,
                positions_generated=0,
                anomaly_types={"malformed_sample"},
            )

        episode_length = len(episode_examples)
        if episode_length == 0:
            active.excluded_game_count_by_reason["empty_game"] += 1
            return EpisodeObservation(
                classification="empty",
                game_length=None,
                positions_generated=0,
                valid_for_scheduler=False,
                anomaly_types=(),
            )

        active.games_completed += 1
        active.positions_generated += episode_length
        active.all_nonempty_game_lengths.append(episode_length)

        anomaly_types: set[str] = set()
        steps: list[int] = []
        declared_lengths: list[int] = []
        values: list[float] = []

        for sample in episode_examples:
            if not isinstance(sample, (tuple, list)) or len(sample) < 6:
                anomaly_types.add("malformed_sample")
                continue

            value = sample[2]
            step_value = sample[4]
            game_length_value = sample[5]

            if not _is_finite_number(value) or float(value) not in (-1.0, 0.0, 1.0):
                anomaly_types.add("invalid_value")
            else:
                values.append(float(value))

            if not _is_integer(step_value) or not _is_integer(game_length_value):
                anomaly_types.add("invalid_step_metadata")
                continue

            step = int(step_value)
            declared_length = int(game_length_value)
            steps.append(step)
            declared_lengths.append(declared_length)
            if declared_length < 1 or step < 1 or step > declared_length:
                anomaly_types.add("invalid_step_range")

        if episode_length > self.config.max_valid_game_length:
            anomaly_types.add("game_length_exceeds_limit")

        if len(steps) == episode_length:
            if steps[0] != 1 or any(
                current != previous + 1
                for previous, current in zip(steps, steps[1:])
            ):
                anomaly_types.add("nonconsecutive_step")
            if len(set(declared_lengths)) > 1:
                anomaly_types.add("inconsistent_declared_game_length")
            if (
                declared_lengths[0] != episode_length
                or steps[-1] != declared_lengths[-1]
            ):
                anomaly_types.add("declared_length_mismatch")

        if len(values) == episode_length:
            has_zero = any(value == 0.0 for value in values)
            has_nonzero = any(value != 0.0 for value in values)
            if has_zero and has_nonzero:
                anomaly_types.add("mixed_terminal_values")

        if anomaly_types:
            return self._record_abnormal_episode(
                active,
                game_length=episode_length,
                positions_generated=episode_length,
                anomaly_types=anomaly_types,
            )

        if all(value == 0.0 for value in values):
            active.excluded_game_count_by_reason["incomplete_game"] += 1
            return EpisodeObservation(
                classification="incomplete",
                game_length=episode_length,
                positions_generated=episode_length,
                valid_for_scheduler=False,
                anomaly_types=(),
            )

        active.valid_game_lengths.append(episode_length)
        return EpisodeObservation(
            classification="valid",
            game_length=episode_length,
            positions_generated=episode_length,
            valid_for_scheduler=True,
            anomaly_types=(),
        )

    def finalize_iteration(
        self,
        coach_metrics: Mapping[str, Any],
    ) -> IterationReplayStats:
        active = self._require_active()
        if active.games_attempted != active.games_planned:
            raise ReplayInstrumentationError(
                f"Replay instrumentation iteration {active.iteration} observed "
                f"{active.games_attempted} attempts but planned "
                f"{active.games_planned}"
            )

        stats = self._build_iteration_stats(active)
        self._validate_coach_metrics(stats, coach_metrics)

        cumulative_excluded = dict(
            self.state.cumulative_excluded_game_count_by_reason
        )
        for reason, count in stats.excluded_game_count_by_reason.items():
            cumulative_excluded[reason] = cumulative_excluded.get(reason, 0) + count
        cumulative_anomalies = dict(self.state.cumulative_anomaly_count_by_type)
        for anomaly, count in stats.anomaly_count_by_type.items():
            cumulative_anomalies[anomaly] = cumulative_anomalies.get(anomaly, 0) + count

        new_state = ReplayInstrumentationState(
            schema_version=self.state.schema_version,
            completed_iteration=stats.iteration,
            total_games_attempted=(
                self.state.total_games_attempted + stats.games_attempted
            ),
            total_games_completed=(
                self.state.total_games_completed + stats.games_completed
            ),
            total_valid_games=self.state.total_valid_games + stats.valid_game_count,
            total_positions_generated=(
                self.state.total_positions_generated + stats.positions_generated
            ),
            cumulative_excluded_game_count_by_reason=cumulative_excluded,
            cumulative_anomaly_count_by_type=cumulative_anomalies,
        )
        _validate_state(self.config, new_state)

        self.state = new_state
        self._active = None
        return stats

    def abort_iteration(self) -> None:
        self._active = None

    def state_dict(self) -> dict[str, Any]:
        if self._active is not None:
            raise ReplayInstrumentationError(
                "instrumentation state can only be saved at an iteration boundary"
            )
        _validate_state(self.config, self.state)
        payload = asdict(self.state)
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("instrumentation state must be JSON serializable") from exc
        return copy.deepcopy(payload)

    @classmethod
    def from_state_dict(
        cls,
        config: ReplayInstrumentationConfig,
        state_dict: dict[str, Any],
    ) -> "ReplayInstrumentation":
        _validate_config(config)
        if not isinstance(state_dict, dict):
            raise ValueError("state_dict must be a dictionary")
        missing = sorted(_STATE_FIELDS - set(state_dict))
        if missing:
            raise ValueError(
                "instrumentation state is missing required field(s): "
                + ", ".join(missing)
            )
        state = ReplayInstrumentationState(
            schema_version=state_dict["schema_version"],
            completed_iteration=state_dict["completed_iteration"],
            total_games_attempted=state_dict["total_games_attempted"],
            total_games_completed=state_dict["total_games_completed"],
            total_valid_games=state_dict["total_valid_games"],
            total_positions_generated=state_dict["total_positions_generated"],
            cumulative_excluded_game_count_by_reason=copy.deepcopy(
                state_dict["cumulative_excluded_game_count_by_reason"]
            ),
            cumulative_anomaly_count_by_type=copy.deepcopy(
                state_dict["cumulative_anomaly_count_by_type"]
            ),
        )
        return cls(config, state)

    def _require_active(self) -> _ActiveIteration:
        if self._active is None:
            raise ValueError("no active iteration")
        return self._active

    @staticmethod
    def _record_abnormal_episode(
        active: _ActiveIteration,
        *,
        game_length: int | None,
        positions_generated: int,
        anomaly_types: set[str],
    ) -> EpisodeObservation:
        active.excluded_game_count_by_reason["abnormal_game"] += 1
        ordered_anomalies = tuple(sorted(anomaly_types))
        active.anomaly_count_by_type.update(ordered_anomalies)
        return EpisodeObservation(
            classification="abnormal",
            game_length=game_length,
            positions_generated=positions_generated,
            valid_for_scheduler=False,
            anomaly_types=ordered_anomalies,
        )

    @staticmethod
    def _build_iteration_stats(active: _ActiveIteration) -> IterationReplayStats:
        valid_lengths = tuple(active.valid_game_lengths)
        nonempty_lengths = active.all_nonempty_game_lengths
        return IterationReplayStats(
            iteration=active.iteration,
            games_planned=active.games_planned,
            games_attempted=active.games_attempted,
            games_completed=active.games_completed,
            valid_game_count=len(valid_lengths),
            valid_game_lengths=valid_lengths,
            mean_valid_game_length=(
                sum(valid_lengths) / len(valid_lengths) if valid_lengths else None
            ),
            realised_valid_states=sum(valid_lengths),
            excluded_game_count_by_reason=dict(
                active.excluded_game_count_by_reason
            ),
            anomaly_count_by_type=dict(sorted(active.anomaly_count_by_type.items())),
            positions_generated=active.positions_generated,
            min_game_length=min(nonempty_lengths) if nonempty_lengths else None,
            max_game_length=max(nonempty_lengths) if nonempty_lengths else None,
            mean_game_length=(
                sum(nonempty_lengths) / len(nonempty_lengths)
                if nonempty_lengths
                else None
            ),
        )

    @staticmethod
    def _validate_coach_metrics(
        stats: IterationReplayStats,
        coach_metrics: Mapping[str, Any],
    ) -> None:
        if not isinstance(coach_metrics, Mapping):
            raise ReplayInstrumentationError("coach_metrics must be a mapping")
        comparisons = (
            ("games_completed", stats.games_completed),
            ("positions_generated", stats.positions_generated),
            ("min_game_length", stats.min_game_length),
            ("max_game_length", stats.max_game_length),
        )
        for field_name, observed in comparisons:
            reported = coach_metrics.get(field_name, "<missing>")
            if reported != observed:
                raise ReplayInstrumentationError(
                    f"Replay instrumentation mismatch at iteration "
                    f"{stats.iteration}: observed {field_name}={observed!r}, "
                    f"Coach reported {field_name}={reported!r}"
                )

        reported_mean = coach_metrics.get("mean_game_length", "<missing>")
        observed_mean = stats.mean_game_length
        means_match = (
            observed_mean is None and reported_mean is None
        ) or (
            observed_mean is not None
            and _is_finite_number(reported_mean)
            and math.isclose(
                observed_mean,
                float(reported_mean),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if not means_match:
            raise ReplayInstrumentationError(
                f"Replay instrumentation mismatch at iteration {stats.iteration}: "
                f"observed mean_game_length={observed_mean!r}, Coach reported "
                f"mean_game_length={reported_mean!r}"
            )


def canonical_state_hash(board: Any) -> bytes:
    """Return the Baseline-compatible SHA-256 canonical-state identity."""

    try:
        return _baseline_canonical_state_hash(board)
    except ReplayAnalysisError as exc:
        raise ReplayInstrumentationError(str(exc)) from exc


def _nearest_rank(values: list[int], quantile: float) -> int:
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


def _analyze_replay_bucket(
    bucket: Sequence[Any],
    *,
    iteration: int,
    expected_board_shape: tuple[int, int, int],
    seen_hashes: set[bytes],
    left_censored: bool,
    expected_games: int | None,
) -> _BucketAnalysis:
    if not _is_integer(iteration) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    if (
        not isinstance(expected_board_shape, tuple)
        or len(expected_board_shape) != 3
        or any(not _is_integer(item) or item < 1 for item in expected_board_shape)
    ):
        raise ValueError("expected_board_shape must contain three positive integers")
    if not isinstance(seen_hashes, set):
        raise ValueError("seen_hashes must be a set")
    if not isinstance(left_censored, bool):
        raise ValueError("left_censored must be a bool")
    if expected_games is not None and (
        not _is_integer(expected_games) or expected_games < 1
    ):
        raise ValueError("expected_games must be an integer >= 1")

    state_counts: Counter[bytes] = Counter()
    anomaly_counts: Counter[str] = Counter()
    game_lengths: list[int] = []
    anomalous_games = 0
    active_game = False
    observed_length = 0
    declared_length: int | None = None
    previous_step = 0
    current_game_anomalous = False

    try:
        samples = iter(bucket)
    except TypeError as exc:
        raise ReplayInstrumentationError(
            "replay history bucket is not iterable"
        ) from exc

    for sample in samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 6:
            anomaly_counts["malformed_sample"] += 1
            if active_game:
                current_game_anomalous = True
            continue

        board, step_value, game_length_value = sample[0], sample[4], sample[5]
        try:
            array = np.asarray(board)
            if tuple(array.shape) != expected_board_shape:
                anomaly_counts["unexpected_board_shape"] += 1
            state_counts[canonical_state_hash(array)] += 1
        except (ReplayInstrumentationError, TypeError, ValueError):
            anomaly_counts["invalid_canonical_board"] += 1

        if not _is_integer(step_value) or not _is_integer(game_length_value):
            anomaly_counts["invalid_step_metadata"] += 1
            if active_game:
                current_game_anomalous = True
            continue
        step = int(step_value)
        game_length = int(game_length_value)
        if step < 1 or game_length < 1 or step > game_length:
            anomaly_counts["invalid_step_range"] += 1
            if active_game:
                current_game_anomalous = True
            continue

        if step == 1:
            if active_game:
                anomaly_counts["premature_step_reset"] += 1
                anomalous_games += _finish_game(
                    game_lengths,
                    observed_length,
                    declared_length,
                    True,
                )
            active_game = True
            observed_length = 0
            declared_length = game_length
            previous_step = 0
            current_game_anomalous = False
        elif not active_game:
            anomaly_counts["missing_step_reset"] += 1
            active_game = True
            observed_length = 0
            declared_length = game_length
            previous_step = 0
            current_game_anomalous = True

        if step != previous_step + 1:
            anomaly_counts["nonconsecutive_step"] += 1
            current_game_anomalous = True
        if declared_length != game_length:
            anomaly_counts["inconsistent_declared_game_length"] += 1
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
            active_game = False
            observed_length = 0
            declared_length = None
            previous_step = 0
            current_game_anomalous = False

    if active_game:
        anomaly_counts["incomplete_final_game"] += 1
        anomalous_games += _finish_game(
            game_lengths,
            observed_length,
            declared_length,
            True,
        )

    states = sum(state_counts.values())
    unique_states = len(state_counts)
    incoming_unique = len(set(state_counts) - seen_hashes)
    duplicate_occurrences = states - unique_states
    duplicate_groups = sum(1 for count in state_counts.values() if count > 1)
    squared_frequency_sum = sum(count * count for count in state_counts.values())
    effective_count = (
        states * states / squared_frequency_sum if squared_frequency_sum else 0.0
    )
    seen_hashes.update(state_counts)

    recovered_games = len(game_lengths)
    planned_games = recovered_games if expected_games is None else expected_games
    bucket_stats = ReplayBucketStats(
        iteration=int(iteration),
        states=states,
        unique_canonical_states=unique_states,
        incoming_unique_states=incoming_unique,
        incoming_unique_state_ratio=(incoming_unique / states if states else 0.0),
        incoming_ratio_left_censored=left_censored,
        duplicate_hash_occurrences=duplicate_occurrences,
        duplicate_hash_groups=duplicate_groups,
        duplicate_rate=(duplicate_occurrences / states if states else 0.0),
        state_effective_count=effective_count,
    )
    trajectory_stats = ReplayTrajectoryStats(
        iteration=int(iteration),
        expected_games=planned_games,
        recovered_games=recovered_games,
        empty_games=max(0, planned_games - recovered_games),
        anomalous_games=anomalous_games,
        min_game_length=min(game_lengths) if game_lengths else None,
        max_game_length=max(game_lengths) if game_lengths else None,
        mean_game_length=(
            statistics.fmean(game_lengths) if game_lengths else None
        ),
        median_game_length=(
            float(statistics.median(game_lengths)) if game_lengths else None
        ),
        p90_game_length=(
            _nearest_rank(game_lengths, 0.90) if game_lengths else None
        ),
        total_recovered_positions=sum(game_lengths),
    )
    return _BucketAnalysis(
        bucket_stats=bucket_stats,
        trajectory_stats=trajectory_stats,
        state_counts=state_counts,
        anomaly_counts=anomaly_counts,
    )


def analyze_replay_bucket(
    bucket: Sequence[Any],
    *,
    iteration: int,
    expected_board_shape: tuple[int, int, int],
    seen_hashes: set[bytes],
    left_censored: bool,
    expected_games: int | None = None,
) -> tuple[ReplayBucketStats, ReplayTrajectoryStats, set[bytes]]:
    """Analyse one retained replay bucket without reading or writing files."""

    analysis = _analyze_replay_bucket(
        bucket,
        iteration=iteration,
        expected_board_shape=expected_board_shape,
        seen_hashes=seen_hashes,
        left_censored=left_censored,
        expected_games=expected_games,
    )
    return analysis.bucket_stats, analysis.trajectory_stats, seen_hashes


def _metric_integer(
    record: Mapping[str, Any],
    field_name: str,
    *,
    minimum: int,
    iteration: int,
) -> int:
    value = record.get(field_name)
    if not _is_integer(value) or value < minimum:
        raise ReplayInstrumentationError(
            f"metrics iteration {iteration} field {field_name} must be an "
            f"integer >= {minimum}"
        )
    return int(value)


def _length_metrics_match(
    trajectory: ReplayTrajectoryStats,
    record: Mapping[str, Any],
) -> bool:
    if record.get("min_game_length") != trajectory.min_game_length:
        return False
    if record.get("max_game_length") != trajectory.max_game_length:
        return False
    reported_mean = record.get("mean_game_length")
    if trajectory.mean_game_length is None:
        return reported_mean is None
    return _is_finite_number(reported_mean) and math.isclose(
        trajectory.mean_game_length,
        float(reported_mean),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def summarize_replay_snapshot(
    replay_iteration: int,
    replay_buckets: Sequence[Sequence[Any]],
    metrics_by_iteration: Mapping[int, Mapping[str, Any]],
    config: ReplayInstrumentationConfig,
) -> dict[str, Any]:
    """Summarise an already-loaded rolling replay snapshot."""

    _validate_config(config)
    if not _is_integer(replay_iteration) or replay_iteration < 1:
        raise ValueError("replay_iteration must be a positive integer")
    if not _episode_is_sequence(replay_buckets):
        raise ValueError("replay_buckets must be a sequence")
    history_buckets = len(replay_buckets)
    if history_buckets < 1:
        raise ValueError("replay_buckets must contain at least one bucket")
    if history_buckets > config.history_iterations:
        raise ValueError("replay bucket count exceeds history_iterations")
    if history_buckets > replay_iteration:
        raise ValueError("replay bucket count exceeds replay_iteration")
    if not isinstance(metrics_by_iteration, Mapping):
        raise ValueError("metrics_by_iteration must be a mapping")

    recovered_start = int(replay_iteration) - history_buckets + 1
    expected_shape = (
        4,
        2 * int(config.board_size) - 1,
        2 * int(config.board_size) - 1,
    )
    seen_hashes: set[bytes] = set()
    global_counts: Counter[bytes] = Counter()
    all_anomalies: Counter[str] = Counter()
    iteration_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    counts_match_metrics = True
    trajectories_match_metrics = True
    empty_buckets = 0
    empty_games = 0
    anomalous_games = 0

    for bucket_index, bucket in enumerate(replay_buckets):
        iteration = recovered_start + bucket_index
        if iteration not in metrics_by_iteration:
            raise ReplayInstrumentationError(
                f"missing metrics record for replay iteration {iteration}"
            )
        metric = metrics_by_iteration[iteration]
        if not isinstance(metric, Mapping):
            raise ReplayInstrumentationError(
                f"metrics record for iteration {iteration} must be a mapping"
            )
        expected_games = _metric_integer(
            metric,
            "games_planned",
            minimum=1,
            iteration=iteration,
        )
        metric_games_completed = _metric_integer(
            metric,
            "games_completed",
            minimum=0,
            iteration=iteration,
        )
        metric_positions = _metric_integer(
            metric,
            "positions_generated",
            minimum=0,
            iteration=iteration,
        )

        analysis = _analyze_replay_bucket(
            bucket,
            iteration=iteration,
            expected_board_shape=expected_shape,
            seen_hashes=seen_hashes,
            left_censored=bucket_index == 0,
            expected_games=expected_games,
        )
        bucket_stats = analysis.bucket_stats
        trajectory = analysis.trajectory_stats
        count_matches = bucket_stats.states == metric_positions
        games_match = trajectory.recovered_games == metric_games_completed
        lengths_match = _length_metrics_match(trajectory, metric)
        counts_match_metrics = counts_match_metrics and count_matches
        trajectories_match_metrics = (
            trajectories_match_metrics and games_match and lengths_match
        )

        iteration_row = asdict(bucket_stats)
        iteration_row.update(
            {
                "metrics_positions_generated": metric_positions,
                "count_matches_metrics": count_matches,
            }
        )
        trajectory_row = asdict(trajectory)
        trajectory_row.update(
            {
                "metrics_games_completed": metric_games_completed,
                "metrics_min_game_length": metric.get("min_game_length"),
                "metrics_max_game_length": metric.get("max_game_length"),
                "metrics_mean_game_length": metric.get("mean_game_length"),
                "games_match_metrics": games_match,
                "length_distribution_matches_metrics": lengths_match,
            }
        )
        iteration_rows.append(iteration_row)
        trajectory_rows.append(trajectory_row)
        global_counts.update(analysis.state_counts)
        all_anomalies.update(analysis.anomaly_counts)
        empty_buckets += int(bucket_stats.states == 0)
        empty_games += trajectory.empty_games
        anomalous_games += trajectory.anomalous_games

    total_states = sum(global_counts.values())
    final_unique_states = len(global_counts)
    duplicate_occurrences = total_states - final_unique_states
    duplicate_groups = sum(1 for count in global_counts.values() if count > 1)
    validations = {
        "history_bucket_count_within_limit": (
            history_buckets <= config.history_iterations
        ),
        "recovered_range_ends_at_replay_iteration": (
            recovered_start + history_buckets - 1 == replay_iteration
        ),
        "counts_match_metrics": counts_match_metrics,
        "trajectories_match_metrics": trajectories_match_metrics,
        "no_empty_buckets": empty_buckets == 0,
        "no_empty_games": empty_games == 0,
        "no_trajectory_anomalies": (
            anomalous_games == 0 and sum(all_anomalies.values()) == 0
        ),
    }
    return {
        "schema_version": 1,
        "status": "completed" if all(validations.values()) else "failed",
        "replay_iteration": int(replay_iteration),
        "history_buckets": history_buckets,
        "recovered_start_iteration": recovered_start,
        "recovered_end_iteration": int(replay_iteration),
        "total_states": total_states,
        "final_unique_states": final_unique_states,
        "final_unique_state_ratio": (
            final_unique_states / total_states if total_states else 0.0
        ),
        "duplicate_hash_occurrences": duplicate_occurrences,
        "duplicate_hash_groups": duplicate_groups,
        "duplicate_rate": (
            duplicate_occurrences / total_states if total_states else 0.0
        ),
        "empty_buckets": empty_buckets,
        "empty_games": empty_games,
        "anomalous_games": anomalous_games,
        "anomaly_count_by_type": dict(sorted(all_anomalies.items())),
        "limitations": list(_LIMITATIONS),
        "validations": validations,
        "iteration_rows": iteration_rows,
        "trajectory_rows": trajectory_rows,
    }


__all__ = [
    "EpisodeObservation",
    "IterationReplayStats",
    "ReplayBucketStats",
    "ReplayInstrumentation",
    "ReplayInstrumentationConfig",
    "ReplayInstrumentationError",
    "ReplayInstrumentationState",
    "ReplayTrajectoryStats",
    "analyze_replay_bucket",
    "canonical_state_hash",
    "summarize_replay_snapshot",
]
