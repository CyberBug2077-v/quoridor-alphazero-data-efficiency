"""Deterministic target-state scheduler for Adaptive self-play.

The scheduler is deliberately independent of configuration files, training,
replay storage, persistence, and budget enforcement.  A runtime supplies the
completed iteration's already-validated game lengths and owns persistence of
the JSON-serializable state returned here.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SchedulerConfig:
    target_states: int
    alpha: float
    minimum_observations: int
    initial_length: float
    first_iteration_games: int
    min_games: int
    max_games: int
    max_valid_game_length: int = 150
    rounding: str = "ceil"
    schema_version: int = 1


@dataclass(frozen=True)
class IterationLengthStats:
    iteration: int
    valid_game_lengths: tuple[int, ...]
    excluded_game_count_by_reason: dict[str, int]


@dataclass
class SchedulerState:
    schema_version: int
    completed_iteration: int
    current_ema_length: float
    next_iteration_games: int
    per_iteration_mean_valid_game_length_history: list[float | None]
    per_iteration_valid_game_count_history: list[int]
    excluded_game_count_by_reason: dict[str, int]
    last_clipping_decision: str | None
    scheduler_rng_state: object | None


@dataclass(frozen=True)
class SchedulerDecision:
    completed_iteration: int
    previous_length_estimate: float
    observed_mean_length: float | None
    updated_length_estimate: float
    raw_next_games: float | None
    rounded_next_games: int | None
    next_iteration_games: int
    update_applied: bool
    skipped_reason: str | None
    clipped: bool
    clipping_bound: str | None
    valid_observations: int
    excluded_observations: int
    # Wall-clock timing is observational and must not change decision equality.
    scheduler_seconds: float = field(compare=False)


_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "completed_iteration",
        "current_ema_length",
        "next_iteration_games",
        "per_iteration_mean_valid_game_length_history",
        "per_iteration_valid_game_count_history",
        "excluded_game_count_by_reason",
        "last_clipping_decision",
        "scheduler_rng_state",
    }
)
_CLIPPING_BOUNDS = frozenset({"min_games", "max_games"})


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_nonnegative_counts(counts: object, *, field_name: str) -> None:
    if not isinstance(counts, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    for reason, count in counts.items():
        if not isinstance(reason, str):
            raise ValueError(f"{field_name} reasons must be strings")
        if not _is_integer(count) or count < 0:
            raise ValueError(
                f"{field_name}[{reason!r}] must be a non-negative integer"
            )


def _validate_config(config: SchedulerConfig) -> None:
    if not isinstance(config, SchedulerConfig):
        raise ValueError("config must be a SchedulerConfig")
    if not _is_integer(config.target_states) or config.target_states <= 0:
        raise ValueError("target_states must be a positive integer")
    if (
        not _is_finite_number(config.alpha)
        or float(config.alpha) <= 0.0
        or float(config.alpha) > 1.0
    ):
        raise ValueError("alpha must be finite and satisfy 0 < alpha <= 1")
    if (
        not _is_integer(config.minimum_observations)
        or config.minimum_observations < 1
    ):
        raise ValueError("minimum_observations must be an integer >= 1")
    if (
        not _is_finite_number(config.initial_length)
        or float(config.initial_length) <= 0.0
    ):
        raise ValueError("initial_length must be finite and greater than 0")
    if not _is_integer(config.min_games) or not _is_integer(config.max_games):
        raise ValueError("min_games and max_games must be integers")
    if config.min_games < 1 or config.min_games > config.max_games:
        raise ValueError("games bounds must satisfy 1 <= min_games <= max_games")
    if (
        not _is_integer(config.first_iteration_games)
        or not config.min_games
        <= config.first_iteration_games
        <= config.max_games
    ):
        raise ValueError(
            "first_iteration_games must be an integer within games bounds"
        )
    if (
        not _is_integer(config.max_valid_game_length)
        or config.max_valid_game_length < 1
    ):
        raise ValueError("max_valid_game_length must be an integer >= 1")
    if config.rounding != "ceil":
        raise ValueError("rounding must be 'ceil'")
    if not _is_integer(config.schema_version) or config.schema_version != 1:
        raise ValueError("SchedulerConfig schema_version must be 1")


def _fresh_state(config: SchedulerConfig) -> SchedulerState:
    return SchedulerState(
        schema_version=1,
        completed_iteration=0,
        current_ema_length=float(config.initial_length),
        next_iteration_games=config.first_iteration_games,
        per_iteration_mean_valid_game_length_history=[],
        per_iteration_valid_game_count_history=[],
        excluded_game_count_by_reason={},
        last_clipping_decision=None,
        scheduler_rng_state=None,
    )


def _copy_state(state: SchedulerState) -> SchedulerState:
    return SchedulerState(
        schema_version=state.schema_version,
        completed_iteration=state.completed_iteration,
        current_ema_length=float(state.current_ema_length),
        next_iteration_games=state.next_iteration_games,
        per_iteration_mean_valid_game_length_history=list(
            state.per_iteration_mean_valid_game_length_history
        ),
        per_iteration_valid_game_count_history=list(
            state.per_iteration_valid_game_count_history
        ),
        excluded_game_count_by_reason=dict(state.excluded_game_count_by_reason),
        last_clipping_decision=state.last_clipping_decision,
        scheduler_rng_state=copy.deepcopy(state.scheduler_rng_state),
    )


def _validate_state(config: SchedulerConfig, state: SchedulerState) -> None:
    if not isinstance(state, SchedulerState):
        raise ValueError("state must be a SchedulerState")
    if (
        not _is_integer(state.schema_version)
        or state.schema_version != 1
        or state.schema_version != config.schema_version
    ):
        raise ValueError("SchedulerState schema_version must match version 1 config")
    if not _is_integer(state.completed_iteration) or state.completed_iteration < 0:
        raise ValueError("completed_iteration must be a non-negative integer")
    if (
        not _is_finite_number(state.current_ema_length)
        or float(state.current_ema_length) <= 0.0
    ):
        raise ValueError("current_ema_length must be finite and greater than 0")
    if (
        not _is_integer(state.next_iteration_games)
        or not config.min_games
        <= state.next_iteration_games
        <= config.max_games
    ):
        raise ValueError("next_iteration_games must be an integer within games bounds")
    if not isinstance(state.per_iteration_mean_valid_game_length_history, list):
        raise ValueError("mean game-length history must be a list")
    if not isinstance(state.per_iteration_valid_game_count_history, list):
        raise ValueError("valid game-count history must be a list")
    expected_length = state.completed_iteration
    if len(state.per_iteration_mean_valid_game_length_history) != expected_length:
        raise ValueError("mean game-length history length must equal completed_iteration")
    if len(state.per_iteration_valid_game_count_history) != expected_length:
        raise ValueError("valid game-count history length must equal completed_iteration")
    for index, mean_length in enumerate(
        state.per_iteration_mean_valid_game_length_history, start=1
    ):
        if mean_length is None:
            continue
        if (
            not _is_finite_number(mean_length)
            or float(mean_length) <= 0.0
            or float(mean_length) > config.max_valid_game_length
        ):
            raise ValueError(
                f"mean game-length history entry {index} must be finite and valid"
            )
    for index, count in enumerate(
        state.per_iteration_valid_game_count_history, start=1
    ):
        if not _is_integer(count) or count < 0:
            raise ValueError(
                f"valid game-count history entry {index} must be non-negative"
            )
    _validate_nonnegative_counts(
        state.excluded_game_count_by_reason,
        field_name="excluded_game_count_by_reason",
    )
    if (
        state.last_clipping_decision is not None
        and state.last_clipping_decision not in _CLIPPING_BOUNDS
    ):
        raise ValueError(
            "last_clipping_decision must be None, 'min_games', or 'max_games'"
        )
    try:
        json.dumps(state.scheduler_rng_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduler_rng_state must be JSON serializable") from exc


def _validate_stats(
    config: SchedulerConfig,
    state: SchedulerState,
    stats: IterationLengthStats,
) -> None:
    if not isinstance(stats, IterationLengthStats):
        raise ValueError("stats must be an IterationLengthStats")
    expected_iteration = state.completed_iteration + 1
    if not _is_integer(stats.iteration) or stats.iteration != expected_iteration:
        raise ValueError(
            f"stats.iteration must be {expected_iteration}, got {stats.iteration!r}"
        )
    for index, length in enumerate(stats.valid_game_lengths):
        if (
            not _is_integer(length)
            or length < 1
            or length > config.max_valid_game_length
        ):
            raise ValueError(
                "valid_game_lengths"
                f"[{index}] must be an integer between 1 and "
                f"{config.max_valid_game_length}"
            )
    _validate_nonnegative_counts(
        stats.excluded_game_count_by_reason,
        field_name="excluded_game_count_by_reason",
    )


class AdaptiveScheduler:
    """Target-state EMA scheduler with exact iteration-boundary resume semantics."""

    def __init__(
        self,
        config: SchedulerConfig,
        state: SchedulerState | None = None,
    ) -> None:
        _validate_config(config)
        candidate = _fresh_state(config) if state is None else state
        _validate_state(config, candidate)
        self.config = config
        self.state = _copy_state(candidate)

    @property
    def next_iteration_games(self) -> int:
        return self.state.next_iteration_games

    def update(self, stats: IterationLengthStats) -> SchedulerDecision:
        started = time.perf_counter()
        _validate_stats(self.config, self.state, stats)

        previous_length = self.state.current_ema_length
        valid_count = len(stats.valid_game_lengths)
        observed_mean = (
            sum(stats.valid_game_lengths) / valid_count if valid_count else None
        )

        if valid_count >= self.config.minimum_observations:
            assert observed_mean is not None
            updated_length = (
                float(self.config.alpha) * observed_mean
                + (1.0 - float(self.config.alpha)) * previous_length
            )
            if not math.isfinite(updated_length) or updated_length <= 0.0:
                raise ValueError("updated EMA length must be finite and greater than 0")
            raw_next_games = self.config.target_states / updated_length
            if not math.isfinite(raw_next_games) or raw_next_games <= 0.0:
                raise ValueError("raw next-game plan must be finite and greater than 0")
            rounded_next_games = math.ceil(raw_next_games)
            if rounded_next_games < self.config.min_games:
                next_games = self.config.min_games
                clipping_bound = "min_games"
            elif rounded_next_games > self.config.max_games:
                next_games = self.config.max_games
                clipping_bound = "max_games"
            else:
                next_games = rounded_next_games
                clipping_bound = None
            update_applied = True
            skipped_reason = None
        else:
            updated_length = previous_length
            raw_next_games = None
            rounded_next_games = None
            next_games = self.state.next_iteration_games
            clipping_bound = None
            update_applied = False
            skipped_reason = "insufficient_valid_observations"

        mean_history = list(
            self.state.per_iteration_mean_valid_game_length_history
        )
        mean_history.append(observed_mean)
        count_history = list(self.state.per_iteration_valid_game_count_history)
        count_history.append(valid_count)
        cumulative_excluded = dict(self.state.excluded_game_count_by_reason)
        for reason, count in stats.excluded_game_count_by_reason.items():
            cumulative_excluded[reason] = cumulative_excluded.get(reason, 0) + count

        new_state = SchedulerState(
            schema_version=self.state.schema_version,
            completed_iteration=stats.iteration,
            current_ema_length=updated_length,
            next_iteration_games=next_games,
            per_iteration_mean_valid_game_length_history=mean_history,
            per_iteration_valid_game_count_history=count_history,
            excluded_game_count_by_reason=cumulative_excluded,
            last_clipping_decision=clipping_bound,
            scheduler_rng_state=copy.deepcopy(self.state.scheduler_rng_state),
        )
        _validate_state(self.config, new_state)

        scheduler_seconds = time.perf_counter() - started
        if not math.isfinite(scheduler_seconds) or scheduler_seconds < 0.0:
            raise ValueError("scheduler timing must be finite and non-negative")
        decision = SchedulerDecision(
            completed_iteration=stats.iteration,
            previous_length_estimate=previous_length,
            observed_mean_length=observed_mean,
            updated_length_estimate=updated_length,
            raw_next_games=raw_next_games,
            rounded_next_games=rounded_next_games,
            next_iteration_games=next_games,
            update_applied=update_applied,
            skipped_reason=skipped_reason,
            clipped=clipping_bound is not None,
            clipping_bound=clipping_bound,
            valid_observations=valid_count,
            excluded_observations=sum(
                stats.excluded_game_count_by_reason.values()
            ),
            scheduler_seconds=scheduler_seconds,
        )
        self.state = new_state
        return decision

    def state_dict(self) -> dict[str, Any]:
        _validate_state(self.config, self.state)
        payload: dict[str, Any] = {
            "schema_version": self.state.schema_version,
            "completed_iteration": self.state.completed_iteration,
            "current_ema_length": self.state.current_ema_length,
            "next_iteration_games": self.state.next_iteration_games,
            "per_iteration_mean_valid_game_length_history": list(
                self.state.per_iteration_mean_valid_game_length_history
            ),
            "per_iteration_valid_game_count_history": list(
                self.state.per_iteration_valid_game_count_history
            ),
            "excluded_game_count_by_reason": dict(
                self.state.excluded_game_count_by_reason
            ),
            "last_clipping_decision": self.state.last_clipping_decision,
            "scheduler_rng_state": copy.deepcopy(self.state.scheduler_rng_state),
        }
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("scheduler state must be JSON serializable") from exc
        return payload

    @classmethod
    def from_state_dict(
        cls,
        config: SchedulerConfig,
        state_dict: dict[str, Any],
    ) -> "AdaptiveScheduler":
        _validate_config(config)
        if not isinstance(state_dict, dict):
            raise ValueError("state_dict must be a dictionary")
        missing = sorted(_STATE_FIELDS - set(state_dict))
        if missing:
            raise ValueError(
                "scheduler state is missing required field(s): " + ", ".join(missing)
            )
        if (
            not _is_integer(state_dict["schema_version"])
            or state_dict["schema_version"] != 1
        ):
            raise ValueError("scheduler state schema_version must be 1")
        state = SchedulerState(
            schema_version=state_dict["schema_version"],
            completed_iteration=state_dict["completed_iteration"],
            current_ema_length=state_dict["current_ema_length"],
            next_iteration_games=state_dict["next_iteration_games"],
            per_iteration_mean_valid_game_length_history=copy.deepcopy(
                state_dict["per_iteration_mean_valid_game_length_history"]
            ),
            per_iteration_valid_game_count_history=copy.deepcopy(
                state_dict["per_iteration_valid_game_count_history"]
            ),
            excluded_game_count_by_reason=copy.deepcopy(
                state_dict["excluded_game_count_by_reason"]
            ),
            last_clipping_decision=state_dict["last_clipping_decision"],
            scheduler_rng_state=copy.deepcopy(state_dict["scheduler_rng_state"]),
        )
        return cls(config, state)


__all__ = [
    "AdaptiveScheduler",
    "IterationLengthStats",
    "SchedulerConfig",
    "SchedulerDecision",
    "SchedulerState",
]
