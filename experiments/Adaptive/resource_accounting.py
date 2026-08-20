"""Deterministic resource accounting for Adaptive training iterations.

This module defines the matched-compute timing boundary and decides whether a
new iteration may start.  It does not load models, save checkpoints, update the
scheduler, or write training metrics.
"""

from __future__ import annotations

import json
import math
import numbers
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


class ResourceAccountingError(RuntimeError):
    """Raised when timing data cannot be safely committed."""


@dataclass(frozen=True)
class ResourceConfig:
    allocated_gpu_count: int
    max_gpu_hours: float
    max_iterations: int
    max_wall_clock_hours: float
    evaluation_time_included: bool = False
    instrumentation_overhead_included: bool = True
    crossing_policy: str = "keep_crossing_iteration"
    schema_version: int = 1


@dataclass(frozen=True)
class IterationResourceRecord:
    iteration: int
    self_play_seconds: float
    training_seconds: float
    instrumentation_seconds: float
    checkpoint_seconds: float
    iteration_seconds: float
    evaluation_seconds: float
    cumulative_training_seconds: float
    cumulative_gpu_hours: float
    self_play_time_fraction: float


@dataclass(frozen=True)
class BudgetDecision:
    continue_run: bool
    stop_reason: str | None
    crossing_iteration: bool
    overshoot_gpu_hours: float


@dataclass
class ResourceAccountingState:
    schema_version: int
    completed_iteration: int
    cumulative_training_seconds: float
    cumulative_gpu_hours: float
    cumulative_evaluation_seconds: float
    cumulative_wall_clock_seconds: float
    last_iteration_wall_clock_seconds: float | None
    last_iteration_record: IterationResourceRecord | None
    last_budget_decision: BudgetDecision | None
    budget_crossing_iteration: int | None


_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "completed_iteration",
        "cumulative_training_seconds",
        "cumulative_gpu_hours",
        "cumulative_evaluation_seconds",
        "cumulative_wall_clock_seconds",
        "last_iteration_wall_clock_seconds",
        "last_iteration_record",
        "last_budget_decision",
        "budget_crossing_iteration",
    }
)
_RECORD_FIELDS = frozenset(IterationResourceRecord.__dataclass_fields__)
_DECISION_FIELDS = frozenset(BudgetDecision.__dataclass_fields__)
_CROSSING_POLICIES = frozenset(
    {
        "keep_crossing_iteration",
        "keep_crossing_iteration_and_do_not_start_another",
    }
)
_STOP_REASONS = frozenset(
    {
        "gpu_hour_budget_reached",
        "max_iterations_reached",
        "wall_clock_safety_limit",
    }
)


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_positive(value: object) -> bool:
    return _is_finite_nonnegative(value) and float(value) > 0.0


def _validate_config(config: ResourceConfig) -> None:
    if not isinstance(config, ResourceConfig):
        raise ValueError("config must be a ResourceConfig")
    if not _is_integer(config.allocated_gpu_count) or config.allocated_gpu_count < 1:
        raise ValueError("allocated_gpu_count must be an integer >= 1")
    if not _is_finite_positive(config.max_gpu_hours):
        raise ValueError("max_gpu_hours must be finite and greater than 0")
    if not _is_integer(config.max_iterations) or config.max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1")
    if not _is_finite_positive(config.max_wall_clock_hours):
        raise ValueError("max_wall_clock_hours must be finite and greater than 0")
    if config.evaluation_time_included is not False:
        raise ValueError(
            "evaluation_time_included must be false under matched-compute v1"
        )
    if config.instrumentation_overhead_included is not True:
        raise ValueError(
            "instrumentation_overhead_included must be true under matched-compute v1"
        )
    if config.crossing_policy not in _CROSSING_POLICIES:
        raise ValueError(
            "crossing_policy must keep the completed crossing iteration"
        )
    if not _is_integer(config.schema_version) or config.schema_version != 1:
        raise ValueError("ResourceConfig schema_version must be 1")


def _fresh_state() -> ResourceAccountingState:
    return ResourceAccountingState(
        schema_version=1,
        completed_iteration=0,
        cumulative_training_seconds=0.0,
        cumulative_gpu_hours=0.0,
        cumulative_evaluation_seconds=0.0,
        cumulative_wall_clock_seconds=0.0,
        last_iteration_wall_clock_seconds=None,
        last_iteration_record=None,
        last_budget_decision=None,
        budget_crossing_iteration=None,
    )


def _copy_state(state: ResourceAccountingState) -> ResourceAccountingState:
    return ResourceAccountingState(
        schema_version=int(state.schema_version),
        completed_iteration=int(state.completed_iteration),
        cumulative_training_seconds=float(state.cumulative_training_seconds),
        cumulative_gpu_hours=float(state.cumulative_gpu_hours),
        cumulative_evaluation_seconds=float(state.cumulative_evaluation_seconds),
        cumulative_wall_clock_seconds=float(state.cumulative_wall_clock_seconds),
        last_iteration_wall_clock_seconds=(
            None
            if state.last_iteration_wall_clock_seconds is None
            else float(state.last_iteration_wall_clock_seconds)
        ),
        last_iteration_record=state.last_iteration_record,
        last_budget_decision=state.last_budget_decision,
        budget_crossing_iteration=state.budget_crossing_iteration,
    )


def _validate_record(record: IterationResourceRecord) -> None:
    if not isinstance(record, IterationResourceRecord):
        raise ValueError("last_iteration_record must be an IterationResourceRecord")
    if not _is_integer(record.iteration) or record.iteration < 1:
        raise ValueError("resource record iteration must be a positive integer")
    timing_fields = (
        "self_play_seconds",
        "training_seconds",
        "instrumentation_seconds",
        "checkpoint_seconds",
        "iteration_seconds",
        "evaluation_seconds",
        "cumulative_training_seconds",
        "cumulative_gpu_hours",
        "self_play_time_fraction",
    )
    for field_name in timing_fields:
        if not _is_finite_nonnegative(getattr(record, field_name)):
            raise ValueError(f"resource record {field_name} must be finite and >= 0")
    expected_iteration_seconds = (
        record.self_play_seconds
        + record.training_seconds
        + record.instrumentation_seconds
        + record.checkpoint_seconds
    )
    if not math.isclose(
        record.iteration_seconds,
        expected_iteration_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("resource record iteration_seconds is inconsistent")
    expected_fraction = (
        record.self_play_seconds / record.iteration_seconds
        if record.iteration_seconds > 0.0
        else 0.0
    )
    if not math.isclose(
        record.self_play_time_fraction,
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("resource record self_play_time_fraction is inconsistent")


def _validate_decision(decision: BudgetDecision) -> None:
    if not isinstance(decision, BudgetDecision):
        raise ValueError("last_budget_decision must be a BudgetDecision")
    if not isinstance(decision.continue_run, bool):
        raise ValueError("BudgetDecision.continue_run must be a bool")
    if not isinstance(decision.crossing_iteration, bool):
        raise ValueError("BudgetDecision.crossing_iteration must be a bool")
    if not _is_finite_nonnegative(decision.overshoot_gpu_hours):
        raise ValueError("overshoot_gpu_hours must be finite and >= 0")
    if decision.continue_run:
        if decision.stop_reason is not None:
            raise ValueError("a continuing BudgetDecision cannot have a stop_reason")
    elif decision.stop_reason not in _STOP_REASONS:
        raise ValueError("a stopping BudgetDecision must have a known stop_reason")
    if decision.crossing_iteration and decision.stop_reason != "gpu_hour_budget_reached":
        raise ValueError("crossing_iteration requires gpu_hour_budget_reached")


def _validate_state(config: ResourceConfig, state: ResourceAccountingState) -> None:
    if not isinstance(state, ResourceAccountingState):
        raise ValueError("state must be a ResourceAccountingState")
    if (
        not _is_integer(state.schema_version)
        or state.schema_version != 1
        or state.schema_version != config.schema_version
    ):
        raise ValueError("ResourceAccountingState schema_version must be 1")
    if not _is_integer(state.completed_iteration) or state.completed_iteration < 0:
        raise ValueError("completed_iteration must be a non-negative integer")
    cumulative_fields = (
        "cumulative_training_seconds",
        "cumulative_gpu_hours",
        "cumulative_evaluation_seconds",
        "cumulative_wall_clock_seconds",
    )
    for field_name in cumulative_fields:
        if not _is_finite_nonnegative(getattr(state, field_name)):
            raise ValueError(f"{field_name} must be finite and >= 0")
    expected_gpu_hours = (
        state.cumulative_training_seconds * config.allocated_gpu_count / 3600.0
    )
    if not math.isclose(
        state.cumulative_gpu_hours,
        expected_gpu_hours,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("cumulative_gpu_hours is inconsistent with training seconds")

    if state.completed_iteration == 0:
        if state.last_iteration_record is not None:
            raise ValueError("fresh resource state cannot contain a resource record")
        if state.last_budget_decision is not None:
            raise ValueError("fresh resource state cannot contain a budget decision")
        if state.last_iteration_wall_clock_seconds is not None:
            raise ValueError("fresh resource state cannot contain iteration wall time")
    else:
        if state.last_iteration_record is None or state.last_budget_decision is None:
            raise ValueError("completed resource state requires record and decision")
        if not _is_finite_nonnegative(state.last_iteration_wall_clock_seconds):
            raise ValueError("last_iteration_wall_clock_seconds must be finite and >= 0")
        _validate_record(state.last_iteration_record)
        _validate_decision(state.last_budget_decision)
        if state.last_iteration_record.iteration != state.completed_iteration:
            raise ValueError("last resource record does not match completed_iteration")
        if not math.isclose(
            state.last_iteration_record.cumulative_training_seconds,
            state.cumulative_training_seconds,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("last resource record has inconsistent training total")
        if not math.isclose(
            state.last_iteration_record.cumulative_gpu_hours,
            state.cumulative_gpu_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("last resource record has inconsistent GPU-hour total")

    crossing = state.budget_crossing_iteration
    if crossing is not None:
        if not _is_integer(crossing) or not 1 <= crossing <= state.completed_iteration:
            raise ValueError("budget_crossing_iteration is invalid")
        if state.cumulative_gpu_hours < config.max_gpu_hours:
            raise ValueError("budget crossing state is inconsistent with GPU hours")
        if crossing != state.completed_iteration:
            raise ValueError("no iteration may follow the budget crossing iteration")
    elif state.cumulative_gpu_hours >= config.max_gpu_hours:
        raise ValueError("GPU-hour budget reached without a crossing iteration")

    if state.completed_iteration > 0:
        assert state.last_budget_decision is not None
        assert state.last_iteration_wall_clock_seconds is not None
        decision = state.last_budget_decision
        if state.cumulative_gpu_hours >= config.max_gpu_hours:
            expected_reason = "gpu_hour_budget_reached"
        elif state.completed_iteration >= config.max_iterations:
            expected_reason = "max_iterations_reached"
        else:
            wall_limit_seconds = config.max_wall_clock_hours * 3600.0
            projected_next_boundary = (
                state.cumulative_wall_clock_seconds
                + state.last_iteration_wall_clock_seconds
            )
            expected_reason = (
                "wall_clock_safety_limit"
                if state.cumulative_wall_clock_seconds >= wall_limit_seconds
                or projected_next_boundary > wall_limit_seconds
                else None
            )
        if expected_reason is None:
            if not decision.continue_run:
                raise ValueError("budget decision stops before any limit is reached")
        elif decision.continue_run or decision.stop_reason != expected_reason:
            raise ValueError("budget decision is inconsistent with restored totals")

        expected_crossing = crossing == state.completed_iteration
        if decision.crossing_iteration != expected_crossing:
            raise ValueError("budget crossing decision is inconsistent with state")
        expected_overshoot = (
            max(0.0, state.cumulative_gpu_hours - config.max_gpu_hours)
            if expected_reason == "gpu_hour_budget_reached"
            else 0.0
        )
        if not math.isclose(
            decision.overshoot_gpu_hours,
            expected_overshoot,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("budget overshoot is inconsistent with restored totals")


def _record_from_dict(payload: object) -> IterationResourceRecord | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("last_iteration_record must be a dictionary or null")
    missing = sorted(_RECORD_FIELDS - set(payload))
    if missing:
        raise ValueError("resource record is missing: " + ", ".join(missing))
    return IterationResourceRecord(**{name: payload[name] for name in _RECORD_FIELDS})


def _decision_from_dict(payload: object) -> BudgetDecision | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("last_budget_decision must be a dictionary or null")
    missing = sorted(_DECISION_FIELDS - set(payload))
    if missing:
        raise ValueError("budget decision is missing: " + ", ".join(missing))
    return BudgetDecision(**{name: payload[name] for name in _DECISION_FIELDS})


class ResourceAccountant:
    """Account completed iterations under the frozen matched-compute contract."""

    def __init__(
        self,
        config: ResourceConfig,
        state: ResourceAccountingState | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        _validate_config(config)
        candidate = _fresh_state() if state is None else state
        _validate_state(config, candidate)
        self.config = config
        self.state = _copy_state(candidate)
        self._clock = time.perf_counter if clock is None else clock
        self._active_iteration: int | None = None
        self._active_started_at: float | None = None

    def begin_iteration(self, iteration: int) -> None:
        if self._active_iteration is not None:
            raise ValueError("an iteration timer is already active")
        self._require_run_can_continue()
        expected_iteration = self.state.completed_iteration + 1
        if not _is_integer(iteration) or iteration != expected_iteration:
            raise ValueError(
                f"iteration must be {expected_iteration}, got {iteration!r}"
            )
        started_at = self._clock()
        if not _is_finite_nonnegative(started_at):
            raise ResourceAccountingError("monotonic clock returned an invalid value")
        self._active_iteration = int(iteration)
        self._active_started_at = float(started_at)

    def abort_iteration(self) -> None:
        self._active_iteration = None
        self._active_started_at = None

    def record_iteration(
        self,
        iteration: int,
        *,
        self_play_seconds: float,
        training_seconds: float,
        instrumentation_seconds: float,
        checkpoint_seconds: float,
        evaluation_seconds: float,
        wall_clock_seconds: float | None = None,
    ) -> tuple[IterationResourceRecord, BudgetDecision]:
        self._require_run_can_continue()
        expected_iteration = self.state.completed_iteration + 1
        if not _is_integer(iteration) or iteration != expected_iteration:
            raise ValueError(
                f"iteration must be {expected_iteration}, got {iteration!r}"
            )
        if self._active_iteration is not None and iteration != self._active_iteration:
            raise ValueError(
                f"active iteration is {self._active_iteration}, got {iteration!r}"
            )

        if wall_clock_seconds is None:
            if self._active_started_at is None:
                raise ValueError(
                    "wall_clock_seconds is required without an active iteration timer"
                )
            finished_at = self._clock()
            if not _is_finite_nonnegative(finished_at):
                raise ResourceAccountingError(
                    "monotonic clock returned an invalid value"
                )
            wall_clock_seconds = float(finished_at) - self._active_started_at

        timings = {
            "self_play_seconds": self_play_seconds,
            "training_seconds": training_seconds,
            "instrumentation_seconds": instrumentation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "evaluation_seconds": evaluation_seconds,
            "wall_clock_seconds": wall_clock_seconds,
        }
        for field_name, value in timings.items():
            if not _is_finite_nonnegative(value):
                raise ValueError(f"{field_name} must be finite and >= 0")

        iteration_seconds = (
            float(self_play_seconds)
            + float(training_seconds)
            + float(instrumentation_seconds)
            + float(checkpoint_seconds)
        )
        cumulative_training_seconds = (
            self.state.cumulative_training_seconds + iteration_seconds
        )
        cumulative_gpu_hours = (
            cumulative_training_seconds * self.config.allocated_gpu_count / 3600.0
        )
        self_play_fraction = (
            float(self_play_seconds) / iteration_seconds
            if iteration_seconds > 0.0
            else 0.0
        )
        record = IterationResourceRecord(
            iteration=int(iteration),
            self_play_seconds=float(self_play_seconds),
            training_seconds=float(training_seconds),
            instrumentation_seconds=float(instrumentation_seconds),
            checkpoint_seconds=float(checkpoint_seconds),
            iteration_seconds=iteration_seconds,
            evaluation_seconds=float(evaluation_seconds),
            cumulative_training_seconds=cumulative_training_seconds,
            cumulative_gpu_hours=cumulative_gpu_hours,
            self_play_time_fraction=self_play_fraction,
        )
        _validate_record(record)

        previous_gpu_hours = self.state.cumulative_gpu_hours
        crossing_iteration = (
            previous_gpu_hours < self.config.max_gpu_hours
            and cumulative_gpu_hours >= self.config.max_gpu_hours
        )
        overshoot_gpu_hours = max(
            0.0, cumulative_gpu_hours - self.config.max_gpu_hours
        )
        cumulative_wall_clock_seconds = (
            self.state.cumulative_wall_clock_seconds + float(wall_clock_seconds)
        )
        decision = self._budget_decision(
            iteration=int(iteration),
            cumulative_gpu_hours=cumulative_gpu_hours,
            cumulative_wall_clock_seconds=cumulative_wall_clock_seconds,
            latest_wall_clock_seconds=float(wall_clock_seconds),
            crossing_iteration=crossing_iteration,
            overshoot_gpu_hours=overshoot_gpu_hours,
        )
        _validate_decision(decision)

        new_state = ResourceAccountingState(
            schema_version=self.state.schema_version,
            completed_iteration=int(iteration),
            cumulative_training_seconds=cumulative_training_seconds,
            cumulative_gpu_hours=cumulative_gpu_hours,
            cumulative_evaluation_seconds=(
                self.state.cumulative_evaluation_seconds
                + float(evaluation_seconds)
            ),
            cumulative_wall_clock_seconds=cumulative_wall_clock_seconds,
            last_iteration_wall_clock_seconds=float(wall_clock_seconds),
            last_iteration_record=record,
            last_budget_decision=decision,
            budget_crossing_iteration=(
                int(iteration)
                if crossing_iteration
                else self.state.budget_crossing_iteration
            ),
        )
        _validate_state(self.config, new_state)

        self.state = new_state
        self._active_iteration = None
        self._active_started_at = None
        return record, decision

    def state_dict(self) -> dict[str, Any]:
        if self._active_iteration is not None:
            raise ResourceAccountingError(
                "resource state can only be saved at an iteration boundary"
            )
        _validate_state(self.config, self.state)
        payload = asdict(self.state)
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("resource state must be JSON serializable") from exc
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._active_iteration is not None:
            raise ResourceAccountingError(
                "cannot load resource state while an iteration is active"
            )
        candidate = self._state_from_dict(state_dict)
        _validate_state(self.config, candidate)
        self.state = _copy_state(candidate)

    @classmethod
    def from_state_dict(
        cls,
        config: ResourceConfig,
        state_dict: dict[str, Any],
        *,
        clock: Callable[[], float] | None = None,
    ) -> "ResourceAccountant":
        accountant = cls(config, clock=clock)
        accountant.load_state_dict(state_dict)
        return accountant

    def _budget_decision(
        self,
        *,
        iteration: int,
        cumulative_gpu_hours: float,
        cumulative_wall_clock_seconds: float,
        latest_wall_clock_seconds: float,
        crossing_iteration: bool,
        overshoot_gpu_hours: float,
    ) -> BudgetDecision:
        if cumulative_gpu_hours >= self.config.max_gpu_hours:
            return BudgetDecision(
                continue_run=False,
                stop_reason="gpu_hour_budget_reached",
                crossing_iteration=crossing_iteration,
                overshoot_gpu_hours=overshoot_gpu_hours,
            )
        if iteration >= self.config.max_iterations:
            return BudgetDecision(
                continue_run=False,
                stop_reason="max_iterations_reached",
                crossing_iteration=False,
                overshoot_gpu_hours=0.0,
            )

        wall_limit_seconds = self.config.max_wall_clock_hours * 3600.0
        next_boundary_projection = (
            cumulative_wall_clock_seconds + latest_wall_clock_seconds
        )
        if (
            cumulative_wall_clock_seconds >= wall_limit_seconds
            or next_boundary_projection > wall_limit_seconds
        ):
            return BudgetDecision(
                continue_run=False,
                stop_reason="wall_clock_safety_limit",
                crossing_iteration=False,
                overshoot_gpu_hours=0.0,
            )
        return BudgetDecision(
            continue_run=True,
            stop_reason=None,
            crossing_iteration=False,
            overshoot_gpu_hours=0.0,
        )

    def _require_run_can_continue(self) -> None:
        decision = self.state.last_budget_decision
        if decision is not None and not decision.continue_run:
            raise ResourceAccountingError(
                f"run already stopped: {decision.stop_reason}"
            )

    def _state_from_dict(self, state_dict: dict[str, Any]) -> ResourceAccountingState:
        if not isinstance(state_dict, dict):
            raise ValueError("state_dict must be a dictionary")
        missing = sorted(_STATE_FIELDS - set(state_dict))
        if missing:
            raise ValueError(
                "resource state is missing required field(s): " + ", ".join(missing)
            )
        return ResourceAccountingState(
            schema_version=state_dict["schema_version"],
            completed_iteration=state_dict["completed_iteration"],
            cumulative_training_seconds=state_dict[
                "cumulative_training_seconds"
            ],
            cumulative_gpu_hours=state_dict["cumulative_gpu_hours"],
            cumulative_evaluation_seconds=state_dict[
                "cumulative_evaluation_seconds"
            ],
            cumulative_wall_clock_seconds=state_dict[
                "cumulative_wall_clock_seconds"
            ],
            last_iteration_wall_clock_seconds=state_dict[
                "last_iteration_wall_clock_seconds"
            ],
            last_iteration_record=_record_from_dict(
                state_dict["last_iteration_record"]
            ),
            last_budget_decision=_decision_from_dict(
                state_dict["last_budget_decision"]
            ),
            budget_crossing_iteration=state_dict["budget_crossing_iteration"],
        )


__all__ = [
    "BudgetDecision",
    "IterationResourceRecord",
    "ResourceAccountant",
    "ResourceAccountingError",
    "ResourceAccountingState",
    "ResourceConfig",
]
