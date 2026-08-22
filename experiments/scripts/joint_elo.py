"""Pure NumPy joint Bradley-Terry/Elo fitting and stratified bootstrap."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


class JointEloError(ValueError):
    """Raised when joint Elo inputs or convergence violate the protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointEloError(message)


def _participant_transform(
    participants: list[str], anchors: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Map free parameters to ratings with mean(anchor Elo) fixed at zero."""
    _require(len(anchors) >= 2, "joint Elo requires at least two shared anchors")
    _require(set(anchors) <= set(participants), "joint Elo anchors are not participants")
    dependent = anchors[-1]
    independent = [participant for participant in participants if participant != dependent]
    columns = {participant: index for index, participant in enumerate(independent)}
    transform = np.zeros((len(participants), len(independent)), dtype=np.float64)
    rows = {participant: index for index, participant in enumerate(participants)}
    for participant in independent:
        transform[rows[participant], columns[participant]] = 1.0
    for anchor in anchors[:-1]:
        transform[rows[dependent], columns[anchor]] = -1.0
    return transform, independent


def aggregate_joint_records(
    records: list[dict[str, Any]],
    *,
    by_bootstrap_stratum: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if by_bootstrap_stratum:
            key = (
                record["condition"],
                int(record["target"]),
                record["opponent"],
                record["model_color"],
            )
        else:
            key = (record["bot1"], record["bot2"])
        group = grouped.setdefault(
            key,
            {
                "key": key,
                "bot1": record["bot1"],
                "bot2": record["bot2"],
                "scores": [],
            },
        )
        _require(
            group["bot1"] == record["bot1"] and group["bot2"] == record["bot2"],
            f"joint Elo stratum {key} mixes participants",
        )
        group["scores"].append(float(record["score_bot1"]))
    return [grouped[key] for key in sorted(grouped, key=lambda item: tuple(map(str, item)))]


def _joint_design(
    participants: list[str],
    anchors: list[str],
    aggregates: list[dict[str, Any]],
    *,
    elo_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform, _ = _participant_transform(participants, anchors)
    participant_index = {name: index for index, name in enumerate(participants)}
    contrast = np.zeros((len(aggregates), len(participants)), dtype=np.float64)
    games = np.zeros(len(aggregates), dtype=np.float64)
    score_sums = np.zeros(len(aggregates), dtype=np.float64)
    for row_index, aggregate in enumerate(aggregates):
        bot1 = aggregate["bot1"]
        bot2 = aggregate["bot2"]
        _require(
            bot1 in participant_index and bot2 in participant_index,
            f"unknown joint Elo participant in {bot1} vs {bot2}",
        )
        scores = np.asarray(aggregate["scores"], dtype=np.float64)
        _require(scores.size > 0, f"empty joint Elo aggregate {bot1} vs {bot2}")
        _require(
            np.isfinite(scores).all()
            and ((0.0 <= scores) & (scores <= 1.0)).all(),
            f"invalid joint Elo scores for {bot1} vs {bot2}",
        )
        contrast[row_index, participant_index[bot1]] = 1.0
        contrast[row_index, participant_index[bot2]] = -1.0
        games[row_index] = float(scores.size)
        score_sums[row_index] = float(scores.sum())
    design = (math.log(10.0) / elo_scale) * (contrast @ transform)
    return design, games, score_sums, transform


def fit_joint_elo(
    participants: list[str],
    anchors: list[str],
    aggregates: list[dict[str, Any]],
    fit_config: dict[str, Any],
    *,
    score_sums_override: np.ndarray | None = None,
    initial_theta: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one regularized Bradley-Terry likelihood on the shared Elo scale."""
    elo_scale = float(fit_config["elo_scale"])
    ridge = float(fit_config["ridge_precision"])
    tolerance = float(fit_config["tolerance"])
    max_iterations = int(fit_config["max_iterations"])
    _require(elo_scale > 0.0, "joint Elo scale must be positive")
    _require(ridge >= 0.0, "joint Elo ridge precision must be non-negative")
    _require(
        tolerance > 0.0 and max_iterations > 0,
        "joint Elo convergence settings are invalid",
    )
    design, games, observed_score_sums, transform = _joint_design(
        participants, anchors, aggregates, elo_scale=elo_scale
    )
    score_sums = (
        observed_score_sums
        if score_sums_override is None
        else np.asarray(score_sums_override, dtype=np.float64)
    )
    _require(score_sums.shape == games.shape, "joint Elo score-sum vector has wrong shape")
    _require(np.isfinite(score_sums).all(), "joint Elo score sums are non-finite")
    _require(
        ((0.0 <= score_sums) & (score_sums <= games)).all(),
        "joint Elo score sums are outside [0, games]",
    )
    theta = (
        np.zeros(transform.shape[1], dtype=np.float64)
        if initial_theta is None
        else np.asarray(initial_theta, dtype=np.float64).copy()
    )
    _require(
        theta.shape == (transform.shape[1],),
        "joint Elo initial parameter vector has wrong shape",
    )
    penalty = transform.T @ transform

    def objective(candidate: np.ndarray) -> float:
        eta = design @ candidate
        log_likelihood = float(
            np.sum(score_sums * eta - games * np.logaddexp(0.0, eta))
        )
        ratings = transform @ candidate
        return log_likelihood - 0.5 * ridge * float(ratings @ ratings)

    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        eta = np.clip(design @ theta, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        gradient = (
            design.T @ (score_sums - games * probabilities)
            - ridge * (penalty @ theta)
        )
        weights = games * probabilities * (1.0 - probabilities)
        information = design.T @ (weights[:, None] * design) + ridge * penalty
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise JointEloError("joint Elo information matrix is singular") from exc
        current_objective = objective(theta)
        step_scale = 1.0
        while step_scale >= 2.0**-20:
            candidate = theta + step_scale * step
            if objective(candidate) >= current_objective - 1.0e-12:
                theta = candidate
                break
            step_scale *= 0.5
        else:
            raise JointEloError(
                "joint Elo Newton step failed to improve the likelihood"
            )
        if float(np.max(np.abs(step_scale * step))) <= tolerance:
            converged = True
            break
    _require(
        converged,
        f"joint Elo did not converge within {max_iterations} iterations",
    )
    ratings_vector = transform @ theta
    ratings = {
        participant: float(ratings_vector[index])
        for index, participant in enumerate(participants)
    }
    _require(
        math.isclose(
            sum(ratings[anchor] for anchor in anchors), 0.0, abs_tol=1.0e-8
        ),
        "joint Elo anchor location constraint was not satisfied",
    )
    return {
        "ratings": ratings,
        "ratings_vector": ratings_vector,
        "theta": theta,
        "iterations": iterations,
        "converged": converged,
        "penalized_log_likelihood": objective(theta),
    }


def bootstrap_joint_elo(
    participants: list[str],
    anchors: list[str],
    strata: list[dict[str, Any]],
    fit_config: dict[str, Any],
    *,
    resamples: int,
    seed: int,
    initial_theta: np.ndarray,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Refit the same joint model after sampling within every frozen stratum."""
    _require(resamples > 0, "joint Elo bootstrap resamples must be positive")
    for stratum in strata:
        _require(
            len(stratum["scores"]) == 25,
            f"joint bootstrap stratum {stratum['key']} must contain exactly 25 games",
        )
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples, len(participants)), dtype=np.float64)
    score_arrays = [
        np.asarray(stratum["scores"], dtype=np.float64) for stratum in strata
    ]
    for sample_index in range(resamples):
        score_sums = np.asarray(
            [
                values[rng.integers(0, values.size, size=values.size)].sum()
                for values in score_arrays
            ],
            dtype=np.float64,
        )
        fitted = fit_joint_elo(
            participants,
            anchors,
            strata,
            fit_config,
            score_sums_override=score_sums,
            initial_theta=initial_theta,
        )
        samples[sample_index, :] = fitted["ratings_vector"]
        if progress is not None and (
            sample_index == 0
            or (sample_index + 1) % 250 == 0
            or sample_index + 1 == resamples
        ):
            progress(f"Joint Elo bootstrap: {sample_index + 1}/{resamples}")
    _require(
        np.isfinite(samples).all(),
        "joint Elo bootstrap produced non-finite ratings",
    )
    return samples

