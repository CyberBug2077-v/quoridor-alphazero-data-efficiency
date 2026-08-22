"""Paired-bootstrap and trajectory-diversity utilities for head-to-head v3."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from head_to_head_stats import adaptive_score


class HeadToHeadStatsV3Error(ValueError):
    """Raised when v3 statistical inputs violate the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadToHeadStatsV3Error(message)


def trajectory_sha256(record: dict[str, Any]) -> str:
    """Return the identity of the complete ordered move sequence."""
    moves = record.get("moves")
    _require(isinstance(moves, list), "game record lacks a move sequence")
    encoded = json.dumps(
        moves,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trajectory_diversity(
    records: list[dict[str, Any]],
    *,
    minimum_unique_per_colour: int,
    expected_per_colour: int = 50,
) -> dict[str, Any]:
    _require(minimum_unique_per_colour >= 2, "trajectory minimum must reject a single trajectory")
    by_colour: dict[str, list[str]] = defaultdict(list)
    for record in records:
        colour = record.get("adaptive_color")
        _require(colour in {"white", "black"}, f"invalid Adaptive colour: {colour!r}")
        digest = trajectory_sha256(record)
        stored = record.get("trajectory_sha256")
        if stored is not None:
            _require(stored == digest, "stored trajectory SHA does not match the move sequence")
        by_colour[str(colour)].append(digest)
    segments: dict[str, Any] = {}
    for colour in ("white", "black"):
        digests = by_colour[colour]
        _require(
            len(digests) == expected_per_colour,
            f"Adaptive {colour} must contain {expected_per_colour} games, found {len(digests)}",
        )
        counts = Counter(digests)
        unique = len(counts)
        passed = unique >= minimum_unique_per_colour
        segments[colour] = {
            "games": len(digests),
            "unique_trajectories": unique,
            "duplicate_games": len(digests) - unique,
            "unique_rate": unique / len(digests),
            "most_common_trajectory_games": max(counts.values()),
            "minimum_unique_required": minimum_unique_per_colour,
            "passed": passed,
        }
    return {
        "identity": "sha256_of_canonical_complete_move_sequence",
        "segments": segments,
        "overall_unique_trajectories": len(
            set(by_colour["white"]) | set(by_colour["black"])
        ),
        "acceptance_passed": all(
            bool(segments[colour]["passed"]) for colour in ("white", "black")
        ),
    }


def seed_pair_bootstrap(
    records: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    expected_pairs: int = 50,
) -> dict[str, float | int | str]:
    """Resample complete colour-swapped seed pairs as the independent units."""
    _require(resamples > 0, "bootstrap resamples must be positive")
    pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_index = record.get("seed_pair_index")
        _require(
            isinstance(pair_index, int) and not isinstance(pair_index, bool),
            "seed_pair_index must be an integer",
        )
        pairs[pair_index].append(record)
    _require(
        set(pairs) == set(range(expected_pairs)),
        f"bootstrap requires seed pairs 0..{expected_pairs - 1}",
    )
    pair_scores: list[float] = []
    for pair_index in range(expected_pairs):
        pair = pairs[pair_index]
        _require(len(pair) == 2, f"seed pair {pair_index} must contain two games")
        _require(
            {item.get("adaptive_color") for item in pair} == {"white", "black"},
            f"seed pair {pair_index} does not swap Adaptive colour",
        )
        _require(
            len({item.get("game_seed") for item in pair}) == 1,
            f"seed pair {pair_index} does not share one game seed",
        )
        pair_scores.append(
            sum(adaptive_score(str(item.get("adaptive_result"))) for item in pair)
            / 2.0
        )
    values = np.asarray(pair_scores, dtype=np.float64)
    _require(np.isfinite(values).all(), "seed-pair scores are non-finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, expected_pairs, size=(resamples, expected_pairs))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    point = float(values.mean())
    _require(
        all(math.isfinite(value) for value in (point, float(low), float(high))),
        "paired bootstrap produced non-finite output",
    )
    return {
        "score_rate": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": resamples,
        "seed": seed,
        "seed_pairs": expected_pairs,
        "games": expected_pairs * 2,
        "resampling_unit": "seed_pair",
    }
