"""Pure deterministic seed, key, and bootstrap utilities for head-to-head v2."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


class HeadToHeadStatsError(ValueError):
    """Raised when head-to-head statistical inputs violate the protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeadToHeadStatsError(message)


def paired_seed(
    config_id: str,
    base_seed: int,
    baseline_checkpoint_sha256: str,
    adaptive_checkpoint_sha256: str,
    seed_pair_index: int,
) -> int:
    """Return the frozen unsigned 64-bit seed shared by a colour-swapped pair."""
    payload = "\x1f".join(
        (
            config_id,
            str(base_seed),
            baseline_checkpoint_sha256,
            adaptive_checkpoint_sha256,
            str(seed_pair_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_game_key(
    baseline_checkpoint_sha256: str,
    adaptive_checkpoint_sha256: str,
    seed_pair_index: int,
    adaptive_color: str,
) -> str:
    payload = {
        "adaptive_checkpoint_sha256": adaptive_checkpoint_sha256,
        "adaptive_color": adaptive_color,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "seed_pair_index": seed_pair_index,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adaptive_score(result: str) -> float:
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    if result == "loss":
        return 0.0
    raise HeadToHeadStatsError(f"invalid Adaptive result: {result!r}")


def colour_stratified_bootstrap(
    records: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    expected_per_colour: int = 50,
) -> dict[str, float | int]:
    """Bootstrap Adaptive score while preserving its white/black sample sizes."""
    _require(resamples > 0, "bootstrap resamples must be positive")
    strata: dict[str, list[dict[str, Any]]] = {
        "white": [],
        "black": [],
    }
    for record in records:
        color = record.get("adaptive_color")
        _require(color in strata, f"invalid Adaptive color: {color!r}")
        strata[color].append(record)
    for color, values in strata.items():
        _require(
            len(values) == expected_per_colour,
            f"Adaptive {color} stratum must contain {expected_per_colour} games, found {len(values)}",
        )
        values.sort(key=lambda item: str(item.get("stable_game_key", "")))

    arrays = {
        color: np.asarray(
            [adaptive_score(str(item["adaptive_result"])) for item in values],
            dtype=np.float64,
        )
        for color, values in strata.items()
    }
    total = sum(values.size for values in arrays.values())
    scores = np.concatenate(list(arrays.values()))
    _require(total > 0 and np.isfinite(scores).all(), "bootstrap scores are invalid")
    rng = np.random.default_rng(seed)
    bootstrap_scores = np.zeros(resamples, dtype=np.float64)
    for values in arrays.values():
        indices = rng.integers(0, values.size, size=(resamples, values.size))
        bootstrap_scores += values[indices].sum(axis=1) / total
    low, high = np.quantile(bootstrap_scores, [0.025, 0.975])
    point = float(scores.mean())
    _require(
        all(math.isfinite(value) for value in (point, float(low), float(high))),
        "bootstrap produced non-finite output",
    )
    return {
        "score_rate": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": resamples,
        "seed": seed,
        "adaptive_white_games": arrays["white"].size,
        "adaptive_black_games": arrays["black"].size,
    }

