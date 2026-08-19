#!/usr/bin/env python3
"""Read-only integrity and semantic verification for fixed_holdout_v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from holdout_common import (
    ARRAY_NAMES,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    HoldoutError,
    build_game,
    concatenate_shards,
    load_json,
    load_jsonl,
    load_npz,
    load_protocol,
    serializable_protocol,
    sha256_dataset_content,
    sha256_file,
    stable_game_seed,
    validate_source_checkpoint,
)


EXPECTED_DTYPES = {
    "boards": np.dtype(np.uint8),
    "policies": np.dtype(np.float32),
    "values": np.dtype(np.float32),
    "valids": np.dtype(np.uint8),
    "game_ids": np.dtype(np.int32),
    "steps": np.dtype(np.int16),
    "game_lengths": np.dtype(np.int16),
}


def validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    action_size: int,
    board_shape: tuple[int, int, int],
    game_id: int | None = None,
    game_length: int | None = None,
    terminal_result: int | None = None,
) -> int:
    if set(arrays) != set(ARRAY_NAMES):
        raise HoldoutError("NPZ array names do not match the frozen schema")
    for name, expected_dtype in EXPECTED_DTYPES.items():
        if arrays[name].dtype != expected_dtype:
            raise HoldoutError(
                f"{name} has dtype {arrays[name].dtype}, expected {expected_dtype}"
            )

    count = int(arrays["boards"].shape[0])
    expected_shapes = {
        "boards": (count, *board_shape),
        "policies": (count, action_size),
        "values": (count,),
        "valids": (count, action_size),
        "game_ids": (count,),
        "steps": (count,),
        "game_lengths": (count,),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise HoldoutError(
                f"{name} has shape {arrays[name].shape}, expected {expected_shape}"
            )
    if count == 0:
        raise HoldoutError("hold-out arrays contain no states")

    policies = arrays["policies"]
    values = arrays["values"]
    valids = arrays["valids"]
    if not np.all(np.isfinite(policies)) or not np.all(np.isfinite(values)):
        raise HoldoutError("policy or value targets contain non-finite values")
    if np.any(policies < 0):
        raise HoldoutError("policy targets contain negative probability")
    if not np.all((valids == 0) | (valids == 1)):
        raise HoldoutError("valid-move masks are not binary")
    if np.any(valids.sum(axis=1) == 0):
        raise HoldoutError("a state has no legal moves")
    if not np.allclose(policies.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise HoldoutError("policy target rows do not sum to one")
    if np.any(np.abs(policies[valids == 0]) > 1e-7):
        raise HoldoutError("policy target assigns mass to an invalid action")
    if not np.all(np.isin(values, (-1.0, 0.0, 1.0))):
        raise HoldoutError("value targets are outside {-1, 0, 1}")

    if game_id is not None:
        if not np.all(arrays["game_ids"] == game_id):
            raise HoldoutError(f"game {game_id} shard contains another game_id")
        if game_length != count:
            raise HoldoutError(f"game {game_id} length does not equal its state count")
        if not np.array_equal(
            arrays["steps"], np.arange(1, count + 1, dtype=np.int16)
        ):
            raise HoldoutError(f"game {game_id} steps are not exactly 1..game_length")
        if not np.all(arrays["game_lengths"] == count):
            raise HoldoutError(f"game {game_id} has inconsistent game_lengths")
        if terminal_result is None:
            raise HoldoutError("terminal_result is required for a game shard")
        players = np.where(arrays["steps"] % 2 == 1, 1.0, -1.0)
        expected_values = (players * terminal_result).astype(np.float32)
        if not np.array_equal(values, expected_values):
            raise HoldoutError(f"game {game_id} value labels have the wrong player sign")
    return count


def _load_resolved(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HoldoutError(f"resolved protocol not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HoldoutError(f"invalid resolved protocol YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HoldoutError("resolved protocol root must be a mapping")
    return loaded


def verify_holdout(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(config_path)
    if dataset_path is not None:
        dataset_path = dataset_path.expanduser().resolve()
        output_dir = dataset_path.parent
    else:
        output_dir = output_dir.expanduser().resolve()
        dataset_path = output_dir / "states.npz"
    checkpoint, checkpoint_sha256 = validate_source_checkpoint(protocol)
    resolved = _load_resolved(output_dir / "protocol.resolved.yaml")
    resolved_base = {key: value for key, value in resolved.items() if key != "runtime"}
    if resolved_base != serializable_protocol(protocol):
        raise HoldoutError("protocol.resolved.yaml differs from holdout_v1.yaml")

    game = build_game(protocol)
    action_size = int(game.getActionSize())
    board_shape = (4, *map(int, game.getBoardSize()))
    runtime = resolved.get("runtime", {})
    if runtime.get("action_size") != action_size:
        raise HoldoutError("resolved action size differs from game.getActionSize()")
    if runtime.get("board_shape") != list(board_shape):
        raise HoldoutError("resolved board shape differs from the game implementation")
    games_expected = runtime.get("games_requested")
    if (
        isinstance(games_expected, bool)
        or not isinstance(games_expected, int)
        or not 1 <= games_expected <= int(protocol["games"])
    ):
        raise HoldoutError("resolved runtime game count is invalid")

    manifest = load_json(output_dir / "manifest.json")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise HoldoutError("manifest protocol_id differs")
    if manifest.get("status") != "completed":
        raise HoldoutError("manifest status is not completed")
    if manifest.get("games_expected") != games_expected:
        raise HoldoutError("manifest expected game count differs")
    if manifest.get("games") != games_expected:
        raise HoldoutError("manifest frozen game count differs")
    if manifest.get("base_seed") != int(protocol["seed"]):
        raise HoldoutError("manifest base seed differs")
    if manifest.get("action_size") != action_size:
        raise HoldoutError("manifest action size differs")
    source = manifest.get("source_checkpoint", {})
    if source.get("sha256") != checkpoint_sha256:
        raise HoldoutError("manifest source checkpoint SHA-256 differs")
    if Path(source.get("path", "")).resolve() != checkpoint.resolve():
        raise HoldoutError("manifest source checkpoint path differs")
    if source.get("size_bytes") != checkpoint.stat().st_size:
        raise HoldoutError("manifest source checkpoint size differs")
    if manifest.get("source_checkpoint_iteration") != 0:
        raise HoldoutError("manifest source checkpoint iteration differs")
    if manifest.get("source_checkpoint_sha256") != checkpoint_sha256:
        raise HoldoutError("manifest flattened source checkpoint SHA-256 differs")
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or len(git_commit) != 40:
        raise HoldoutError("manifest git_commit is invalid")
    source_integrity = manifest.get("source_integrity", {})
    if source_integrity.get("enabled"):
        if (
            source_integrity.get("status") != "passed"
            or source_integrity.get("changed") is not False
            or source_integrity.get("before_sha256") != checkpoint_sha256
            or source_integrity.get("after_sha256") != checkpoint_sha256
        ):
            raise HoldoutError("source checkpoint integrity check did not pass")

    records = load_jsonl(output_dir / "games.jsonl")
    if len(records) != games_expected:
        raise HoldoutError(
            f"games.jsonl has {len(records)} rows, expected {games_expected}"
        )
    records_by_id: dict[int, dict[str, Any]] = {}
    total_states = 0
    states_per_game: dict[int, int] = {}
    game_seeds: list[int] = []
    max_policy_sum_error = 0.0
    max_invalid_policy = 0.0
    observed_values: set[float] = set()
    max_game_length = int(protocol["self_play"]["max_game_length"])
    for line_number, record in enumerate(records, start=1):
        game_id = record.get("game_id")
        if isinstance(game_id, bool) or not isinstance(game_id, int):
            raise HoldoutError(f"games.jsonl line {line_number} has invalid game_id")
        if game_id in records_by_id:
            raise HoldoutError(f"duplicate game_id in games.jsonl: {game_id}")
        if game_id < 0 or game_id >= games_expected:
            raise HoldoutError(f"game_id is outside the fixed range: {game_id}")
        if record.get("schema_version") != 1:
            raise HoldoutError(f"game {game_id} schema_version differs")
        if record.get("protocol_id") != protocol["protocol_id"]:
            raise HoldoutError(f"game {game_id} protocol_id differs")
        if record.get("game_seed") != stable_game_seed(int(protocol["seed"]), game_id):
            raise HoldoutError(f"game {game_id} seed differs")
        expected_shard = Path("shards") / f"game_{game_id:04d}.npz"
        if record.get("shard_path") != expected_shard.as_posix():
            raise HoldoutError(f"game {game_id} shard path differs")
        shard_path = output_dir / expected_shard
        if sha256_file(shard_path) != record.get("shard_sha256"):
            raise HoldoutError(f"game {game_id} shard SHA-256 differs")
        termination = record.get("termination")
        terminal_result = record.get("terminal_result")
        game_length = record.get("game_length")
        if termination == "win":
            if terminal_result not in {-1, 1}:
                raise HoldoutError(f"game {game_id} win has invalid terminal_result")
            if not isinstance(game_length, int) or not 1 <= game_length <= max_game_length:
                raise HoldoutError(f"game {game_id} has invalid length")
        elif termination == "max_turns":
            if terminal_result != 0 or game_length != max_game_length:
                raise HoldoutError(f"game {game_id} max-turn draw is inconsistent")
        else:
            raise HoldoutError(f"game {game_id} has unexpected termination")
        arrays = load_npz(shard_path)
        count = validate_arrays(
            arrays,
            action_size=action_size,
            board_shape=board_shape,
            game_id=game_id,
            game_length=game_length,
            terminal_result=terminal_result,
        )
        if record.get("positions") != count:
            raise HoldoutError(f"game {game_id} record position count differs")
        if record.get("action_size") != action_size:
            raise HoldoutError(f"game {game_id} record action size differs")
        if record.get("illegal_actions") != 0:
            raise HoldoutError(f"game {game_id} records an illegal action")
        duration = record.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration < 0:
            raise HoldoutError(f"game {game_id} duration is invalid")
        records_by_id[game_id] = record
        states_per_game[game_id] = count
        game_seeds.append(int(record["game_seed"]))
        max_policy_sum_error = max(
            max_policy_sum_error,
            float(np.max(np.abs(arrays["policies"].sum(axis=1) - 1.0))),
        )
        invalid_probabilities = np.abs(arrays["policies"][arrays["valids"] == 0])
        if invalid_probabilities.size:
            max_invalid_policy = max(
                max_invalid_policy, float(np.max(invalid_probabilities))
            )
        observed_values.update(float(item) for item in np.unique(arrays["values"]))
        total_states += count

    expected_ids = set(range(games_expected))
    if set(records_by_id) != expected_ids:
        raise HoldoutError(
            f"games.jsonl does not contain exactly game_id 0..{games_expected - 1}"
        )
    expected_shards = {f"game_{game_id:04d}.npz" for game_id in expected_ids}
    shards_directory = output_dir / "shards"
    observed_shards = {
        path.name for path in shards_directory.iterdir() if path.is_file()
    }
    if observed_shards != expected_shards:
        raise HoldoutError("shards directory does not match the resolved game range")
    if manifest.get("games_completed") != len(records_by_id):
        raise HoldoutError("manifest completed game count differs")
    if manifest.get("states_completed") != total_states:
        raise HoldoutError("manifest completed state count differs")

    states_path = dataset_path
    states_record = manifest.get("dataset", {})
    if states_record.get("path") != "states.npz":
        raise HoldoutError("manifest states path differs")
    states_file_sha256 = sha256_file(states_path)
    if states_file_sha256 != states_record.get("sha256"):
        raise HoldoutError("states.npz SHA-256 differs from the manifest")
    if manifest.get("dataset_file_sha256") != states_file_sha256:
        raise HoldoutError("manifest dataset file SHA-256 differs")
    if states_record.get("size_bytes") != states_path.stat().st_size:
        raise HoldoutError("states.npz size differs from the manifest")
    combined = load_npz(states_path)
    validate_arrays(
        combined,
        action_size=action_size,
        board_shape=board_shape,
    )
    if int(combined["boards"].shape[0]) != total_states:
        raise HoldoutError("states.npz sample count differs from the shards")
    if manifest.get("states") != total_states:
        raise HoldoutError("manifest frozen state count differs")
    if states_record.get("samples") != total_states:
        raise HoldoutError("manifest states sample count differs")
    expected_combined = concatenate_shards(records_by_id.values(), output_dir)
    for name in ARRAY_NAMES:
        if not np.array_equal(combined[name], expected_combined[name]):
            raise HoldoutError(f"states.npz {name} differs from ordered shard concatenation")
    content_sha256 = sha256_dataset_content(combined)
    if (
        manifest.get("dataset_content_sha256") != content_sha256
        or states_record.get("content_sha256") != content_sha256
    ):
        raise HoldoutError("dataset logical content SHA-256 differs from the manifest")
    if manifest.get("illegal_actions") != 0:
        raise HoldoutError("manifest records illegal actions")

    return {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "games": len(records_by_id),
        "states": total_states,
        "shards": len(records_by_id),
        "action_size": action_size,
        "board_shape": list(board_shape),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_unchanged": (
            not source_integrity.get("enabled")
            or source_integrity.get("changed") is False
        ),
        "states_sha256": states_record["sha256"],
        "dataset_content_sha256": content_sha256,
        "combined_matches_shards": True,
        "game_seeds": sorted(game_seeds),
        "states_per_game": [states_per_game[game_id] for game_id in sorted(states_per_game)],
        "max_policy_sum_abs_error": max_policy_sum_error,
        "max_invalid_action_probability": max_invalid_policy,
        "observed_value_labels": sorted(observed_values),
        "steps_contiguous": True,
        "unique_seed_count": len(set(game_seeds)),
        "duplicate_game_ids": 0,
        "empty_games": 0,
        "illegal_actions": 0,
        "invalid_values": 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    location.add_argument("--dataset", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_holdout(args.config, args.output_dir, args.dataset)
    except (HoldoutError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Protocol: {result['protocol_id']}")
    print(f"Games: {result['games']}")
    print(f"States: {result['states']}")
    print(f"Shards: {result['shards']}")
    print(f"Action size: {result['action_size']}")
    print(f"Board shape: {result['board_shape']}")
    print(f"Game seeds: {result['game_seeds']}")
    print(f"Unique seeds: {result['unique_seed_count']}")
    print(f"Duplicate game IDs: {result['duplicate_game_ids']}")
    print(f"Empty games: {result['empty_games']}")
    print(f"Illegal actions: {result['illegal_actions']}")
    print(f"States per game: {result['states_per_game']}")
    print(
        "Max policy-sum absolute error: "
        f"{result['max_policy_sum_abs_error']:.9g}"
    )
    print(
        "Max invalid-action probability: "
        f"{result['max_invalid_action_probability']:.9g}"
    )
    print(f"Observed value labels: {result['observed_value_labels']}")
    print(f"Invalid values: {result['invalid_values']}")
    print("Steps contiguous: yes")
    print(
        "Source checkpoint unchanged: "
        f"{'yes' if result['source_checkpoint_unchanged'] else 'no'}"
    )
    print("Combined matches shards: yes")
    print(f"Dataset content SHA-256: {result['dataset_content_sha256']}")
    print("Output status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
