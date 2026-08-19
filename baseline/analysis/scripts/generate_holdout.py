#!/usr/bin/env python3
"""Generate the frozen fixed_holdout_v1 dataset from checkpoint 0 self-play."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from holdout_common import (
    ARRAY_NAMES,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    HoldoutError,
    append_jsonl_record,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_yaml,
    build_mcts,
    build_network,
    concatenate_shards,
    git_revision,
    load_json,
    load_jsonl,
    load_npz,
    load_protocol,
    serializable_protocol,
    set_all_seeds,
    sha256_file,
    sha256_dataset_content,
    stable_game_seed,
    validate_source_checkpoint,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class GenerationLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        self._stream.write(line + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _normalized_policy(pi: Any, valids: np.ndarray, action_size: int) -> np.ndarray:
    policy = np.asarray(pi, dtype=np.float64)
    if policy.shape != (action_size,):
        raise HoldoutError(
            f"MCTS policy shape is {policy.shape}, expected {(action_size,)}"
        )
    if not np.all(np.isfinite(policy)) or np.any(policy < 0):
        raise HoldoutError("MCTS returned a non-finite or negative policy")
    if float(policy[valids == 0].sum()) > 1e-7:
        raise HoldoutError("MCTS assigned probability to an illegal action")
    mass = float(policy.sum())
    if mass <= 0:
        valid_count = int(valids.sum())
        if valid_count == 0:
            raise HoldoutError("no legal moves are available")
        policy = valids.astype(np.float64) / valid_count
    else:
        policy /= mass
    return policy.astype(np.float32)


def sample_action(policy: np.ndarray) -> int:
    """Sample from a stored float32 target after restoring an exact double sum."""
    sampling_policy = policy.astype(np.float64)
    sampling_policy /= float(sampling_policy.sum())
    return int(np.random.choice(len(sampling_policy), p=sampling_policy))


def finalize_episode(
    history: list[tuple[np.ndarray, int, np.ndarray, np.ndarray, int]],
    *,
    game_id: int,
    terminal_result: int,
    game_length: int,
) -> dict[str, np.ndarray]:
    """Convert Coach-compatible episode history into the frozen shard arrays."""
    if not history:
        raise HoldoutError("episode ended without any recorded states")
    values = np.asarray(
        [terminal_result * player for _, player, _, _, _ in history],
        dtype=np.float32,
    )
    count = len(history)
    return {
        "boards": np.stack([item[0] for item in history]).astype(np.uint8),
        "policies": np.stack([item[2] for item in history]).astype(np.float32),
        "values": values,
        "valids": np.stack([item[3] for item in history]).astype(np.uint8),
        "game_ids": np.full(count, game_id, dtype=np.int32),
        "steps": np.asarray([item[4] for item in history], dtype=np.int16),
        "game_lengths": np.full(count, game_length, dtype=np.int16),
    }


def run_episode(game, network, protocol: dict[str, Any], game_id: int):
    """Run one independent self-play game using the baseline episode semantics."""
    action_size = int(game.getActionSize())
    self_play = protocol["self_play"]
    mcts = build_mcts(game, network, protocol)
    board = game.getInitBoard()
    current_player = 1
    history: list[tuple[np.ndarray, int, np.ndarray, np.ndarray, int]] = []
    terminal_result = 0
    termination = "max_turns"

    for step in range(1, int(self_play["max_game_length"]) + 1):
        canonical = np.ascontiguousarray(
            game.getCanonicalForm(board, current_player), dtype=np.uint8
        )
        valids = np.asarray(game.getValidMoves(canonical, 1), dtype=np.uint8)
        if valids.shape != (action_size,):
            raise HoldoutError(
                f"valid-move shape is {valids.shape}, expected {(action_size,)}"
            )
        if not np.all((valids == 0) | (valids == 1)):
            raise HoldoutError("valid-move mask is not binary")

        temperature = int(step < int(self_play["temperature_threshold"]))
        raw_policy = mcts.getActionProb(
            canonical,
            temp=temperature,
            add_dirichlet_noise=bool(self_play["dirichlet_noise"]),
        )
        policy = _normalized_policy(raw_policy, valids, action_size)
        action = sample_action(policy)
        if not valids[action]:
            raise HoldoutError(f"self-play selected illegal action {action} at step {step}")

        history.append(
            (canonical.copy(), current_player, policy.copy(), valids.copy(), step)
        )
        board, current_player = game.getNextState(board, current_player, action)
        result = int(game.getGameEnded(board, current_player))
        if result != 0:
            if result not in {-1, 1}:
                raise HoldoutError(f"unexpected terminal result: {result}")
            terminal_result = result
            termination = "win"
            break

    game_length = len(history)
    arrays = finalize_episode(
        history,
        game_id=game_id,
        terminal_result=terminal_result,
        game_length=game_length,
    )
    return arrays, terminal_result, termination


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "protocol": output_dir / "protocol.resolved.yaml",
        "manifest": output_dir / "manifest.json",
        "games": output_dir / "games.jsonl",
        "shards": output_dir / "shards",
        "states": output_dir / "states.npz",
        "log": output_dir / "generation.log",
    }


def _relative_shard(game_id: int) -> Path:
    return Path("shards") / f"game_{game_id:04d}.npz"


def resolve_game_count(protocol: dict[str, Any], requested: int | None) -> int:
    configured = int(protocol["games"])
    if requested is None:
        return configured
    if isinstance(requested, bool) or requested < 1 or requested > configured:
        raise HoldoutError(f"--games must be between 1 and {configured}")
    return requested


def _base_protocol_from_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    return {key: resolved[key] for key in serializable_protocol(resolved) if key != "runtime"}


def _validate_resume_records(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    output_dir: Path,
    expected_games: int | None = None,
) -> dict[int, dict[str, Any]]:
    game_count = int(protocol["games"]) if expected_games is None else expected_games
    completed: dict[int, dict[str, Any]] = {}
    for line_number, record in enumerate(records, start=1):
        game_id = record.get("game_id")
        if isinstance(game_id, bool) or not isinstance(game_id, int):
            raise HoldoutError(f"games.jsonl line {line_number} has invalid game_id")
        if game_id < 0 or game_id >= game_count:
            raise HoldoutError(f"games.jsonl line {line_number} game_id is out of range")
        if game_id in completed:
            raise HoldoutError(f"duplicate completed game_id: {game_id}")
        expected_seed = stable_game_seed(int(protocol["seed"]), game_id)
        if record.get("game_seed") != expected_seed:
            raise HoldoutError(f"game {game_id} has the wrong deterministic seed")
        expected_relative = _relative_shard(game_id).as_posix()
        if record.get("shard_path") != expected_relative:
            raise HoldoutError(f"game {game_id} has an unexpected shard path")
        shard = output_dir / expected_relative
        digest = sha256_file(shard) if shard.is_file() else None
        if digest != record.get("shard_sha256"):
            raise HoldoutError(f"game {game_id} shard is missing or its SHA-256 differs")
        arrays = load_npz(shard)
        positions = int(arrays["boards"].shape[0])
        if record.get("positions") != positions:
            raise HoldoutError(f"game {game_id} position count differs from its shard")
        if record.get("illegal_actions") != 0:
            raise HoldoutError(f"game {game_id} records an illegal action")
        completed[game_id] = record
    return completed


def _initial_manifest(
    protocol: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    action_size: int,
    games_expected: int,
    verify_source_integrity: bool,
    git_commit: str,
    git_worktree_dirty: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "generating",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "source_checkpoint": {
            "iteration": 0,
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint.stat().st_size,
        },
        "games_expected": games_expected,
        "games": games_expected,
        "games_completed": 0,
        "states_completed": 0,
        "states": 0,
        "base_seed": int(protocol["seed"]),
        "source_checkpoint_iteration": 0,
        "source_checkpoint_sha256": checkpoint_sha256,
        "dataset_file_sha256": None,
        "dataset_content_sha256": None,
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "illegal_actions": 0,
        "action_size": action_size,
        "board_shape": [4, 17, 17],
        "game_seed_formula": "seed + zero_based_game_id",
        "canonical_view": "current player to move; no non-acting-player duplicate",
        "arrays": list(ARRAY_NAMES),
        "resume_count": 0,
        "dataset": None,
        "source_integrity": {
            "enabled": verify_source_integrity,
            "before_sha256": checkpoint_sha256,
            "after_sha256": None,
            "status": "pending" if verify_source_integrity else "not_requested",
            "changed": None if verify_source_integrity else False,
        },
    }


def generate(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    paths = _artifact_paths(output_dir)
    checkpoint, checkpoint_sha256 = validate_source_checkpoint(protocol)
    games_expected = resolve_game_count(protocol, args.games)

    controlled = [paths[name] for name in ("protocol", "manifest", "games", "states")]
    if not args.resume and any(path.exists() for path in controlled):
        raise HoldoutError(
            f"hold-out output already exists at {output_dir}; use --resume to continue"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["shards"].mkdir(parents=True, exist_ok=True)
    logger = GenerationLogger(paths["log"])

    try:
        game, network = build_network(protocol, checkpoint, device=args.device)
        action_size = int(game.getActionSize())
        board_shape = [4, *map(int, game.getBoardSize())]
        resume_metadata = paths["protocol"].is_file() or paths["manifest"].is_file()
        if args.resume and resume_metadata and not (
            paths["protocol"].is_file() and paths["manifest"].is_file()
        ):
            raise HoldoutError(
                "--resume found only one of protocol.resolved.yaml and manifest.json"
            )
        if args.resume and not resume_metadata and any(
            path.exists() for path in (paths["games"], paths["states"])
        ):
            raise HoldoutError("--resume found data artifacts without resume metadata")
        continuing = args.resume and resume_metadata

        if continuing:
            resolved = load_protocol(args.config)
            recorded_resolved = yaml.safe_load(
                paths["protocol"].read_text(encoding="utf-8")
            )
            if not isinstance(recorded_resolved, dict):
                raise HoldoutError("protocol.resolved.yaml is not a mapping")
            if _base_protocol_from_resolved(recorded_resolved) != serializable_protocol(resolved):
                raise HoldoutError("resolved protocol differs from the requested protocol")
            recorded_runtime = recorded_resolved.get("runtime", {})
            if recorded_runtime.get("games_requested") != games_expected:
                raise HoldoutError("resume game count differs from the resolved protocol")
            manifest = load_json(paths["manifest"])
            source = manifest.get("source_checkpoint", {})
            if source.get("sha256") != checkpoint_sha256:
                raise HoldoutError("resume source checkpoint SHA-256 differs")
            if manifest.get("action_size") != action_size:
                raise HoldoutError("resume action size differs from the game implementation")
            if manifest.get("games_expected") != games_expected:
                raise HoldoutError("resume game count differs from the manifest")
            integrity = manifest.get("source_integrity", {})
            if bool(integrity.get("enabled")) != bool(args.verify_source_integrity):
                raise HoldoutError(
                    "resume --verify-source-integrity setting differs from the manifest"
                )
            records = load_jsonl(paths["games"]) if paths["games"].is_file() else []
            completed = _validate_resume_records(
                records, protocol, output_dir, games_expected
            )
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            if (
                len(completed) == games_expected
                and manifest.get("status") == "completed"
                and paths["states"].is_file()
                and manifest.get("dataset", {}).get("sha256")
                == sha256_file(paths["states"])
            ):
                if args.verify_source_integrity:
                    after_sha256 = sha256_file(checkpoint)
                    if after_sha256 != checkpoint_sha256:
                        raise HoldoutError("source checkpoint changed after generation")
                logger.log("Hold-out already completed; all recorded artifacts match.")
                return
        else:
            git_commit, git_worktree_dirty = git_revision()
            resolved_payload = serializable_protocol(protocol)
            resolved_payload["runtime"] = {
                "resolved_source_checkpoint": checkpoint.as_posix(),
                "source_checkpoint_sha256": checkpoint_sha256,
                "action_size": action_size,
                "board_shape": board_shape,
                "game_seed_formula": "seed + zero_based_game_id",
                "canonical_view": (
                    "one canonical state for the player to move; the non-acting-player "
                    "view is not duplicated"
                ),
                "output_directory": output_dir.as_posix(),
                "games_requested": games_expected,
                "execution_scope": (
                    "formal"
                    if games_expected == int(protocol["games"])
                    else "pilot"
                ),
            }
            atomic_write_yaml(paths["protocol"], resolved_payload)
            manifest = _initial_manifest(
                protocol,
                checkpoint,
                checkpoint_sha256,
                action_size,
                games_expected,
                args.verify_source_integrity,
                git_commit,
                git_worktree_dirty,
            )
            manifest["board_shape"] = board_shape
            atomic_write_json(paths["manifest"], manifest)
            paths["games"].touch(exist_ok=False)
            completed = {}

        manifest["status"] = "generating"
        manifest.pop("error", None)
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(paths["manifest"], manifest)
        logger.log(
            f"Starting fixed_holdout_v1: {len(completed)}/{games_expected} games present; "
            f"action_size={action_size}; device={args.device}."
        )

        with paths["games"].open("a", encoding="utf-8", newline="\n") as games_stream:
            for game_id in range(games_expected):
                if game_id in completed:
                    continue
                game_seed = stable_game_seed(int(protocol["seed"]), game_id)
                set_all_seeds(game_seed)
                started = time.perf_counter()
                arrays, terminal_result, termination = run_episode(
                    game, network, protocol, game_id
                )
                relative_shard = _relative_shard(game_id)
                shard_path = output_dir / relative_shard
                atomic_write_npz(shard_path, arrays)
                shard_sha256 = sha256_file(shard_path)
                duration = time.perf_counter() - started
                record = {
                    "schema_version": 1,
                    "protocol_id": protocol["protocol_id"],
                    "game_id": game_id,
                    "game_seed": game_seed,
                    "shard_path": relative_shard.as_posix(),
                    "shard_sha256": shard_sha256,
                    "positions": int(arrays["boards"].shape[0]),
                    "game_length": int(arrays["boards"].shape[0]),
                    "terminal_result": terminal_result,
                    "termination": termination,
                    "action_size": action_size,
                    "duration_seconds": duration,
                    "illegal_actions": 0,
                }
                append_jsonl_record(games_stream, record)
                completed[game_id] = record
                manifest["games_completed"] = len(completed)
                manifest["states_completed"] = sum(
                    int(item["positions"]) for item in completed.values()
                )
                manifest["states"] = manifest["states_completed"]
                manifest["updated_at_utc"] = utc_now()
                atomic_write_json(paths["manifest"], manifest)
                logger.log(
                    f"Game {game_id:04d} completed: states={record['positions']}, "
                    f"termination={termination}, seed={game_seed}."
                )

        ordered = [completed[game_id] for game_id in range(games_expected)]
        combined = concatenate_shards(ordered, output_dir)
        atomic_write_npz(paths["states"], combined)
        states_sha256 = sha256_file(paths["states"])
        content_sha256 = sha256_dataset_content(combined)
        if args.verify_source_integrity:
            after_sha256 = sha256_file(checkpoint)
            source_changed = after_sha256 != checkpoint_sha256
            manifest["source_integrity"] = {
                "enabled": True,
                "before_sha256": checkpoint_sha256,
                "after_sha256": after_sha256,
                "status": "failed" if source_changed else "passed",
                "changed": source_changed,
            }
            if source_changed:
                raise HoldoutError("source checkpoint changed during generation")
        manifest["status"] = "completed"
        manifest["games_completed"] = len(ordered)
        manifest["states_completed"] = int(combined["boards"].shape[0])
        manifest["games"] = len(ordered)
        manifest["states"] = int(combined["boards"].shape[0])
        manifest["dataset_file_sha256"] = states_sha256
        manifest["dataset_content_sha256"] = content_sha256
        manifest["dataset"] = {
            "path": paths["states"].name,
            "sha256": states_sha256,
            "content_sha256": content_sha256,
            "size_bytes": paths["states"].stat().st_size,
            "samples": int(combined["boards"].shape[0]),
        }
        manifest["completed_at_utc"] = utc_now()
        manifest["updated_at_utc"] = manifest["completed_at_utc"]
        manifest["freeze_policy"] = {
            "regenerate": False,
            "manual_state_deletion": False,
            "select_samples_by_loss": False,
            "add_to_baseline_replay": False,
            "add_to_adaptive_replay": False,
        }
        atomic_write_json(paths["manifest"], manifest)
        logger.log(
            f"Output status: completed; games={len(ordered)}; "
            f"states={combined['boards'].shape[0]}; states_sha256={states_sha256}."
        )
    except KeyboardInterrupt:
        if "manifest" in locals():
            manifest["status"] = "interrupted"
            manifest["updated_at_utc"] = utc_now()
            atomic_write_json(paths["manifest"], manifest)
        logger.log("Generation interrupted; rerun with --resume.")
        raise
    except Exception as exc:
        if "manifest" in locals():
            manifest["status"] = "failed"
            manifest["updated_at_utc"] = utc_now()
            manifest["error"] = str(exc)
            atomic_write_json(paths["manifest"], manifest)
        logger.log(f"Generation failed: {exc}")
        raise
    finally:
        logger.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--games",
        type=int,
        help="pilot-only runtime game count; must not exceed the frozen protocol count",
    )
    parser.add_argument("--verify-source-integrity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        generate(parse_args(argv))
    except (HoldoutError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
