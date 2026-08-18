from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[2]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from arena.arena import MatchResult, _prep_bot, _validate_move, play_game
from arena.bot_alphazero import AlphaZeroBot
from arena.bot_greedy import GreedyBot
from arena.bot_js_mcts import JSBot
from arena.bot_random import RandomBot
from arena.bot_random_greedy import RandomGreedyBot
from pyquoridor.board import Board
from pyquoridor.exceptions import InvalidFence, InvalidMove


DEFAULT_CONFIG = BASELINE_ROOT / "analysis" / "configs" / "fixed_basket_v1.yaml"
DEFAULT_RUN_DIR = BASELINE_ROOT / "outputs" / "baseline_reproduction_seed1001_4090"
ANALYSIS_OUTPUT_ROOT = (
    BASELINE_ROOT / "outputs" / "baseline_seed1001_4090_analysis"
)
DEFAULT_FORMAL_OUTPUT_DIR = ANALYSIS_OUTPUT_ROOT / "fixed_basket_v1"
DEFAULT_PILOT_OUTPUT_DIR = ANALYSIS_OUTPUT_ROOT / "fixed_basket_v1_pilot"
DEFAULT_OUTPUT_DIR = DEFAULT_FORMAL_OUTPUT_DIR
SEEDED_JS_BRIDGE = BASELINE_ROOT / "analysis" / "js" / "seeded_bot.js"
CHECKPOINT_EXTENSIONS = (".pth.tar", ".pth", ".pt", ".ckpt")
CHECKPOINT_ITERATION = re.compile(
    r"(?:checkpoint|iteration|iter)[_-]?(\d+)(?=\D|$)", re.IGNORECASE
)
REQUIRED_GAME_FIELDS = frozenset(
    {
        "protocol_id",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_sha256",
        "opponent",
        "game_index",
        "game_seed",
        "model_color",
        "winner",
        "model_result",
        "termination",
        "turns",
        "duration_seconds",
        "fault",
        "moves",
        "model_temperatures",
    }
)


class FixedBasketError(ValueError):
    """Raised when the fixed-basket protocol or artifacts are invalid."""


class EvaluationLogger:
    """Append-only evaluation log that is flushed after every message."""

    def __init__(self, path: Path):
        self.path = path
        self._destination = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._destination = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._destination is not None:
            self._destination.close()

    def write(self, message: str) -> None:
        if self._destination is None:
            raise RuntimeError("evaluation logger is not open")
        timestamp = datetime.now(timezone.utc).isoformat()
        self._destination.write(f"{timestamp} {message}\n")
        self._destination.flush()
        os.fsync(self._destination.fileno())
        print(message)


class ScheduledTemperatureAlphaZeroBot(AlphaZeroBot):
    def __init__(
        self,
        *args,
        early_temp=0.18,
        early_moves=5,
        later_temp=0.0,
        **kwargs,
    ):
        super().__init__(*args, temp=early_temp, **kwargs)
        self.early_temp = early_temp
        self.early_moves = early_moves
        self.later_temp = later_temp
        self.temperature_history = []
        self.fallback_events = []

    def select_move(self, board):
        self.temp = (
            self.early_temp
            if self.turn_count < self.early_moves
            else self.later_temp
        )
        self.temperature_history.append(self.temp)
        return super().select_move(board)

    def _log_fallback_debug(
        self,
        board,
        az_board,
        player,
        probs,
        valid_moves,
        legal_pawn_moves,
        legal_fence_moves,
        failed_actions,
    ):
        """Record fallback diagnostics without serializing raw board state.

        The shared arena implementation writes a debug JSONL entry containing
        values supplied by NumPy.  Some late-game states include ``np.int64``
        coordinates, which can make that diagnostic write raise and incorrectly
        divert an otherwise valid MCTS selection into the raw-policy path.  The
        fixed-basket protocol keeps compact, JSON-safe per-game diagnostics
        instead and leaves the shared sanity-evaluation implementation untouched.
        """
        self.fallback_events.append(
            {
                "model_color": self.color,
                "player": int(player),
                "failed_actions": int(len(failed_actions)),
                "alphazero_valid_actions": int(np.sum(valid_moves)),
                "reason": (
                    "all_policy_actions_failed_arena_legality"
                    if failed_actions
                    else "no_policy_mass_on_arena_legal_actions"
                ),
            }
        )


def stable_seed(
    protocol_id: str,
    checkpoint: int,
    opponent_id: str,
    game_index: int,
    *,
    base_seed: int = 0,
) -> int:
    payload = (
        f"{protocol_id}\x1f{base_seed}\x1f{checkpoint}\x1f"
        f"{opponent_id}\x1f{game_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _request_seed(game_seed: int, request_index: int) -> int:
    payload = f"fixed_basket_v1-js\x1f{game_seed}\x1f{request_index}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class SeededJSBot(JSBot):
    """JSBot transport that supplies one deterministic uint32 seed per move."""

    MAX_INVALID_PROPOSAL_RETRIES = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._game_seed: int | None = None
        self._request_index = 0
        self.invalid_proposals = 0

    def set_game_seed(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise FixedBasketError("JS game seed must be a uint32")
        self._game_seed = seed
        self._request_index = 0
        self.invalid_proposals = 0

    def select_move(self, board: Board):
        for _ in range(self.MAX_INVALID_PROPOSAL_RETRIES + 1):
            move = super().select_move(board)
            legal_pawn_moves, legal_fence_moves = board.legal_moves(player=self.color)
            try:
                _validate_move(
                    self.color,
                    move,
                    legal_pawn_moves,
                    legal_fence_moves,
                )
            except (InvalidMove, InvalidFence):
                self.invalid_proposals += 1
                continue
            return move
        raise FixedBasketError(
            "seeded JS opponent exceeded the invalid-proposal retry limit"
        )

    def _writeln(self, obj):
        if self._game_seed is None:
            raise FixedBasketError("set_game_seed must be called before JS selection")
        payload = dict(obj)
        payload["seed"] = _request_seed(self._game_seed, self._request_index)
        self._request_index += 1
        super()._writeln(payload)

    def cleanup(self) -> None:
        process = getattr(self, "proc", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait(timeout=5)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints against the fixed opponent basket."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--mode",
        choices=("pilot", "formal"),
        default="formal",
        help="Keep pilot and formal artifacts in separate output directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the mode-specific output directory.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        help="Evaluate a configured checkpoint subset; the manifest still covers all.",
    )
    parser.add_argument(
        "--games-per-opponent",
        type=int,
        help="Pilot-only even game count override; formal evaluation remains fixed at 50.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible games.jsonl and skip completed game keys.",
    )
    parser.add_argument(
        "--retry-termination",
        action="append",
        choices=("invalid_move", "bot_error"),
        default=[],
        help="With --resume, remove and replay records with this termination.",
    )
    parser.add_argument(
        "--retry-game",
        action="append",
        default=[],
        metavar="CHECKPOINT:OPPONENT:GAME_INDEX",
        help="With --resume, remove and replay one stable game key.",
    )
    parser.add_argument(
        "--verify-source-integrity",
        action="store_true",
        help="Hash every file in the source training run before and after evaluation.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve/hash checkpoints and verify JS determinism without playing games.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_directory_hashes(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FixedBasketError(f"source integrity directory not found: {root}")
    files = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise FixedBasketError(
                f"source file changed while hashing: {path.relative_to(root)}"
            )
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": before.st_size,
                "sha256": digest,
            }
        )
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "root": root.as_posix(),
        "file_count": len(files),
        "total_size_bytes": sum(entry["size_bytes"] for entry in files),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def compare_directory_snapshots(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_files = {entry["path"]: entry for entry in before["files"]}
    after_files = {entry["path"]: entry for entry in after["files"]}
    changed_paths = sorted(
        path
        for path in before_files.keys() | after_files.keys()
        if before_files.get(path) != after_files.get(path)
    )
    return {
        "root": before["root"],
        "before_file_count": before["file_count"],
        "after_file_count": after["file_count"],
        "before_tree_sha256": before["tree_sha256"],
        "after_tree_sha256": after["tree_sha256"],
        "unchanged": not changed_paths,
        "changed_paths": changed_paths,
        "files": before["files"],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            yaml.safe_dump(payload, destination, sort_keys=False, allow_unicode=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_output_dir(mode: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if mode == "pilot":
        return DEFAULT_PILOT_OUTPUT_DIR.resolve()
    if mode == "formal":
        return DEFAULT_FORMAL_OUTPUT_DIR.resolve()
    raise FixedBasketError(f"unsupported evaluation mode: {mode}")


def write_resolved_protocol(protocol: dict[str, Any], output_dir: Path) -> Path:
    destination = output_dir / "protocol.resolved.yaml"
    serializable = {key: value for key, value in protocol.items() if key != "_path"}
    _atomic_write_yaml(destination, serializable)
    return destination


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FixedBasketError(f"required JSON file not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixedBasketError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise FixedBasketError(f"JSON root must be an object: {path}")
    return loaded


def load_protocol(
    path: Path, *, allow_games_per_opponent_override: bool = False
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FixedBasketError(f"protocol configuration not found: {path}")
    try:
        protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FixedBasketError(f"invalid protocol YAML: {exc}") from exc
    if not isinstance(protocol, dict):
        raise FixedBasketError("protocol configuration must contain a mapping")
    validate_protocol(
        protocol,
        allow_games_per_opponent_override=allow_games_per_opponent_override,
    )
    protocol["_path"] = path
    return protocol


def validate_protocol(
    protocol: dict[str, Any], *, allow_games_per_opponent_override: bool = False
) -> None:
    expected_checkpoints = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 210]
    if protocol.get("schema_version") != 1:
        raise FixedBasketError("schema_version must be 1")
    if protocol.get("protocol_id") != "fixed_basket_v1":
        raise FixedBasketError("protocol_id must be fixed_basket_v1")
    if protocol.get("base_seed") != 1001:
        raise FixedBasketError("base_seed must be 1001")
    if protocol.get("checkpoints") != expected_checkpoints:
        raise FixedBasketError("checkpoint sequence does not match fixed_basket_v1")
    games = protocol.get("games_per_opponent")
    valid_game_count = (
        not isinstance(games, bool)
        and isinstance(games, int)
        and games > 0
        and games % 2 == 0
        and (games <= 50 if allow_games_per_opponent_override else games == 50)
    )
    if not valid_game_count:
        expected = "an even integer from 2 to 50" if allow_games_per_opponent_override else "50"
        raise FixedBasketError(f"games_per_opponent must be {expected}")
    if games % 2 or protocol.get("alternate_sides") is not True:
        raise FixedBasketError("games must split evenly across alternating sides")
    if protocol.get("max_turns") != 150:
        raise FixedBasketError("max_turns must be the baseline protocol value 150")
    opponents = protocol.get("opponents")
    expected_opponents = [
        {"id": "heuristic_20", "type": "js_mcts", "rollouts": 20},
        {"id": "heuristic_200", "type": "js_mcts", "rollouts": 200},
        {
            "id": "greedy_random_50",
            "type": "random_greedy",
            "greedy_probability": 0.5,
        },
        {"id": "random", "type": "random"},
    ]
    if opponents != expected_opponents:
        raise FixedBasketError("opponent basket does not match fixed_basket_v1")
    model = protocol.get("model")
    if not isinstance(model, dict):
        raise FixedBasketError("model must be a mapping")
    required_model = {
        "use_mcts": True,
        "num_mcts_sims": 200,
        "eval_mcts_in_batch": 4,
        "cpuct": 1.25,
        "clear_tree_each_move": True,
        "early_temperature": 0.18,
        "early_temperature_moves_per_player": 5,
        "later_temperature": 0.0,
        "dirichlet_noise": False,
    }
    if model != required_model:
        raise FixedBasketError("model settings do not match fixed_basket_v1")


def apply_execution_overrides(
    protocol: dict[str, Any], *, mode: str, games_per_opponent: int | None
) -> dict[str, Any]:
    if games_per_opponent is None:
        return protocol
    if mode != "pilot":
        raise FixedBasketError("--games-per-opponent is only allowed in pilot mode")
    effective = dict(protocol)
    effective["games_per_opponent"] = games_per_opponent
    validate_protocol(effective, allow_games_per_opponent_override=True)
    effective["_path"] = protocol["_path"]
    return effective


def _resolve_recorded_path(recorded: str, run_dir: Path) -> Path:
    raw = Path(recorded)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([BASELINE_ROOT / raw, run_dir / raw])
    normalized = recorded.replace("\\", "/")
    marker = "/baseline/"
    if marker in normalized:
        candidates.append(BASELINE_ROOT / normalized.split(marker, 1)[1])
    elif normalized.startswith("baseline/"):
        candidates.append(BASELINE_ROOT / normalized[len("baseline/") :])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FixedBasketError(f"recorded checkpoint path cannot be resolved locally: {recorded}")


def _resolve_recorded_directory(recorded: str, run_dir: Path) -> Path:
    raw = Path(recorded)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([BASELINE_ROOT / raw, run_dir / raw])
    normalized = recorded.replace("\\", "/")
    marker = "/baseline/"
    if marker in normalized:
        candidates.append(BASELINE_ROOT / normalized.split(marker, 1)[1])
    elif normalized.startswith("baseline/"):
        candidates.append(BASELINE_ROOT / normalized[len("baseline/") :])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise FixedBasketError(f"recorded checkpoint directory cannot be resolved: {recorded}")


def _iteration_from_path(path: Path) -> int | None:
    match = CHECKPOINT_ITERATION.search(path.name)
    return int(match.group(1)) if match else None


def _is_checkpoint(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(CHECKPOINT_EXTENSIONS)


def _manifest_checkpoint_candidates(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for manifest_path in run_dir.rglob("*.sha256"):
        for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            recorded = parts[1].lstrip("*")
            try:
                candidate = _resolve_recorded_path(recorded, run_dir)
            except FixedBasketError:
                continue
            if _is_checkpoint(candidate):
                candidates.append(candidate)
    return candidates


def discover_checkpoints(
    protocol: dict[str, Any], run_dir: Path, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    initial_recorded = metadata.get("initial_checkpoint_path")
    if not isinstance(initial_recorded, str) or not initial_recorded.strip():
        raise FixedBasketError("run_metadata.json lacks initial_checkpoint_path")
    initial_path = _resolve_recorded_path(initial_recorded, run_dir)
    initial_expected_hash = metadata.get("initial_checkpoint_sha256")

    resolved_config = metadata.get("resolved_config", {})
    configured_directory = (
        resolved_config.get("checkpoint", {}).get("directory")
        if isinstance(resolved_config, dict)
        else None
    )
    checkpoint_directory = run_dir / "checkpoints"
    if isinstance(configured_directory, str):
        try:
            checkpoint_directory = _resolve_recorded_directory(
                configured_directory, run_dir
            )
        except FixedBasketError:
            pass
    if not checkpoint_directory.is_dir():
        raise FixedBasketError(f"checkpoint directory not found: {checkpoint_directory}")

    candidates: dict[int, list[tuple[Path, str]]] = {}
    for path in _manifest_checkpoint_candidates(run_dir):
        iteration = _iteration_from_path(path)
        if iteration is not None:
            candidates.setdefault(iteration, []).append((path.resolve(), "existing_manifest"))
    for path in checkpoint_directory.iterdir():
        if not _is_checkpoint(path):
            continue
        iteration = _iteration_from_path(path)
        if iteration is not None:
            candidates.setdefault(iteration, []).append((path.resolve(), "directory_scan"))

    entries: list[dict[str, Any]] = []
    for iteration in protocol["checkpoints"]:
        if iteration == 0:
            path = initial_path
            source = "run_metadata.initial_checkpoint_path"
            expected_hash = initial_expected_hash
        else:
            options = candidates.get(iteration, [])
            unique_options: dict[Path, str] = {}
            for path, source_name in options:
                unique_options.setdefault(path, source_name)
            if not unique_options:
                raise FixedBasketError(
                    f"no checkpoint artifact discovered for iteration {iteration}"
                )
            manifest_options = [
                (path, source_name)
                for path, source_name in unique_options.items()
                if source_name == "existing_manifest"
            ]
            selected = manifest_options[0] if manifest_options else next(iter(unique_options.items()))
            path, source = selected
            expected_hash = None
            if len(unique_options) > 1:
                digests = {_sha256(candidate) for candidate in unique_options}
                if len(digests) != 1:
                    raise FixedBasketError(
                        f"ambiguous checkpoint artifacts for iteration {iteration}"
                    )
        digest = _sha256(path)
        if isinstance(expected_hash, str) and digest != expected_hash:
            raise FixedBasketError("checkpoint 0 hash differs from run metadata")
        entries.append(
            {
                "iteration": iteration,
                "path": path.as_posix(),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "resolution_source": source,
            }
        )
    return entries


def build_evaluation_manifest(
    protocol: dict[str, Any],
    run_dir: Path,
    output_dir: Path,
    *,
    mode: str = "formal",
    selected_checkpoints: Sequence[int] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    metadata = _load_json(metadata_path)
    entries = discover_checkpoints(protocol, run_dir, metadata)
    selected = (
        list(protocol["checkpoints"])
        if selected_checkpoints is None
        else list(selected_checkpoints)
    )
    resolved_protocol_path = output_dir / "protocol.resolved.yaml"
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "evaluation_mode": mode,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_config": protocol["_path"].as_posix(),
        "protocol_config_sha256": _sha256(protocol["_path"]),
        "protocol_resolved": resolved_protocol_path.as_posix(),
        "protocol_resolved_sha256": _sha256(resolved_protocol_path),
        "run_metadata": metadata_path.as_posix(),
        "run_metadata_sha256": _sha256(metadata_path),
        "checkpoints": entries,
        "selected_checkpoints": selected,
        "expected_protocol_games": (
            len(protocol["checkpoints"])
            * len(protocol["opponents"])
            * protocol["games_per_opponent"]
        ),
        "expected_evaluation_games": (
            len(selected)
            * len(protocol["opponents"])
            * protocol["games_per_opponent"]
        ),
        "game_seed_derivation": (
            "first 32 bits of SHA-256(protocol_id + base_seed + checkpoint + "
            "opponent_id + game_index)"
        ),
        "side_schedule": {
            "alpha_zero_white_games_per_opponent": protocol["games_per_opponent"] // 2,
            "alpha_zero_black_games_per_opponent": protocol["games_per_opponent"] // 2,
        },
        "max_turns": protocol["max_turns"],
        "js_move_legality_policy": {
            "validation": "arena._validate_move against the current board",
            "on_invalid_proposal": (
                "reject and deterministically request the next seeded JS proposal"
            ),
            "max_retries": SeededJSBot.MAX_INVALID_PROPOSAL_RETRIES,
            "record_field": "opponent_invalid_proposals",
        },
        "model_fallback_policy": {
            "scope": "ScheduledTemperatureAlphaZeroBot only",
            "shared_arena_modified": False,
            "raw_state_persisted": False,
            "record_fields": ["model_fallback_count", "model_fallback_events"],
            "interpretation": (
                "arena-legal deterministic fallback after AlphaZero policy actions "
                "cannot be mapped to a legal pyquoridor move"
            ),
        },
        "outputs": {
            "protocol_resolved": resolved_protocol_path.as_posix(),
            "evaluation_manifest": (output_dir / "evaluation_manifest.json").as_posix(),
            "games": (output_dir / "games.jsonl").as_posix(),
            "evaluation_log": (output_dir / "evaluation.log").as_posix(),
        },
        "sanity_evaluation_modified": False,
    }
    _atomic_write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest


def _make_probe_board() -> Board:
    board = Board()
    bots = {"white": GreedyBot("white"), "black": GreedyBot("black")}
    for _ in range(8):
        color = board.current_player()
        move = bots[color].select_move(board)
        bots[color].apply_move(board, move)
    return board


def _move_signature(move: Any) -> str:
    return json.dumps(move.to_dict(), sort_keys=True, separators=(",", ":"))


def _cleanup_bot(bot: Any) -> None:
    cleanup = getattr(bot, "cleanup", None)
    if callable(cleanup):
        cleanup()
        return
    process = getattr(bot, "proc", None)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait(timeout=5)


def check_js_determinism(
    protocol: dict[str, Any], *, repeats: int = 12
) -> dict[str, Any]:
    js_entry = BASELINE_ROOT / "external" / "js-mcts" / "bot.js"
    ai_source = BASELINE_ROOT / "external" / "js-mcts" / "src" / "js" / "ai.js"
    source_uses_math_random = "Math.random" in ai_source.read_text(
        encoding="utf-8", errors="replace"
    )
    board = _make_probe_board()
    color = board.current_player()
    checks = []
    for opponent in protocol["opponents"]:
        if opponent["type"] != "js_mcts":
            continue
        rollouts = int(opponent["rollouts"])
        raw_bot = JSBot(color, str(js_entry), rollouts=rollouts)
        try:
            raw_signatures = [
                _move_signature(raw_bot.select_move(board)) for _ in range(repeats)
            ]
        finally:
            _cleanup_bot(raw_bot)

        seeded_bot = SeededJSBot(color, str(SEEDED_JS_BRIDGE), rollouts=rollouts)
        seeded_signatures = []
        probe_seed = stable_seed(
            protocol["protocol_id"],
            -1,
            opponent["id"],
            0,
            base_seed=int(protocol["base_seed"]),
        )
        try:
            for _ in range(repeats):
                seeded_bot.set_game_seed(probe_seed)
                seeded_signatures.append(_move_signature(seeded_bot.select_move(board)))
        finally:
            seeded_bot.cleanup()
        seeded_passed = len(set(seeded_signatures)) == 1
        checks.append(
            {
                "opponent_id": opponent["id"],
                "rollouts": rollouts,
                "probe_repeats": repeats,
                "source_uses_math_random": source_uses_math_random,
                "unseeded_distinct_actions": len(set(raw_signatures)),
                "unseeded_repeated_action_consistent": len(set(raw_signatures)) == 1,
                "unseeded_classification": (
                    "not_fully_reproducible_unseeded_rng"
                    if source_uses_math_random or len(set(raw_signatures)) != 1
                    else "deterministic_implementation"
                ),
                "seeded_bridge": SEEDED_JS_BRIDGE.as_posix(),
                "seeded_bridge_sha256": _sha256(SEEDED_JS_BRIDGE),
                "seeded_distinct_actions": len(set(seeded_signatures)),
                "seeded_repeated_action_consistent": seeded_passed,
                "evaluation_mode": "seeded_js_bridge",
            }
        )
        if not seeded_passed:
            raise FixedBasketError(
                f"seeded JS determinism check failed for {opponent['id']}"
            )
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "passed",
        "probe_board": "fixed non-initial board after eight deterministic greedy moves",
        "checks": checks,
    }
    return report


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_bot(bot: Any, color: str) -> Any:
    prepared = _prep_bot(bot, color)
    if hasattr(prepared, "temperature_history"):
        prepared.temperature_history = []
    if hasattr(prepared, "fallback_events"):
        prepared.fallback_events = []
    for child_name in ("_random", "_greedy"):
        child = getattr(prepared, child_name, None)
        if child is not None:
            child.color = color
    return prepared


def make_model(entry: dict[str, Any], protocol: dict[str, Any], board_size: int):
    path = Path(entry["path"])
    model = protocol["model"]
    return ScheduledTemperatureAlphaZeroBot(
        "white",
        str(path.parent),
        path.name,
        board_size=board_size,
        use_mcts=model["use_mcts"],
        clear_tree_each_move=model["clear_tree_each_move"],
        numMCTSSims=model["num_mcts_sims"],
        cpuct=model["cpuct"],
        eval_mcts_in_batch=model["eval_mcts_in_batch"],
        early_temp=model["early_temperature"],
        early_moves=model["early_temperature_moves_per_player"],
        later_temp=model["later_temperature"],
    )


def make_opponent(spec: dict[str, Any], color: str):
    opponent_type = spec["type"]
    if opponent_type == "js_mcts":
        return SeededJSBot(
            color,
            str(SEEDED_JS_BRIDGE),
            rollouts=int(spec["rollouts"]),
        )
    if opponent_type == "random_greedy":
        return RandomGreedyBot(
            color, greedy_prob=float(spec["greedy_probability"])
        )
    if opponent_type == "random":
        return RandomBot(color)
    raise FixedBasketError(f"unsupported opponent type: {opponent_type}")


def _model_result(result: MatchResult, model_color: str) -> str:
    opponent_color = "black" if model_color == "white" else "white"
    if result.fault == model_color:
        return "loss"
    if result.fault == opponent_color:
        return "win"
    if result.winner == model_color:
        return "win"
    if result.winner == opponent_color:
        return "loss"
    return "draw"


def _serialize_moves(result: MatchResult) -> list[dict[str, Any]]:
    serialized = []
    for move in result.moves:
        to_dict = getattr(move, "to_dict", None)
        if not callable(to_dict):
            raise FixedBasketError(
                f"match result contains a non-serializable move: {type(move).__name__}"
            )
        payload = to_dict()
        if not isinstance(payload, dict):
            raise FixedBasketError("move.to_dict() must return an object")
        serialized.append(payload)
    return serialized


def _game_record(
    *,
    protocol: dict[str, Any],
    entry: dict[str, Any],
    opponent: dict[str, Any],
    game_index: int,
    game_seed: int,
    model_color: str,
    result: MatchResult,
    model_temperatures: Sequence[float],
    opponent_invalid_proposals: int,
    model_fallback_events: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    opponent_color = "black" if model_color == "white" else "white"
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "checkpoint": entry["iteration"],
        "checkpoint_path": entry["path"],
        "checkpoint_sha256": entry["sha256"],
        "opponent": opponent["id"],
        "opponent_type": opponent["type"],
        "game_index": game_index,
        "game_seed": game_seed,
        "model_color": model_color,
        "opponent_color": opponent_color,
        "winner": result.winner,
        "model_result": _model_result(result, model_color),
        "termination": result.termination,
        "fault": result.fault,
        "turns": result.turns,
        "total_moves": result.total_moves,
        "duration_seconds": result.match_duration,
        "model_move_seconds": result.move_times.get(model_color, 0.0),
        "opponent_move_seconds": result.move_times.get(opponent_color, 0.0),
        "model_moves": result.move_counts.get(model_color, 0),
        "opponent_moves": result.move_counts.get(opponent_color, 0),
        "opponent_invalid_proposals": opponent_invalid_proposals,
        "model_fallback_count": len(model_fallback_events),
        "model_fallback_events": list(model_fallback_events),
        "model_temperatures": list(model_temperatures),
        "max_turns": protocol["max_turns"],
        "moves": _serialize_moves(result),
    }


def _load_existing_games(
    games_path: Path,
    protocol: dict[str, Any],
    manifest_by_iteration: dict[int, dict[str, Any]],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    completed: dict[tuple[int, str, int], dict[str, Any]] = {}
    configured_opponents = {opponent["id"] for opponent in protocol["opponents"]}
    if not games_path.exists():
        return completed
    with games_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixedBasketError(
                    f"invalid games JSONL line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise FixedBasketError(
                    f"games JSONL line {line_number} must contain an object"
                )
            missing = REQUIRED_GAME_FIELDS - record.keys()
            if missing:
                raise FixedBasketError(
                    f"games JSONL line {line_number} lacks required fields: "
                    f"{sorted(missing)}"
                )
            key = (
                int(record.get("checkpoint")),
                str(record.get("opponent")),
                int(record.get("game_index")),
            )
            if key in completed:
                raise FixedBasketError(f"duplicate completed game key: {key}")
            entry = manifest_by_iteration.get(key[0])
            if (
                key[1] not in configured_opponents
                or not 0 <= key[2] < protocol["games_per_opponent"]
            ):
                raise FixedBasketError(f"existing game has an invalid key: {key}")
            expected_color = (
                "white"
                if key[2] < protocol["games_per_opponent"] // 2
                else "black"
            )
            if (
                record.get("protocol_id") != protocol["protocol_id"]
                or entry is None
                or record.get("checkpoint_path") != entry["path"]
                or record.get("checkpoint_sha256") != entry["sha256"]
                or record.get("model_color") != expected_color
                or record.get("max_turns") != protocol["max_turns"]
                or record.get("model_result") not in {"win", "draw", "loss"}
                or not isinstance(record.get("moves"), list)
            ):
                raise FixedBasketError(
                    f"existing game is incompatible with current manifest: {key}"
                )
            expected_seed = stable_seed(
                protocol["protocol_id"],
                key[0],
                key[1],
                key[2],
                base_seed=int(protocol["base_seed"]),
            )
            if record.get("game_seed") != expected_seed:
                raise FixedBasketError(f"existing game has wrong stable seed: {key}")
            completed[key] = record
    return completed


def _count_jsonl_records(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def remove_retryable_game_records(
    games_path: Path, terminations: set[str]
) -> list[tuple[int, str, int]]:
    if not terminations:
        return []
    kept_lines = []
    removed_keys = []
    with games_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixedBasketError(
                    f"invalid games JSONL line {line_number}: {exc}"
                ) from exc
            if record.get("termination") in terminations:
                removed_keys.append(
                    (
                        int(record["checkpoint"]),
                        str(record["opponent"]),
                        int(record["game_index"]),
                    )
                )
            else:
                kept_lines.append(
                    json.dumps(
                        record,
                        sort_keys=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
    temporary = games_path.with_name(f".{games_path.name}.{os.getpid()}.retry.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for line in kept_lines:
                destination.write(line + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(games_path)
    finally:
        temporary.unlink(missing_ok=True)
    return removed_keys


def parse_retry_game_keys(values: Sequence[str]) -> set[tuple[int, str, int]]:
    keys: set[tuple[int, str, int]] = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise FixedBasketError(
                "--retry-game must use CHECKPOINT:OPPONENT:GAME_INDEX"
            )
        try:
            key = (int(parts[0]), parts[1], int(parts[2]))
        except ValueError as exc:
            raise FixedBasketError(
                "--retry-game checkpoint and game index must be integers"
            ) from exc
        if not key[1] or key[0] < 0 or key[2] < 0:
            raise FixedBasketError(f"invalid --retry-game key: {value}")
        keys.add(key)
    return keys


def remove_selected_game_records(
    games_path: Path, keys: set[tuple[int, str, int]]
) -> list[tuple[int, str, int]]:
    if not keys:
        return []
    kept_records = []
    removed_keys = []
    with games_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixedBasketError(
                    f"invalid games JSONL line {line_number}: {exc}"
                ) from exc
            key = (
                int(record["checkpoint"]),
                str(record["opponent"]),
                int(record["game_index"]),
            )
            if key in keys:
                removed_keys.append(key)
                continue
            # Normalize diagnostics introduced after a long-running resume so
            # every retained formal record has the same explicit schema.
            record.setdefault("model_fallback_count", 0)
            record.setdefault("model_fallback_events", [])
            kept_records.append(record)
    missing = keys - set(removed_keys)
    if missing:
        raise FixedBasketError(
            f"--retry-game keys were not present in games.jsonl: {sorted(missing)}"
        )
    temporary = games_path.with_name(f".{games_path.name}.{os.getpid()}.retry.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for record in kept_records:
                destination.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(games_path)
    finally:
        temporary.unlink(missing_ok=True)
    return sorted(removed_keys)


def evaluate_matchups(
    protocol: dict[str, Any],
    entries: list[dict[str, Any]],
    games_path: Path,
    *,
    board_size: int,
    selected_checkpoints: set[int] | None = None,
    model_factory: Callable[[dict[str, Any], dict[str, Any], int], Any] = make_model,
    opponent_factory: Callable[[dict[str, Any], str], Any] = make_opponent,
    play_game_fn: Callable[..., MatchResult] = play_game,
    on_game_completed: Callable[[dict[str, Any]], None] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> int:
    games_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_by_iteration = {entry["iteration"]: entry for entry in entries}
    completed = _load_existing_games(games_path, protocol, manifest_by_iteration)
    new_games = 0
    games_per_side = protocol["games_per_opponent"] // 2

    with games_path.open("a", encoding="utf-8", newline="\n") as destination:
        for entry in entries:
            checkpoint = int(entry["iteration"])
            if selected_checkpoints is not None and checkpoint not in selected_checkpoints:
                continue
            checkpoint_has_pending_games = any(
                (checkpoint, opponent["id"], game_index) not in completed
                for opponent in protocol["opponents"]
                for game_index in range(protocol["games_per_opponent"])
            )
            if not checkpoint_has_pending_games:
                continue
            if _sha256(Path(entry["path"])) != entry["sha256"]:
                raise FixedBasketError(
                    f"checkpoint changed after manifest creation: {checkpoint}"
                )
            model_bot = model_factory(entry, protocol, board_size)
            if on_event is not None:
                on_event(f"Checkpoint model loaded: checkpoint={checkpoint}")
            try:
                for opponent_spec in protocol["opponents"]:
                    opponent_has_pending_games = any(
                        (checkpoint, opponent_spec["id"], game_index) not in completed
                        for game_index in range(protocol["games_per_opponent"])
                    )
                    if not opponent_has_pending_games:
                        continue
                    opponent_bot = opponent_factory(opponent_spec, "black")
                    if isinstance(opponent_bot, SeededJSBot):
                        process = getattr(opponent_bot, "proc", None)
                        if process is None or process.poll() is not None:
                            raise FixedBasketError(
                                f"JS opponent failed to start: {opponent_spec['id']}"
                            )
                    if on_event is not None:
                        on_event(
                            "Opponent started: "
                            f"checkpoint={checkpoint} "
                            f"opponent={opponent_spec['id']} "
                            f"type={opponent_spec['type']}"
                        )
                    try:
                        for game_index in range(protocol["games_per_opponent"]):
                            key = (checkpoint, opponent_spec["id"], game_index)
                            if key in completed:
                                continue
                            model_color = (
                                "white" if game_index < games_per_side else "black"
                            )
                            opponent_color = (
                                "black" if model_color == "white" else "white"
                            )
                            game_seed = stable_seed(
                                protocol["protocol_id"],
                                checkpoint,
                                opponent_spec["id"],
                                game_index,
                                base_seed=int(protocol["base_seed"]),
                            )
                            set_all_seeds(game_seed)
                            _prepare_bot(model_bot, model_color)
                            _prepare_bot(opponent_bot, opponent_color)
                            if isinstance(opponent_bot, SeededJSBot):
                                opponent_bot.set_game_seed(game_seed)
                            if model_color == "white":
                                result = play_game_fn(
                                    model_bot,
                                    opponent_bot,
                                    max_turns=protocol["max_turns"],
                                )
                            else:
                                result = play_game_fn(
                                    opponent_bot,
                                    model_bot,
                                    max_turns=protocol["max_turns"],
                                )
                            record = _game_record(
                                protocol=protocol,
                                entry=entry,
                                opponent=opponent_spec,
                                game_index=game_index,
                                game_seed=game_seed,
                                model_color=model_color,
                                result=result,
                                model_temperatures=getattr(
                                    model_bot, "temperature_history", []
                                ),
                                opponent_invalid_proposals=int(
                                    getattr(opponent_bot, "invalid_proposals", 0)
                                ),
                                model_fallback_events=getattr(
                                    model_bot, "fallback_events", []
                                ),
                            )
                            encoded = json.dumps(
                                record,
                                sort_keys=True,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            destination.write(encoded + "\n")
                            destination.flush()
                            os.fsync(destination.fileno())
                            completed[key] = record
                            new_games += 1
                            if on_game_completed is not None:
                                on_game_completed(record)
                    finally:
                        _cleanup_bot(opponent_bot)
                        if isinstance(opponent_bot, SeededJSBot):
                            process = getattr(opponent_bot, "proc", None)
                            if process is not None and process.poll() is None:
                                raise FixedBasketError(
                                    f"JS opponent failed to exit: {opponent_spec['id']}"
                                )
                        if on_event is not None:
                            on_event(
                                "Opponent stopped: "
                                f"checkpoint={checkpoint} "
                                f"opponent={opponent_spec['id']} "
                                f"type={opponent_spec['type']}"
                            )
            finally:
                _cleanup_bot(model_bot)
                if on_event is not None:
                    on_event(f"Checkpoint model released: checkpoint={checkpoint}")
                del model_bot
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return new_games


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args.mode, args.output_dir)
    manifest_path = output_dir / "evaluation_manifest.json"
    games_path = output_dir / "games.jsonl"
    with EvaluationLogger(output_dir / "evaluation.log") as logger:
        manifest: dict[str, Any] | None = None
        try:
            protocol = load_protocol(args.config)
            protocol = apply_execution_overrides(
                protocol,
                mode=args.mode,
                games_per_opponent=args.games_per_opponent,
            )
            run_dir = args.run_dir.expanduser().resolve()
            configured = set(protocol["checkpoints"])
            requested = set(args.checkpoints) if args.checkpoints else configured
            if not requested <= configured:
                raise FixedBasketError("--checkpoints contains an unconfigured iteration")
            selected_order = [
                checkpoint
                for checkpoint in protocol["checkpoints"]
                if checkpoint in requested
            ]
            existing_manifest = None
            if games_path.exists() and not args.resume:
                raise FixedBasketError(
                    "games.jsonl already exists; pass --resume to continue without "
                    "duplicating completed games"
                )
            if args.resume and not games_path.exists():
                raise FixedBasketError("--resume requires an existing games.jsonl")
            if (args.retry_termination or args.retry_game) and not args.resume:
                raise FixedBasketError(
                    "--retry-termination and --retry-game require --resume"
                )
            if args.resume:
                existing_manifest = _load_json(manifest_path)
                expected_games = (
                    len(selected_order)
                    * len(protocol["opponents"])
                    * protocol["games_per_opponent"]
                )
                if (
                    existing_manifest.get("protocol_id") != protocol["protocol_id"]
                    or existing_manifest.get("evaluation_mode") != args.mode
                    or existing_manifest.get("selected_checkpoints") != selected_order
                    or existing_manifest.get("expected_evaluation_games")
                    != expected_games
                ):
                    raise FixedBasketError(
                        "existing games.jsonl belongs to a different protocol, mode, "
                        "or checkpoint selection"
                    )
                removed_keys = remove_retryable_game_records(
                    games_path, set(args.retry_termination)
                )
                if removed_keys:
                    logger.write(
                        "Removed retryable game records: "
                        + ", ".join(str(key) for key in removed_keys)
                    )
                selected_retry_keys = parse_retry_game_keys(args.retry_game)
                selected_removed_keys = remove_selected_game_records(
                    games_path, selected_retry_keys
                )
                if selected_removed_keys:
                    logger.write(
                        "Removed explicitly selected game records: "
                        + ", ".join(str(key) for key in selected_removed_keys)
                    )
            write_resolved_protocol(protocol, output_dir)
            manifest = build_evaluation_manifest(
                protocol,
                run_dir,
                output_dir,
                mode=args.mode,
                selected_checkpoints=selected_order,
            )
            if existing_manifest is not None:
                manifest["created_at_utc"] = existing_manifest.get(
                    "created_at_utc", manifest["created_at_utc"]
                )
                manifest["resume_count"] = int(
                    existing_manifest.get("resume_count", 0)
                ) + 1
                manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
                if "source_integrity" in existing_manifest:
                    manifest["source_integrity"] = existing_manifest[
                        "source_integrity"
                    ]
            logger.write(
                f"Evaluation mode: {args.mode}; selected checkpoints: {selected_order}"
            )
            determinism = check_js_determinism(protocol)
            manifest["js_determinism"] = determinism
            manifest["js_determinism_status"] = determinism["status"]
            _atomic_write_json(manifest_path, manifest)
            logger.write("JS determinism status: passed (seeded bridge)")
            if args.prepare_only:
                logger.write(f"Evaluation manifest: {manifest_path}")
                logger.write("Evaluation status: prepared")
                return 0

            metadata = _load_json(run_dir / "run_metadata.json")
            board_size = int(metadata["resolved_config"]["model"]["board_size"])
            source_before = None
            if args.verify_source_integrity:
                logger.write(f"Hashing source training directory before run: {run_dir}")
                source_before = snapshot_directory_hashes(run_dir)
                manifest["source_integrity"] = {
                    "status": "before_snapshot_complete",
                    "root": source_before["root"],
                    "before_file_count": source_before["file_count"],
                    "before_tree_sha256": source_before["tree_sha256"],
                    "files": source_before["files"],
                }
            manifest["status"] = "running"
            manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(manifest_path, manifest)

            def log_completed_game(record: dict[str, Any]) -> None:
                logger.write(
                    "Game completed and flushed: "
                    f"checkpoint={record['checkpoint']} "
                    f"opponent={record['opponent']} "
                    f"game_index={record['game_index']} "
                    f"model_result={record['model_result']}"
                )

            selected_filter = requested if args.checkpoints else None
            new_games = evaluate_matchups(
                protocol,
                manifest["checkpoints"],
                games_path,
                board_size=board_size,
                selected_checkpoints=selected_filter,
                on_game_completed=log_completed_game,
                on_event=logger.write,
            )
            games_recorded = _count_jsonl_records(games_path)
            if games_recorded != manifest["expected_evaluation_games"]:
                raise FixedBasketError(
                    "games.jsonl count mismatch after evaluation: "
                    f"expected {manifest['expected_evaluation_games']}, "
                    f"found {games_recorded}"
                )
            if source_before is not None:
                logger.write(f"Hashing source training directory after run: {run_dir}")
                source_after = snapshot_directory_hashes(run_dir)
                integrity = compare_directory_snapshots(source_before, source_after)
                integrity["status"] = "passed" if integrity["unchanged"] else "failed"
                manifest["source_integrity"] = integrity
                if not integrity["unchanged"]:
                    raise FixedBasketError(
                        "source training directory changed during pilot: "
                        f"{integrity['changed_paths']}"
                    )
                logger.write(
                    "Source training directory integrity: passed; "
                    f"files={integrity['before_file_count']}"
                )
            is_full_protocol = requested == configured
            manifest["status"] = (
                "completed"
                if args.mode == "pilot" or is_full_protocol
                else "partial"
            )
            manifest["full_protocol_completed"] = is_full_protocol
            manifest["new_games_completed"] = new_games
            manifest["games_file"] = games_path.as_posix()
            manifest["games_recorded"] = games_recorded
            manifest["games_sha256"] = _sha256(games_path)
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(manifest_path, manifest)
            logger.write(f"New games completed: {new_games}")
            logger.write(f"Evaluation status: {manifest['status']}")
            return 0
        except (
            FixedBasketError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            if manifest is not None:
                manifest["status"] = "failed"
                manifest["failure"] = str(exc)
                manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_write_json(manifest_path, manifest)
            logger.write(f"Fixed-basket evaluation failed: {exc}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
