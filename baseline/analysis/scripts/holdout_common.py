"""Shared protocol and artifact helpers for fixed_holdout_v1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = BASELINE_ROOT.parent
DEFAULT_CONFIG = BASELINE_ROOT / "analysis" / "configs" / "holdout_v1.yaml"
DEFAULT_OUTPUT_DIR = (
    BASELINE_ROOT / "outputs" / "baseline_seed1001_4090_analysis" / "holdout_v1"
)
DEFAULT_BASELINE_RUN_DIR = (
    BASELINE_ROOT / "outputs" / "baseline_reproduction_seed1001_4090"
)
EXPECTED_CHECKPOINTS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 210]
ARRAY_NAMES = (
    "boards",
    "policies",
    "values",
    "valids",
    "game_ids",
    "steps",
    "game_lengths",
)


class HoldoutError(ValueError):
    """Raised when the fixed hold-out protocol or artifacts are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataset_content(arrays: dict[str, np.ndarray]) -> str:
    """Hash the ordered logical arrays independently of NPZ container bytes."""
    digest = hashlib.sha256(b"fixed_holdout_v1_dataset_content\0")
    for name in ARRAY_NAMES:
        if name not in arrays:
            raise HoldoutError(f"dataset content hash is missing array: {name}")
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def git_revision() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASELINE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=BASELINE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HoldoutError(f"could not record git revision: {exc}") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise HoldoutError("git rev-parse HEAD did not return a full commit hash")
    return commit, dirty


def _require_exact_mapping(
    value: Any, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or value != expected:
        raise HoldoutError(f"{label} does not match fixed_holdout_v1")
    return value


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise HoldoutError("schema_version must be 1")
    if protocol.get("protocol_id") != "fixed_holdout_v1":
        raise HoldoutError("protocol_id must be fixed_holdout_v1")
    if protocol.get("seed") != 71001:
        raise HoldoutError("seed must be 71001")
    if protocol.get("games") != 200:
        raise HoldoutError("games must be 200")
    _require_exact_mapping(
        protocol.get("source"),
        {
            "type": "alphazero_self_play",
            "checkpoint_iteration": 0,
            "checkpoint_path": (
                "outputs/pretraining_reproduction_seed1001/checkpoints/"
                "checkpoint_0.pth.tar"
            ),
            "expected_sha256": (
                "4824a2a8ba1c1ebb5a38a992af075a45a033b87b403973b583ab98a079f35667"
            ),
        },
        "source",
    )
    _require_exact_mapping(
        protocol.get("model"),
        {
            "board_size": 9,
            "num_channels": 128,
            "num_res_blocks": 6,
            "attn_depth": 1,
            "num_heads": 8,
            "se_enabled": False,
        },
        "model",
    )
    _require_exact_mapping(
        protocol.get("self_play"),
        {
            "mcts_simulations": 200,
            "eval_mcts_in_batch": 10,
            "cpuct": 1.25,
            "temperature_threshold": 15,
            "dirichlet_noise": True,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.25,
            "max_game_length": 150,
        },
        "self_play",
    )
    _require_exact_mapping(
        protocol.get("storage"),
        {
            "save_dtype_board": "uint8",
            "save_dtype_policy": "float32",
            "save_dtype_value": "float32",
            "save_dtype_valids": "uint8",
            "save_per_game_shards": True,
        },
        "storage",
    )
    _require_exact_mapping(
        protocol.get("evaluation"),
        {
            "checkpoints": EXPECTED_CHECKPOINTS,
            "batch_size": 1024,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 72001,
            "confidence_level": 0.95,
        },
        "evaluation",
    )
    if protocol["self_play"]["mcts_simulations"] % protocol["self_play"][
        "eval_mcts_in_batch"
    ]:
        raise HoldoutError(
            "self_play.mcts_simulations must be divisible by eval_mcts_in_batch"
        )


def load_protocol(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise HoldoutError(f"hold-out protocol not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HoldoutError(f"invalid hold-out protocol YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HoldoutError("hold-out protocol root must be a mapping")
    validate_protocol(loaded)
    loaded["_path"] = path
    return loaded


def serializable_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in protocol.items() if not key.startswith("_")}


def resolve_source_checkpoint(protocol: dict[str, Any]) -> Path:
    recorded = Path(protocol["source"]["checkpoint_path"])
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.extend((BASELINE_ROOT / recorded, SOURCE_ROOT / recorded))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise HoldoutError(
        f"source checkpoint cannot be resolved: {protocol['source']['checkpoint_path']}"
    )


def validate_source_checkpoint(protocol: dict[str, Any]) -> tuple[Path, str]:
    checkpoint = resolve_source_checkpoint(protocol)
    digest = sha256_file(checkpoint)
    if digest != protocol["source"]["expected_sha256"]:
        raise HoldoutError("checkpoint_0 SHA-256 does not match holdout_v1.yaml")
    return checkpoint, digest


def stable_game_seed(base_seed: int, game_id: int) -> int:
    if isinstance(game_id, bool) or not isinstance(game_id, int) or game_id < 0:
        raise HoldoutError("game_id must be a non-negative integer")
    seed = base_seed + game_id
    if seed > 0xFFFFFFFF:
        raise HoldoutError("derived game seed exceeds uint32")
    return seed


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_alphazero_paths() -> None:
    alphazero_root = BASELINE_ROOT / "external" / "alphazero"
    pathfinder_root = alphazero_root / "quoridor" / "pathFinder-module"
    for path in (alphazero_root, pathfinder_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_game(protocol: dict[str, Any]):
    add_alphazero_paths()
    from quoridor.QuoridorGame import QuoridorGame

    return QuoridorGame(int(protocol["model"]["board_size"]))


def build_network(
    protocol: dict[str, Any], checkpoint: Path, *, device: str = "cuda"
):
    if device not in {"cpu", "cuda"}:
        raise HoldoutError("device must be cpu or cuda")
    use_cuda = device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise HoldoutError("CUDA was requested but is unavailable")
    add_alphazero_paths()
    from quoridor.pytorch import NNet as nnet_module
    from quoridor.pytorch.NNet import NNetWrapper
    from utils import dotdict

    game = build_game(protocol)
    model = protocol["model"]
    nn_args = dotdict(
        {
            "cuda": use_cuda,
            "lr": 0.0002,
            "dropout": 0.3,
            "weight_decay": 0.0001,
            "lr_decay_gamma": 1.0,
            "epochs": 1,
            "batch_size": int(protocol["evaluation"]["batch_size"]),
            "micro_batch_size": int(protocol["evaluation"]["batch_size"]),
            "num_channels": int(model["num_channels"]),
            "num_res_blocks": int(model["num_res_blocks"]),
            "attn_depth": int(model["attn_depth"]),
            "num_heads": int(model["num_heads"]),
            "se_enabled": bool(model["se_enabled"]),
            "fast_opts": False,
            "clip": 1.0,
            "use_amp": False,
            "amp_dtype": "bf16",
        }
    )
    nnet_module.args.update(nn_args)
    network = NNetWrapper(game, custom_args=nn_args)
    network.load_checkpoint(str(checkpoint.parent), checkpoint.name)
    network.nnet.eval()
    return game, network


def build_mcts(game, network, protocol: dict[str, Any]):
    add_alphazero_paths()
    from MCTS import MCTS
    from utils import dotdict

    self_play = protocol["self_play"]
    args = dotdict(
        {
            "numMCTSSims": int(self_play["mcts_simulations"]),
            "eval_mcts_in_batch": int(self_play["eval_mcts_in_batch"]),
            "cpuct": float(self_play["cpuct"]),
            "dirichlet_alpha": float(self_play["dirichlet_alpha"]),
            "dirichlet_epsilon": (
                float(self_play["dirichlet_epsilon"])
                if self_play["dirichlet_noise"]
                else 0.0
            ),
            "heuristic_alpha": 0.0,
            "heuristic_decay_iters": 1,
        }
    )
    return MCTS(game, network, args, heuristic_prior_fn=None, iteration=0)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            yaml.safe_dump(payload, destination, sort_keys=False, allow_unicode=True)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise HoldoutError(f"NPZ artifact not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(set(ARRAY_NAMES) - set(loaded.files))
            extra = sorted(set(loaded.files) - set(ARRAY_NAMES))
            if missing or extra:
                raise HoldoutError(
                    f"NPZ arrays differ: missing={missing}, extra={extra}"
                )
            return {name: np.array(loaded[name], copy=True) for name in ARRAY_NAMES}
    except (OSError, ValueError) as exc:
        raise HoldoutError(f"could not read NPZ artifact {path}: {exc}") from exc


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HoldoutError(f"JSON artifact not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HoldoutError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HoldoutError(f"JSON root must be an object: {path}")
    return loaded


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HoldoutError(f"JSONL artifact not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise HoldoutError(f"blank JSONL line at {line_number}: {path}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HoldoutError(
                    f"invalid JSONL line {line_number} in {path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise HoldoutError(f"JSONL line {line_number} must be an object")
            records.append(record)
    return records


def append_jsonl_record(destination, record: dict[str, Any]) -> None:
    destination.write(
        json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    )
    destination.flush()
    os.fsync(destination.fileno())


def concatenate_shards(
    records: Iterable[dict[str, Any]], output_dir: Path
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_NAMES}
    for record in sorted(records, key=lambda item: int(item["game_id"])):
        shard = Path(record["shard_path"])
        if not shard.is_absolute():
            shard = output_dir / shard
        arrays = load_npz(shard.resolve())
        for name in ARRAY_NAMES:
            chunks[name].append(arrays[name])
    if not all(chunks.values()):
        raise HoldoutError("cannot merge an empty hold-out")
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
