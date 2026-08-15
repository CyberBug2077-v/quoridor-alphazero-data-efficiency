#!/usr/bin/env python3
"""Create the frozen checkpoint_0 used by Baseline and Adaptive runs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from runtime.artifacts import (
    JsonlWriter,
    atomic_write_json,
    atomic_write_yaml,
    find_existing_run_artifacts,
)
from runtime.checkpointing import save_numbered_checkpoint
from runtime.config import ConfigError, load_yaml, resolve_pretraining_config
from runtime.metadata import sha256_file, write_run_metadata


BASELINE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = BASELINE_ROOT.parent
DEFAULT_CONFIG = BASELINE_ROOT / "configs" / "pretraining_reproduction.yaml"


class _Tee:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", newline="\n")
        self.stdout = _Tee(sys.stdout, self.file)
        self.stderr = _Tee(sys.stderr, self.file)
        self.redirect_stdout = redirect_stdout(self.stdout)
        self.redirect_stderr = redirect_stderr(self.stderr)
        self.redirect_stdout.__enter__()
        self.redirect_stderr.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.redirect_stderr.__exit__(exc_type, exc_value, traceback)
        self.redirect_stdout.__exit__(exc_type, exc_value, traceback)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "fresh"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def initialize_cuda(resolved: dict[str, Any]) -> torch.device:
    run = resolved["run"]
    if run["device"] != "cuda":
        raise ConfigError("pretraining requires run.device: cuda")
    if not torch.cuda.is_available():
        raise ConfigError("CUDA is not available to PyTorch")
    gpu_index = int(run["gpu_index"])
    if gpu_index >= torch.cuda.device_count():
        raise ConfigError(
            f"run.gpu_index {gpu_index} is invalid; detected "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )
    torch.cuda.set_device(gpu_index)
    return torch.device("cuda", gpu_index)


def require_inside_baseline(resolved: dict[str, Any]) -> None:
    for label, path in (
        ("run.output_dir", resolved["_output_path"]),
        ("checkpoint.directory", resolved["checkpoint"]["_directory_path"]),
    ):
        if not Path(path).is_relative_to(BASELINE_ROOT):
            raise ConfigError(f"{label} must be inside {BASELINE_ROOT}")
    expected_checkpoint_dir = resolved["_output_path"] / "checkpoints"
    if resolved["checkpoint"]["_directory_path"] != expected_checkpoint_dir:
        raise ConfigError("checkpoint.directory must equal run.output_dir/checkpoints")


def ensure_output_is_unused(output_dir: Path) -> None:
    conflicts = find_existing_run_artifacts(output_dir)
    if output_dir.is_dir():
        conflicts.extend(path for path in output_dir.iterdir() if path not in conflicts)
    if conflicts:
        joined = "\n  - ".join(str(path) for path in sorted(set(conflicts)))
        raise ConfigError(f"run would overwrite existing output:\n  - {joined}")


def load_and_validate_dataset(resolved: dict[str, Any]) -> list | tuple:
    data = resolved["data"]
    archive_path = Path(data["_archive_path"])
    pickle_path = Path(data["_extracted_path"])
    if not archive_path.is_file():
        raise ConfigError(f"data archive not found: {archive_path}")
    if not pickle_path.is_file():
        raise ConfigError(f"extracted dataset not found: {pickle_path}")
    actual_hash = sha256_file(pickle_path)
    if actual_hash != data["expected_sha256"].lower():
        raise ConfigError(
            f"dataset SHA-256 mismatch: expected={data['expected_sha256']} "
            f"actual={actual_hash}"
        )
    try:
        with pickle_path.open("rb") as source:
            examples = pickle.load(source)
    except Exception as exc:
        raise ConfigError(f"could not load pretraining pickle: {exc}") from exc
    if not isinstance(examples, (list, tuple)) or not examples:
        raise ConfigError("pretraining pickle must contain a non-empty list or tuple")

    expected_board_shape = (4, 17, 17)
    expected_policy_size = 136
    zero_policy_examples = 0
    policy_on_invalid_examples = 0
    for index, sample in enumerate(examples):
        if not isinstance(sample, (list, tuple)) or len(sample) < 3:
            raise ConfigError(f"sample {index} must contain at least (board, policy, value)")
        board = np.asarray(sample[0])
        policy = np.asarray(sample[1])
        value_array = np.asarray(sample[2])
        if board.shape != expected_board_shape or not np.isfinite(board).all():
            raise ConfigError(f"sample {index} has invalid board shape or values")
        if policy.shape != (expected_policy_size,) or not np.isfinite(policy).all():
            raise ConfigError(f"sample {index} has invalid policy shape or values")
        policy_sum = float(policy.sum())
        if np.any(policy < 0):
            raise ConfigError(f"sample {index} policy must be non-negative")
        if math.isclose(policy_sum, 0.0, rel_tol=0.0, abs_tol=1e-8):
            zero_policy_examples += 1
        elif not math.isclose(policy_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ConfigError(f"sample {index} policy must sum to 0 or 1")
        if value_array.size != 1:
            raise ConfigError(f"sample {index} value must be scalar")
        value = float(value_array.reshape(-1)[0])
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ConfigError(f"sample {index} value must be finite and in [-1, 1]")
        if len(sample) >= 4:
            valids = np.asarray(sample[3])
            if valids.shape != (expected_policy_size,) or not np.isfinite(valids).all():
                raise ConfigError(f"sample {index} has invalid legal-action mask")
            if np.any((valids != 0) & (valids != 1)) or not np.any(valids == 1):
                raise ConfigError(f"sample {index} legal-action mask must be binary and non-empty")
            if np.any((policy > 0) & (valids != 1)):
                policy_on_invalid_examples += 1
        if index and index % 500_000 == 0:
            print(f"Validated {index:,}/{len(examples):,} pretraining samples")
    resolved["_dataset_validation"] = {
        "examples": len(examples),
        "zero_policy_examples": zero_policy_examples,
        "zero_policy_fraction": zero_policy_examples / len(examples),
        "policy_on_invalid_examples": policy_on_invalid_examples,
        "policy_on_invalid_fraction": policy_on_invalid_examples / len(examples),
    }
    print(
        f"Validated {len(examples):,} pretraining samples "
        f"({zero_policy_examples:,} zero-policy and "
        f"{policy_on_invalid_examples:,} masked-policy samples)"
    )
    return examples


def create_network(resolved: dict[str, Any]):
    alphazero_root = BASELINE_ROOT / "external" / "alphazero"
    pathfinder_root = alphazero_root / "quoridor" / "pathFinder-module"
    for path in (alphazero_root, pathfinder_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from quoridor.QuoridorGame import QuoridorGame
    from quoridor.pytorch.NNet import NNetWrapper
    from utils import dotdict

    game = QuoridorGame(resolved["model"]["board_size"])
    nn_args = dotdict(copy.deepcopy(resolved["mapped_args"]["nn_args"]))
    network = NNetWrapper(game, custom_args=nn_args)
    if game.getActionSize() != 136 or game.getInitBoard().shape != (4, 17, 17):
        raise ConfigError("constructed Quoridor network has an unexpected board/action shape")
    if not network.nnet.state_dict():
        raise ConfigError("constructed network has an empty state dict")
    return game, network


def serializable_config(resolved: dict[str, Any], *, mode: str) -> dict[str, Any]:
    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(nested)
                for key, nested in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    payload = clean(resolved)
    payload["schema_version"] = 1
    payload["mode"] = mode
    return payload


def save_best_copy(checkpoint_0: Path, best_path: Path) -> str:
    temporary = best_path.with_name(best_path.name + ".tmp")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with checkpoint_0.open("rb") as source, temporary.open("wb") as destination:
            while chunk := source.read(8 * 1024 * 1024):
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, best_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(best_path)


def prepare(config_path: Path) -> tuple[dict[str, Any], list | tuple, Any]:
    resolved = resolve_pretraining_config(load_yaml(config_path), config_path)
    require_inside_baseline(resolved)
    ensure_output_is_unused(resolved["_output_path"])
    initialize_cuda(resolved)
    set_seed(resolved["run"]["seed"], resolved["run"]["deterministic"])
    examples = load_and_validate_dataset(resolved)
    _, network = create_network(resolved)
    return resolved, examples, network


def main() -> int:
    args = parse_args()
    try:
        resolved, examples, network = prepare(args.config)
        printable = serializable_config(resolved, mode="pretraining")
        if args.mode == "dry-run":
            print(yaml.safe_dump(printable, sort_keys=False, allow_unicode=True))
            print("Pretraining dry-run validation passed; no files were written.")
            return 0

        output_dir = resolved["_output_path"]
        checkpoint_dir = resolved["checkpoint"]["_directory_path"]
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(output_dir / "resolved_config.yaml", printable)
        metadata_path = output_dir / resolved["logging"]["metadata_file"]
        write_run_metadata(
            metadata_path,
            project_root=SOURCE_ROOT,
            resolved_config=printable,
            input_hashes={"pretraining_dataset": resolved["data"]["expected_sha256"]},
            extra={
                "run_type": "pretraining",
                "dataset_validation": resolved["_dataset_validation"],
            },
        )
        with RunLog(output_dir / "run.log"):
            print(yaml.safe_dump(printable, sort_keys=False, allow_unicode=True))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            train_metrics = network.train(
                examples,
                print_summary=True,
                lr_override=resolved["pretraining"]["learning_rate"],
                available_examples=len(examples),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if train_metrics["optimizer_steps"] <= 0:
                raise RuntimeError("pretraining completed without an optimizer step")
            metrics = {
                "schema_version": 1,
                "phase": "pretraining",
                "epochs": resolved["pretraining"]["epochs"],
                **train_metrics,
                "training_seconds": elapsed,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                **resolved["_dataset_validation"],
            }
            metrics_path = output_dir / resolved["logging"]["metrics_file"]
            with JsonlWriter(metrics_path) as writer:
                writer(metrics)
            saved = save_numbered_checkpoint(network, checkpoint_dir, 0)
            best_path = checkpoint_dir / resolved["checkpoint"]["best_filename"]
            best_hash = save_best_copy(saved["path"], best_path)
            if best_hash != saved["sha256"]:
                raise RuntimeError("best.pth.tar differs from checkpoint_0.pth.tar")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["output_hashes"] = {
                "checkpoint_0": saved["sha256"],
                "best": best_hash,
            }
            atomic_write_json(metadata_path, metadata)
            summary = {
                "schema_version": 1,
                "status": "completed",
                "run_id": resolved["run"]["id"],
                "examples": len(examples),
                **resolved["_dataset_validation"],
                "epochs": resolved["pretraining"]["epochs"],
                "optimizer_steps": train_metrics["optimizer_steps"],
                "training_seconds": elapsed,
                "checkpoint_0_path": saved["path"].as_posix(),
                "checkpoint_0_sha256": saved["sha256"],
                "best_path": best_path.as_posix(),
                "best_sha256": best_hash,
            }
            atomic_write_json(
                output_dir / resolved["logging"]["summary_file"], summary
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
        print(f"Pretraining error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
