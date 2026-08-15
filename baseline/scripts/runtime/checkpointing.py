from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metadata import sha256_file


def validate_checkpoint_hash(
    path: Path | str,
    expected_sha256: str | None,
) -> str:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    actual = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {checkpoint_path}: "
            f"expected={expected_sha256.lower()} actual={actual}"
        )
    return actual


def load_model_checkpoint(
    model: Any,
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    digest = validate_checkpoint_hash(checkpoint_path, expected_sha256)
    if hasattr(model, "load_checkpoint"):
        model.load_checkpoint(str(checkpoint_path.parent), checkpoint_path.name)
    elif hasattr(model, "load_state_dict"):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        raise TypeError("model must implement load_checkpoint or load_state_dict")
    return {"path": checkpoint_path, "sha256": digest}


def save_numbered_checkpoint(
    model: Any,
    directory: Path | str,
    iteration: int,
    *,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be an integer >= 0")
    checkpoint_dir = Path(directory).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"checkpoint_{iteration}.pth.tar"
    temporary_name = destination.name + ".tmp"
    temporary = checkpoint_dir / temporary_name
    try:
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(str(checkpoint_dir), temporary_name)
        elif hasattr(model, "state_dict"):
            torch.save({"state_dict": model.state_dict()}, temporary)
        else:
            raise TypeError("model must implement save_checkpoint or state_dict")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": destination,
        "sha256": sha256_file(destination) if compute_sha256 else None,
    }


def save_run_state(path: Path | str, state: dict[str, Any]) -> Path:
    if not isinstance(state, dict):
        raise TypeError("run state must be a mapping")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_run_state(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state_path = Path(path).expanduser().resolve()
    if not state_path.is_file():
        raise FileNotFoundError(f"run state not found: {state_path}")
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"run state must contain a mapping: {state_path}")
    return state


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"RNG state missing field(s): {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state is present but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
