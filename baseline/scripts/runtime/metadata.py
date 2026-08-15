from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: Path | str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_metadata(project_root: Path | str) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", *args],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "root": root.as_posix(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def collect_python_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "PyYAML", "torch", "torchvision", "einops", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "packages": packages,
    }


def collect_cuda_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "torch_importable": False}
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "available": available,
        "torch_importable": True,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": len(devices),
        "devices": devices,
    }


def collect_hardware_metadata() -> dict[str, Any]:
    memory_bytes = None
    try:
        import psutil

        memory_bytes = psutil.virtual_memory().total
    except ImportError:
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
    }


def write_run_metadata(
    path: Path | str,
    *,
    project_root: Path | str,
    resolved_config: dict[str, Any] | None = None,
    input_hashes: dict[str, str | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": collect_git_metadata(project_root),
        "python": collect_python_environment(),
        "cuda": collect_cuda_environment(),
        "hardware": collect_hardware_metadata(),
        "input_hashes": input_hashes or {},
        "resolved_config": _jsonable(resolved_config),
    }
    if extra:
        payload.update(_jsonable(extra))
    return atomic_write_json(path, payload)
