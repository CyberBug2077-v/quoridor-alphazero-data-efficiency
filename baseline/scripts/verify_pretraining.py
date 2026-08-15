#!/usr/bin/env python3
"""Read-only verification of a completed formal pretraining run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from runtime.metadata import sha256_file
from verify_baseline import (
    VerificationError,
    expected_model_shapes,
    load_checkpoint,
    load_json_mapping,
    load_yaml_mapping,
    require,
    require_finite,
)


BASELINE_ROOT = Path(__file__).resolve().parents[1]
MODEL_FIELDS = {
    "board_size",
    "num_channels",
    "num_res_blocks",
    "attn_depth",
    "num_heads",
    "se_enabled",
    "dropout",
}


def require_finite_number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{path} must be a finite number",
    )
    numeric = float(value)
    if positive:
        require(numeric > 0, f"{path} must be > 0")
    if nonnegative:
        require(numeric >= 0, f"{path} must be >= 0")
    return numeric


def resolve_inside(path: Path, root: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.expanduser().resolve()
    require(resolved.is_relative_to(root), f"{label} must be inside {root}")
    return resolved


def configured_artifact(run_dir: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and bool(value.strip()), f"{label} is missing")
    return resolve_inside(Path(value), run_dir, label)


def load_pretraining_metrics(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing metrics file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"could not read metrics file {path}: {exc}") from exc
    lines = text.splitlines()
    require(bool(lines), "pretraining metrics file is empty")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"blank metrics line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"invalid metrics line {line_number}: {exc}"
            ) from exc
        require(
            isinstance(record, dict),
            f"metrics line {line_number} is not an object",
        )
        require_finite(record, f"metrics[{line_number}]")
        records.append(record)
    return records


def verify_metadata(
    metadata: dict[str, Any], resolved: dict[str, Any]
) -> None:
    require(metadata.get("run_type") == "pretraining", "metadata run_type must be pretraining")
    embedded = metadata.get("resolved_config")
    require(isinstance(embedded, dict), "metadata resolved_config is missing")
    require(embedded.get("mode") == "pretraining", "metadata mode must be pretraining")
    require(
        embedded.get("run", {}).get("id") == resolved["run"]["id"],
        "metadata run ID differs from resolved config",
    )
    require(
        embedded.get("run", {}).get("seed") == resolved["run"]["seed"],
        "metadata seed differs from resolved config",
    )
    require(
        embedded.get("model") == resolved["model"],
        "metadata model structure differs from resolved config",
    )

    git = metadata.get("git")
    require(isinstance(git, dict), "Git metadata is missing")
    require(
        isinstance(git.get("commit"), str) and bool(git["commit"].strip()),
        "Git commit is missing from metadata",
    )
    require(isinstance(git.get("dirty"), bool), "Git dirty status is missing")

    cuda = metadata.get("cuda")
    require(isinstance(cuda, dict), "CUDA metadata is missing")
    require(cuda.get("torch_importable") is True, "CUDA PyTorch metadata is incomplete")
    require(cuda.get("available") is True, "metadata does not record available CUDA")
    require(
        isinstance(cuda.get("torch_cuda_version"), str)
        and bool(cuda["torch_cuda_version"].strip()),
        "CUDA toolkit version is missing",
    )
    require(
        isinstance(cuda.get("cudnn_version"), int) and cuda["cudnn_version"] > 0,
        "cuDNN version is missing",
    )
    devices = cuda.get("devices")
    device_count = cuda.get("device_count")
    require(
        isinstance(device_count, int)
        and device_count > 0
        and isinstance(devices, list)
        and len(devices) == device_count,
        "CUDA device inventory is incomplete",
    )
    gpu_index = resolved["run"].get("gpu_index")
    selected = next(
        (
            device
            for device in devices
            if isinstance(device, dict) and device.get("index") == gpu_index
        ),
        None,
    )
    require(selected is not None, "configured GPU is absent from CUDA metadata")
    require(
        isinstance(selected.get("name"), str)
        and bool(selected["name"].strip())
        and isinstance(selected.get("total_memory_bytes"), int)
        and selected["total_memory_bytes"] > 0
        and isinstance(selected.get("compute_capability"), str)
        and bool(selected["compute_capability"].strip()),
        "configured GPU metadata is incomplete",
    )


def verify_metrics(
    records: list[dict[str, Any]], resolved: dict[str, Any]
) -> dict[str, Any]:
    expected_epochs = resolved["pretraining"].get("epochs")
    require(
        isinstance(expected_epochs, int) and not isinstance(expected_epochs, bool)
        and expected_epochs > 0,
        "configured pretraining epochs are invalid",
    )
    expected_effective_batch = resolved["pretraining"].get("batch_size")
    expected_micro_batch = resolved["pretraining"].get("micro_batch_size")
    require(
        isinstance(expected_effective_batch, int)
        and expected_effective_batch > 0
        and isinstance(expected_micro_batch, int)
        and expected_micro_batch > 0
        and expected_effective_batch % expected_micro_batch == 0,
        "configured pretraining batch sizes are invalid",
    )
    expected_accumulation = expected_effective_batch // expected_micro_batch
    for index, record in enumerate(records, 1):
        prefix = f"metrics[{index}]"
        require(record.get("phase") == "pretraining", f"{prefix}.phase must be pretraining")
        require(record.get("epochs") == expected_epochs, f"{prefix}.epochs differs from config")
        require_finite_number(record.get("optimizer_steps"), f"{prefix}.optimizer_steps", positive=True)
        require_finite_number(record.get("samples_seen"), f"{prefix}.samples_seen", positive=True)
        require(
            record.get("effective_batch_size") == expected_effective_batch,
            f"{prefix}.effective_batch_size differs from config",
        )
        require(
            record.get("micro_batch_size") == expected_micro_batch,
            f"{prefix}.micro_batch_size differs from config",
        )
        require(
            record.get("gradient_accumulation_steps") == expected_accumulation,
            f"{prefix}.gradient_accumulation_steps differs from config",
        )
        require(
            record.get("micro_batches_processed")
            == record["optimizer_steps"] * expected_accumulation,
            f"{prefix}.micro_batches_processed is inconsistent",
        )
        require(
            record["samples_seen"]
            == record["optimizer_steps"] * expected_effective_batch,
            f"{prefix}.samples_seen is inconsistent",
        )
        for field in ("policy_loss", "value_loss", "total_loss"):
            require_finite_number(record.get(field), f"{prefix}.{field}")
        require_finite_number(record.get("mean_grad_norm"), f"{prefix}.mean_grad_norm", nonnegative=True)
        if "max_grad_norm" in record:
            require_finite_number(record["max_grad_norm"], f"{prefix}.max_grad_norm", nonnegative=True)
        require_finite_number(
            record.get("peak_gpu_memory_mb"),
            f"{prefix}.peak_gpu_memory_mb",
            nonnegative=True,
        )
    return records[-1]


def verify_checkpoint(
    path: Path,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    state_dict = load_checkpoint(path)
    actual_shapes = {name: tuple(tensor.shape) for name, tensor in state_dict.items()}
    require(
        actual_shapes == expected_shapes,
        f"checkpoint parameter names or shapes differ from configured model: {path}",
    )
    return state_dict


def verify_pretraining(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    require(run_dir.is_dir(), f"run directory not found: {run_dir}")
    require(run_dir.is_relative_to(BASELINE_ROOT), "run directory must be inside baseline")

    temporary_files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and (path.name.endswith(".tmp") or ".tmp." in path.name)
    )
    require(
        not temporary_files,
        "temporary files remain in run directory: "
        + ", ".join(path.relative_to(run_dir).as_posix() for path in temporary_files),
    )

    resolved = load_yaml_mapping(run_dir / "resolved_config.yaml")
    require(resolved.get("mode") == "pretraining", "resolved configuration mode must be pretraining")
    run = resolved.get("run")
    require(isinstance(run, dict), "resolved run configuration is missing")
    require(run.get("id") == run_dir.name, "run ID does not match directory")
    require(
        isinstance(run.get("seed"), int) and not isinstance(run.get("seed"), bool),
        "run seed is missing",
    )
    require(run.get("device") == "cuda", "pretraining device must be CUDA")
    require(
        isinstance(run.get("gpu_index"), int) and run["gpu_index"] >= 0,
        "configured GPU index is invalid",
    )
    model = resolved.get("model")
    require(isinstance(model, dict), "resolved model configuration is missing")
    require(MODEL_FIELDS <= set(model), "model structure record is incomplete")
    pretraining = resolved.get("pretraining")
    checkpoint_config = resolved.get("checkpoint")
    logging = resolved.get("logging")
    data = resolved.get("data")
    for value, label in (
        (pretraining, "pretraining"),
        (checkpoint_config, "checkpoint"),
        (logging, "logging"),
        (data, "data"),
    ):
        require(isinstance(value, dict), f"resolved {label} configuration is missing")

    metadata_path = configured_artifact(run_dir, logging.get("metadata_file"), "logging.metadata_file")
    metrics_path = configured_artifact(run_dir, logging.get("metrics_file"), "logging.metrics_file")
    summary_path = configured_artifact(run_dir, logging.get("summary_file"), "logging.summary_file")
    metadata = load_json_mapping(metadata_path)
    verify_metadata(metadata, resolved)

    expected_dataset_hash = data.get("expected_sha256")
    require(
        isinstance(expected_dataset_hash, str) and len(expected_dataset_hash) == 64,
        "configured dataset SHA-256 is missing",
    )
    metadata_dataset_hash = metadata.get("input_hashes", {}).get("pretraining_dataset")
    embedded_dataset_hash = (
        metadata.get("resolved_config", {}).get("data", {}).get("expected_sha256")
    )
    require(
        isinstance(metadata_dataset_hash, str)
        and isinstance(embedded_dataset_hash, str)
        and metadata_dataset_hash.lower()
        == expected_dataset_hash.lower()
        == embedded_dataset_hash.lower(),
        "dataset SHA-256 differs between metadata and resolved config",
    )
    dataset_value = data.get("extracted_path")
    require(
        isinstance(dataset_value, str) and bool(dataset_value.strip()),
        "configured extracted dataset path is missing",
    )
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = BASELINE_ROOT / dataset_path
    dataset_path = dataset_path.expanduser().resolve()
    require(dataset_path.is_file(), f"pretraining dataset no longer exists: {dataset_path}")
    actual_dataset_hash = sha256_file(dataset_path)
    require(
        actual_dataset_hash.lower() == expected_dataset_hash.lower(),
        "actual pretraining dataset SHA-256 differs from metadata/config",
    )

    metrics = load_pretraining_metrics(metrics_path)
    final_metrics = verify_metrics(metrics, resolved)
    summary = load_json_mapping(summary_path)
    require(summary.get("status") == "completed", "summary status must be completed")
    require(summary.get("run_id") == run["id"], "summary run ID differs")
    require(summary.get("epochs") == pretraining["epochs"], "summary epochs differ from config")
    require(
        summary.get("optimizer_steps") == final_metrics["optimizer_steps"],
        "summary optimizer steps differ from final metrics",
    )

    configured_checkpoint_dir = Path(checkpoint_config.get("directory", ""))
    checkpoint_dir = resolve_inside(
        configured_checkpoint_dir,
        BASELINE_ROOT,
        "checkpoint.directory",
    )
    require(checkpoint_dir == run_dir / "checkpoints", "checkpoint directory differs from run directory")
    checkpoint_0 = checkpoint_dir / checkpoint_config.get("checkpoint_0_filename", "")
    best = checkpoint_dir / checkpoint_config.get("best_filename", "")
    require(checkpoint_0.name == "checkpoint_0.pth.tar", "checkpoint_0 filename is invalid")
    require(best.name == "best.pth.tar", "best checkpoint filename is invalid")
    require(checkpoint_0.is_file(), f"missing checkpoint: {checkpoint_0}")
    require(best.is_file(), f"missing checkpoint: {best}")
    checkpoint_hash = sha256_file(checkpoint_0)
    best_hash = sha256_file(best)
    require(checkpoint_hash == best_hash, "checkpoint_0 and best SHA-256 differ")
    require(
        summary.get("checkpoint_0_sha256") == checkpoint_hash,
        "summary checkpoint_0 SHA-256 differs from actual file",
    )
    require(
        summary.get("best_sha256") == best_hash,
        "summary best SHA-256 differs from actual file",
    )
    summary_checkpoint = configured_artifact(
        run_dir, summary.get("checkpoint_0_path"), "summary.checkpoint_0_path"
    )
    summary_best = configured_artifact(
        run_dir, summary.get("best_path"), "summary.best_path"
    )
    require(summary_checkpoint == checkpoint_0, "summary checkpoint_0 path differs")
    require(summary_best == best, "summary best path differs")

    expected_shapes = expected_model_shapes(model)
    checkpoint_state = verify_checkpoint(checkpoint_0, expected_shapes)
    best_state = verify_checkpoint(best, expected_shapes)
    require(
        set(checkpoint_state) == set(best_state),
        "checkpoint_0 and best parameter names differ",
    )

    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": run["id"],
        "epochs": pretraining["epochs"],
        "optimizer_steps": final_metrics["optimizer_steps"],
        "dataset_verified": True,
        "metadata_verified": True,
        "metrics_verified": True,
        "checkpoints_verified": True,
        "temporary_files_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(json.dumps(verify_pretraining(args.run_dir), indent=2, ensure_ascii=False))
        return 0
    except (
        VerificationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"Pretraining verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
