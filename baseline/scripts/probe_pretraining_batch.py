#!/usr/bin/env python3
"""Probe one formal-model pretraining batch without creating experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_pretraining import (
    BASELINE_ROOT,
    DEFAULT_CONFIG,
    create_network,
    initialize_cuda,
    load_and_validate_dataset,
    require_inside_baseline,
    set_seed,
)
from runtime.artifacts import atomic_write_json
from runtime.config import ConfigError, load_yaml, resolve_pretraining_config


MINIMUM_FREE_BYTES = 1024**3
MINIMUM_FREE_FRACTION = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    if args.effective_batch_size is not None and args.effective_batch_size < 1:
        parser.error("--effective-batch-size must be >= 1")
    if args.micro_batch_size is not None and args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be >= 1")
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.trial < 1:
        parser.error("--trial must be >= 1")
    return args


def tree_snapshot(path: Path) -> dict[str, tuple[int, int]] | None:
    if not path.exists():
        return None
    return {
        item.relative_to(path).as_posix(): (item.stat().st_size, item.stat().st_mtime_ns)
        for item in path.rglob("*")
        if item.is_file()
    }


def memory_margin(
    *,
    peak_allocated_bytes: int,
    total_memory_bytes: int,
    peak_reserved_bytes: int | None = None,
    free_after_bytes: int | None = None,
) -> dict[str, Any]:
    peak_reserved = (
        peak_allocated_bytes
        if peak_reserved_bytes is None
        else max(peak_allocated_bytes, peak_reserved_bytes)
    )
    candidates = [max(0, total_memory_bytes - peak_reserved)]
    if free_after_bytes is not None:
        candidates.append(max(0, free_after_bytes))
    remaining = min(candidates)
    fraction = remaining / total_memory_bytes if total_memory_bytes else 0.0
    return {
        "estimated_remaining_bytes": remaining,
        "estimated_remaining_mb": remaining / (1024**2),
        "estimated_remaining_fraction": fraction,
        "basis": "min(total_minus_peak_reserved, device_free_after)",
        "minimum_required_bytes": MINIMUM_FREE_BYTES,
        "minimum_required_fraction": MINIMUM_FREE_FRACTION,
        "ok": remaining >= MINIMUM_FREE_BYTES and fraction >= MINIMUM_FREE_FRACTION,
    }


def validate_formal_model(resolved: dict[str, Any]) -> None:
    expected = {
        "num_channels": 128,
        "num_res_blocks": 6,
        "attn_depth": 1,
    }
    actual = {key: resolved["model"].get(key) for key in expected}
    if actual != expected:
        raise ConfigError(
            f"probe requires the formal model architecture: expected={expected} actual={actual}"
        )


def validate_requested_batches(
    effective_batch_size: int,
    micro_batch_size: int,
) -> int:
    if effective_batch_size < 1:
        raise ConfigError("effective batch size must be >= 1")
    if micro_batch_size < 1:
        raise ConfigError("micro batch size must be >= 1")
    if micro_batch_size > effective_batch_size:
        raise ConfigError("micro batch size must be <= effective batch size")
    if effective_batch_size % micro_batch_size != 0:
        raise ConfigError("effective batch size must be divisible by micro batch size")
    return effective_batch_size // micro_batch_size


def fixed_batch(
    examples: list | tuple,
    *,
    batch_size: int,
    seed: int,
) -> tuple[list, str]:
    if batch_size > len(examples):
        raise ConfigError(
            f"requested batch size {batch_size} exceeds dataset size {len(examples)}"
        )
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(examples), size=batch_size, replace=False)
    index_hash = hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()
    return [examples[int(index)] for index in indices], index_hash


def batch_tensors(batch: list, device: torch.device):
    columns = list(zip(*batch))
    if len(columns) < 4:
        raise ConfigError("formal pretraining samples must include legal-action masks")
    boards = torch.as_tensor(
        np.asarray(columns[0], dtype=np.uint8), dtype=torch.float32, device=device
    )
    policies = torch.as_tensor(
        np.asarray(columns[1]), dtype=torch.float32, device=device
    )
    values = torch.as_tensor(
        np.asarray(columns[2]), dtype=torch.float32, device=device
    )
    valids = torch.as_tensor(
        np.asarray(columns[3], dtype=np.uint8), dtype=torch.float32, device=device
    )
    return boards, policies, values, valids


def execute_probe(
    resolved: dict[str, Any],
    examples: list | tuple,
    *,
    effective_batch_size: int,
    micro_batch_size: int,
    steps: int,
    trial: int,
) -> dict[str, Any]:
    seed = int(resolved["run"]["seed"])
    accumulation_steps = validate_requested_batches(
        effective_batch_size, micro_batch_size
    )
    batch, sample_index_sha256 = fixed_batch(
        examples, batch_size=effective_batch_size, seed=seed
    )
    _, network = create_network(resolved)
    device = torch.device("cuda", int(resolved["run"]["gpu_index"]))
    properties = torch.cuda.get_device_properties(device)
    total_memory = int(properties.total_memory)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "probe_only": True,
        "run_id": resolved["run"]["id"],
        "trial": trial,
        "configured_effective_batch_size": resolved["pretraining"]["batch_size"],
        "configured_micro_batch_size": resolved["pretraining"]["micro_batch_size"],
        "requested_effective_batch_size": effective_batch_size,
        "requested_micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "requested_optimizer_steps": steps,
        "optimizer_steps": 0,
        "micro_batches_processed": 0,
        "samples_seen": 0,
        "sample_index_sha256": sample_index_sha256,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": total_memory,
        "oom": False,
        "policy_loss": None,
        "value_loss": None,
        "total_loss": None,
        "gradient_norm": None,
        "execution_seconds": None,
        "peak_gpu_memory_bytes": None,
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_reserved_bytes": None,
        "peak_gpu_memory_reserved_mb": None,
        "gpu_free_memory_before_bytes": None,
        "gpu_free_memory_after_bytes": None,
        "memory_margin": None,
        "acceptance": {
            "effective_batch_matches_config": False,
            "optimizer_steps_ok": False,
            "micro_batch_count_ok": False,
            "samples_seen_ok": False,
            "finite_metrics": False,
            "memory_margin_ok": False,
            "no_formal_checkpoint": False,
            "passed": False,
        },
    }
    optimizer = torch.optim.AdamW(
        network.nnet.parameters(),
        lr=resolved["pretraining"]["learning_rate"],
        weight_decay=resolved["pretraining"]["weight_decay"],
    )
    try:
        network.nnet.train()
        torch.cuda.synchronize(device)
        result["gpu_free_memory_before_bytes"] = int(torch.cuda.mem_get_info(device)[0])
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        total_loss_sum = 0.0
        gradient_norm_sum = 0.0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            step_policy_loss = 0.0
            step_value_loss = 0.0
            step_total_loss = 0.0
            for micro_index in range(accumulation_steps):
                start = micro_index * micro_batch_size
                stop = start + micro_batch_size
                boards, policies, values, valids = batch_tensors(
                    batch[start:stop], device
                )
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=resolved["pretraining"]["amp"],
                    dtype=(
                        torch.bfloat16
                        if resolved["pretraining"]["amp_dtype"] == "bf16"
                        else torch.float16
                    ),
                ):
                    logits, predicted_values = network.nnet(boards, logits=True)
                    masked_logits = logits.masked_fill(valids == 0, float("-inf"))
                    probabilities = torch.softmax(masked_logits, dim=1)
                    policy_loss = network.loss_pi(policies, probabilities)
                    value_loss = network.loss_v(values, predicted_values)
                    total_loss = policy_loss + value_loss
                    backward_loss = total_loss / accumulation_steps
                if not torch.isfinite(total_loss):
                    raise RuntimeError("probe produced a non-finite loss")
                if (
                    resolved["pretraining"]["amp"]
                    and resolved["pretraining"]["amp_dtype"] == "fp16"
                ):
                    network.scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
                result["micro_batches_processed"] += 1
                step_policy_loss += float(policy_loss.detach()) / accumulation_steps
                step_value_loss += float(value_loss.detach()) / accumulation_steps
                step_total_loss += float(total_loss.detach()) / accumulation_steps
            if (
                resolved["pretraining"]["amp"]
                and resolved["pretraining"]["amp_dtype"] == "fp16"
            ):
                network.scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                network.nnet.parameters(),
                resolved["pretraining"]["gradient_clip"],
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("probe produced a non-finite gradient norm")
            if (
                resolved["pretraining"]["amp"]
                and resolved["pretraining"]["amp_dtype"] == "fp16"
            ):
                network.scaler.step(optimizer)
                network.scaler.update()
            else:
                optimizer.step()
            result["optimizer_steps"] += 1
            result["samples_seen"] += effective_batch_size
            policy_loss_sum += step_policy_loss
            value_loss_sum += step_value_loss
            total_loss_sum += step_total_loss
            gradient_norm_sum += float(gradient_norm.detach())
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_after = int(torch.cuda.mem_get_info(device)[0])
        result.update(
            {
                "policy_loss": policy_loss_sum / steps,
                "value_loss": value_loss_sum / steps,
                "total_loss": total_loss_sum / steps,
                "gradient_norm": gradient_norm_sum / steps,
                "execution_seconds": elapsed,
                "peak_gpu_memory_bytes": peak_memory,
                "peak_gpu_memory_mb": peak_memory / (1024**2),
                "peak_gpu_memory_reserved_bytes": peak_reserved,
                "peak_gpu_memory_reserved_mb": peak_reserved / (1024**2),
                "gpu_free_memory_after_bytes": free_after,
                "memory_margin": memory_margin(
                    peak_allocated_bytes=peak_memory,
                    total_memory_bytes=total_memory,
                    peak_reserved_bytes=peak_reserved,
                    free_after_bytes=free_after,
                ),
            }
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        result["oom"] = True
        result["error"] = str(exc)
        result["execution_seconds"] = (
            time.perf_counter() - started if "started" in locals() else 0.0
        )
        result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_gpu_memory_mb"] = result["peak_gpu_memory_bytes"] / (1024**2)
        result["peak_gpu_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        result["peak_gpu_memory_reserved_mb"] = (
            result["peak_gpu_memory_reserved_bytes"] / (1024**2)
        )
        result["gpu_free_memory_after_bytes"] = int(torch.cuda.mem_get_info(device)[0])
        result["memory_margin"] = memory_margin(
            peak_allocated_bytes=result["peak_gpu_memory_bytes"],
            total_memory_bytes=total_memory,
            peak_reserved_bytes=result["peak_gpu_memory_reserved_bytes"],
            free_after_bytes=result["gpu_free_memory_after_bytes"],
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    finite_values = all(
        isinstance(result[field], (int, float))
        and math.isfinite(float(result[field]))
        for field in ("policy_loss", "value_loss", "total_loss", "gradient_norm")
    )
    result["acceptance"].update(
        {
            "effective_batch_matches_config": (
                effective_batch_size == resolved["pretraining"]["batch_size"]
            ),
            "optimizer_steps_ok": result["optimizer_steps"] == steps,
            "micro_batch_count_ok": (
                result["micro_batches_processed"]
                == steps * accumulation_steps
            ),
            "samples_seen_ok": (
                result["samples_seen"] == steps * effective_batch_size
            ),
            "finite_metrics": finite_values,
            "memory_margin_ok": bool(result["memory_margin"] and result["memory_margin"]["ok"]),
        }
    )
    return result


def main() -> int:
    args = parse_args()
    try:
        resolved = resolve_pretraining_config(load_yaml(args.config), args.config)
        require_inside_baseline(resolved)
        validate_formal_model(resolved)
        effective_batch_size = (
            args.effective_batch_size
            or resolved["pretraining"]["batch_size"]
        )
        micro_batch_size = (
            args.micro_batch_size
            or resolved["pretraining"]["micro_batch_size"]
        )
        validate_requested_batches(effective_batch_size, micro_batch_size)
        output_path = (
            BASELINE_ROOT
            / "outputs"
            / "pretraining_probe"
            / (
                f"effective_{effective_batch_size}_micro_{micro_batch_size}_"
                f"steps_{args.steps}_trial_{args.trial}.json"
            )
        )
        if output_path.is_relative_to(resolved["_output_path"]):
            raise ConfigError("probe output must not be inside the formal pretraining output")
        formal_before = tree_snapshot(resolved["_output_path"])
        data_path = Path(resolved["data"]["_extracted_path"])
        data_before = (data_path.stat().st_size, data_path.stat().st_mtime_ns)
        initialize_cuda(resolved)
        set_seed(resolved["run"]["seed"], resolved["run"]["deterministic"])
        examples = load_and_validate_dataset(resolved)
        result = execute_probe(
            resolved,
            examples,
            effective_batch_size=effective_batch_size,
            micro_batch_size=micro_batch_size,
            steps=args.steps,
            trial=args.trial,
        )
        formal_unchanged = tree_snapshot(resolved["_output_path"]) == formal_before
        data_unchanged = (
            data_path.stat().st_size,
            data_path.stat().st_mtime_ns,
        ) == data_before
        no_formal_checkpoint = formal_unchanged and data_unchanged
        result["acceptance"]["no_formal_checkpoint"] = no_formal_checkpoint
        result["acceptance"]["passed"] = (
            result["acceptance"]["effective_batch_matches_config"]
            and result["acceptance"]["optimizer_steps_ok"]
            and result["acceptance"]["micro_batch_count_ok"]
            and result["acceptance"]["samples_seen_ok"]
            and result["acceptance"]["finite_metrics"]
            and result["acceptance"]["memory_margin_ok"]
            and no_formal_checkpoint
            and not result["oom"]
        )
        result["status"] = "passed" if result["acceptance"]["passed"] else "failed"
        result["dataset_validation"] = resolved["_dataset_validation"]
        atomic_write_json(output_path, result)
        print(output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["acceptance"]["passed"] else 2
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"Pretraining probe error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
