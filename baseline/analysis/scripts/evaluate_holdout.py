#!/usr/bin/env python3
"""Evaluate baseline checkpoints on the frozen fixed_holdout_v1 states."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_fixed_basket import discover_checkpoints
from holdout_common import (
    DEFAULT_BASELINE_RUN_DIR,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_CHECKPOINTS,
    HoldoutError,
    atomic_write_csv,
    atomic_write_json,
    build_network,
    load_json,
    load_jsonl,
    load_npz,
    load_protocol,
    sha256_file,
)
from verify_holdout import verify_holdout


CHECKPOINT_FIELDS = (
    "checkpoint",
    "gpu_hours",
    "states",
    "games",
    "holdout_policy_loss",
    "holdout_policy_loss_ci_low",
    "holdout_policy_loss_ci_high",
    "holdout_value_loss",
    "holdout_value_loss_ci_low",
    "holdout_value_loss_ci_high",
    "holdout_total_loss",
    "logged_train_policy_loss",
    "logged_train_value_loss",
    "approx_policy_gap",
    "approx_value_gap",
    "evaluation_seconds",
    "checkpoint_sha256",
    "dataset_content_sha256",
)

TRAJECTORY_FIELDS = (
    "checkpoint",
    "game_id",
    "states",
    "policy_loss",
    "value_loss",
    "total_loss",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def loss_arrays(
    network,
    boards: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    valids: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-state losses using the exact valid-mask training definition."""
    count = int(boards.shape[0])
    policy_losses = np.empty(count, dtype=np.float64)
    value_losses = np.empty(count, dtype=np.float64)
    torch_device = torch.device(device)
    network.nnet.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch_boards = torch.as_tensor(
                boards[start:stop], dtype=torch.float32, device=torch_device
            )
            batch_targets = torch.as_tensor(
                policies[start:stop], dtype=torch.float32, device=torch_device
            )
            batch_values = torch.as_tensor(
                values[start:stop], dtype=torch.float32, device=torch_device
            )
            batch_valids = torch.as_tensor(
                valids[start:stop], dtype=torch.float32, device=torch_device
            )
            logits, predictions = network._fwd(batch_boards, True)
            logits = logits * batch_valids
            logits[batch_valids == 0.0] = float("-inf")
            probabilities = F.softmax(logits, dim=1)
            policy_batch = (
                -batch_targets * torch.log(probabilities + 1.0e-8)
            ).sum(dim=-1)
            value_batch = (batch_values - predictions.view(-1)) ** 2
            policy_losses[start:stop] = policy_batch.float().cpu().numpy()
            value_losses[start:stop] = value_batch.float().cpu().numpy()
    total_losses = policy_losses + value_losses
    if not (
        np.all(np.isfinite(policy_losses))
        and np.all(np.isfinite(value_losses))
        and np.all(np.isfinite(total_losses))
    ):
        raise HoldoutError("checkpoint evaluation produced non-finite losses")
    return policy_losses, value_losses, total_losses


def trajectory_rows(
    checkpoint: int,
    game_ids: np.ndarray,
    policy_losses: np.ndarray,
    value_losses: np.ndarray,
    total_losses: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for game_id in sorted(int(item) for item in np.unique(game_ids)):
        selected = game_ids == game_id
        states = int(selected.sum())
        rows.append(
            {
                "checkpoint": checkpoint,
                "game_id": game_id,
                "states": states,
                "policy_loss": float(policy_losses[selected].mean()),
                "value_loss": float(value_losses[selected].mean()),
                "total_loss": float(total_losses[selected].mean()),
            }
        )
    return rows


def cluster_bootstrap_intervals(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, tuple[float, float]]:
    """Bootstrap whole games, then compute state-weighted losses in each draw."""
    if not rows:
        raise HoldoutError("cannot bootstrap an empty trajectory table")
    counts = np.asarray([row["states"] for row in rows], dtype=np.float64)
    metrics = ("policy_loss", "value_loss", "total_loss")
    sums = {
        metric: np.asarray([row[metric] for row in rows], dtype=np.float64) * counts
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(resamples, len(rows)))
    denominators = counts[draws].sum(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    intervals = {}
    for metric in metrics:
        estimates = sums[metric][draws].sum(axis=1) / denominators
        low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
        intervals[metric] = (float(low), float(high))
    return intervals


def _read_training_metrics(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise HoldoutError(f"training metrics not found: {path}")
    rows = load_jsonl(path)
    by_iteration: dict[int, dict[str, Any]] = {}
    for row in rows:
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise HoldoutError("training metrics contain an invalid iteration")
        if iteration in by_iteration:
            raise HoldoutError(f"duplicate training metrics iteration: {iteration}")
        by_iteration[iteration] = row
    return by_iteration


def _training_columns(
    checkpoint: int, training: dict[int, dict[str, Any]], holdout: dict[str, float]
) -> dict[str, Any]:
    if checkpoint == 0:
        return {
            "logged_train_policy_loss": None,
            "logged_train_value_loss": None,
            "approx_policy_gap": None,
            "approx_value_gap": None,
            "gpu_hours": 0.0,
        }
    row = training.get(checkpoint)
    if row is None:
        raise HoldoutError(f"training metrics lack checkpoint iteration {checkpoint}")
    output: dict[str, Any] = {}
    for metric, logged_key, gap_key in (
        ("policy_loss", "logged_train_policy_loss", "approx_policy_gap"),
        ("value_loss", "logged_train_value_loss", "approx_value_gap"),
    ):
        train_value = row.get(metric)
        if not isinstance(train_value, (int, float)) or not np.isfinite(train_value):
            raise HoldoutError(f"training {metric} is invalid at iteration {checkpoint}")
        output[logged_key] = float(train_value)
        output[gap_key] = float(holdout[metric] - train_value)
    gpu_hours = row.get("cumulative_gpu_hours")
    if not isinstance(gpu_hours, (int, float)) or not np.isfinite(gpu_hours):
        raise HoldoutError(f"cumulative_gpu_hours is invalid at iteration {checkpoint}")
    output["gpu_hours"] = float(gpu_hours)
    return output


def _selected_checkpoints(
    requested: Iterable[int] | None, configured: list[int]
) -> list[int]:
    if requested is None:
        return list(configured)
    selected = list(requested)
    if len(selected) != len(set(selected)):
        raise HoldoutError("--checkpoints contains duplicates")
    if any(item not in configured for item in selected):
        raise HoldoutError("--checkpoints contains an iteration outside holdout_v1.yaml")
    return selected


def evaluate(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    states_path = args.dataset.expanduser().resolve()
    verification = verify_holdout(
        args.config, states_path.parent, dataset_path=states_path
    )
    if args.verify_dataset_hash and verification.get("status") != "passed":
        raise HoldoutError("dataset hash verification did not pass")
    arrays = load_npz(states_path)
    checkpoints = _selected_checkpoints(
        args.checkpoints, list(protocol["evaluation"]["checkpoints"])
    )

    metadata = load_json(run_dir / "run_metadata.json")
    discovered = discover_checkpoints(
        {"checkpoints": checkpoints}, run_dir, metadata
    )
    source_sha = verification["source_checkpoint_sha256"]
    checkpoint_zero = next(
        (item for item in discovered if item["iteration"] == 0),
        None,
    )
    if checkpoint_zero is None:
        checkpoint_zero = discover_checkpoints(
            {"checkpoints": [0]}, run_dir, metadata
        )[0]
    if checkpoint_zero["sha256"] != source_sha:
        raise HoldoutError(
            "baseline checkpoint 0 differs from the checkpoint used to generate hold-out"
        )
    training_metrics = (
        args.training_metrics.expanduser().resolve()
        if args.training_metrics is not None
        else run_dir / "metrics.jsonl"
    )
    training = _read_training_metrics(training_metrics)
    dataset_content_sha256 = verification["dataset_content_sha256"]

    all_trajectory_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for entry in discovered:
        checkpoint = int(entry["iteration"])
        print(f"Evaluating checkpoint {checkpoint}...", flush=True)
        evaluation_started = time.perf_counter()
        _, network = build_network(
            protocol, Path(entry["path"]), device=args.device
        )
        policy_loss, value_loss, total_loss = loss_arrays(
            network,
            arrays["boards"],
            arrays["policies"],
            arrays["values"],
            arrays["valids"],
            batch_size=int(protocol["evaluation"]["batch_size"]),
            device=args.device,
        )
        rows = trajectory_rows(
            checkpoint,
            arrays["game_ids"],
            policy_loss,
            value_loss,
            total_loss,
        )
        all_trajectory_rows.extend(rows)
        intervals = cluster_bootstrap_intervals(
            rows,
            resamples=int(protocol["evaluation"]["bootstrap_resamples"]),
            seed=int(protocol["evaluation"]["bootstrap_seed"]) + checkpoint,
            confidence_level=float(protocol["evaluation"]["confidence_level"]),
        )
        means = {
            "policy_loss": float(policy_loss.mean()),
            "value_loss": float(value_loss.mean()),
            "total_loss": float(total_loss.mean()),
        }
        row = {
            "checkpoint": checkpoint,
            "states": int(arrays["boards"].shape[0]),
            "games": int(verification["games"]),
            "holdout_policy_loss": means["policy_loss"],
            "holdout_policy_loss_ci_low": intervals["policy_loss"][0],
            "holdout_policy_loss_ci_high": intervals["policy_loss"][1],
            "holdout_value_loss": means["value_loss"],
            "holdout_value_loss_ci_low": intervals["value_loss"][0],
            "holdout_value_loss_ci_high": intervals["value_loss"][1],
            "holdout_total_loss": means["total_loss"],
            "checkpoint_sha256": entry["sha256"],
            "dataset_content_sha256": dataset_content_sha256,
        }
        row.update(_training_columns(checkpoint, training, means))
        row["evaluation_seconds"] = time.perf_counter() - evaluation_started
        checkpoint_rows.append(row)
        print(
            f"Checkpoint {checkpoint} completed: "
            f"policy_loss={row['holdout_policy_loss']:.6f}, "
            f"value_loss={row['holdout_value_loss']:.6f}, "
            f"seconds={row['evaluation_seconds']:.3f}.",
            flush=True,
        )
        del network
        if args.device == "cuda":
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_metrics.csv"
    trajectory_path = output_dir / "trajectory_checkpoint_metrics.csv"
    summary_path = output_dir / "summary.json"
    atomic_write_csv(checkpoint_path, checkpoint_rows, CHECKPOINT_FIELDS)
    atomic_write_csv(trajectory_path, all_trajectory_rows, TRAJECTORY_FIELDS)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "completed",
        "completed_at_utc": utc_now(),
        "evaluation_scope": (
            "formal" if checkpoints == EXPECTED_CHECKPOINTS else "explicit_subset"
        ),
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "games": int(protocol["games"]),
        "states": int(arrays["boards"].shape[0]),
        "dataset": {
            "path": states_path.as_posix(),
            "sha256": sha256_file(states_path),
            "content_sha256": dataset_content_sha256,
            "hash_verification_requested": bool(args.verify_dataset_hash),
            "hash_verification_status": "passed",
            "verification": verification,
        },
        "loss_definition": {
            "policy_loss": (
                "mean state-level cross entropy against the MCTS root policy, after "
                "the same valid-action masking and softmax used by training"
            ),
            "value_loss": "mean state-level squared error",
            "total_loss": "policy_loss + value_loss",
            "gap_name": "approximate online train–hold-out gap",
            "gap_formula": (
                "holdout loss minus the same-iteration logged online training loss"
            ),
            "gap_limitation": (
                "Only final latest.examples was retained. Per-checkpoint replay snapshots "
                "are unavailable, so these gaps are not posterior losses recomputed on each "
                "checkpoint's complete training replay."
            ),
        },
        "confidence_interval": {
            "method": "cluster bootstrap over complete game trajectories",
            "resamples": int(protocol["evaluation"]["bootstrap_resamples"]),
            "seed_per_checkpoint": (
                f"{protocol['evaluation']['bootstrap_seed']} + checkpoint_iteration"
            ),
            "confidence_level": float(protocol["evaluation"]["confidence_level"]),
        },
        "outputs": {
            "checkpoint_metrics": checkpoint_path.as_posix(),
            "trajectory_checkpoint_metrics": trajectory_path.as_posix(),
        },
    }
    atomic_write_json(summary_path, summary)
    print(f"Checkpoints evaluated: {len(checkpoints)}")
    print(f"States evaluated per checkpoint: {arrays['boards'].shape[0]}")
    print("Output status: completed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_OUTPUT_DIR / "states.npz"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_BASELINE_RUN_DIR)
    parser.add_argument(
        "--training-metrics",
        type=Path,
    )
    parser.add_argument("--verify-dataset-hash", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoints", type=int, nargs="+")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        evaluate(parse_args(argv))
    except (HoldoutError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
