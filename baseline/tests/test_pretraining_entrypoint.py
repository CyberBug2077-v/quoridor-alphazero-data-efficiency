from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_pretraining
from runtime.metadata import sha256_file


class MockPretrainingNetwork:
    def __init__(self, metrics: dict[str, Any], *, fail_save: bool = False):
        self.nnet = torch.nn.Linear(2, 1)
        self.metrics = metrics
        self.fail_save = fail_save

    def train(self, examples, **kwargs):
        assert examples
        assert kwargs["available_examples"] == len(examples)
        return dict(self.metrics)

    def save_checkpoint(self, folder: str, filename: str) -> None:
        destination = Path(folder) / filename
        torch.save({"state_dict": self.nnet.state_dict()}, destination)
        if self.fail_save:
            raise RuntimeError("synthetic checkpoint save failure")


def successful_metrics() -> dict[str, Any]:
    return {
        "available_examples": 2,
        "examples_used": 2,
        "samples_seen": 2,
        "training_batches": 1,
        "optimizer_steps": 1,
        "effective_batch_size": 2,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "micro_batches_processed": 2,
        "policy_loss": 2.0,
        "value_loss": 0.5,
        "total_loss": 2.5,
        "mean_grad_norm": 1.25,
        "max_grad_norm": 1.25,
    }


def make_pretraining_config(tmp_path: Path, run_id: str) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "configs"
    data_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    archive = data_dir / "heuristic_games.zip"
    archive.write_bytes(b"synthetic archive")
    policy = np.zeros(136, dtype=np.float32)
    policy[0] = 1.0
    valids = np.ones(136, dtype=np.uint8)
    examples = [
        (np.zeros((4, 17, 17), dtype=np.uint8), policy, 1.0, valids),
        (np.ones((4, 17, 17), dtype=np.uint8), policy, -1.0, valids),
    ]
    dataset = data_dir / "heuristic_games.pkl"
    with dataset.open("wb") as output:
        pickle.dump(examples, output)
    output_dir = tmp_path / "outputs" / run_id
    config = {
        "run": {
            "id": run_id,
            "seed": 1001,
            "device": "cuda",
            "gpu_index": 0,
            "deterministic": True,
            "output_dir": f"outputs/{run_id}",
        },
        "data": {
            "archive_path": "data/heuristic_games.zip",
            "extracted_path": "data/heuristic_games.pkl",
            "expected_sha256": sha256_file(dataset),
            "validate_before_training": True,
        },
        "model": {
            "board_size": 9,
            "num_channels": 8,
            "num_res_blocks": 1,
            "attn_depth": 0,
            "num_heads": 1,
            "se_enabled": False,
            "dropout": 0.0,
        },
        "pretraining": {
            "epochs": 1,
            "batch_size": 2,
            "micro_batch_size": 1,
            "learning_rate": 0.0005,
            "optimizer": "adamw",
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "amp": False,
            "amp_dtype": "bf16",
        },
        "checkpoint": {
            "directory": f"outputs/{run_id}/checkpoints",
            "checkpoint_0_filename": "checkpoint_0.pth.tar",
            "best_filename": "best.pth.tar",
            "save_rng_state": True,
            "compute_sha256": True,
        },
        "logging": {
            "metrics_file": "pretraining_metrics.jsonl",
            "metadata_file": "run_metadata.json",
            "evaluation_file": "evaluation.json",
            "summary_file": "summary.json",
            "log_every_optimizer_steps": 1,
        },
    }
    config_path = config_dir / f"{run_id}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return config_path, output_dir


def patch_pretraining_runtime(
    monkeypatch,
    tmp_path: Path,
    metrics: dict[str, Any],
    *,
    fail_save: bool = False,
) -> list[tuple[int, bool]]:
    seed_calls: list[tuple[int, bool]] = []

    def fixed_seed(seed: int, deterministic: bool) -> None:
        seed_calls.append((seed, deterministic))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    monkeypatch.setattr(run_pretraining, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(run_pretraining, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(run_pretraining, "initialize_cuda", lambda resolved: None)
    monkeypatch.setattr(run_pretraining, "set_seed", fixed_seed)
    monkeypatch.setattr(
        run_pretraining,
        "create_network",
        lambda resolved: (None, MockPretrainingNetwork(metrics, fail_save=fail_save)),
    )
    monkeypatch.setattr(run_pretraining.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        run_pretraining.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(
        run_pretraining.torch.cuda, "max_memory_allocated", lambda: 64 * 1024**2
    )
    return seed_calls


def invoke_fresh(monkeypatch, config_path: Path) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pretraining.py", "fresh", "--config", str(config_path)],
    )
    return run_pretraining.main()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_fresh_entrypoint_creates_complete_artifact_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, output_dir = make_pretraining_config(
        tmp_path, "pretraining_entrypoint_seed1001"
    )
    seed_calls = patch_pretraining_runtime(
        monkeypatch, tmp_path, successful_metrics()
    )

    assert invoke_fresh(monkeypatch, config_path) == 0

    expected = {
        "resolved_config.yaml",
        "run_metadata.json",
        "run.log",
        "pretraining_metrics.jsonl",
        "summary.json",
        "checkpoints/checkpoint_0.pth.tar",
        "checkpoints/best.pth.tar",
    }
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert expected <= actual
    assert seed_calls == [(1001, True)]
    resolved = yaml.safe_load(
        (output_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (output_dir / "pretraining_metrics.jsonl").read_text(encoding="utf-8")
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = output_dir / "checkpoints" / "checkpoint_0.pth.tar"
    best = output_dir / "checkpoints" / "best.pth.tar"
    assert resolved["mode"] == "pretraining"
    assert metadata["run_type"] == "pretraining"
    assert metadata["input_hashes"]["pretraining_dataset"]
    assert metrics["optimizer_steps"] == 1
    assert metrics["effective_batch_size"] == 2
    assert metrics["micro_batch_size"] == 1
    assert metrics["gradient_accumulation_steps"] == 2
    assert metrics["micro_batches_processed"] == 2
    assert summary["status"] == "completed"
    assert sha256_file(checkpoint) == sha256_file(best)
    assert metadata["output_hashes"]["checkpoint_0"] == sha256_file(checkpoint)
    assert metadata["output_hashes"]["best"] == sha256_file(best)
    assert summary["checkpoint_0_sha256"] == summary["best_sha256"]
    assert not any(path.name.endswith(".tmp") for path in output_dir.rglob("*"))

    before = file_hashes(output_dir)
    assert invoke_fresh(monkeypatch, config_path) == 2
    assert file_hashes(output_dir) == before


def test_fresh_entrypoint_rejects_zero_optimizer_steps(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, output_dir = make_pretraining_config(
        tmp_path, "pretraining_zero_steps_seed1001"
    )
    metrics = successful_metrics()
    metrics.update(
        {
            "samples_seen": 0,
            "training_batches": 0,
            "optimizer_steps": 0,
            "policy_loss": None,
            "value_loss": None,
            "total_loss": None,
            "mean_grad_norm": None,
            "max_grad_norm": None,
        }
    )
    patch_pretraining_runtime(monkeypatch, tmp_path, metrics)

    assert invoke_fresh(monkeypatch, config_path) == 2
    assert (output_dir / "resolved_config.yaml").is_file()
    assert (output_dir / "run_metadata.json").is_file()
    assert (output_dir / "run.log").is_file()
    assert not (output_dir / "pretraining_metrics.jsonl").exists()
    assert not (output_dir / "checkpoints" / "checkpoint_0.pth.tar").exists()
    assert not (output_dir / "summary.json").exists()


def test_checkpoint_save_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, output_dir = make_pretraining_config(
        tmp_path, "pretraining_save_failure_seed1001"
    )
    patch_pretraining_runtime(
        monkeypatch, tmp_path, successful_metrics(), fail_save=True
    )

    assert invoke_fresh(monkeypatch, config_path) == 2
    assert not any(
        path.is_file() and path.name.endswith(".tmp")
        for path in output_dir.rglob("*")
    )
    assert not (output_dir / "checkpoints" / "checkpoint_0.pth.tar").exists()
