from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

import verify_pretraining
from runtime.artifacts import atomic_write_json, atomic_write_yaml
from runtime.metadata import sha256_file


def write_jsonl(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


@pytest.fixture
def pretraining_artifacts(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(verify_pretraining, "BASELINE_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "pretraining_test_seed1"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    dataset = tmp_path / "data" / "examples.pkl"
    dataset.parent.mkdir()
    dataset.write_bytes(b"frozen pretraining dataset")
    dataset_hash = sha256_file(dataset)

    from quoridor.QuoridorGame import QuoridorGame
    from quoridor.pytorch.QuoridorNNet import QuoridorNNet
    from utils import dotdict

    model_config = {
        "board_size": 9,
        "num_channels": 8,
        "num_res_blocks": 1,
        "attn_depth": 0,
        "num_heads": 1,
        "se_enabled": False,
        "dropout": 0.3,
    }
    state_dict = QuoridorNNet(
        QuoridorGame(9), dotdict(model_config)
    ).state_dict()
    checkpoint = checkpoint_dir / "checkpoint_0.pth.tar"
    best = checkpoint_dir / "best.pth.tar"
    torch.save({"state_dict": state_dict}, checkpoint)
    shutil.copyfile(checkpoint, best)
    checkpoint_hash = sha256_file(checkpoint)
    resolved = {
        "schema_version": 1,
        "mode": "pretraining",
        "run": {
            "id": run_dir.name,
            "seed": 1,
            "device": "cuda",
            "gpu_index": 0,
        },
        "data": {
            "extracted_path": dataset.as_posix(),
            "expected_sha256": dataset_hash,
        },
        "model": model_config,
        "pretraining": {
            "epochs": 10,
            "batch_size": 2,
            "micro_batch_size": 1,
        },
        "checkpoint": {
            "directory": f"outputs/{run_dir.name}/checkpoints",
            "checkpoint_0_filename": checkpoint.name,
            "best_filename": best.name,
        },
        "logging": {
            "metrics_file": "pretraining_metrics.jsonl",
            "metadata_file": "run_metadata.json",
            "summary_file": "summary.json",
        },
    }
    atomic_write_yaml(run_dir / "resolved_config.yaml", resolved)
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "run_type": "pretraining",
            "git": {"commit": "abc", "dirty": False},
            "cuda": {
                "torch_importable": True,
                "available": True,
                "torch_cuda_version": "12.8",
                "cudnn_version": 91002,
                "device_count": 1,
                "devices": [
                    {
                        "index": 0,
                        "name": "test GPU",
                        "total_memory_bytes": 8 * 1024**3,
                        "compute_capability": "8.9",
                    }
                ],
            },
            "input_hashes": {"pretraining_dataset": dataset_hash},
            "output_hashes": {
                "checkpoint_0": checkpoint_hash,
                "best": checkpoint_hash,
            },
            "resolved_config": resolved,
        },
    )
    write_jsonl(
        run_dir / "pretraining_metrics.jsonl",
        {
            "schema_version": 1,
            "phase": "pretraining",
            "epochs": 10,
            "optimizer_steps": 2,
            "samples_seen": 4,
            "effective_batch_size": 2,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "micro_batches_processed": 4,
            "policy_loss": 2.0,
            "value_loss": 0.5,
            "total_loss": 2.5,
            "mean_grad_norm": 1.0,
            "max_grad_norm": 1.5,
            "peak_gpu_memory_mb": 512.0,
        },
    )
    atomic_write_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "run_id": run_dir.name,
            "epochs": 10,
            "optimizer_steps": 2,
            "checkpoint_0_path": checkpoint.as_posix(),
            "checkpoint_0_sha256": checkpoint_hash,
            "best_path": best.as_posix(),
            "best_sha256": checkpoint_hash,
        },
    )
    return run_dir


def read_metrics(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "pretraining_metrics.jsonl").read_text(encoding="utf-8")
    )


def read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def test_complete_pretraining_artifacts_pass_without_modification(
    pretraining_artifacts: Path,
) -> None:
    root = pretraining_artifacts.parents[1]
    before = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }

    result = verify_pretraining.verify_pretraining(pretraining_artifacts)

    after = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert result["status"] == "verified"
    assert result["epochs"] == 10
    assert result["optimizer_steps"] == 2
    assert before == after


def test_missing_checkpoint_fails(pretraining_artifacts: Path) -> None:
    (pretraining_artifacts / "checkpoints" / "checkpoint_0.pth.tar").unlink()

    with pytest.raises(verify_pretraining.VerificationError, match="missing checkpoint"):
        verify_pretraining.verify_pretraining(pretraining_artifacts)


def test_summary_checkpoint_hash_mismatch_fails(
    pretraining_artifacts: Path,
) -> None:
    summary = read_summary(pretraining_artifacts)
    summary["checkpoint_0_sha256"] = "0" * 64
    atomic_write_json(pretraining_artifacts / "summary.json", summary)

    with pytest.raises(
        verify_pretraining.VerificationError,
        match="summary checkpoint_0 SHA-256 differs",
    ):
        verify_pretraining.verify_pretraining(pretraining_artifacts)


def test_best_and_checkpoint_mismatch_fails(pretraining_artifacts: Path) -> None:
    best = pretraining_artifacts / "checkpoints" / "best.pth.tar"
    best.write_bytes(best.read_bytes() + b"different")

    with pytest.raises(
        verify_pretraining.VerificationError,
        match="checkpoint_0 and best SHA-256 differ",
    ):
        verify_pretraining.verify_pretraining(pretraining_artifacts)


def test_zero_optimizer_steps_fails(pretraining_artifacts: Path) -> None:
    metrics = read_metrics(pretraining_artifacts)
    metrics["optimizer_steps"] = 0
    write_jsonl(pretraining_artifacts / "pretraining_metrics.jsonl", metrics)

    with pytest.raises(
        verify_pretraining.VerificationError,
        match="optimizer_steps must be > 0",
    ):
        verify_pretraining.verify_pretraining(pretraining_artifacts)


@pytest.mark.parametrize("invalid_loss", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_loss_fails(
    pretraining_artifacts: Path, invalid_loss: float
) -> None:
    metrics = read_metrics(pretraining_artifacts)
    metrics["total_loss"] = invalid_loss
    write_jsonl(pretraining_artifacts / "pretraining_metrics.jsonl", metrics)

    with pytest.raises(verify_pretraining.VerificationError, match="non-finite"):
        verify_pretraining.verify_pretraining(pretraining_artifacts)


def test_state_dict_shape_mismatch_fails(pretraining_artifacts: Path) -> None:
    checkpoint = pretraining_artifacts / "checkpoints" / "checkpoint_0.pth.tar"
    best = pretraining_artifacts / "checkpoints" / "best.pth.tar"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_name = next(iter(payload["state_dict"]))
    original = payload["state_dict"][first_name]
    payload["state_dict"][first_name] = original.reshape(-1)[:-1]
    torch.save(payload, checkpoint)
    shutil.copyfile(checkpoint, best)
    digest = sha256_file(checkpoint)
    summary = read_summary(pretraining_artifacts)
    summary["checkpoint_0_sha256"] = digest
    summary["best_sha256"] = digest
    atomic_write_json(pretraining_artifacts / "summary.json", summary)
    metadata_path = pretraining_artifacts / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_hashes"] = {"checkpoint_0": digest, "best": digest}
    atomic_write_json(metadata_path, metadata)

    with pytest.raises(
        verify_pretraining.VerificationError,
        match="parameter names or shapes differ",
    ):
        verify_pretraining.verify_pretraining(pretraining_artifacts)
