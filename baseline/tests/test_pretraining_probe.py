from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import probe_pretraining_batch
import run_pretraining
from runtime.metadata import sha256_file


BASELINE_ROOT = Path(__file__).resolve().parents[1]


def make_probe_config(
    tmp_path: Path,
    *,
    output_dir: str = "outputs/formal_pretraining_seed1001",
) -> tuple[Path, Path]:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    archive = data_dir / "heuristic_games.zip"
    dataset = data_dir / "heuristic_games.pkl"
    archive.write_bytes(b"archive")
    dataset.write_bytes(b"immutable synthetic dataset")
    config = {
        "run": {
            "id": "formal_pretraining_seed1001",
            "seed": 1001,
            "device": "cuda",
            "gpu_index": 0,
            "deterministic": True,
            "output_dir": output_dir,
        },
        "data": {
            "archive_path": "data/heuristic_games.zip",
            "extracted_path": "data/heuristic_games.pkl",
            "expected_sha256": sha256_file(dataset),
            "validate_before_training": True,
        },
        "model": {
            "board_size": 9,
            "num_channels": 128,
            "num_res_blocks": 6,
            "attn_depth": 1,
            "num_heads": 8,
            "se_enabled": False,
            "dropout": 0.3,
        },
        "pretraining": {
            "epochs": 10,
            "batch_size": 8,
            "micro_batch_size": 4,
            "learning_rate": 0.0005,
            "optimizer": "adamw",
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "amp": True,
            "amp_dtype": "bf16",
        },
        "checkpoint": {
            "directory": f"{output_dir}/checkpoints",
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
            "log_every_optimizer_steps": 100,
        },
    }
    config_path = config_dir / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return config_path, dataset


def successful_probe_result() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "probe_only": True,
        "configured_effective_batch_size": 8,
        "configured_micro_batch_size": 4,
        "requested_effective_batch_size": 8,
        "requested_micro_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "requested_optimizer_steps": 1,
        "optimizer_steps": 1,
        "micro_batches_processed": 2,
        "samples_seen": 8,
        "oom": False,
        "policy_loss": 2.0,
        "value_loss": 0.5,
        "total_loss": 2.5,
        "gradient_norm": 1.0,
        "execution_seconds": 0.1,
        "peak_gpu_memory_bytes": 1024,
        "gpu_name": "synthetic GPU",
        "memory_margin": {"ok": True},
        "acceptance": {
            "effective_batch_matches_config": True,
            "optimizer_steps_ok": True,
            "micro_batch_count_ok": True,
            "samples_seen_ok": True,
            "finite_metrics": True,
            "memory_margin_ok": True,
            "no_formal_checkpoint": False,
            "passed": False,
        },
    }


def patch_probe_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(probe_pretraining_batch, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(run_pretraining, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(
        probe_pretraining_batch, "initialize_cuda", lambda resolved: None
    )
    monkeypatch.setattr(
        probe_pretraining_batch, "set_seed", lambda seed, deterministic: None
    )

    def load_dataset(resolved):
        resolved["_dataset_validation"] = {
            "examples": 1,
            "zero_policy_examples": 0,
            "zero_policy_fraction": 0.0,
            "policy_on_invalid_examples": 0,
            "policy_on_invalid_fraction": 0.0,
        }
        return ["sample"]

    monkeypatch.setattr(
        probe_pretraining_batch, "load_and_validate_dataset", load_dataset
    )


def invoke_probe(
    monkeypatch,
    config_path: Path,
    *,
    effective_batch_size: int | None = None,
    micro_batch_size: int | None = None,
    steps: int = 1,
    trial: int = 1,
) -> int:
    argv = [
        "probe_pretraining_batch.py",
        "--config",
        str(config_path),
        "--steps",
        str(steps),
        "--trial",
        str(trial),
    ]
    if effective_batch_size is not None:
        argv.extend(["--effective-batch-size", str(effective_batch_size)])
    if micro_batch_size is not None:
        argv.extend(["--micro-batch-size", str(micro_batch_size)])
    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )
    return probe_pretraining_batch.main()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("--steps", "0"),
        ("--trial", "0"),
        ("--effective-batch-size", "0"),
        ("--micro-batch-size", "-1"),
    ],
)
def test_probe_requires_positive_steps_and_batch_size(
    monkeypatch, argument: str, value: str
) -> None:
    argv = [
        "probe_pretraining_batch.py",
        "--config",
        "unused.yaml",
        "--effective-batch-size",
        "1",
        "--micro-batch-size",
        "1",
        "--steps",
        "1",
        "--trial",
        "1",
    ]
    argv[argv.index(argument) + 1] = value
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as raised:
        probe_pretraining_batch.parse_args()

    assert raised.value.code == 2


def test_probe_rejects_output_inside_formal_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path, _ = make_probe_config(tmp_path, output_dir="outputs")
    monkeypatch.setattr(probe_pretraining_batch, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(run_pretraining, "BASELINE_ROOT", tmp_path)

    assert invoke_probe(monkeypatch, config_path) == 2
    assert "probe output must not be inside" in capsys.readouterr().err
    assert not (tmp_path / "outputs" / "pretraining_probe").exists()


@pytest.mark.parametrize(
    "error",
    [
        torch.cuda.OutOfMemoryError("synthetic OOM"),
        RuntimeError("synthetic runtime failure"),
    ],
    ids=["oom", "runtime-error"],
)
def test_probe_runtime_failures_return_nonzero(
    tmp_path: Path, monkeypatch, error: RuntimeError
) -> None:
    config_path, _ = make_probe_config(tmp_path)
    patch_probe_runtime(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(probe_pretraining_batch, "execute_probe", fail)

    assert invoke_probe(monkeypatch, config_path) == 2


def test_successful_probe_writes_json_without_mutating_inputs_or_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, dataset = make_probe_config(tmp_path)
    patch_probe_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        probe_pretraining_batch,
        "execute_probe",
        lambda *args, **kwargs: successful_probe_result(),
    )
    config_hash = sha256_file(config_path)
    dataset_hash = sha256_file(dataset)

    assert invoke_probe(monkeypatch, config_path) == 0

    output = (
        tmp_path
        / "outputs"
        / "pretraining_probe"
        / "effective_8_micro_4_steps_1_trial_1.json"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["optimizer_steps"] == 1
    assert payload["micro_batches_processed"] == 2
    assert payload["samples_seen"] == 8
    assert payload["requested_effective_batch_size"] == 8
    assert payload["requested_micro_batch_size"] == 4
    assert payload["acceptance"]["passed"] is True
    assert payload["acceptance"]["no_formal_checkpoint"] is True
    assert sha256_file(config_path) == config_hash
    assert sha256_file(dataset) == dataset_hash
    assert not any(tmp_path.rglob("checkpoint_0.pth.tar"))


def test_probe_memory_margin_failure_writes_json_and_returns_two(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _ = make_probe_config(tmp_path)
    patch_probe_runtime(monkeypatch, tmp_path)
    failed_result = successful_probe_result()
    failed_result["memory_margin"] = {"ok": False}
    failed_result["acceptance"]["memory_margin_ok"] = False
    monkeypatch.setattr(
        probe_pretraining_batch,
        "execute_probe",
        lambda *args, **kwargs: failed_result,
    )

    assert invoke_probe(monkeypatch, config_path) == 2

    output = (
        tmp_path
        / "outputs"
        / "pretraining_probe"
        / "effective_8_micro_4_steps_1_trial_1.json"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["acceptance"]["memory_margin_ok"] is False
    assert payload["acceptance"]["passed"] is False
    assert not any(tmp_path.rglob("checkpoint_0.pth.tar"))


def test_probe_batch_selection_and_memory_margin_are_deterministic() -> None:
    examples = list(range(20))
    first, first_hash = probe_pretraining_batch.fixed_batch(
        examples, batch_size=5, seed=1001
    )
    second, second_hash = probe_pretraining_batch.fixed_batch(
        examples, batch_size=5, seed=1001
    )
    assert first == second
    assert first_hash == second_hash
    assert len(set(first)) == 5
    assert probe_pretraining_batch.memory_margin(
        peak_allocated_bytes=7 * 1024**3,
        total_memory_bytes=8 * 1024**3,
    )["ok"] is True
    assert probe_pretraining_batch.memory_margin(
        peak_allocated_bytes=6 * 1024**3,
        peak_reserved_bytes=int(7.5 * 1024**3),
        free_after_bytes=3 * 1024**3,
        total_memory_bytes=8 * 1024**3,
    )["ok"] is False


@pytest.mark.parametrize(
    ("effective", "micro", "message"),
    [(1024, 2048, "must be <="), (2048, 1500, "must be divisible")],
)
def test_probe_rejects_invalid_accumulation_configuration(
    effective: int, micro: int, message: str
) -> None:
    with pytest.raises(probe_pretraining_batch.ConfigError, match=message):
        probe_pretraining_batch.validate_requested_batches(effective, micro)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_FORMAL_PRETRAINING_PROBE") != "1",
    reason="set RUN_FORMAL_PRETRAINING_PROBE=1 to run the formal GPU probe",
)
def test_formal_scale_gpu_probe() -> None:
    config_path = BASELINE_ROOT / "configs" / "pretraining_reproduction.yaml"
    command = [
        sys.executable,
        str(BASELINE_ROOT / "scripts" / "probe_pretraining_batch.py"),
        "--config",
        str(config_path),
        "--effective-batch-size",
        "2048",
        "--micro-batch-size",
        "1024",
        "--steps",
        "20",
        "--trial",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=BASELINE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report_path = (
        BASELINE_ROOT
        / "outputs"
        / "pretraining_probe"
        / "effective_2048_micro_1024_steps_20_trial_1.json"
    )
    assert completed.returncode in {0, 2}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["optimizer_steps"] in {0, 20}
    if not report["oom"]:
        assert report["optimizer_steps"] == 20
        assert report["micro_batches_processed"] == 40
        assert report["samples_seen"] == 40960
        for field in ("policy_loss", "value_loss", "total_loss", "gradient_norm"):
            assert math.isfinite(report[field])
