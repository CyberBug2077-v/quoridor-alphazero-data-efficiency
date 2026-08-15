from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

import run_baseline
import run_pretraining
import verify_baseline
from runtime.artifacts import atomic_write_json, atomic_write_yaml
from runtime.checkpointing import save_run_state
from runtime.metadata import sha256_file


def test_pretraining_dataset_validation(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"archive")
    policy = np.zeros(136, dtype=np.float32)
    policy[0] = 1.0
    examples = [(np.zeros((4, 17, 17), dtype=np.uint8), policy, 1.0)]
    pickle_path = tmp_path / "examples.pkl"
    with pickle_path.open("wb") as output:
        pickle.dump(examples, output)
    resolved = {
        "data": {
            "_archive_path": archive,
            "_extracted_path": pickle_path,
            "expected_sha256": sha256_file(pickle_path),
        }
    }

    loaded = run_pretraining.load_and_validate_dataset(resolved)

    assert len(loaded) == 1


def test_strict_initial_checkpoint_loading(tmp_path: Path) -> None:
    class Wrapper:
        def __init__(self):
            self.nnet = torch.nn.Linear(3, 2)

    source = Wrapper()
    checkpoint = tmp_path / "checkpoint_0.pth.tar"
    torch.save({"state_dict": source.nnet.state_dict()}, checkpoint)
    restored = Wrapper()

    digest = run_baseline.strict_load_initial_weights(
        restored, checkpoint, sha256_file(checkpoint)
    )

    assert digest == sha256_file(checkpoint)
    for expected, actual in zip(source.nnet.parameters(), restored.nnet.parameters()):
        assert torch.equal(expected, actual)


def test_verify_baseline_reads_complete_artifacts_without_writing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(verify_baseline, "BASELINE_ROOT", tmp_path)
    run_dir = tmp_path / "baseline_test_seed1"
    checkpoint_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluations"
    checkpoint_dir.mkdir(parents=True)
    evaluation_dir.mkdir()
    initial = tmp_path / "pretraining" / "checkpoints" / "checkpoint_0.pth.tar"
    initial.parent.mkdir(parents=True)
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
    torch.save({"state_dict": state_dict}, initial)
    checkpoint = checkpoint_dir / "checkpoint_1.pth.tar"
    torch.save({"state_dict": state_dict}, checkpoint)
    initial_hash = sha256_file(initial)
    resolved = {
        "schema_version": 1,
        "mode": "baseline",
        "run": {"id": run_dir.name, "seed": 1},
        "initialization": {
            "mode": "pretrained_checkpoint",
            "checkpoint_path": initial.as_posix(),
            "expected_sha256": initial_hash,
            "load_replay": False,
        },
        "model": model_config,
        "self_play": {"iterations": 1, "games_per_iteration": 2},
        "training": {"batch_size": 2, "micro_batch_size": 1},
        "replay": {"history_iterations": 150},
        "checkpoint": {"save_every_iterations": 1},
        "logging": {
            "metrics_file": "metrics.jsonl",
            "metadata_file": "run_metadata.json",
            "summary_file": "summary.json",
        },
        "evaluation": {
            "opponents": ["random"],
            "games_per_opponent": 2,
        },
    }
    atomic_write_yaml(run_dir / "resolved_config.yaml", resolved)
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "git": {"commit": "abc"},
            "initial_checkpoint_path": initial.as_posix(),
            "initial_checkpoint_sha256": initial_hash,
        },
    )
    metrics = {
        "iteration": 1,
        "games_completed": 2,
        "optimizer_steps": 1,
        "effective_batch_size": 2,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "micro_batches_processed": 2,
        "samples_seen": 2,
        "illegal_action_count": 0,
        "replay_buffer_size": 1,
        "checkpoint_path": checkpoint.as_posix(),
    }
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8", newline="\n"
    )
    with (checkpoint_dir / "latest.examples").open("wb") as output:
        pickle.dump({"iteration": 1, "examples": [[("sample",)]]}, output)
    save_run_state(
        checkpoint_dir / "latest.state.pt",
        {
            "iteration": 1,
            "python_rng_state": (3, (), None),
            "numpy_rng_state": ("MT19937", np.zeros(624, dtype=np.uint32), 0, 0, 0.0),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": [torch.zeros(16, dtype=torch.uint8)],
            "cumulative_gpu_hours": 0.1,
            "instrumentation_state": {},
        },
    )
    atomic_write_json(
        evaluation_dir / "evaluation_checkpoint_1.json",
        {
            "checkpoint_path": checkpoint.as_posix(),
            "opponents": {
                "random": {
                    "wins": 1,
                    "draws": 1,
                    "losses": 0,
                    "illegal_actions": 0,
                    "games": [
                        {"model_side": "first"},
                        {"model_side": "second"},
                    ],
                }
            },
        },
    )
    atomic_write_json(
        run_dir / "summary.json",
        {"status": "completed", "completed_iterations": [1]},
    )
    before = {path: sha256_file(path) for path in run_dir.rglob("*") if path.is_file()}

    result = verify_baseline.verify_baseline(run_dir)

    after = {path: sha256_file(path) for path in run_dir.rglob("*") if path.is_file()}
    assert result["status"] == "verified"
    assert before == after
