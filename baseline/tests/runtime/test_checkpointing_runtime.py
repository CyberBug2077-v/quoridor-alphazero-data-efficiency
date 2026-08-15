from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from runtime.checkpointing import (
    capture_rng_state,
    load_model_checkpoint,
    load_run_state,
    restore_rng_state,
    save_numbered_checkpoint,
    save_run_state,
    validate_checkpoint_hash,
)


def test_numbered_checkpoint_round_trip_and_hash(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    saved = save_numbered_checkpoint(source, tmp_path, 4)
    restored = torch.nn.Linear(3, 2)

    loaded = load_model_checkpoint(
        restored,
        saved["path"],
        expected_sha256=saved["sha256"],
    )

    assert loaded["sha256"] == saved["sha256"]
    for source_value, restored_value in zip(source.parameters(), restored.parameters()):
        assert torch.equal(source_value, restored_value)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_checkpoint_hash(saved["path"], "0" * 64)


def test_run_state_and_rng_round_trip(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    rng_state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))

    random.random()
    np.random.random()
    torch.rand(1)
    restore_rng_state(rng_state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(1)))
    assert actual == expected

    path = save_run_state(
        tmp_path / "run_state.pth.tar",
        {"iteration": 5, "rng_state": rng_state},
    )
    loaded = load_run_state(path)
    assert loaded["iteration"] == 5
    assert set(loaded["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
