from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
ALPHAZERO_ROOT = BASELINE_ROOT / "external" / "alphazero"
PATHFINDER_ROOT = ALPHAZERO_ROOT / "quoridor" / "pathFinder-module"
SCRIPTS_ROOT = BASELINE_ROOT / "scripts"

for import_root in (ALPHAZERO_ROOT, PATHFINDER_ROOT, SCRIPTS_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import run_smoke
from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch import NNet as nnet_module
from quoridor.pytorch.NNet import NNetWrapper
from utils import dotdict


TEST_SEED = 1001


@pytest.fixture(autouse=True)
def fixed_seed() -> int:
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TEST_SEED)
    return TEST_SEED


@pytest.fixture(scope="session")
def game() -> QuoridorGame:
    return QuoridorGame(9)


@pytest.fixture
def minimal_network_args() -> dotdict:
    return dotdict(
        {
            "cuda": torch.cuda.is_available(),
            "lr": 0.0002,
            "dropout": 0.0,
            "epochs": 1,
            "batch_size": 2,
            "num_channels": 8,
            "num_res_blocks": 1,
            "attn_depth": 0,
            "num_heads": 1,
            "se_enabled": False,
            "fast_opts": False,
            "clip": 1.0,
            "weight_decay": 0.0001,
            "lr_decay_gamma": 1.0,
            "use_amp": False,
            "amp_dtype": "bf16",
        }
    )


@pytest.fixture
def cuda_network(
    game: QuoridorGame,
    minimal_network_args: dotdict,
    cuda_device: torch.device,
) -> NNetWrapper:
    nnet_module.args.update(minimal_network_args)
    network = NNetWrapper(game, custom_args=minimal_network_args)
    assert next(network.nnet.parameters()).device == cuda_device
    return network


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU baseline test suite")
    return torch.device("cuda", 0)


@pytest.fixture
def temporary_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "smoke-output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def initial_state(game: QuoridorGame) -> np.ndarray:
    return game.getInitBoard().copy()


@pytest.fixture
def fixed_legal_action(game: QuoridorGame, initial_state: np.ndarray) -> int:
    action = 0
    assert game.getValidMoves(initial_state, 1)[action] == 1
    return action


@pytest.fixture
def training_examples(
    game: QuoridorGame,
    initial_state: np.ndarray,
    fixed_legal_action: int,
) -> list[tuple]:
    valids = game.getValidMoves(initial_state, 1)
    policy = np.zeros(game.getActionSize(), dtype=np.float32)
    policy[fixed_legal_action] = 1.0
    return [
        (
            initial_state.copy(),
            policy.copy(),
            1.0,
            valids.copy(),
            1,
            1,
        )
        for _ in range(4)
    ]


@pytest.fixture
def smoke_config() -> dict:
    config_path = BASELINE_ROOT / "configs" / "smoke_gpu.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(config: dict) -> Path:
        config_path = tmp_path / "smoke.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        return config_path

    return _write


@pytest.fixture
def loaded_config(smoke_config: dict, write_config) -> tuple[dict, Path]:
    config_path = write_config(smoke_config)
    return run_smoke.load_config(config_path), config_path


@pytest.fixture
def resolved_config(
    loaded_config: tuple[dict, Path],
    cuda_device: torch.device,
) -> dict:
    config, config_path = loaded_config
    return run_smoke.resolve_config(copy.deepcopy(config), config_path)
