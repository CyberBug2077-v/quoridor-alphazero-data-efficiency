from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper
from utils import dotdict


def checkpoint_args(checkpoint_dir: Path, num_iterations: int) -> dotdict:
    return dotdict(
        {
            "numIters": num_iterations,
            "numEps": 1,
            "max_game_length": 4,
            "tempThreshold": 2,
            "numMCTSSims": 1,
            "eval_mcts_in_batch": 1,
            "maxlenOfQueue": 64,
            "max_train_size": 2,
            "batch_size": 2,
            "arenaCompare": 0,
            "updateThreshold": -0.51,
            "cpuct": 1.25,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.0,
            "lr": 0.0002,
            "checkpoint": str(checkpoint_dir),
            "save_every_n_iterations": 1,
            "load_folder_examples_file": [
                str(checkpoint_dir),
                "latest.examples",
            ],
            "numItersForTrainExamplesHistory": 10,
            "print_summary": False,
        }
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_weight_round_trip(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    minimal_network_args: dotdict,
    initial_state: np.ndarray,
    temporary_output_dir: Path,
) -> None:
    checkpoint_dir = temporary_output_dir / "checkpoints"
    batch = np.stack([initial_state, initial_state])
    valids = np.stack([game.getValidMoves(initial_state, 1)] * 2)
    before_policy, before_value = cuda_network.predict(batch, batch_valids=valids)

    cuda_network.save_checkpoint(str(checkpoint_dir), "roundtrip.pth.tar")
    saved = torch.load(
        checkpoint_dir / "roundtrip.pth.tar",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    restored = NNetWrapper(game, custom_args=minimal_network_args)
    incompatible = restored.nnet.load_state_dict(saved, strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    restored.load_checkpoint(str(checkpoint_dir), "roundtrip.pth.tar")
    after_policy, after_value = restored.predict(batch, batch_valids=valids)
    np.testing.assert_array_equal(after_policy, before_policy)
    np.testing.assert_array_equal(after_value, before_value)


def test_save_frequency_creates_four_numbered_checkpoints(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    training_examples: list[tuple],
    temporary_output_dir: Path,
) -> None:
    checkpoint_dir = temporary_output_dir / "checkpoints"
    coach = Coach(game, cuda_network, checkpoint_args(checkpoint_dir, 4))
    coach.executeEpisode = lambda: training_examples[:2]

    coach.learn()

    assert sorted(path.name for path in checkpoint_dir.glob("checkpoint_*.pth.tar")) == [
        "checkpoint_1.pth.tar",
        "checkpoint_2.pth.tar",
        "checkpoint_3.pth.tar",
        "checkpoint_4.pth.tar",
    ]


def test_invocation_boundary_is_checkpointed_between_cadence(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    training_examples: list[tuple],
    temporary_output_dir: Path,
) -> None:
    checkpoint_dir = temporary_output_dir / "boundary-checkpoints"
    args = checkpoint_args(checkpoint_dir, 2)
    args.save_every_n_iterations = 10
    coach = Coach(game, cuda_network, args)
    coach.executeEpisode = lambda: training_examples[:2]

    coach.learn()

    assert sorted(path.name for path in checkpoint_dir.glob("checkpoint_*.pth.tar")) == [
        "checkpoint_2.pth.tar"
    ]


def test_resume_from_iteration_two_preserves_history_weights_and_old_files(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    minimal_network_args: dotdict,
    training_examples: list[tuple],
    initial_state: np.ndarray,
    temporary_output_dir: Path,
) -> None:
    checkpoint_dir = temporary_output_dir / "checkpoints"
    first_coach = Coach(game, cuda_network, checkpoint_args(checkpoint_dir, 2))
    first_coach.executeEpisode = lambda: training_examples[:2]
    first_coach.learn()

    old_checkpoint_digests = {
        iteration: file_digest(checkpoint_dir / f"checkpoint_{iteration}.pth.tar")
        for iteration in (1, 2)
    }
    batch = np.stack([initial_state])
    valids = np.stack([game.getValidMoves(initial_state, 1)])
    expected_policy, expected_value = cuda_network.predict(batch, batch_valids=valids)

    restored_network = NNetWrapper(game, custom_args=minimal_network_args)
    restored_network.load_checkpoint(str(checkpoint_dir), "best.pth.tar")
    restored_policy, restored_value = restored_network.predict(batch, batch_valids=valids)
    np.testing.assert_array_equal(restored_policy, expected_policy)
    np.testing.assert_array_equal(restored_value, expected_value)

    resumed_iterations = []
    resumed_coach = Coach(
        game,
        restored_network,
        checkpoint_args(checkpoint_dir, 3),
        iteration_callback=lambda metrics: resumed_iterations.append(metrics["iteration"]),
    )
    resumed_coach.loadTrainExamples()
    assert resumed_coach.current_iteration == 2
    assert len(resumed_coach.trainExamplesHistory) == 2
    assert sum(len(batch) for batch in resumed_coach.trainExamplesHistory) == 4
    assert resumed_coach.skipFirstSelfPlay is True

    resumed_coach.executeEpisode = lambda: training_examples[:2]
    resumed_coach.learn()

    assert resumed_iterations == [3]
    assert (checkpoint_dir / "checkpoint_3.pth.tar").is_file()
    for iteration, digest in old_checkpoint_digests.items():
        assert file_digest(checkpoint_dir / f"checkpoint_{iteration}.pth.tar") == digest
