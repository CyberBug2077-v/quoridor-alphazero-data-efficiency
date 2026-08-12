from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from MCTS import MCTS
from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper
from utils import dotdict


def mcts_args(eval_mcts_in_batch: int) -> dotdict:
    return dotdict(
        {
            "numMCTSSims": 8,
            "eval_mcts_in_batch": eval_mcts_in_batch,
            "cpuct": 1.25,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.0,
            "heuristic_alpha": 0.0,
        }
    )


def assert_tree_is_finite(mcts: MCTS) -> None:
    for cache in (mcts.Qsa, mcts.Nsa, mcts.Ns, mcts.Ps):
        for value in cache.values():
            assert np.isfinite(np.asarray(value)).all()


@pytest.mark.parametrize("eval_mcts_in_batch", [1, 4])
def test_mcts_action_probabilities_and_tree_integrity(
    eval_mcts_in_batch: int,
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    initial_state: np.ndarray,
) -> None:
    root_before = initial_state.copy()
    legal = game.getValidMoves(initial_state, 1)
    mcts = MCTS(game, cuda_network, mcts_args(eval_mcts_in_batch))

    probabilities = np.asarray(
        mcts.getActionProb(
            initial_state,
            temp=1,
            add_dirichlet_noise=False,
        ),
        dtype=np.float64,
    )

    assert probabilities.shape == (136,)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-7)
    assert np.count_nonzero(probabilities[legal == 0]) == 0
    selected_action = int(np.argmax(probabilities))
    assert legal[selected_action] == 1
    np.testing.assert_array_equal(initial_state, root_before)
    assert_tree_is_finite(mcts)


@pytest.mark.parametrize("eval_mcts_in_batch", [1, 4])
def test_terminal_state_is_not_expanded(
    eval_mcts_in_batch: int,
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    initial_state: np.ndarray,
) -> None:
    terminal = initial_state.copy()
    terminal[0].fill(0)
    terminal[0][0, 8] = 1
    terminal_before = terminal.copy()
    mcts = MCTS(game, cuda_network, mcts_args(eval_mcts_in_batch))

    if eval_mcts_in_batch == 1:
        assert mcts.search(terminal) == -1
    else:
        mcts._search_batch(terminal, eval_mcts_in_batch)

    root_key = game.stringRepresentation(terminal)
    assert mcts.Es[root_key] == 1
    assert root_key not in mcts.Ps
    assert root_key not in mcts.Vs
    assert root_key not in mcts.Ns
    assert mcts.Qsa == {}
    assert mcts.Nsa == {}
    np.testing.assert_array_equal(terminal, terminal_before)


@pytest.mark.parametrize("eval_mcts_in_batch", [1, 4])
def test_noise_free_mcts_is_reproducible_after_seed_reset(
    eval_mcts_in_batch: int,
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    initial_state: np.ndarray,
) -> None:
    results = []
    for _ in range(2):
        random.seed(1001)
        np.random.seed(1001)
        torch.manual_seed(1001)
        torch.cuda.manual_seed_all(1001)
        mcts = MCTS(game, cuda_network, mcts_args(eval_mcts_in_batch))
        results.append(
            np.asarray(
                mcts.getActionProb(
                    initial_state.copy(),
                    temp=1,
                    add_dirichlet_noise=False,
                )
            )
        )

    np.testing.assert_array_equal(results[0], results[1])
