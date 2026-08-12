from __future__ import annotations

import numpy as np
import pytest

from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper
from utils import dotdict


def test_real_self_play_sample_contract(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    temporary_output_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = dotdict(
        {
            "checkpoint": str(temporary_output_dir / "checkpoints"),
            "save_every_n_iterations": 1,
            "numMCTSSims": 8,
            "eval_mcts_in_batch": 4,
            "cpuct": 1.25,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.0,
            "heuristic_alpha": 0.0,
            "tempThreshold": 15,
            "max_game_length": 150,
        }
    )
    coach = Coach(game, cuda_network, args)

    original_get_action_prob = coach.mcts.getActionProb
    roots_checked = 0

    def immutable_root_search(board, *search_args, **search_kwargs):
        nonlocal roots_checked
        before = board.copy()
        probabilities = original_get_action_prob(
            board,
            *search_args,
            **search_kwargs,
        )
        np.testing.assert_array_equal(board, before)
        roots_checked += 1
        return probabilities

    coach.mcts.getActionProb = immutable_root_search

    original_choice = np.random.choice
    chosen_probabilities = []

    def checked_choice(options, *choice_args, **choice_kwargs):
        probability = choice_kwargs.get("p")
        selected = original_choice(options, *choice_args, **choice_kwargs)
        if probability is not None and int(options) == game.getActionSize():
            assert probability[selected] > 0
            chosen_probabilities.append(float(probability[selected]))
        return selected

    monkeypatch.setattr(np.random, "choice", checked_choice)
    examples = coach.executeEpisode()

    assert examples
    assert roots_checked == len(examples)
    assert len(chosen_probabilities) == len(examples)

    episode_length = examples[0][5]
    assert episode_length == len(examples)
    assert episode_length <= 150
    values = []
    for expected_step, sample in enumerate(examples, start=1):
        assert len(sample) == 6
        board, policy, value, valids, step, sample_episode_length = sample
        policy = np.asarray(policy)
        valids = np.asarray(valids)
        assert board.shape == (4, 17, 17)
        assert policy.shape == (136,)
        assert valids.shape == (136,)
        assert policy.sum() == pytest.approx(1.0, abs=1e-7)
        assert np.count_nonzero(policy[valids == 0]) == 0
        assert value in (-1, 0, 1)
        assert step == expected_step
        assert sample_episode_length == episode_length
        values.append(value)

    if values[0] == 0:
        assert set(values) == {0}
    else:
        for index, value in enumerate(values):
            assert value == values[0] * (1 if index % 2 == 0 else -1)
    if episode_length == 150:
        assert set(values) == {0}


def test_max_length_episode_is_recorded_as_draw(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    temporary_output_dir,
) -> None:
    args = dotdict(
        {
            "checkpoint": str(temporary_output_dir / "draw-checkpoints"),
            "save_every_n_iterations": 1,
            "numMCTSSims": 1,
            "eval_mcts_in_batch": 1,
            "cpuct": 1.25,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.0,
            "heuristic_alpha": 0.0,
            "tempThreshold": 0,
            "max_game_length": 150,
        }
    )
    coach = Coach(game, cuda_network, args)
    calls = 0

    def cycling_policy(board, temp, add_dirichlet_noise):
        nonlocal calls
        action = 2 if (calls // 2) % 2 == 0 else 3
        calls += 1
        valids = game.getValidMoves(board, 1)
        assert valids[action] == 1
        policy = np.zeros(game.getActionSize(), dtype=np.float64)
        policy[action] = 1.0
        return policy

    coach.mcts.getActionProb = cycling_policy
    examples = coach.executeEpisode()

    assert len(examples) == 150
    assert calls == 150
    assert all(sample[2] == 0 for sample in examples)
    assert all(sample[5] == 150 for sample in examples)
