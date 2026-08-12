from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

import evaluate
from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper


def test_independent_checkpoint_evaluation(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    resolved_config: dict,
    temporary_output_dir: Path,
    monkeypatch,
) -> None:
    resolved = copy.deepcopy(resolved_config)
    checkpoint_dir = temporary_output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    resolved["_output_path"] = temporary_output_dir
    resolved["run"]["output_dir"] = str(temporary_output_dir)
    resolved["checkpoint"]["directory"] = str(checkpoint_dir)
    resolved["self_play"]["iterations"] = 4
    resolved["evaluation"].update(
        {
            "enabled": True,
            "games_per_opponent": 2,
            "opponents": ["random", "greedy"],
            "model_mcts_simulations": 8,
            "temperature": 0.0,
            "dirichlet_noise": False,
            "max_game_length": 150,
        }
    )
    cuda_network.save_checkpoint(str(checkpoint_dir), "checkpoint_4.pth.tar")
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in cuda_network.nnet.named_parameters()
    }

    eval_mode_observations = []

    def record_eval_mode(module, _inputs):
        eval_mode_observations.append(not module.training)

    hook = cuda_network.nnet.register_forward_pre_hook(record_eval_mode)
    noise_observations = []
    original_get_action_prob = evaluate.MCTS.getActionProb

    def record_noise(self, *args, **kwargs):
        noise_observations.append(kwargs.get("add_dirichlet_noise"))
        return original_get_action_prob(self, *args, **kwargs)

    monkeypatch.setattr(evaluate.MCTS, "getActionProb", record_noise)
    try:
        result, result_path = evaluate.evaluate_checkpoint(
            resolved,
            game,
            cuda_network,
        )
    finally:
        hook.remove()

    assert result_path == temporary_output_dir / "evaluation.json"
    assert result_path.is_file()
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert result["checkpoint_path"] == (
        checkpoint_dir / "checkpoint_4.pth.tar"
    ).as_posix()
    assert result["model"] == {
        "mcts_simulations": 8,
        "temperature": 0.0,
        "dirichlet_noise": False,
    }

    for opponent in ("random", "greedy"):
        opponent_result = result["opponents"][opponent]
        assert (
            opponent_result["wins"]
            + opponent_result["draws"]
            + opponent_result["losses"]
            == 2
        )
        assert len(opponent_result["games"]) == 2
        assert opponent_result["mean_game_length"] >= 0
        assert opponent_result["illegal_actions"] == 0
        assert [game_result["model_side"] for game_result in opponent_result["games"]] == [
            "first",
            "second",
        ]
        assert len({game_result["seed"] for game_result in opponent_result["games"]}) == 2
        assert all(game_result["illegal_actions"] == 0 for game_result in opponent_result["games"])

    assert eval_mode_observations
    assert all(eval_mode_observations)
    assert noise_observations
    assert all(observation is False for observation in noise_observations)
    for name, parameter in cuda_network.nnet.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name], rtol=0, atol=0)
