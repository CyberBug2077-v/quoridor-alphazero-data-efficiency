from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.artifacts import atomic_write_json


BASELINE_ROOT = Path(__file__).resolve().parents[1]
ALPHAZERO_ROOT = BASELINE_ROOT / "external" / "alphazero"
if str(ALPHAZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHAZERO_ROOT))

from MCTS import MCTS
from utils import dotdict


def _random_action(valids: np.ndarray) -> int:
    return int(np.random.choice(np.flatnonzero(valids)))


def _greedy_action(valids: np.ndarray) -> int:
    # Boards are canonical: action 0 advances the current player toward its goal.
    for action in (0, 4, 7, 2, 3, 1, 6, 5):
        if valids[action]:
            return action
    return _random_action(valids)


def _model_action(game, nnet, board, evaluation: dict[str, Any]) -> int:
    mcts = MCTS(
        game,
        nnet,
        dotdict(
            {
                "numMCTSSims": evaluation["model_mcts_simulations"],
                "eval_mcts_in_batch": min(
                    4,
                    evaluation["model_mcts_simulations"],
                ),
                "cpuct": 1.25,
                "dirichlet_alpha": 0.15,
                "dirichlet_epsilon": 0.0,
                "heuristic_alpha": 0.0,
            }
        ),
    )
    probabilities = np.asarray(
        mcts.getActionProb(
            board,
            temp=evaluation["temperature"],
            add_dirichlet_noise=False,
        ),
        dtype=np.float64,
    )
    if evaluation["temperature"] == 0:
        return int(np.argmax(probabilities))
    return int(np.random.choice(len(probabilities), p=probabilities))


def _play_game(game, nnet, opponent: str, model_side: int, evaluation: dict):
    board = game.getInitBoard()
    current_player = 1
    moves = 0
    illegal_actions = 0

    while game.getGameEnded(board, 1) == 0 and moves < evaluation["max_game_length"]:
        canonical = game.getCanonicalForm(board, current_player)
        valids = game.getValidMoves(canonical, 1)
        if current_player == model_side:
            action = _model_action(game, nnet, canonical, evaluation)
        elif opponent == "random":
            action = _random_action(valids)
        elif opponent == "greedy":
            action = _greedy_action(valids)
        else:
            raise ValueError(f"unsupported evaluation opponent: {opponent}")
        if not valids[action]:
            illegal_actions += 1
            outcome = "loss" if current_player == model_side else "win"
            return outcome, moves, illegal_actions
        board, current_player = game.getNextState(board, current_player, action)
        moves += 1

    absolute_result = game.getGameEnded(board, 1)
    model_result = absolute_result * model_side
    outcome = "win" if model_result > 0 else "loss" if model_result < 0 else "draw"
    return outcome, moves, illegal_actions


def evaluate_checkpoint(
    resolved: dict[str, Any],
    game,
    nnet,
    checkpoint: Path | None = None,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    evaluation_start = time.perf_counter()
    evaluation = resolved["evaluation"]
    final_iteration = resolved["self_play"].get("iterations")
    if final_iteration is None:
        final_iteration = resolved.get("budget", {}).get("max_iterations")
    checkpoint_dir = Path(resolved["checkpoint"]["directory"])
    numbered_checkpoint = checkpoint_dir / f"checkpoint_{final_iteration}.pth.tar"
    checkpoint_path = checkpoint or (
        numbered_checkpoint if numbered_checkpoint.is_file() else checkpoint_dir / "best.pth.tar"
    )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"final checkpoint not found: {checkpoint_path}")

    nnet.load_checkpoint(str(checkpoint_path.parent), checkpoint_path.name)
    nnet.nnet.eval()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in nnet.nnet.named_parameters()
    }

    result = {
        "schema_version": 1,
        "checkpoint_path": checkpoint_path.as_posix(),
        "model": {
            "mcts_simulations": evaluation["model_mcts_simulations"],
            "temperature": evaluation["temperature"],
            "dirichlet_noise": False,
        },
        "opponents": {},
    }
    base_seed = int(resolved["run"]["seed"])
    opponent_seconds: dict[str, float] = {}
    for opponent_index, opponent in enumerate(evaluation["opponents"]):
        opponent_start = time.perf_counter()
        games = []
        counts = {"wins": 0, "draws": 0, "losses": 0}
        total_moves = 0
        illegal_actions = 0
        for game_index in range(evaluation["games_per_opponent"]):
            evaluation_seed = base_seed + opponent_index * 10_000 + game_index
            np.random.seed(evaluation_seed)
            torch.manual_seed(evaluation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(evaluation_seed)
            model_side = 1 if game_index % 2 == 0 else -1
            outcome, moves, game_illegal_actions = _play_game(
                game,
                nnet,
                opponent,
                model_side,
                evaluation,
            )
            counts[{"win": "wins", "draw": "draws", "loss": "losses"}[outcome]] += 1
            total_moves += moves
            illegal_actions += game_illegal_actions
            games.append(
                {
                    "seed": evaluation_seed,
                    "model_side": "first" if model_side == 1 else "second",
                    "outcome": outcome,
                    "moves": moves,
                    "illegal_actions": game_illegal_actions,
                }
            )
        result["opponents"][opponent] = {
            **counts,
            "mean_game_length": total_moves / len(games) if games else None,
            "illegal_actions": illegal_actions,
            "games": games,
        }
        opponent_seconds[opponent] = time.perf_counter() - opponent_start

    for name, parameter in nnet.nnet.named_parameters():
        if not torch.equal(parameter.detach(), parameters_before[name]):
            raise RuntimeError(f"evaluation modified model parameter: {name}")

    evaluation_seconds = time.perf_counter() - evaluation_start
    games_evaluated = sum(
        len(opponent_result["games"])
        for opponent_result in result["opponents"].values()
    )
    result["timing"] = {
        "evaluation_seconds": evaluation_seconds,
        "per_opponent_seconds": opponent_seconds,
        "games_evaluated": games_evaluated,
        "seconds_per_game": (
            evaluation_seconds / games_evaluated if games_evaluated else None
        ),
    }

    destination = output_path or (resolved["_output_path"] / "evaluation.json")
    return result, atomic_write_json(destination, result)
