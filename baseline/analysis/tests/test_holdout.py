from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ANALYSIS = Path(__file__).resolve().parents[1]
SCRIPTS = ANALYSIS / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_holdout as evaluate
import generate_holdout as generate
import holdout_common as common
import verify_holdout as verify


def protocol() -> dict:
    return common.load_protocol(ANALYSIS / "configs" / "holdout_v1.yaml")


def test_fixed_holdout_protocol_is_exact() -> None:
    config = protocol()

    assert config["protocol_id"] == "fixed_holdout_v1"
    assert config["seed"] == 71001
    assert config["games"] == 200
    assert config["source"]["checkpoint_iteration"] == 0
    assert config["source"]["expected_sha256"] == (
        "4824a2a8ba1c1ebb5a38a992af075a45a033b87b403973b583ab98a079f35667"
    )
    assert config["self_play"] == {
        "mcts_simulations": 200,
        "eval_mcts_in_batch": 10,
        "cpuct": 1.25,
        "temperature_threshold": 15,
        "dirichlet_noise": True,
        "dirichlet_alpha": 0.15,
        "dirichlet_epsilon": 0.25,
        "max_game_length": 150,
    }
    assert config["evaluation"]["checkpoints"] == common.EXPECTED_CHECKPOINTS


def test_action_size_is_taken_from_the_game_implementation() -> None:
    game = common.build_game(protocol())

    assert game.getActionSize() == 8 + 2 * (9 - 1) ** 2
    assert game.getActionSize() == 136
    assert game.getBoardSize() == (17, 17)


def test_game_seeds_are_independent_and_stable() -> None:
    assert common.stable_game_seed(71001, 0) == 71001
    assert common.stable_game_seed(71001, 199) == 71200
    with pytest.raises(common.HoldoutError):
        common.stable_game_seed(71001, -1)


def test_pilot_game_count_is_a_bounded_runtime_override() -> None:
    config = protocol()

    assert generate.resolve_game_count(config, None) == 200
    assert generate.resolve_game_count(config, 4) == 4
    with pytest.raises(common.HoldoutError):
        generate.resolve_game_count(config, 0)
    with pytest.raises(common.HoldoutError):
        generate.resolve_game_count(config, 201)


def test_float32_policy_is_renormalized_for_numpy_sampling() -> None:
    policy = np.full(136, np.float32(1.0 / 136.0), dtype=np.float32)

    np.random.seed(123)
    actions = [generate.sample_action(policy) for _ in range(20)]

    assert all(0 <= action < 136 for action in actions)


def _history() -> list[tuple[np.ndarray, int, np.ndarray, np.ndarray, int]]:
    board = np.zeros((4, 1, 1), dtype=np.uint8)
    policy = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    valids = np.asarray([1, 0, 0], dtype=np.uint8)
    return [
        (board.copy(), 1, policy.copy(), valids.copy(), 1),
        (board.copy(), -1, policy.copy(), valids.copy(), 2),
    ]


def test_episode_finalization_uses_current_player_value_sign_without_duplication() -> None:
    arrays = generate.finalize_episode(
        _history(), game_id=7, terminal_result=1, game_length=2
    )

    assert arrays["boards"].shape == (2, 4, 1, 1)
    assert arrays["values"].tolist() == [1.0, -1.0]
    assert arrays["game_ids"].tolist() == [7, 7]
    assert arrays["steps"].tolist() == [1, 2]
    assert arrays["game_lengths"].tolist() == [2, 2]


class FakeGame:
    def getActionSize(self) -> int:
        return 3

    def getInitBoard(self) -> np.ndarray:
        return np.zeros((4, 1, 1), dtype=np.uint8)

    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        return board.copy()

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        return np.asarray([1, 0, 0], dtype=np.uint8)

    def getNextState(self, board: np.ndarray, player: int, action: int):
        next_board = board.copy()
        next_board[0, 0, 0] += 1
        return next_board, -player

    def getGameEnded(self, board: np.ndarray, player: int) -> int:
        return 1 if int(board[0, 0, 0]) >= 16 else 0


class FakeMCTS:
    def __init__(self) -> None:
        self.temperatures: list[int] = []
        self.noise_flags: list[bool] = []

    def getActionProb(self, board, temp: int, add_dirichlet_noise: bool):
        self.temperatures.append(temp)
        self.noise_flags.append(add_dirichlet_noise)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def test_episode_creates_fresh_mcts_and_matches_temperature_threshold(monkeypatch) -> None:
    created: list[FakeMCTS] = []

    def fake_build_mcts(game, network, config):
        instance = FakeMCTS()
        created.append(instance)
        return instance

    monkeypatch.setattr(generate, "build_mcts", fake_build_mcts)
    config = protocol()
    first, result, termination = generate.run_episode(FakeGame(), object(), config, 0)
    second, _, _ = generate.run_episode(FakeGame(), object(), config, 1)

    assert len(created) == 2
    assert created[0].temperatures == [1] * 14 + [0] * 2
    assert all(created[0].noise_flags)
    assert result == 1
    assert termination == "win"
    assert first["values"].tolist() == [1.0, -1.0] * 8
    assert second["game_ids"].tolist() == [1] * 16


def test_verifier_enforces_schema_policy_support_and_value_sign() -> None:
    arrays = generate.finalize_episode(
        _history(), game_id=7, terminal_result=1, game_length=2
    )

    assert (
        verify.validate_arrays(
            arrays,
            action_size=3,
            board_shape=(4, 1, 1),
            game_id=7,
            game_length=2,
            terminal_result=1,
        )
        == 2
    )
    invalid = {name: value.copy() for name, value in arrays.items()}
    invalid["policies"][0] = np.asarray([0.9, 0.1, 0.0], dtype=np.float32)
    with pytest.raises(common.HoldoutError, match="invalid action"):
        verify.validate_arrays(
            invalid,
            action_size=3,
            board_shape=(4, 1, 1),
            game_id=7,
            game_length=2,
            terminal_result=1,
        )


class TinyModule(torch.nn.Module):
    def eval(self):
        return self


class TinyNetwork:
    def __init__(self) -> None:
        self.nnet = TinyModule()

    def _fwd(self, boards: torch.Tensor, logits: bool):
        assert logits is True
        batch = boards.shape[0]
        raw = torch.tensor([[2.0, -3.0, 0.0]], device=boards.device).repeat(batch, 1)
        values = torch.tensor([[0.25]], device=boards.device).repeat(batch, 1)
        return raw, values


def test_holdout_loss_matches_training_mask_softmax_and_mse() -> None:
    boards = np.zeros((1, 4, 1, 1), dtype=np.uint8)
    policies = np.asarray([[0.75, 0.0, 0.25]], dtype=np.float32)
    values = np.asarray([1.0], dtype=np.float32)
    valids = np.asarray([[1, 0, 1]], dtype=np.uint8)

    policy_loss, value_loss, total_loss = evaluate.loss_arrays(
        TinyNetwork(),
        boards,
        policies,
        values,
        valids,
        batch_size=1,
        device="cpu",
    )
    probabilities = torch.softmax(torch.tensor([2.0, float("-inf"), 0.0]), dim=0)
    expected_policy = float(
        -(0.75 * torch.log(probabilities[0] + 1e-8))
        - (0.25 * torch.log(probabilities[2] + 1e-8))
    )
    assert policy_loss[0] == pytest.approx(expected_policy)
    assert value_loss[0] == pytest.approx((1.0 - 0.25) ** 2)
    assert total_loss[0] == pytest.approx(policy_loss[0] + value_loss[0])


def test_cluster_bootstrap_is_seeded_and_resamples_complete_games() -> None:
    rows = [
        {
            "states": 2,
            "policy_loss": 1.0,
            "value_loss": 0.25,
            "total_loss": 1.25,
        },
        {
            "states": 8,
            "policy_loss": 3.0,
            "value_loss": 0.75,
            "total_loss": 3.75,
        },
    ]

    first = evaluate.cluster_bootstrap_intervals(
        rows, resamples=100, seed=72001, confidence_level=0.95
    )
    second = evaluate.cluster_bootstrap_intervals(
        rows, resamples=100, seed=72001, confidence_level=0.95
    )

    assert first == second
    assert set(first) == {"policy_loss", "value_loss", "total_loss"}


def test_shard_round_trip_preserves_exact_array_schema(tmp_path: Path) -> None:
    arrays = generate.finalize_episode(
        _history(), game_id=7, terminal_result=1, game_length=2
    )
    shard = tmp_path / "shards" / "game_0007.npz"

    common.atomic_write_npz(shard, arrays)
    loaded = common.load_npz(shard)

    assert set(loaded) == set(common.ARRAY_NAMES)
    for name in common.ARRAY_NAMES:
        assert np.array_equal(loaded[name], arrays[name])

    assert common.sha256_dataset_content(loaded) == common.sha256_dataset_content(
        arrays
    )
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["boards"][0, 0, 0, 0] = 1
    assert common.sha256_dataset_content(changed) != common.sha256_dataset_content(
        arrays
    )


def test_resume_accepts_one_committed_game_and_rejects_duplicate_record(
    tmp_path: Path,
) -> None:
    arrays = generate.finalize_episode(
        _history(), game_id=0, terminal_result=1, game_length=2
    )
    shard = tmp_path / "shards" / "game_0000.npz"
    common.atomic_write_npz(shard, arrays)
    record = {
        "game_id": 0,
        "game_seed": 71001,
        "shard_path": "shards/game_0000.npz",
        "shard_sha256": common.sha256_file(shard),
        "positions": 2,
        "illegal_actions": 0,
    }

    completed = generate._validate_resume_records([record], protocol(), tmp_path)
    assert list(completed) == [0]
    with pytest.raises(common.HoldoutError, match="duplicate"):
        generate._validate_resume_records([record, record], protocol(), tmp_path)


def test_holdout_scripts_are_not_imported_by_training_entrypoints() -> None:
    baseline_root = ANALYSIS.parent
    training_files = [
        baseline_root / "external" / "alphazero" / "Coach.py",
        baseline_root / "main.py",
    ]
    existing = [path for path in training_files if path.is_file()]
    assert existing
    for path in existing:
        assert "generate_holdout" not in path.read_text(encoding="utf-8")
        assert "evaluate_holdout" not in path.read_text(encoding="utf-8")


def test_adaptive_config_freezes_the_formal_holdout_content_hash() -> None:
    config_path = ANALYSIS.parents[1] / "extension" / "configs" / "adaptive_holdout_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    holdout = config["holdout"]

    assert holdout["dataset_path"] == (
        "../baseline/outputs/baseline_seed1001_4090_analysis/holdout_v1/states.npz"
    )
    assert holdout["expected_games"] == 200
    assert holdout["expected_states"] == 9259
    assert holdout["expected_content_sha256"] == (
        "33d77cab4fd08bbad5b66c5ef7e9f359a1aacfbc078f440fe180385228879194"
    )
    assert holdout["read_only"] is True
    assert holdout["excluded_from_replay"] is True
