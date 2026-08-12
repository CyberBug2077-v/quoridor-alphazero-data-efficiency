from __future__ import annotations

import copy
from pathlib import Path

import pytest

import run_smoke


def resolve_from_dict(config: dict, write_config) -> dict:
    config_path = write_config(config)
    loaded = run_smoke.load_config(config_path)
    return run_smoke.resolve_config(loaded, config_path)


def test_all_required_fields_are_present(smoke_config: dict) -> None:
    assert set(run_smoke.REQUIRED_FIELDS) <= set(smoke_config)
    for section, required_fields in run_smoke.REQUIRED_FIELDS.items():
        assert required_fields <= set(smoke_config[section])


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("run", "id"),
        ("self_play", "iterations"),
        ("training", "batch_size"),
    ],
)
def test_missing_required_field_is_rejected(
    smoke_config: dict,
    write_config,
    section: str,
    field: str,
) -> None:
    config = copy.deepcopy(smoke_config)
    del config[section][field]

    with pytest.raises(run_smoke.ConfigError, match="missing required field"):
        run_smoke.load_config(write_config(config))


def test_cuda_device_requires_cuda(
    loaded_config: tuple[dict, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path = loaded_config
    monkeypatch.setattr(run_smoke.torch.cuda, "is_available", lambda: False)

    with pytest.raises(run_smoke.ConfigError, match="CUDA is not available"):
        run_smoke.resolve_config(config, config_path)


def test_gpu_index_must_be_in_range(
    loaded_config: tuple[dict, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path = loaded_config
    config["run"]["gpu_index"] = 1
    monkeypatch.setattr(run_smoke.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_smoke.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(run_smoke.ConfigError, match="gpu_index 1 is invalid"):
        run_smoke.resolve_config(config, config_path)


def test_max_samples_must_cover_at_least_one_batch(
    smoke_config: dict,
    write_config,
) -> None:
    config = copy.deepcopy(smoke_config)
    config["training"]["max_samples"] = config["training"]["batch_size"] - 1

    with pytest.raises(run_smoke.ConfigError, match="max_samples must be >="):
        resolve_from_dict(config, write_config)


def test_mcts_simulations_must_cover_evaluation_batch(
    smoke_config: dict,
    write_config,
) -> None:
    config = copy.deepcopy(smoke_config)
    config["self_play"]["mcts_simulations"] = 3
    config["self_play"]["eval_mcts_in_batch"] = 4

    with pytest.raises(run_smoke.ConfigError, match="mcts_simulations must be >="):
        resolve_from_dict(config, write_config)


def test_mcts_simulations_must_be_divisible_by_evaluation_batch(
    smoke_config: dict,
    write_config,
) -> None:
    config = copy.deepcopy(smoke_config)
    config["self_play"]["mcts_simulations"] = 10
    config["self_play"]["eval_mcts_in_batch"] = 4

    with pytest.raises(run_smoke.ConfigError, match="must be divisible"):
        resolve_from_dict(config, write_config)


def test_disabled_update_gating_maps_to_zero_arena_games(
    resolved_config: dict,
) -> None:
    assert resolved_config["training"]["update_gating"] is False
    assert resolved_config["mapped_args"]["train_args"]["arenaCompare"] == 0


def test_disabled_dirichlet_noise_zeroes_effective_epsilon(
    resolved_config: dict,
) -> None:
    assert resolved_config["self_play"]["dirichlet_noise"] is False
    assert resolved_config["self_play"]["dirichlet_epsilon"] == 0.0
    assert (
        resolved_config["mapped_args"]["train_args"]["dirichlet_epsilon"]
        == 0.0
    )


def test_checkpoint_directory_is_inside_output_directory(
    resolved_config: dict,
) -> None:
    output_dir = Path(resolved_config["run"]["output_dir"])
    checkpoint_dir = Path(resolved_config["checkpoint"]["directory"])

    assert checkpoint_dir.is_relative_to(output_dir)
    assert checkpoint_dir == output_dir / "checkpoints"


def test_evaluation_games_per_opponent_must_be_even(
    smoke_config: dict,
    write_config,
) -> None:
    config = copy.deepcopy(smoke_config)
    config["evaluation"]["games_per_opponent"] = 3

    with pytest.raises(run_smoke.ConfigError, match="must be even"):
        resolve_from_dict(config, write_config)


def test_top_level_and_mapped_arguments_are_consistent(
    resolved_config: dict,
) -> None:
    self_play = resolved_config["self_play"]
    training = resolved_config["training"]
    checkpoint = resolved_config["checkpoint"]
    train_args = resolved_config["mapped_args"]["train_args"]
    nn_args = resolved_config["mapped_args"]["nn_args"]
    evaluation = resolved_config["evaluation"]
    evaluation_args = resolved_config["mapped_args"]["evaluation_args"]

    assert train_args["numIters"] == self_play["iterations"]
    assert train_args["numEps"] == self_play["games_per_iteration"]
    assert train_args["numMCTSSims"] == self_play["mcts_simulations"]
    assert train_args["eval_mcts_in_batch"] == self_play["eval_mcts_in_batch"]
    assert train_args["max_train_size"] == training["max_samples"]
    assert train_args["batch_size"] == training["batch_size"]
    assert train_args["lr"] == training["learning_rate"]
    assert train_args["checkpoint"] == checkpoint["directory"]

    assert nn_args["cuda"] is True
    assert nn_args["lr"] == training["learning_rate"]
    assert nn_args["epochs"] == training["epochs"]
    assert nn_args["batch_size"] == training["batch_size"]
    assert nn_args["num_channels"] == training["num_channels"]
    assert nn_args["num_res_blocks"] == training["num_res_blocks"]
    assert nn_args["attn_depth"] == training["attn_depth"]
    assert nn_args["num_heads"] == training["num_heads"]
    assert nn_args["clip"] == training["gradient_clip"]
    assert nn_args["use_amp"] == training["amp"]
    assert nn_args["amp_dtype"] == training["amp_dtype"]
    assert evaluation_args == evaluation


def test_save_interval_mapping_regression(resolved_config: dict) -> None:
    assert (
        resolved_config["mapped_args"]["train_args"][
            "save_every_n_iterations"
        ]
        == resolved_config["checkpoint"]["save_every_iterations"]
    )
