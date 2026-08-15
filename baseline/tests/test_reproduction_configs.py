from __future__ import annotations

from pathlib import Path

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = BASELINE_ROOT / "configs"


def load_config(name: str) -> dict:
    path = CONFIG_ROOT / name
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{name} must contain a YAML mapping"
    return loaded


def walk_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(walk_keys(nested))
    return keys


def test_pretraining_is_supervised_only() -> None:
    config = load_config("pretraining_reproduction.yaml")

    assert set(config) == {
        "run",
        "data",
        "model",
        "pretraining",
        "checkpoint",
        "logging",
    }
    forbidden_keys = {
        "self_play",
        "games_per_iteration",
        "mcts_simulations",
        "replay",
        "adaptive_scheduler",
        "evaluation",
        "expert_top_up",
        "expert_data",
    }
    assert walk_keys(config).isdisjoint(forbidden_keys)

    assert config["pretraining"] == {
        "epochs": 10,
        "batch_size": 2048,
        "learning_rate": 0.0005,
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "amp": True,
        "amp_dtype": "bf16",
    }
    assert config["checkpoint"]["checkpoint_0_filename"] == "checkpoint_0.pth.tar"
    assert config["checkpoint"]["best_filename"] == "best.pth.tar"


def test_all_reproduction_configs_share_the_model() -> None:
    pretraining = load_config("pretraining_reproduction.yaml")
    pilot = load_config("baseline_pilot.yaml")
    baseline = load_config("baseline_reproduction.yaml")

    assert pretraining["model"] == pilot["model"] == baseline["model"]


def test_pilot_and_baseline_share_one_fresh_initialization() -> None:
    pilot = load_config("baseline_pilot.yaml")
    baseline = load_config("baseline_reproduction.yaml")

    assert pilot["initialization"] == baseline["initialization"]
    initialization = pilot["initialization"]
    assert initialization["mode"] == "pretrained_checkpoint"
    assert initialization["checkpoint_path"].endswith("/checkpoint_0.pth.tar")
    assert initialization["load_weights"] is True
    assert initialization["load_replay"] is False
    assert initialization["load_optimizer_state"] is False
    assert initialization["load_tracker_state"] is False
    assert initialization["start_iteration"] == 1


def test_pilot_and_baseline_algorithm_conditions_match() -> None:
    pilot = load_config("baseline_pilot.yaml")
    baseline = load_config("baseline_reproduction.yaml")

    for section in ("model", "training", "replay"):
        assert pilot[section] == baseline[section], section
    pilot_self_play = {key: value for key, value in pilot["self_play"].items() if key != "iterations"}
    baseline_self_play = {key: value for key, value in baseline["self_play"].items() if key != "iterations"}
    assert pilot_self_play == baseline_self_play

    assert pilot["self_play"]["iterations"] == 7
    assert pilot["training"]["update_gating"] is False
    assert "expert_top_up" not in walk_keys(pilot)
    assert "expert_top_up" not in walk_keys(baseline)


def test_pilot_exercises_resume_metrics_and_read_only_evaluation() -> None:
    pilot = load_config("baseline_pilot.yaml")

    assert pilot["checkpoint"] == {
        "directory": "outputs/baseline_pilot_seed1001/checkpoints",
        "save_every_iterations": 1,
        "save_replay_state": True,
        "save_instrumentation_state": True,
        "save_rng_state": True,
        "compute_sha256": True,
    }
    assert pilot["evaluation"]["evaluate_every_iterations"] == 1
    assert pilot["evaluation"]["games_per_opponent"] == 2
    assert pilot["evaluation"]["alternate_starting_player"] is True
    assert all(
        value is True
        for key, value in pilot["instrumentation"].items()
        if key != "tracker_file"
    )


def test_formal_budget_and_cadence_await_benchmark() -> None:
    baseline = load_config("baseline_reproduction.yaml")

    assert baseline["budget"] == {
        "max_gpu_hours": None,
        "max_wall_clock_hours": None,
        "max_iterations": None,
    }
    assert baseline["checkpoint"]["save_every_iterations"] is None
    assert baseline["evaluation"]["evaluate_every_iterations"] is None
    assert baseline["evaluation"]["games_per_opponent"] is None


def test_every_config_has_standard_logging_files() -> None:
    expected = {
        "metrics_file": "metrics.jsonl",
        "metadata_file": "run_metadata.json",
        "evaluation_file": "evaluation.json",
        "summary_file": "summary.json",
    }

    for name in (
        "pretraining_reproduction.yaml",
        "baseline_pilot.yaml",
        "baseline_reproduction.yaml",
    ):
        logging = load_config(name)["logging"]
        expected_for_config = dict(expected)
        if name == "pretraining_reproduction.yaml":
            expected_for_config["metrics_file"] = "pretraining_metrics.jsonl"
        assert {key: logging[key] for key in expected_for_config} == expected_for_config
