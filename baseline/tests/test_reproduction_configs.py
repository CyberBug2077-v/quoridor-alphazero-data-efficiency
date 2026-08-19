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
        "micro_batch_size": 1024,
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
    assert pretraining["pretraining"]["batch_size"] == 2048
    assert pretraining["pretraining"]["micro_batch_size"] == 1024
    assert pilot["training"]["batch_size"] == 2048
    assert pilot["training"]["micro_batch_size"] == 2048
    assert pilot["training"] == baseline["training"]


def test_pilot_and_baseline_share_one_fresh_initialization() -> None:
    pilot = load_config("baseline_pilot.yaml")
    baseline = load_config("baseline_reproduction.yaml")

    assert pilot["initialization"] == baseline["initialization"]
    initialization = pilot["initialization"]
    assert initialization["mode"] == "pretrained_checkpoint"
    assert initialization["checkpoint_path"] == (
        "outputs/pretraining_reproduction_seed1001/checkpoints/checkpoint_0.pth.tar"
    )
    assert initialization["expected_sha256"] == (
        "4824a2a8ba1c1ebb5a38a992af075a45a033b87b403973b583ab98a079f35667"
    )
    assert initialization["load_weights"] is True
    assert initialization["load_replay"] is False
    assert initialization["load_optimizer_state"] is False
    assert initialization["load_tracker_state"] is False
    assert initialization["start_iteration"] == 1


def test_pilot_and_baseline_algorithm_conditions_match() -> None:
    pilot = load_config("baseline_pilot.yaml")
    baseline = load_config("baseline_reproduction.yaml")

    for section in ("model", "training", "replay", "instrumentation", "logging"):
        assert pilot[section] == baseline[section], section
    pilot_self_play = {key: value for key, value in pilot["self_play"].items() if key != "iterations"}
    baseline_self_play = {key: value for key, value in baseline["self_play"].items() if key != "iterations"}
    assert pilot_self_play == baseline_self_play
    pilot_evaluation = {
        key: value
        for key, value in pilot["evaluation"].items()
        if key != "evaluate_every_iterations"
    }
    baseline_evaluation = {
        key: value
        for key, value in baseline["evaluation"].items()
        if key != "evaluate_every_iterations"
    }
    assert pilot_evaluation == baseline_evaluation

    assert pilot["self_play"]["iterations"] == 7
    assert pilot["self_play"]["eval_mcts_in_batch"] == 10
    assert baseline["self_play"]["eval_mcts_in_batch"] == 10
    assert pilot["evaluation"]["eval_mcts_in_batch"] == 4
    assert baseline["evaluation"]["eval_mcts_in_batch"] == 4
    assert pilot["training"]["update_gating"] is False
    assert pilot["training"]["batch_size"] == 2048
    assert pilot["training"]["micro_batch_size"] == 2048
    assert "expert_top_up" not in walk_keys(pilot)
    assert "expert_top_up" not in walk_keys(baseline)


def test_pilot_exercises_resume_metrics_and_read_only_evaluation() -> None:
    pilot = load_config("baseline_pilot.yaml")

    assert pilot["checkpoint"] == {
        "directory": "outputs/baseline_pilot_seed1001_4090/checkpoints",
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


def test_formal_budget_and_cadence_are_frozen() -> None:
    baseline = load_config("baseline_reproduction.yaml")

    assert baseline["run"]["id"] == "baseline_reproduction_seed1001_4090"
    assert baseline["run"]["seed"] == 1001
    assert baseline["run"]["output_dir"] == (
        "outputs/baseline_reproduction_seed1001_4090"
    )
    assert baseline["budget"] == {
        "max_gpu_hours": 24,
        "max_wall_clock_hours": None,
        "max_iterations": 210,
    }
    assert baseline["self_play"]["iterations"] == 210
    assert baseline["checkpoint"] == {
        "directory": "outputs/baseline_reproduction_seed1001_4090/checkpoints",
        "save_every_iterations": 10,
        "save_replay_state": True,
        "save_instrumentation_state": True,
        "save_rng_state": True,
        "compute_sha256": True,
    }
    assert baseline["evaluation"]["evaluate_every_iterations"] == 20
    assert baseline["evaluation"]["games_per_opponent"] == 2
    cadence = baseline["evaluation"]["evaluate_every_iterations"]
    final_iteration = baseline["self_play"]["iterations"]
    planned_evaluations = [
        0,
        *range(cadence, final_iteration, cadence),
        final_iteration,
    ]
    assert planned_evaluations == [
        0,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        210,
    ]


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


def test_baseline_gate2_analysis_protocol_is_frozen() -> None:
    path = BASELINE_ROOT / "analysis" / "configs" / "baseline_gate2.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)

    assert config["analysis"]["input_run_id"] == (
        "baseline_reproduction_seed1001_4090"
    )
    assert config["checkpoints"]["iterations"] == [
        0,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        210,
    ]
    assert config["replay"]["history_iterations"] == 150
    assert config["state_hash"]["definition"] == (
        "SHA256(dtype + shape + contiguous canonical-board bytes)"
    )

    # The fixed-basket protocol is authoritative only in fixed_basket_v1.yaml.
    assert "fixed_basket" not in config

    assert config["holdout"]["seed"] == 71001
    assert config["holdout"]["games"] == 200
    assert config["plateau"]["method"] == (
        "rolling_ols_slope_with_paired_stratified_bootstrap"
    )
    assert config["outputs"]["root"] == (
        "outputs/baseline_reproduction_seed1001_4090/gate2"
    )
    assert config["outputs"]["derived_metrics"].endswith("/derived_metrics.csv")
    assert config["outputs"]["baseline_resource_summary"].endswith(
        "/baseline_resource_summary.json"
    )
    assert config["outputs"]["data_quality_report"].endswith(
        "/data_quality_report.json"
    )
    fixed_basket_root = (
        "outputs/baseline_seed1001_4090_analysis/fixed_basket_v1"
    )
    assert {
        key: value
        for key, value in config["outputs"].items()
        if key.startswith("fixed_basket_")
    } == {
        "fixed_basket_root": fixed_basket_root,
        "fixed_basket_protocol": f"{fixed_basket_root}/protocol.resolved.yaml",
        "fixed_basket_manifest": f"{fixed_basket_root}/evaluation_manifest.json",
        "fixed_basket_games": f"{fixed_basket_root}/games.jsonl",
        "fixed_basket_checkpoint_summary": (
            f"{fixed_basket_root}/checkpoint_summary.csv"
        ),
        "fixed_basket_opponent_summary": (
            f"{fixed_basket_root}/opponent_summary.csv"
        ),
        "fixed_basket_elo_summary": f"{fixed_basket_root}/elo_summary.csv",
        "fixed_basket_evaluation_log": f"{fixed_basket_root}/evaluation.log",
    }
    assert config["baseline_metrics"]["buffer_fraction_consumed"] == (
        "examples_used / replay_buffer_size"
    )
    assert config["baseline_metrics"]["mean_sample_exposure"] == (
        "samples_seen / replay_buffer_size"
    )
    assert config["baseline_metrics"]["selected_sample_reuse"] == (
        "samples_seen / examples_used"
    )
    assert config["outputs"]["replay_iteration_metrics"] == (
        "outputs/baseline_seed1001_4090_analysis/replay/replay_iteration_stats.csv"
    )
    assert config["outputs"]["replay_summary"] == (
        "outputs/baseline_seed1001_4090_analysis/replay/replay_final_summary.json"
    )
    assert config["outputs"]["trajectory_stats"] == (
        "outputs/baseline_seed1001_4090_analysis/replay/trajectory_stats.csv"
    )
    assert config["outputs"]["plateau_windows"] == (
        "outputs/baseline_seed1001_4090_analysis/h1_v1/plateau_windows.csv"
    )
    assert config["outputs"]["plateau_result"] == (
        "outputs/baseline_seed1001_4090_analysis/h1_v1/plateau.json"
    )


def test_h1_analysis_protocol_is_frozen() -> None:
    path = BASELINE_ROOT / "analysis" / "configs" / "h1_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)

    assert config["protocol_id"] == "h1_v1"
    assert config["checkpoint_grid"]["iterations"] == [
        0,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        210,
    ]
    assert config["interval_aggregation"]["interval"] == (
        "previous_checkpoint < iteration <= checkpoint"
    )
    assert config["trend_analysis"]["minimum_valid_checkpoints"] == 4
    assert config["plateau_source"] == {
        "config": "analysis/configs/baseline_gate2.yaml",
        "section": "plateau",
        "metric": "fixed_basket_macro_score",
        "copy_thresholds_into_h1_config": False,
    }

    expected_directions = {
        "fresh_states_per_update": "decrease",
        "mean_sample_exposure": "increase",
        "mean_sample_age": "increase",
        "p90_sample_age": "increase",
        "turnover_fraction": "decrease",
        "incoming_unique_state_ratio": "decrease",
        "duplicate_rate": "increase",
        "state_effective_ratio": "decrease",
        "buffer_inflow_fraction": "decrease",
        "states_per_gpu_hour": "not_increase",
        "approx_policy_gap": "increase",
        "approx_value_gap": "increase",
        "approx_total_gap": "increase",
        "fixed_basket_macro_score": "later_plateau",
    }
    assert {
        metric: definition["expected_direction"]
        for metric, definition in config["metrics"].items()
    } == expected_directions
    assert config["metrics"]["turnover_fraction"][
        "eligible_iteration_start"
    ] == 151
    assert config["metrics"]["incoming_unique_state_ratio"][
        "exclude_rows_where"
    ] == {"incoming_ratio_left_censored": True}
    assert config["missing_values"][
        "fewer_than_four_valid_trend_points"
    ] == "unavailable"
    assert config["h1_classification"]["score_or_point_total"] == "none"
    assert config["outputs"]["root"] == (
        "outputs/baseline_seed1001_4090_analysis/h1_v1"
    )
    assert config["outputs"]["plateau_windows"] == (
        "outputs/baseline_seed1001_4090_analysis/h1_v1/plateau_windows.csv"
    )
    assert config["outputs"]["plateau_result"] == (
        "outputs/baseline_seed1001_4090_analysis/h1_v1/plateau.json"
    )
