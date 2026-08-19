from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Callable

import pytest
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detect = load_script("detect_plateau")
merge = load_script("merge_h1_evidence")

CHECKPOINTS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 210]
OPPONENTS = ["heuristic_20", "heuristic_200", "greedy_random_50", "random"]


def make_protocol(
    *, checkpoints: list[int] | None = None, games_per_opponent: int = 50
) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "fixed_basket_v1",
        "checkpoints": list(CHECKPOINTS if checkpoints is None else checkpoints),
        "games_per_opponent": games_per_opponent,
        "alternate_sides": True,
        "opponents": [{"id": opponent} for opponent in OPPONENTS],
    }


def make_games(
    protocol: dict,
    result_for: Callable[[int, str, int], str] | None = None,
) -> list[dict]:
    games = []
    games_per_opponent = protocol["games_per_opponent"]
    games_per_side = games_per_opponent // 2
    if result_for is None:
        outcomes = ("win", "loss", "draw", "win")
        result_for = lambda checkpoint, opponent, index: outcomes[index % 4]
    for checkpoint in protocol["checkpoints"]:
        for opponent in OPPONENTS:
            for game_index in range(games_per_opponent):
                result = result_for(checkpoint, opponent, game_index)
                games.append(
                    {
                        "checkpoint": checkpoint,
                        "opponent": opponent,
                        "game_index": game_index,
                        "model_color": (
                            "white" if game_index < games_per_side else "black"
                        ),
                        "model_result": result,
                        "termination": "max_turns" if result == "draw" else "win",
                        "fault": None,
                    }
                )
    return games


def make_summary_rows(games: list[dict], protocol: dict) -> list[dict]:
    reconstructed = detect.reconstruct_checkpoint_scores(games, protocol)
    rows = []
    for checkpoint in protocol["checkpoints"]:
        result = reconstructed[checkpoint]
        row = {
            "checkpoint": checkpoint,
            "total_games": result["total_games"],
            "wins": result["wins"],
            "draws": result["draws"],
            "losses": result["losses"],
            "score_rate": result["macro_score"],
        }
        row.update(
            {
                f"{opponent}_score": result["opponent_scores"][opponent]
                for opponent in OPPONENTS
            }
        )
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def formal_protocol() -> dict:
    return make_protocol()


@pytest.fixture(scope="module")
def formal_games(formal_protocol: dict) -> list[dict]:
    return make_games(formal_protocol)


def test_normal_plateau_case() -> None:
    checkpoints = [0, 20, 40, 60, 80, 100]
    protocol = make_protocol(checkpoints=checkpoints, games_per_opponent=4)

    def result_for(checkpoint: int, opponent: str, game_index: int) -> str:
        wins = 1 if checkpoint == 0 else 2
        return "win" if game_index < wins else "loss"

    games = make_games(protocol, result_for)
    reconstructed = detect.reconstruct_checkpoint_scores(games, protocol)
    results = []
    for index, window in enumerate(detect.build_rolling_windows(checkpoints)):
        bootstrap = detect.paired_stratified_bootstrap_slope(
            games,
            window,
            OPPONENTS,
            bootstrap_resamples=200,
            seed=1001 + index,
        )
        start_score = reconstructed[window[0]]["macro_score"]
        end_score = reconstructed[window[-1]]["macro_score"]
        results.append(
            {
                "start_checkpoint": window[0],
                "end_checkpoint": window[-1],
                "qualifies": detect.qualify_flat_window(
                    bootstrap["ci_low"],
                    bootstrap["ci_high"],
                    start_score,
                    end_score,
                ),
            }
        )

    assert [result["qualifies"] for result in results] == [False, True, True]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_detected"] is True
    assert decision["plateau_iteration"] == 20


def test_no_plateau_case() -> None:
    results = [
        {"start_checkpoint": 0, "end_checkpoint": 60, "qualifies": False},
        {"start_checkpoint": 20, "end_checkpoint": 80, "qualifies": False},
        {"start_checkpoint": 40, "end_checkpoint": 100, "qualifies": False},
    ]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_detected"] is False
    assert decision["plateau_iteration"] is None


def test_only_one_qualifying_window_is_not_a_plateau() -> None:
    results = [
        {"start_checkpoint": 0, "end_checkpoint": 60, "qualifies": False},
        {"start_checkpoint": 20, "end_checkpoint": 80, "qualifies": True},
        {"start_checkpoint": 40, "end_checkpoint": 100, "qualifies": False},
    ]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_detected"] is False
    assert decision["qualifying_window_indices"] == [1]


def test_plateau_decision_reports_all_qualifying_windows_and_full_first_run() -> None:
    results = [
        {"start_checkpoint": 0, "end_checkpoint": 60, "qualifies": True},
        {"start_checkpoint": 20, "end_checkpoint": 80, "qualifies": False},
        {"start_checkpoint": 40, "end_checkpoint": 100, "qualifies": True},
        {"start_checkpoint": 60, "end_checkpoint": 120, "qualifies": True},
        {"start_checkpoint": 80, "end_checkpoint": 140, "qualifies": True},
    ]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_iteration"] == 40
    assert decision["plateau_start_checkpoint"] == 40
    assert decision["first_qualifying_window_start"] == 40
    assert decision["first_qualifying_window_end"] == 100
    assert decision["confirmation_window_start"] == 60
    assert decision["confirmation_window_end"] == 120
    assert decision["plateau_confirmation_checkpoint"] == 120
    assert decision["consecutive_qualifying_windows"] == 3
    assert decision["qualifying_window_indices"] == [0, 2, 3, 4]


def test_rolling_windows_include_all_nine_and_the_short_final_spacing() -> None:
    windows = detect.build_rolling_windows(CHECKPOINTS)
    assert len(windows) == 9
    assert windows[0] == (0, 20, 40, 60)
    assert windows[-1] == (160, 180, 200, 210)


def test_ols_uses_real_checkpoint_iterations_for_200_to_210_spacing() -> None:
    checkpoints = [160, 180, 200, 210]
    scores = [0.16, 0.18, 0.20, 0.21]
    assert detect.ols_slope(checkpoints, scores) == pytest.approx(0.001)


def test_bootstrap_reuses_the_same_game_indices_across_checkpoints() -> None:
    checkpoints = [0, 20, 40, 60]
    protocol = make_protocol(checkpoints=checkpoints, games_per_opponent=4)
    games = make_games(protocol)
    result = detect.paired_stratified_bootstrap_slope(
        games,
        checkpoints,
        OPPONENTS,
        bootstrap_resamples=500,
        seed=31001,
    )
    assert result["observed_slope"] == pytest.approx(0.0, abs=1e-15)
    assert result["ci_low"] == pytest.approx(0.0, abs=1e-15)
    assert result["ci_high"] == pytest.approx(0.0, abs=1e-15)


def test_reconstructed_macro_score_counts_draws_and_is_not_raw_win_rate() -> None:
    protocol = make_protocol(checkpoints=[0], games_per_opponent=2)
    outcomes = {
        "heuristic_20": ("win", "draw"),
        "heuristic_200": ("loss", "loss"),
        "greedy_random_50": ("win", "win"),
        "random": ("draw", "draw"),
    }
    games = make_games(
        protocol,
        lambda checkpoint, opponent, index: outcomes[opponent][index],
    )
    reconstructed = detect.reconstruct_checkpoint_scores(games, protocol)[0]
    assert reconstructed["opponent_scores"] == {
        "heuristic_20": 0.75,
        "heuristic_200": 0.0,
        "greedy_random_50": 1.0,
        "random": 0.5,
    }
    assert reconstructed["macro_score"] == pytest.approx(0.5625)
    assert reconstructed["wins"] / reconstructed["total_games"] == pytest.approx(
        0.375
    )


def test_flat_window_requires_both_ci_and_endpoint_conditions() -> None:
    assert detect.qualify_flat_window(-0.01, 0.01, 0.80, 0.83) is True
    assert detect.qualify_flat_window(-0.01, 0.01, 0.80, 0.831) is False
    assert detect.qualify_flat_window(0.001, 0.01, 0.80, 0.81) is False


def test_incomplete_manifest_is_rejected(formal_protocol: dict) -> None:
    expected_games = 12 * 4 * 50
    manifest = {
        "protocol_id": "fixed_basket_v1",
        "evaluation_mode": "formal",
        "status": "running",
        "selected_checkpoints": CHECKPOINTS,
        "expected_evaluation_games": expected_games,
        "summary": {"status": "completed", "observed_games": expected_games},
    }
    with pytest.raises(detect.PlateauInputError, match="status is not completed"):
        detect.validate_evaluation_status(manifest, formal_protocol)


def test_missing_game_is_rejected(
    formal_protocol: dict, formal_games: list[dict]
) -> None:
    with pytest.raises(detect.PlateauInputError, match="expected 2400 games"):
        detect.validate_game_coverage(formal_games[:-1], formal_protocol)


def test_duplicate_game_key_is_rejected(
    formal_protocol: dict, formal_games: list[dict]
) -> None:
    games = list(formal_games)
    games[-1] = dict(games[0])
    with pytest.raises(detect.PlateauInputError, match="unique game keys"):
        detect.validate_unique_game_keys(games, formal_protocol)


def test_side_imbalance_is_rejected(
    formal_protocol: dict, formal_games: list[dict]
) -> None:
    games = list(formal_games)
    games[25] = {**games[25], "model_color": "white"}
    with pytest.raises(detect.PlateauInputError, match="side balance"):
        detect.validate_side_balance(games, formal_protocol)


def test_error_termination_is_rejected(formal_games: list[dict]) -> None:
    games = list(formal_games)
    games[0] = {**games[0], "termination": "bot_error"}
    with pytest.raises(detect.PlateauInputError, match="error termination"):
        detect.validate_terminations(games)


def test_summary_score_mismatch_is_rejected(
    formal_protocol: dict, formal_games: list[dict]
) -> None:
    reconstructed = detect.reconstruct_checkpoint_scores(
        formal_games, formal_protocol
    )
    rows = make_summary_rows(formal_games, formal_protocol)
    rows[0]["score_rate"] = float(rows[0]["score_rate"]) + 0.01
    with pytest.raises(detect.PlateauInputError, match="does not match reconstructed"):
        detect.validate_summary_scores(reconstructed, rows, formal_protocol)


def test_invalid_input_returns_2_without_formal_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], formal_protocol: dict
) -> None:
    gate_config = {
        "checkpoints": {"iterations": CHECKPOINTS},
        "plateau": {
            "metric": "fixed_basket_macro_score",
            "method": "rolling_ols_slope_with_paired_stratified_bootstrap",
            "window_checkpoints": 4,
            "bootstrap_seed": 81001,
            "bootstrap_resamples": 10000,
            "confidence_level": 0.95,
            "slope_unit_iterations": 20,
            "maximum_absolute_score_change_over_window": 0.03,
            "require_slope_confidence_interval_contains_zero": True,
            "consecutive_qualifying_windows": 2,
            "plateau_iteration": "first_checkpoint_of_first_qualifying_window",
        },
    }
    expected_games = 12 * 4 * 50
    manifest = {
        "protocol_id": "fixed_basket_v1",
        "evaluation_mode": "formal",
        "status": "completed",
        "selected_checkpoints": CHECKPOINTS,
        "expected_evaluation_games": expected_games,
        "summary": {"status": "completed", "observed_games": expected_games},
    }
    gate_path = tmp_path / "baseline_gate2.yaml"
    basket_path = tmp_path / "fixed_basket_v1.yaml"
    games_path = tmp_path / "games.jsonl"
    summary_path = tmp_path / "checkpoint_summary.csv"
    manifest_path = tmp_path / "evaluation_manifest.json"
    output_dir = tmp_path / "h1_v1"
    gate_path.write_text(yaml.safe_dump(gate_config), encoding="utf-8")
    basket_path.write_text(yaml.safe_dump(formal_protocol), encoding="utf-8")
    games_path.write_text("", encoding="utf-8")
    with summary_path.open("w", encoding="utf-8", newline="") as destination:
        csv.writer(destination).writerow(
            [
                "checkpoint",
                "total_games",
                "wins",
                "draws",
                "losses",
                "score_rate",
                *[f"{opponent}_score" for opponent in OPPONENTS],
            ]
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = detect.main(
        [
            "--gate-config",
            str(gate_path),
            "--basket-config",
            str(basket_path),
            "--games",
            str(games_path),
            "--checkpoint-summary",
            str(summary_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    assert "Input validation failed" in capsys.readouterr().err
    assert not (output_dir / "plateau_windows.csv").exists()
    assert not (output_dir / "plateau.json").exists()


def test_ratio_of_sums_is_not_mean_of_iteration_ratios() -> None:
    rows = [
        {"positions_generated": 10, "optimizer_steps": 2},
        {"positions_generated": 10, "optimizer_steps": 8},
    ]
    assert merge.aggregate_ratio_of_sums(
        rows, "positions_generated", "optimizer_steps"
    ) == pytest.approx(2.0)
    assert sum(
        row["positions_generated"] / row["optimizer_steps"] for row in rows
    ) / len(rows) == pytest.approx(3.125)


def test_checkpoint_block_boundaries_are_previous_plus_one_through_checkpoint() -> None:
    blocks = merge.build_checkpoint_blocks(CHECKPOINTS)
    assert blocks[0] == {
        "checkpoint": 0,
        "block_start_iteration": None,
        "block_end_iteration": None,
        "block_iterations": 0,
        "is_short_block": False,
    }
    assert blocks[1]["block_start_iteration"] == 1
    assert blocks[1]["block_end_iteration"] == 20
    assert blocks[10]["block_start_iteration"] == 181
    assert blocks[10]["block_end_iteration"] == 200


def test_checkpoint_210_is_a_ten_iteration_block() -> None:
    final = merge.build_checkpoint_blocks(CHECKPOINTS)[-1]
    assert final["block_start_iteration"] == 201
    assert final["block_end_iteration"] == 210
    assert final["block_iterations"] == 10
    assert final["is_short_block"] is True


def test_checkpoint_zero_gaps_are_missing_not_zero() -> None:
    [row] = merge.apply_observability_rules(
        [
            {
                "checkpoint": 0,
                "approx_policy_gap": 0.0,
                "approx_value_gap": 0.0,
                "approx_total_gap": 0.0,
            }
        ]
    )
    assert row["approx_policy_gap"] is None
    assert row["approx_value_gap"] is None
    assert row["approx_total_gap"] is None


def test_training_replay_proxies_are_available_before_iteration_61() -> None:
    [row] = merge.apply_observability_rules(
        [
            {
                "checkpoint": 40,
                "mean_sample_exposure": 1.0,
                "mean_sample_age": 2.0,
                "p90_sample_age": 3.0,
            }
        ]
    )
    assert row["mean_sample_exposure"] == pytest.approx(1.0)
    assert row["mean_sample_age"] == pytest.approx(2.0)
    assert row["p90_sample_age"] == pytest.approx(3.0)


def test_snapshot_replay_metrics_are_left_truncated_before_iteration_61() -> None:
    rows = merge.apply_observability_rules(
        [
            {
                "checkpoint": checkpoint,
                "incoming_unique_state_ratio": 0.5,
                "duplicate_rate": 0.4,
                "state_effective_ratio": 0.6,
            }
            for checkpoint in (60, 80)
        ]
    )
    assert rows[0]["incoming_unique_state_ratio"] is None
    assert rows[0]["duplicate_rate"] is None
    assert rows[0]["state_effective_ratio"] is None
    assert rows[1]["incoming_unique_state_ratio"] == pytest.approx(0.5)
    assert rows[1]["duplicate_rate"] == pytest.approx(0.4)
    assert rows[1]["state_effective_ratio"] == pytest.approx(0.6)


def test_turnover_is_unobservable_until_iteration_151() -> None:
    rows = merge.apply_observability_rules(
        [
            {"checkpoint": 140, "turnover_fraction": 0.2},
            {"checkpoint": 160, "turnover_fraction": 0.3},
        ]
    )
    assert rows[0]["turnover_fraction"] is None
    assert rows[0]["turnover_observable"] is False
    assert rows[1]["turnover_fraction"] == pytest.approx(0.3)


def _synthetic_merge_sources() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    baseline = []
    replay = []
    holdout = []
    fixed = []
    for block in merge.build_checkpoint_blocks(CHECKPOINTS):
        checkpoint = block["checkpoint"]
        baseline.append({**block, "fresh_states_per_update": 1.0})
        replay.append(
            {
                "checkpoint": checkpoint,
                "incoming_unique_state_ratio": 0.8,
                "duplicate_rate": 0.2,
                "state_effective_ratio": 0.8,
                "replay_observable": checkpoint >= 80,
            }
        )
        holdout.append(
            {
                "checkpoint": checkpoint,
                "holdout_policy_loss": 1.0,
                "holdout_value_loss": 1.0,
                "holdout_total_loss": 2.0,
                "logged_train_policy_loss": 0.9,
                "logged_train_value_loss": 0.9,
                "approx_policy_gap": 0.1,
                "approx_value_gap": 0.1,
            }
        )
        fixed.append(
            {
                "checkpoint": checkpoint,
                "score_rate": 0.5,
                "score_rate_ci95_low": 0.45,
                "score_rate_ci95_high": 0.55,
                **{field: 0.5 for field in merge.FIXED_OPPONENT_SCORE_COLUMNS},
            }
        )
    return baseline, replay, holdout, fixed


def test_multi_source_join_has_exactly_twelve_checkpoint_rows() -> None:
    baseline, replay, holdout, fixed = _synthetic_merge_sources()
    rows = merge.merge_checkpoint_sources(
        baseline,
        replay,
        holdout,
        fixed,
        {"plateau_detected": True, "plateau_iteration": 100},
    )
    merge.validate_aligned_table(rows, CHECKPOINTS)
    assert len(rows) == 12
    assert [row["checkpoint"] for row in rows] == CHECKPOINTS
    assert rows[0]["approx_policy_gap"] is None


def test_best_observed_tie_selects_earliest_checkpoint() -> None:
    rows = [
        {"checkpoint": 0, "fixed_basket_macro_score": 0.7},
        {"checkpoint": 20, "fixed_basket_macro_score": 0.9},
        {"checkpoint": 40, "fixed_basket_macro_score": 0.9},
    ]
    assert merge.find_best_observed_checkpoint(rows) == 20


def test_maximum_drawdown_marks_its_peak_and_trough() -> None:
    rows = [
        {"checkpoint": checkpoint, "fixed_basket_macro_score": score}
        for checkpoint, score in (
            (0, 0.8),
            (20, 0.9),
            (40, 0.7),
            (60, 0.95),
            (80, 0.65),
        )
    ]
    rows = merge.calculate_drawdowns(merge.calculate_running_best(rows))
    result = merge.find_maximum_drawdown(rows)
    assert result is not None
    assert result["maximum_drawdown"] == pytest.approx(0.30)
    assert result["peak_checkpoint"] == 60
    assert result["trough_checkpoint"] == 80


@pytest.mark.parametrize(
    (
        "input_status",
        "plateau",
        "supply",
        "replay",
        "snapshot",
        "gap",
        "temporal",
        "expected",
    ),
    [
        (
            "passed",
            True,
            "consistent",
            "consistent",
            "consistent",
            "consistent",
            "consistent_before_or_at_onset",
            "supported_with_limitations",
        ),
        (
            "passed",
            True,
            "consistent",
            "mixed",
            "unavailable",
            "unavailable",
            "consistent",
            "partially_supported",
        ),
        (
            "passed",
            True,
            "inconsistent",
            "consistent",
            "consistent",
            "consistent",
            "consistent",
            "not_supported",
        ),
        (
            "passed",
            False,
            "consistent",
            "consistent",
            "consistent",
            "consistent",
            "consistent",
            "not_assessable",
        ),
        (
            "failed",
            True,
            "consistent",
            "consistent",
            "consistent",
            "consistent",
            "consistent_before_or_at_onset",
            "not_assessable",
        ),
        (
            "passed",
            True,
            "unavailable",
            "consistent",
            "unavailable",
            "unavailable",
            "unavailable",
            "not_assessable",
        ),
    ],
)
def test_four_h1_outcomes(
    input_status: str,
    plateau: bool,
    supply: str,
    replay: str,
    snapshot: str,
    gap: str,
    temporal: str,
    expected: str,
) -> None:
    assert merge.judge_h1(
        input_status=input_status,
        plateau_detected=plateau,
        supply=supply,
        replay=replay,
        snapshot_diversity=snapshot,
        generalisation=gap,
        temporal_alignment=temporal,
    ) == expected


def test_no_plateau_uses_full_run_in_descriptive_only_mode() -> None:
    rows = [
        {
            "metric": "fresh_states_per_update",
            "analysis_grain": "training_iteration",
            "x_iteration": iteration,
            "x_gpu_hours": float(iteration),
            "value": value,
        }
        for iteration, value in ((20, 4.0), (40, 3.0), (60, 2.0), (80, 1.0))
    ]
    selected = merge.select_pre_plateau_range(
        rows,
        "fresh_states_per_update",
        "training_iteration",
        {"plateau_detected": False, "plateau_iteration": None},
    )
    assert selected["analysis_mode"] == "descriptive_full_run"
    assert selected["availability_status"] == "available"
    assert len(selected["rows"]) == 4
    assert "descriptive only" in selected["limitation"]


def test_early_plateau_uses_iteration_level_supply() -> None:
    raw_metrics = []
    derived_metrics = []
    for iteration in range(1, 41):
        positions = 1000 - iteration
        raw_metrics.append(
            {
                "iteration": iteration,
                "positions_generated": positions,
                "games_completed": 10,
                "optimizer_steps": 10,
                "replay_buffer_size": 10000 + iteration,
                "examples_used": 100,
                "samples_seen": 200,
                "iteration_seconds": 60.0,
                "self_play_seconds": 40.0,
                "training_seconds": 20.0,
                "cumulative_gpu_hours": iteration / 60.0,
            }
        )
        derived_metrics.append(
            {
                "iteration": iteration,
                "positions_generated": positions,
                "mean_sample_age": iteration / 2.0,
                "p90_sample_age": float(iteration),
                "turnover_fraction": 0.01,
            }
        )
    training_rows = merge.build_training_iteration_effect_rows(
        raw_metrics,
        derived_metrics,
        {
            "plateau_detected": True,
            "plateau_start_checkpoint": 40,
            "plateau_iteration": 40,
        },
    )
    effects = merge.build_effect_rows(
        training_rows,
        [],
        [],
        {
            "plateau_detected": True,
            "plateau_start_checkpoint": 40,
            "plateau_iteration": 40,
        },
        minimum_points=4,
    )
    fresh = next(
        row for row in effects if row["metric"] == "fresh_states_per_update"
    )
    assert fresh["analysis_grain"] == "training_iteration"
    assert fresh["start_iteration"] == 1
    assert fresh["end_iteration"] == 40
    assert fresh["valid_points"] == 40
    assert fresh["availability_status"] == "available"
    for metric in ("mean_sample_exposure", "mean_sample_age", "p90_sample_age"):
        effect = next(row for row in effects if row["metric"] == metric)
        assert effect["valid_points"] == 40
        assert effect["availability_status"] == "available"


def test_early_plateau_has_no_snapshot_effect_points() -> None:
    replay_rows = [
        {
            "iteration": iteration,
            "incoming_unique_state_ratio": 0.8,
            "incoming_ratio_left_censored": iteration == 61,
            "duplicate_rate": 0.2,
            "state_effective_ratio": 0.8,
        }
        for iteration in range(61, 81)
    ]
    points = merge.build_replay_iteration_effect_rows(
        replay_rows,
        {"plateau_detected": True, "plateau_start_checkpoint": 40},
    )
    effects = merge.build_effect_rows(
        [],
        points,
        [],
        {"plateau_detected": True, "plateau_start_checkpoint": 40},
    )
    assert points == []
    for metric in (
        "incoming_unique_state_ratio",
        "duplicate_rate",
        "state_effective_ratio",
    ):
        effect = next(row for row in effects if row["metric"] == metric)
        assert effect["analysis_grain"] == "replay_iteration"
        assert effect["valid_points"] == 0
        assert effect["availability_status"] == "unavailable"


def test_early_plateau_keeps_two_point_gap_unavailable() -> None:
    holdout_rows = [
        {
            "checkpoint": checkpoint,
            "gpu_hours": float(checkpoint),
            "approx_policy_gap": value,
            "approx_value_gap": value / 2.0,
        }
        for checkpoint, value in ((20, 0.1), (40, 0.2))
    ]
    checkpoint_rows = merge.build_checkpoint_effect_rows(
        holdout_rows,
        {"plateau_detected": True, "plateau_start_checkpoint": 40},
    )
    effects = merge.build_effect_rows(
        [],
        [],
        checkpoint_rows,
        {"plateau_detected": True, "plateau_start_checkpoint": 40},
        minimum_points=4,
    )
    policy_gap = next(
        row for row in effects if row["metric"] == "approx_policy_gap"
    )
    assert policy_gap["availability_status"] == "unavailable"
    assert policy_gap["analysis_grain"] == "evaluation_checkpoint"
    assert policy_gap["start_checkpoint"] == 20
    assert policy_gap["end_checkpoint"] == 40
    assert policy_gap["valid_points"] == 2


def test_h1_is_partial_when_gap_is_unavailable_but_other_stages_are_consistent() -> None:
    assert merge.judge_h1(
        input_status="passed",
        plateau_detected=True,
        supply="consistent",
        replay="consistent",
        snapshot_diversity="unavailable",
        generalisation="unavailable",
        temporal_alignment="consistent_before_or_at_onset",
    ) == "partially_supported"


@pytest.mark.parametrize(
    ("slope", "endpoint", "expected", "status"),
    [
        (-0.1, -0.2, "decrease", "consistent"),
        (0.1, 0.2, "decrease", "inconsistent"),
        (-0.1, 0.2, "decrease", "mixed"),
        (0.0, -0.2, "decrease", "mixed"),
    ],
)
def test_descriptive_direction_uses_slope_and_endpoint(
    slope: float, endpoint: float, expected: str, status: str
) -> None:
    result = merge.classify_metric_direction(slope, endpoint, expected)
    assert result["effect_status"] == status


def test_temporal_alignment_is_limited_at_early_onset() -> None:
    effects = [
        {
            "metric": "fresh_states_per_update",
            "evidence_stage": "supply",
            "analysis_grain": "training_iteration",
            "availability_status": "available",
            "effect_status": "consistent",
            "start_iteration": 1,
            "end_iteration": 40,
        },
        {
            "metric": "mean_sample_age",
            "evidence_stage": "replay",
            "analysis_grain": "training_iteration",
            "availability_status": "available",
            "effect_status": "consistent",
            "start_iteration": 1,
            "end_iteration": 40,
        },
    ]
    result = merge.judge_temporal_alignment(
        effects,
        {"plateau_detected": True, "plateau_start_checkpoint": 40},
    )
    assert result["status"] == "consistent_before_or_at_onset"
    assert "not change-point detection" in result["rationale"]


def test_h1_v1_1_freezes_source_native_grains_and_observability() -> None:
    path = SCRIPTS.parent / "configs" / "h1_v1_1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["protocol_id"] == "h1_v1_1"
    assert config["revision"]["supersedes"] == "h1_v1"
    assert config["trend_analysis"]["minimum_valid_points"] == 4
    assert config["trend_analysis"]["method"] == "descriptive_ols"
    assert config["trend_analysis"]["significance_test"] == "none"
    assert config["trend_analysis"]["classification"] == {
        "consistent": "slope_and_endpoint_match_expected_direction",
        "mixed": "slope_and_endpoint_disagree_or_one_is_flat",
        "inconsistent": "slope_and_endpoint_both_opposite",
        "unavailable": "fewer_than_minimum_valid_points",
    }
    assert config["effect_output"]["fields"] == list(merge.EFFECT_FIELDS)
    assert "minimum_valid_checkpoints" not in config["trend_analysis"]

    expected_grains = {
        "fresh_states_per_update": "training_iteration",
        "buffer_inflow_fraction": "training_iteration",
        "states_per_gpu_hour": "training_iteration",
        "mean_sample_exposure": "training_iteration",
        "mean_sample_age": "training_iteration",
        "p90_sample_age": "training_iteration",
        "incoming_unique_state_ratio": "replay_iteration",
        "duplicate_rate": "replay_iteration",
        "state_effective_ratio": "replay_iteration",
        "approx_policy_gap": "evaluation_checkpoint",
        "approx_value_gap": "evaluation_checkpoint",
    }
    assert {
        metric: config["metrics"][metric]["analysis_grain"]
        for metric in expected_grains
    } == expected_grains
    assert {
        metric: config["metrics"][metric]["eligible_iteration_start"]
        for metric in (
            "mean_sample_exposure",
            "mean_sample_age",
            "p90_sample_age",
            "incoming_unique_state_ratio",
            "duplicate_rate",
            "state_effective_ratio",
            "turnover_fraction",
        )
    } == {
        "mean_sample_exposure": 1,
        "mean_sample_age": 1,
        "p90_sample_age": 1,
        "incoming_unique_state_ratio": 61,
        "duplicate_rate": 61,
        "state_effective_ratio": 61,
        "turnover_fraction": 151,
    }
    assert config["outputs"]["root"] == (
        "outputs/baseline_seed1001_4090_analysis/h1_v1_1"
    )
