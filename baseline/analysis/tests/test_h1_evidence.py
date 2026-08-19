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
        {"start_checkpoint": 0, "qualifies": False},
        {"start_checkpoint": 20, "qualifies": False},
        {"start_checkpoint": 40, "qualifies": False},
    ]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_detected"] is False
    assert decision["plateau_iteration"] is None


def test_only_one_qualifying_window_is_not_a_plateau() -> None:
    results = [
        {"start_checkpoint": 0, "qualifies": False},
        {"start_checkpoint": 20, "qualifies": True},
        {"start_checkpoint": 40, "qualifies": False},
    ]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_detected"] is False
    assert decision["qualifying_window_indices"] == [1]


def test_plateau_decision_reports_all_qualifying_windows_and_full_first_run() -> None:
    results = [
        {"start_checkpoint": 0, "qualifies": True},
        {"start_checkpoint": 20, "qualifies": False},
        {"start_checkpoint": 40, "qualifies": True},
        {"start_checkpoint": 60, "qualifies": True},
        {"start_checkpoint": 80, "qualifies": True},
    ]
    decision = detect.detect_consecutive_plateau_windows(results)
    assert decision["plateau_iteration"] == 40
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
