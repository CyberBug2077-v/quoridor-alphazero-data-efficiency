from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = SOURCE_ROOT / "experiments"
SCRIPTS = EXPERIMENTS / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_h3 as h3
from head_to_head_stats import colour_stratified_bootstrap
from head_to_head_stats_v3 import seed_pair_bootstrap, trajectory_diversity


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_v2_protocol_points_to_v2_evaluations_and_output() -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "h3_v2.yaml").read_text(encoding="utf-8")
    )

    assert config["config_id"] == "h3_v2"
    assert config["outputs"]["root"].endswith("/h3_v2")
    assert config["inputs"]["adaptive_fixed_basket_records"]["path"].endswith(
        "/fixed_basket_v2/games.jsonl"
    )
    assert config["inputs"]["head_to_head_records"]["path"].endswith(
        "/head_to_head_v2/games.jsonl"
    )
    assert config["inputs"]["h2_protocol"]["path"] == "configs/h2_v2.yaml"
    assert config["attribution"]["h3_decision_independent_of_h2_decision"] is True


def test_v3_protocol_points_to_trajectory_audited_head_to_head() -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "h3_v3.yaml").read_text(encoding="utf-8")
    )

    assert config["config_id"] == "h3_v3"
    assert config["outputs"]["root"].endswith("/h3_v3")
    assert config["inputs"]["head_to_head_protocol"]["path"] == (
        "configs/head_to_head_v3.yaml"
    )
    assert config["inputs"]["head_to_head_records"]["path"].endswith(
        "/head_to_head_v3/games.jsonl"
    )
    assert config["other_strength_metrics"]["final_head_to_head"][
        "minimum_unique_trajectories_per_adaptive_colour"
    ] == 2


def _v3_head_records(*, diverse: bool) -> list[dict]:
    records: list[dict] = []
    for pair_index in range(50):
        for colour in ("white", "black"):
            variant = pair_index % 2 if diverse else 0
            records.append(
                {
                    "stable_game_key": f"{pair_index}-{colour}",
                    "seed_pair_index": pair_index,
                    "game_seed": pair_index + 500,
                    "adaptive_color": colour,
                    "adaptive_result": "win" if colour == "white" else "draw",
                    "technically_valid": True,
                    "moves": [
                        {
                            "player": colour,
                            "type": "pawn",
                            "row": variant,
                            "col": 4,
                        }
                    ],
                }
            )
    return records


def _v3_head_summary(records: list[dict]) -> dict:
    interval = seed_pair_bootstrap(records, resamples=10000, seed=92001)
    diversity = trajectory_diversity(records, minimum_unique_per_colour=2)
    return {
        "status": "completed",
        "output_sha256": {"games": "f" * 64},
        "checkpoint_selection": {
            "best_checkpoint_selection_used": False,
            "baseline": {"actual_gpu_hours": 20.0},
            "adaptive": {"actual_gpu_hours": 19.9},
        },
        "trajectory_diversity": diversity,
        "adaptive_score": {
            "score_rate": interval["score_rate"],
            "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"],
            "bootstrap": {
                "method": "nonparametric_seed_pair_bootstrap",
                "resamples": 10000,
                "seed": 92001,
                "preserve_seed_pairs": 50,
            },
        },
    }


def test_h3_v3_recomputes_paired_ci_and_rejects_duplicate_trajectories() -> None:
    diverse = _v3_head_records(diverse=True)
    interval = h3._validate_head_to_head(
        diverse,
        _v3_head_summary(diverse),
        20.004163361943395,
        "f" * 64,
        2,
    )

    assert interval["method"] == "nonparametric_seed_pair_bootstrap"
    assert interval["seed_pairs"] == 50
    duplicated = _v3_head_records(diverse=False)
    with pytest.raises(h3.H3Error, match="trajectory diversity failed"):
        h3._validate_head_to_head(
            duplicated,
            _v3_head_summary(duplicated),
            20.004163361943395,
            "f" * 64,
            2,
        )


def _make_complete_inputs(tmp_path: Path) -> Path:
    frozen = yaml.safe_load(
        (EXPERIMENTS / "configs" / "h3_v2.yaml").read_text(encoding="utf-8")
    )
    source_matched = yaml.safe_load(
        (EXPERIMENTS / "configs" / "matched_compute_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    targets = source_matched["pairing_and_randomness"]["checkpoint_grid"]["targets"]
    matched = {
        "compute_budget": {"common_horizon": {"maximum_gpu_hours": 24.0}},
        "pairing_and_randomness": {"checkpoint_grid": {"targets": targets}},
    }
    matched_path = tmp_path / "matched.yaml"
    matched_path.write_text(yaml.safe_dump(matched, sort_keys=False), encoding="utf-8")

    baseline_games: list[dict] = []
    adaptive_games: list[dict] = []
    baseline_summary: list[dict] = []
    adaptive_summary: list[dict] = []
    joint: list[dict] = []
    for target_index, target_specification in enumerate(targets):
        target = int(target_specification["baseline_checkpoint_iteration"])
        gpu_hours = float(target_specification["gpu_hours"])
        baseline_score = 0.80
        adaptive_score = 0.84
        summary_common = {
            "checkpoint": target,
            "gpu_hours": 0.1,
            "total_games": 200,
            "wins": 160,
            "losses": 40,
            "draws": 0,
            "win_rate": baseline_score,
            "draw_rate": 0.0,
            "mean_game_length": 30.0,
            "mean_move_time": 0.01,
            "invalid_moves": 0,
            "bot_errors": 0,
            "max_turn_draws": 0,
            "js_invalid_proposals": 0,
            "model_fallbacks": 0,
            "score_rate_ci95_low": 0.7,
            "score_rate_ci95_high": 0.9,
            "bootstrap_iterations": 10000,
        }
        baseline_summary.append(
            {
                **summary_common,
                "score_rate": baseline_score,
                **{field: baseline_score for field in h3.OPPONENT_SCORE_FIELDS},
            }
        )
        adaptive_summary.append(
            {
                **summary_common,
                "wins": 168,
                "losses": 32,
                "win_rate": adaptive_score,
                "score_rate": adaptive_score,
                **{field: adaptive_score for field in h3.OPPONENT_SCORE_FIELDS},
                "target_baseline_iteration": target,
                "target_gpu_hours": gpu_hours,
                "selected_adaptive_iteration": target_index,
                "selected_adaptive_gpu_hours": max(0.0, gpu_hours - 0.01),
            }
        )
        for opponent in h3.EXPECTED_OPPONENTS:
            for game_index in range(50):
                common = {
                    "checkpoint": target,
                    "opponent": opponent,
                    "game_index": game_index,
                    "game_seed": target_index * 1000 + game_index,
                    "model_color": "white" if game_index < 25 else "black",
                    "fault": None,
                    "termination": "win",
                }
                baseline_games.append(
                    {
                        **common,
                        "checkpoint_sha256": "a" * 64 if target else "0" * 64,
                        "model_result": "win" if game_index < 40 else "loss",
                    }
                )
                adaptive_games.append(
                    {
                        **common,
                        "checkpoint_sha256": "b" * 64 if target else "0" * 64,
                        "model_result": "win" if game_index < 42 else "loss",
                    }
                )
        baseline_elo = 400.0 + target_index
        adaptive_elo = baseline_elo + 30.0
        joint.append(
            {
                "target_baseline_iteration": target,
                "target_gpu_hours": gpu_hours,
                "selected_adaptive_iteration": target_index,
                "selected_adaptive_gpu_hours": max(0.0, gpu_hours - 0.01),
                "baseline_participant": f"baseline_checkpoint_{target}",
                "adaptive_participant": f"adaptive_checkpoint_{target}",
                "baseline_joint_elo": baseline_elo,
                "baseline_joint_elo_ci95_low": baseline_elo - 10.0,
                "baseline_joint_elo_ci95_high": baseline_elo + 10.0,
                "adaptive_joint_elo": adaptive_elo,
                "adaptive_joint_elo_ci95_low": adaptive_elo - 10.0,
                "adaptive_joint_elo_ci95_high": adaptive_elo + 10.0,
                "adaptive_minus_baseline_elo": 30.0,
                "adaptive_minus_baseline_elo_ci95_low": 10.0,
                "adaptive_minus_baseline_elo_ci95_high": 50.0,
                "fit_id": "joint-fit-1",
                "fit_scope": "one_joint_model_over_all_baseline_adaptive_and_shared_anchor_games",
                "fit_method": "synthetic",
                "bootstrap_iterations": 10000,
                "bootstrap_seed": 94001,
            }
        )

    baseline_games_path = tmp_path / "baseline_games.jsonl"
    adaptive_games_path = tmp_path / "adaptive_games.jsonl"
    baseline_summary_path = tmp_path / "baseline_summary.csv"
    adaptive_summary_path = tmp_path / "adaptive_summary.csv"
    joint_path = tmp_path / "joint.csv"
    _write_jsonl(baseline_games_path, baseline_games)
    _write_jsonl(adaptive_games_path, adaptive_games)
    _write_csv(baseline_summary_path, baseline_summary)
    _write_csv(adaptive_summary_path, adaptive_summary)
    _write_csv(joint_path, joint)

    head_records: list[dict] = []
    for pair_index in range(50):
        for color in ("white", "black"):
            position = pair_index
            result = "win" if position < 30 else "draw" if position < 40 else "loss"
            head_records.append(
                {
                    "stable_game_key": f"{pair_index:02d}-{color}",
                    "seed_pair_index": pair_index,
                    "game_seed": pair_index + 100,
                    "adaptive_color": color,
                    "adaptive_result": result,
                    "technically_valid": True,
                }
            )
    interval = colour_stratified_bootstrap(head_records, resamples=10000, seed=92001)
    head_games_path = tmp_path / "head_games.jsonl"
    head_summary_path = tmp_path / "head_summary.json"
    _write_jsonl(head_games_path, head_records)
    head_games_sha256 = h3._sha256_file(head_games_path)
    head_summary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "output_sha256": {"games": head_games_sha256},
                "checkpoint_selection": {
                    "best_checkpoint_selection_used": False,
                    "baseline": {"actual_gpu_hours": targets[-1]["gpu_hours"]},
                    "adaptive": {
                        "actual_gpu_hours": targets[-1]["gpu_hours"] - 0.1
                    },
                },
                "adaptive_score": {
                    "score_rate": interval["score_rate"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                    "bootstrap": {
                        "resamples": 10000,
                        "seed": 92001,
                        "preserve_games_per_colour": 50,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    h2_decision_path = tmp_path / "h2_decision.json"
    h2_decision_path.write_text(
        json.dumps({"hypothesis": "H2", "status": "supported"}),
        encoding="utf-8",
    )
    dummy_paths = {
        "adaptive_fixed_basket_protocol": tmp_path / "adaptive_protocol.yaml",
        "head_to_head_protocol": tmp_path / "head_protocol.yaml",
        "h2_protocol": tmp_path / "h2_protocol.yaml",
    }
    for path in dummy_paths.values():
        path.write_text("config_id: synthetic\n", encoding="utf-8")
    paths = {
        "matched_compute": matched_path,
        "baseline_fixed_basket_records": baseline_games_path,
        "baseline_fixed_basket_checkpoint_summary": baseline_summary_path,
        "adaptive_fixed_basket_records": adaptive_games_path,
        "adaptive_fixed_basket_checkpoint_summary": adaptive_summary_path,
        "joint_elo_summary": joint_path,
        "head_to_head_records": head_games_path,
        "head_to_head_summary": head_summary_path,
        "h2_decision": h2_decision_path,
        **dummy_paths,
    }
    for name, path in paths.items():
        frozen["inputs"][name] = {"path": path.as_posix(), "sha256": None}
    config_path = tmp_path / "h3.yaml"
    config_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    return config_path


def test_complete_h3_is_recomputable_and_h2_only_controls_attribution(
    tmp_path: Path,
) -> None:
    config_path = _make_complete_inputs(tmp_path)
    output = tmp_path / "output"

    status = h3.execute(config_path, output)

    assert status == "supported"
    aligned = _read_csv(output / "aligned_strength.csv")
    utility = _read_csv(output / "utility_curve.csv")
    effects = _read_csv(output / "effects.csv")
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(aligned) == 12
    assert aligned[0]["target_baseline_iteration"] == "0"
    assert float(aligned[-1]["target_gpu_hours"]) == pytest.approx(
        summary["common_horizon_gpu_hours"]
    )
    assert len(utility) == 11
    assert {row["extrapolated"] for row in utility} == {"False"}
    horizon = summary["common_horizon_gpu_hours"]
    baseline_aulc = sum(float(row["baseline_interval_area"]) for row in utility) / horizon
    adaptive_aulc = sum(float(row["adaptive_interval_area"]) for row in utility) / horizon
    primary = next(row for row in effects if row["metric"] == "normalized_joint_elo_aulc")
    assert baseline_aulc == pytest.approx(float(primary["baseline_value"]))
    assert adaptive_aulc == pytest.approx(float(primary["adaptive_value"]))
    assert adaptive_aulc - baseline_aulc == pytest.approx(float(primary["effect"]))
    assert decision["status"] == h3.decide_h3(
        effects, audit_passed=True, h2_status="not_supported"
    )["status"]
    assert decision["mechanism_attribution"]["status"] == (
        "attributed_to_replay_adaptive_mechanism"
    )
    without_h2 = h3.decide_h3(effects, audit_passed=True, h2_status="not_supported")
    assert without_h2["status"] == "supported"
    assert without_h2["mechanism_attribution"]["status"] == (
        "performance_improvement_without_replay_mechanism_attribution"
    )


def test_missing_required_input_writes_not_assessable(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "h3_v2.yaml").read_text(encoding="utf-8")
    )
    config["inputs"]["head_to_head_records"] = {
        "path": (tmp_path / "missing-games.jsonl").as_posix(),
        "sha256": None,
    }
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "output"

    status = h3.execute(config_path, output)

    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "input_audit.json").read_text(encoding="utf-8"))
    assert status == "not_assessable"
    assert decision["status"] == "not_assessable"
    assert audit["status"] == "failed"
