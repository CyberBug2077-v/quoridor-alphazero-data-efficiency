from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = SOURCE_ROOT / "experiments"
SCRIPTS = EXPERIMENTS / "scripts"
BASELINE_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for root in (SCRIPTS, BASELINE_SCRIPTS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_head_to_head as head
from head_to_head_stats import (
    colour_stratified_bootstrap,
    paired_seed,
    stable_game_key,
)


def _default_args(output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=EXPERIMENTS / "configs" / "head_to_head_v2.yaml",
        matched_compute=None,
        baseline_run_dir=None,
        adaptive_run_dir=None,
        baseline_checkpoint_manifest=None,
        adaptive_checkpoint_manifest=None,
        output_dir=output_dir,
    )


def test_v2_protocol_freezes_requested_parameters_and_output() -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "head_to_head_v2.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["config_id"] == "head_to_head_v2"
    assert config["outputs"]["root"] == (
        "outputs/adaptive_seed1001_4090_v2_analysis/head_to_head_v2"
    )
    assert config["model_selection"]["allow_best_checkpoint"] is False
    assert config["model_protocol"]["mcts_simulations"] == 200
    assert config["model_protocol"]["temperature"] == 0.0
    assert config["model_protocol"]["dirichlet_noise"] is False
    assert config["model_protocol"]["clear_tree_each_move"] is True
    assert config["games"]["max_turns"] == 150
    assert config["analysis"]["bootstrap"]["resamples"] == 10000
    assert config["analysis"]["bootstrap"]["strata"] == ["adaptive_colour"]


def test_dynamic_common_horizon_selection_and_task_pairs(tmp_path: Path) -> None:
    context = head.resolve_context(_default_args(tmp_path))
    tasks = head.build_tasks(context)

    assert context.common_horizon == 20.004163361943395
    assert context.baseline["iteration"] == 210
    assert context.baseline["actual_gpu_hours"] == context.common_horizon
    assert context.adaptive["iteration"] == 179
    assert context.adaptive["actual_gpu_hours"] == 19.90316950874274
    assert context.baseline["actual_gpu_hours"] <= context.common_horizon
    assert context.adaptive["actual_gpu_hours"] <= context.common_horizon
    assert len(tasks) == len({task["stable_game_key"] for task in tasks}) == 100
    for seed_pair_index in range(50):
        pair = [
            task for task in tasks if task["seed_pair_index"] == seed_pair_index
        ]
        assert len(pair) == 2
        assert len({task["game_seed"] for task in pair}) == 1
        assert {task["adaptive_color"] for task in pair} == {"white", "black"}


def test_seed_and_game_key_are_deterministic_and_colour_specific() -> None:
    baseline_sha = "a" * 64
    adaptive_sha = "b" * 64
    seed = paired_seed("head_to_head_v2", 91001, baseline_sha, adaptive_sha, 7)

    assert seed == paired_seed(
        "head_to_head_v2", 91001, baseline_sha, adaptive_sha, 7
    )
    assert seed != paired_seed(
        "head_to_head_v2", 91001, baseline_sha, adaptive_sha, 8
    )
    assert stable_game_key(baseline_sha, adaptive_sha, 7, "white") != stable_game_key(
        baseline_sha, adaptive_sha, 7, "black"
    )


def test_colour_stratified_bootstrap_matches_registered_score_and_repeats() -> None:
    records = []
    for color in ("white", "black"):
        for index in range(50):
            result = "win" if index < 30 else "draw" if index < 40 else "loss"
            records.append(
                {
                    "stable_game_key": f"{color}-{index:02d}",
                    "adaptive_color": color,
                    "adaptive_result": result,
                }
            )
    first = colour_stratified_bootstrap(
        records, resamples=10000, seed=92001
    )
    second = colour_stratified_bootstrap(
        list(reversed(records)), resamples=10000, seed=92001
    )

    assert first == second
    assert first["score_rate"] == pytest.approx((60 + 0.5 * 20) / 100)
    assert first["adaptive_white_games"] == 50
    assert first["adaptive_black_games"] == 50
    assert first["ci95_low"] <= first["score_rate"] <= first["ci95_high"]


def _synthetic_task() -> dict:
    return {
        "stable_game_key": "c" * 64,
        "seed_pair_index": 0,
        "game_seed": 123,
        "baseline_color": "black",
        "adaptive_color": "white",
        "baseline_iteration": 210,
        "baseline_gpu_hours": 20.0,
        "baseline_checkpoint_path": "baseline.pth.tar",
        "baseline_checkpoint_sha256": "a" * 64,
        "adaptive_iteration": 179,
        "adaptive_gpu_hours": 19.9,
        "adaptive_checkpoint_path": "adaptive.pth.tar",
        "adaptive_checkpoint_sha256": "b" * 64,
    }


def test_valid_attempt_is_recovered_without_replaying_game(tmp_path: Path) -> None:
    task = _synthetic_task()
    attempt = {
        "schema_version": 2,
        "config_id": "head_to_head_v2",
        "record_type": "attempt",
        **task,
        "attempt_index": 1,
        "termination": "win",
        "fault": None,
        "technically_valid": True,
        "adaptive_result": "win",
    }
    context = SimpleNamespace(
        attempts_path=tmp_path / "attempts.jsonl",
        games_path=tmp_path / "games.jsonl",
    )
    head._append_jsonl_fsync(context.attempts_path, attempt)

    attempts, games = head.load_and_validate_state(
        context, [task], recover=True
    )

    assert len(attempts[task["stable_game_key"]]) == 1
    assert set(games) == {task["stable_game_key"]}
    persisted = head._load_jsonl(context.games_path, "games.jsonl")
    assert persisted == [head._game_from_attempt(attempt)]


def test_technical_failure_requires_same_key_explicit_retry(tmp_path: Path) -> None:
    task = _synthetic_task()
    failed = {
        "schema_version": 2,
        "config_id": "head_to_head_v2",
        "record_type": "attempt",
        **task,
        "attempt_index": 1,
        "termination": "bot_error",
        "fault": "white",
        "technically_valid": False,
        "adaptive_result": None,
    }
    context = SimpleNamespace(
        attempts_path=tmp_path / "attempts.jsonl",
        games_path=tmp_path / "games.jsonl",
    )
    head._append_jsonl_fsync(context.attempts_path, failed)
    attempts, games = head.load_and_validate_state(
        context, [task], recover=True
    )

    assert attempts[task["stable_game_key"]][0]["technically_valid"] is False
    assert games == {}
    assert head.parse_retry_keys(["0:white"], [task]) == {
        task["stable_game_key"]
    }
    with pytest.raises(head.HeadToHeadError):
        head.parse_retry_keys(["0:black"], [task])


def test_evaluation_loop_excludes_technical_attempt_from_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _synthetic_task()
    second = {
        **first,
        "stable_game_key": "d" * 64,
        "seed_pair_index": 1,
        "game_seed": 456,
        "baseline_color": "white",
        "adaptive_color": "black",
    }

    class FakeBot:
        def __init__(self) -> None:
            self.color = "white"
            self.temperature_history: list[float] = []
            self.fallback_events: list[dict] = []
            self.turn_count = 0
            self.valid_moves_cache = {}

        def reset_mcts(self) -> None:
            return None

    bots = iter((FakeBot(), FakeBot()))
    monkeypatch.setattr(head, "make_model", lambda checkpoint, context: next(bots))
    outcomes = iter(
        (
            head.MatchResult(
                winner=None,
                termination="bot_error",
                fault="white",
                message="synthetic failure",
            ),
            head.MatchResult(
                winner=None,
                termination="max_turns",
                fault=None,
                message="synthetic draw",
            ),
        )
    )
    monkeypatch.setattr(head, "play_game", lambda *args, **kwargs: next(outcomes))

    class Logger:
        def write(self, message: str) -> None:
            return None

    context = SimpleNamespace(
        baseline={"iteration": 210},
        adaptive={"iteration": 179},
        config={"games": {"max_turns": 150}},
        attempts_path=tmp_path / "attempts.jsonl",
        games_path=tmp_path / "games.jsonl",
    )
    attempts: dict[str, list[dict]] = defaultdict(list)
    games: dict[str, dict] = {}
    count = head.evaluate_pending_tasks(
        context,
        [first, second],
        attempts,
        games,
        retry_keys=set(),
        implementation_set_sha256="e" * 64,
        logger=Logger(),
    )

    assert count == 2
    assert len(head._load_jsonl(context.attempts_path, "attempts")) == 2
    persisted_games = head._load_jsonl(context.games_path, "games")
    assert len(persisted_games) == 1
    assert persisted_games[0]["stable_game_key"] == second["stable_game_key"]
    assert persisted_games[0]["adaptive_result"] == "draw"
