from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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


evaluate = load_script("evaluate_fixed_basket")
summarize = load_script("summarize_fixed_basket")

from arena.bot_interface import PawnMove
from pyquoridor.board import Board


def formal_protocol() -> dict:
    return evaluate.load_protocol(
        Path(__file__).resolve().parents[1] / "configs" / "fixed_basket_v1.yaml"
    )


def test_fixed_basket_protocol_is_exact_and_totals_2400_games() -> None:
    protocol = formal_protocol()

    assert protocol["checkpoints"] == [
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
    assert protocol["games_per_opponent"] == 50
    assert protocol["max_turns"] == 150
    assert protocol["alternate_sides"] is True
    assert [opponent["id"] for opponent in protocol["opponents"]] == [
        "heuristic_20",
        "heuristic_200",
        "greedy_random_50",
        "random",
    ]
    assert (
        len(protocol["checkpoints"])
        * len(protocol["opponents"])
        * protocol["games_per_opponent"]
        == 2400
    )


def test_pilot_and_formal_default_output_directories_are_separate() -> None:
    pilot = evaluate.resolve_output_dir("pilot", None)
    formal = evaluate.resolve_output_dir("formal", None)

    assert pilot.name == "fixed_basket_v1_pilot"
    assert formal.name == "fixed_basket_v1"
    assert pilot.parent == formal.parent
    assert pilot != formal

    protocol = formal_protocol()
    pilot_protocol = evaluate.apply_execution_overrides(
        protocol, mode="pilot", games_per_opponent=2
    )
    assert pilot_protocol["games_per_opponent"] == 2
    assert protocol["games_per_opponent"] == 50
    with pytest.raises(evaluate.FixedBasketError):
        evaluate.apply_execution_overrides(
            protocol, mode="formal", games_per_opponent=2
        )


def test_scheduled_temperature_uses_five_early_model_moves(monkeypatch) -> None:
    observed = []

    def fake_init(self, *args, temp, **kwargs):
        self.temp = temp
        self.turn_count = 0

    def fake_select(self, board):
        observed.append(self.temp)
        self.turn_count += 1
        return self.temp

    monkeypatch.setattr(evaluate.AlphaZeroBot, "__init__", fake_init)
    monkeypatch.setattr(evaluate.AlphaZeroBot, "select_move", fake_select)
    bot = evaluate.ScheduledTemperatureAlphaZeroBot(
        "white",
        ".",
        "unused",
        early_temp=0.18,
        early_moves=5,
        later_temp=0.0,
    )

    for _ in range(7):
        bot.select_move(None)

    assert observed == [0.18, 0.18, 0.18, 0.18, 0.18, 0.0, 0.0]
    assert bot.temperature_history == observed


def test_stable_game_seed_is_repeatable_distinct_and_uint32() -> None:
    first = evaluate.stable_seed("fixed_basket_v1", 20, "random", 0, base_seed=1001)
    repeated = evaluate.stable_seed(
        "fixed_basket_v1", 20, "random", 0, base_seed=1001
    )
    other_game = evaluate.stable_seed(
        "fixed_basket_v1", 20, "random", 1, base_seed=1001
    )
    other_opponent = evaluate.stable_seed(
        "fixed_basket_v1", 20, "heuristic_20", 0, base_seed=1001
    )

    assert first == repeated
    assert len({first, other_game, other_opponent}) == 3
    assert all(0 <= seed <= 0xFFFFFFFF for seed in (first, other_game, other_opponent))


def test_source_directory_hash_snapshot_detects_changes(tmp_path: Path) -> None:
    source = tmp_path / "training_run"
    source.mkdir()
    artifact = source / "checkpoint.bin"
    artifact.write_bytes(b"unchanged")

    before = evaluate.snapshot_directory_hashes(source)
    same = evaluate.snapshot_directory_hashes(source)
    assert evaluate.compare_directory_snapshots(before, same)["unchanged"] is True

    artifact.write_bytes(b"changed")
    after = evaluate.snapshot_directory_hashes(source)
    comparison = evaluate.compare_directory_snapshots(before, after)
    assert comparison["unchanged"] is False
    assert comparison["changed_paths"] == ["checkpoint.bin"]


def test_checkpoint_zero_comes_from_metadata_and_other_names_are_discovered(
    tmp_path: Path,
) -> None:
    protocol = formal_protocol()
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    initial = tmp_path / "pretraining" / "initial-model.pth.tar"
    initial.parent.mkdir()
    initial.write_bytes(b"pretrained")
    for iteration in protocol["checkpoints"][1:]:
        (checkpoint_dir / f"model-iteration-{iteration}.pth.tar").write_bytes(
            f"checkpoint-{iteration}".encode()
        )
    metadata = {
        "initial_checkpoint_path": str(initial),
        "initial_checkpoint_sha256": evaluate._sha256(initial),
        "resolved_config": {"checkpoint": {"directory": str(checkpoint_dir)}},
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    entries = evaluate.discover_checkpoints(protocol, run_dir, metadata)

    assert [entry["iteration"] for entry in entries] == protocol["checkpoints"]
    assert Path(entries[0]["path"]) == initial.resolve()
    assert entries[0]["resolution_source"] == "run_metadata.initial_checkpoint_path"
    assert entries[1]["filename"] == "model-iteration-20.pth.tar"
    assert all(len(entry["sha256"]) == 64 for entry in entries)

    output_dir = tmp_path / "fixed_basket_v1_pilot"
    evaluate.write_resolved_protocol(protocol, output_dir)
    manifest = evaluate.build_evaluation_manifest(
        protocol,
        run_dir,
        output_dir,
        mode="pilot",
        selected_checkpoints=[0],
    )
    assert (output_dir / "protocol.resolved.yaml").is_file()
    assert (output_dir / "evaluation_manifest.json").is_file()
    assert manifest["evaluation_mode"] == "pilot"
    assert manifest["selected_checkpoints"] == [0]
    assert manifest["expected_evaluation_games"] == 200
    assert manifest["outputs"]["games"].endswith("/games.jsonl")
    assert manifest["outputs"]["evaluation_log"].endswith("/evaluation.log")


class FakeBot:
    def __init__(self, tag: str, color: str, cleanup_counts: dict[str, int]):
        self.tag = tag
        self.color = color
        self.turn_count = 0
        self.cleanup_counts = cleanup_counts

    def reset_mcts(self):
        return None

    def cleanup(self):
        self.cleanup_counts[self.tag] = self.cleanup_counts.get(self.tag, 0) + 1


class FakeMove:
    def __init__(self, index: int):
        self.index = index

    def to_dict(self):
        return {"move_index": self.index}


def test_evaluation_loads_once_splits_sides_and_forces_150_turns(
    tmp_path: Path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "model-iteration-20.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    entry = {
        "iteration": 20,
        "path": checkpoint.as_posix(),
        "sha256": evaluate._sha256(checkpoint),
    }
    protocol = {
        "protocol_id": "fixed_basket_v1",
        "base_seed": 1001,
        "games_per_opponent": 2,
        "max_turns": 150,
        "opponents": [
            {"id": "one", "type": "random"},
            {"id": "two", "type": "random"},
        ],
    }
    creations = {"model": 0, "opponents": 0}
    cleanup_counts: dict[str, int] = {}
    observed_games = []
    fsync_calls = []
    monkeypatch.setattr(evaluate.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))

    def model_factory(entry, protocol, board_size):
        creations["model"] += 1
        return FakeBot("model", "white", cleanup_counts)

    def opponent_factory(spec, color):
        creations["opponents"] += 1
        return FakeBot(spec["id"], color, cleanup_counts)

    def fake_play_game(white, black, *, max_turns):
        assert max_turns == 150
        observed_games.append((white.tag, black.tag, white.color, black.color))
        return evaluate.MatchResult(
            winner="white",
            termination="win",
            turns=10,
            moves=[FakeMove(index) for index in range(10)],
            move_times={"white": 1.0, "black": 2.0},
            move_counts={"white": 5, "black": 5},
            match_duration=3.0,
        )

    games_path = tmp_path / "games.jsonl"
    count = evaluate.evaluate_matchups(
        protocol,
        [entry],
        games_path,
        board_size=9,
        model_factory=model_factory,
        opponent_factory=opponent_factory,
        play_game_fn=fake_play_game,
    )

    assert count == 4
    assert creations == {"model": 1, "opponents": 2}
    assert cleanup_counts == {"one": 1, "two": 1, "model": 1}
    assert observed_games == [
        ("model", "one", "white", "black"),
        ("one", "model", "white", "black"),
        ("model", "two", "white", "black"),
        ("two", "model", "white", "black"),
    ]
    records = [json.loads(line) for line in games_path.read_text().splitlines()]
    assert [record["model_color"] for record in records] == [
        "white",
        "black",
        "white",
        "black",
    ]
    assert all(record["max_turns"] == 150 for record in records)
    assert len(fsync_calls) == 4
    required_fields = {
        "protocol_id",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_sha256",
        "opponent",
        "game_index",
        "game_seed",
        "model_color",
        "winner",
        "model_result",
        "termination",
        "turns",
        "duration_seconds",
        "fault",
        "moves",
        "model_temperatures",
    }
    assert all(required_fields <= record.keys() for record in records)
    assert all(len(record["moves"]) == 10 for record in records)

    resumed = evaluate.evaluate_matchups(
        protocol,
        [entry],
        games_path,
        board_size=9,
        model_factory=model_factory,
        opponent_factory=opponent_factory,
        play_game_fn=fake_play_game,
    )
    assert resumed == 0
    assert len(games_path.read_text().splitlines()) == 4
    assert creations == {"model": 1, "opponents": 2}

    interrupted_path = tmp_path / "interrupted-games.jsonl"
    interrupted_path.write_text(
        "\n".join(games_path.read_text().splitlines()[:3]) + "\n",
        encoding="utf-8",
    )
    resumed_partial = evaluate.evaluate_matchups(
        protocol,
        [entry],
        interrupted_path,
        board_size=9,
        model_factory=model_factory,
        opponent_factory=opponent_factory,
        play_game_fn=fake_play_game,
    )
    interrupted_records = [
        json.loads(line) for line in interrupted_path.read_text().splitlines()
    ]
    interrupted_keys = {
        (record["checkpoint"], record["opponent"], record["game_index"])
        for record in interrupted_records
    }
    assert resumed_partial == 1
    assert len(interrupted_records) == len(interrupted_keys) == 4


def test_summarizer_requires_50_games_and_25_per_side(tmp_path: Path) -> None:
    protocol = formal_protocol()
    manifest_entries = [
        {
            "iteration": iteration,
            "path": f"/checkpoints/model-{iteration}.pth.tar",
            "sha256": f"{iteration:064x}",
        }
        for iteration in protocol["checkpoints"]
    ]
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "evaluation_mode": "formal",
        "selected_checkpoints": protocol["checkpoints"],
        "expected_evaluation_games": 2400,
        "checkpoints": manifest_entries,
    }
    games = []
    for entry in manifest_entries:
        checkpoint = entry["iteration"]
        for opponent in protocol["opponents"]:
            for game_index in range(50):
                model_color = "white" if game_index < 25 else "black"
                opponent_color = "black" if model_color == "white" else "white"
                model_result = "win" if game_index % 2 == 0 else "loss"
                games.append(
                    {
                        "protocol_id": protocol["protocol_id"],
                        "checkpoint": checkpoint,
                        "checkpoint_path": entry["path"],
                        "checkpoint_sha256": entry["sha256"],
                        "opponent": opponent["id"],
                        "game_index": game_index,
                        "game_seed": evaluate.stable_seed(
                            protocol["protocol_id"],
                            checkpoint,
                            opponent["id"],
                            game_index,
                            base_seed=protocol["base_seed"],
                        ),
                        "model_color": model_color,
                        "winner": model_color if model_result == "win" else opponent_color,
                        "model_result": model_result,
                        "termination": "win",
                        "max_turns": 150,
                        "turns": 20,
                        "total_moves": 20,
                        "duration_seconds": 1.0,
                        "model_move_seconds": 1.0,
                        "model_moves": 10,
                        "fault": None,
                        "moves": [],
                        "model_temperatures": [0.18] * 5 + [0.0],
                    }
                )
    games_path = tmp_path / "games.jsonl"
    games_path.write_text(
        "".join(json.dumps(game) + "\n" for game in games), encoding="utf-8"
    )
    manifest_path = tmp_path / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = summarize.summarize_results(
        protocol,
        manifest,
        games,
        tmp_path / "summary",
        games_path=games_path,
        manifest_path=manifest_path,
        elo_calculator=lambda results, names: {name: 1000.0 for name in names},
    )

    assert summary["status"] == "completed"
    assert summary["expected_games"] == summary["observed_games"] == 2400
    assert summary["data_quality"] == {
        "unique_game_keys": 2400,
        "invalid_records": 0,
        "faults": 0,
        "termination_counts": {"win": 2400},
        "js_invalid_proposals": 0,
        "model_fallbacks": 0,
        "errors": [],
        "invalid_record_details": [],
    }
    checkpoint_rows = list(
        __import__("csv").DictReader(
            (tmp_path / "summary" / "checkpoint_summary.csv").open(
                encoding="utf-8"
            )
        )
    )
    opponent_rows = list(
        __import__("csv").DictReader(
            (tmp_path / "summary" / "opponent_summary.csv").open(encoding="utf-8")
        )
    )
    elo_rows = list(
        __import__("csv").DictReader(
            (tmp_path / "summary" / "elo_summary.csv").open(encoding="utf-8")
        )
    )
    assert len(checkpoint_rows) == 12
    required_checkpoint_fields = {
        "checkpoint",
        "gpu_hours",
        "total_games",
        "wins",
        "losses",
        "draws",
        "score_rate",
        "win_rate",
        "draw_rate",
        "heuristic_20_score",
        "heuristic_200_score",
        "greedy_random_50_score",
        "random_score",
        "mean_game_length",
        "mean_move_time",
        "invalid_moves",
        "bot_errors",
        "max_turn_draws",
        "js_invalid_proposals",
        "model_fallbacks",
    }
    assert all(required_checkpoint_fields <= row.keys() for row in checkpoint_rows)
    assert all(int(row["total_games"]) == 200 for row in checkpoint_rows)
    assert all(int(row["wins"]) == 100 for row in checkpoint_rows)
    assert all(int(row["losses"]) == 100 for row in checkpoint_rows)
    assert all(float(row["score_rate"]) == pytest.approx(0.5) for row in checkpoint_rows)
    assert all(float(row["mean_move_time"]) == pytest.approx(0.1) for row in checkpoint_rows)
    assert all(
        float(row["score_rate_ci95_low"]) <= 0.5 <= float(row["score_rate_ci95_high"])
        for row in checkpoint_rows
    )
    assert len(opponent_rows) == 48
    assert all(int(row["games"]) == 50 for row in opponent_rows)
    assert len(elo_rows) == 16
    assert all(row["status"] == "provisional" for row in elo_rows)
    assert all(int(row["random_seed"]) == 1001 for row in elo_rows)
    assert summary["elo"]["status"] == "provisional"
    assert summary["confidence_intervals"]["strata"] == [
        "opponent",
        "model_color",
    ]


def test_pilot_summary_is_isolated_and_only_writes_checkpoint_csv(
    tmp_path: Path,
) -> None:
    protocol = formal_protocol()
    checkpoint = protocol["checkpoints"][0]
    entry = {
        "iteration": checkpoint,
        "path": "/checkpoints/initial-model.pth.tar",
        "sha256": "0" * 64,
    }
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "evaluation_mode": "pilot",
        "selected_checkpoints": [checkpoint],
        "expected_evaluation_games": 200,
        "checkpoints": [entry],
    }
    games = []
    for opponent in protocol["opponents"]:
        for game_index in range(50):
            model_color = "white" if game_index < 25 else "black"
            games.append(
                {
                    "protocol_id": protocol["protocol_id"],
                    "checkpoint": checkpoint,
                    "checkpoint_path": entry["path"],
                    "checkpoint_sha256": entry["sha256"],
                    "opponent": opponent["id"],
                    "game_index": game_index,
                    "game_seed": evaluate.stable_seed(
                        protocol["protocol_id"],
                        checkpoint,
                        opponent["id"],
                        game_index,
                        base_seed=protocol["base_seed"],
                    ),
                    "model_color": model_color,
                    "winner": None,
                    "model_result": "draw",
                    "termination": "max_turns",
                    "max_turns": 150,
                    "turns": 150,
                    "duration_seconds": 1.0,
                    "fault": None,
                    "moves": [],
                    "model_temperatures": [0.18] * 5 + [0.0],
                }
            )
    pilot_dir = tmp_path / "fixed_basket_v1_pilot"
    pilot_dir.mkdir()
    games_path = pilot_dir / "games.jsonl"
    games_path.write_text(
        "".join(json.dumps(game) + "\n" for game in games), encoding="utf-8"
    )
    manifest_path = pilot_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = summarize.summarize_results(
        protocol,
        manifest,
        games,
        pilot_dir,
        games_path=games_path,
        manifest_path=manifest_path,
        mode="pilot",
        elo_calculator=lambda *_: pytest.fail("pilot must not calculate Elo"),
    )

    assert summary["status"] == "completed"
    assert (pilot_dir / "checkpoint_summary.csv").is_file()
    assert not (pilot_dir / "opponent_summary.csv").exists()
    assert not (pilot_dir / "elo_summary.csv").exists()


def test_seeded_js_bridge_requires_an_explicit_uint32_seed() -> None:
    bridge = Path(__file__).resolve().parents[1] / "js" / "seeded_bot.js"
    source = bridge.read_text(encoding="utf-8")

    assert "mulberry32" in source
    assert "Number.isInteger(seed)" in source
    assert "Math.random = mulberry32(seed)" in source
    assert "Math.random = originalRandom" in source


def test_seeded_js_bot_rejects_illegal_proposals_deterministically(
    monkeypatch,
) -> None:
    proposals = iter(
        [
            PawnMove("white", (8, 8)),
            PawnMove("white", (1, 4)),
        ]
    )
    monkeypatch.setattr(
        evaluate.JSBot,
        "select_move",
        lambda self, board: next(proposals),
    )
    bot = object.__new__(evaluate.SeededJSBot)
    bot.color = "white"
    bot.invalid_proposals = 0

    selected = bot.select_move(Board())

    assert selected.target == (1, 4)
    assert bot.invalid_proposals == 1


def test_retryable_termination_records_are_atomically_removed(tmp_path: Path) -> None:
    games_path = tmp_path / "games.jsonl"
    records = [
        {
            "checkpoint": 0,
            "opponent": "heuristic_200",
            "game_index": 49,
            "termination": "invalid_move",
        },
        {
            "checkpoint": 0,
            "opponent": "random",
            "game_index": 0,
            "termination": "win",
        },
    ]
    games_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    removed = evaluate.remove_retryable_game_records(
        games_path, {"invalid_move"}
    )
    remaining = [json.loads(line) for line in games_path.read_text().splitlines()]

    assert removed == [(0, "heuristic_200", 49)]
    assert remaining == [records[1]]


def test_fixed_basket_model_fallback_diagnostics_are_json_safe() -> None:
    bot = object.__new__(evaluate.ScheduledTemperatureAlphaZeroBot)
    bot.color = "black"
    bot.fallback_events = []

    bot._log_fallback_debug(
        Board(),
        object(),
        -1,
        [0.5, 0.5],
        __import__("numpy").array([1, 0], dtype=__import__("numpy").int64),
        {},
        ({}, {}),
        [(int(1), __import__("numpy").float64(0.5), "invalid")],
    )

    assert bot.fallback_events == [
        {
            "model_color": "black",
            "player": -1,
            "failed_actions": 1,
            "alphazero_valid_actions": 1,
            "reason": "all_policy_actions_failed_arena_legality",
        }
    ]
    json.dumps(bot.fallback_events, allow_nan=False)


def test_selected_game_records_are_atomically_removed_and_normalized(
    tmp_path: Path,
) -> None:
    games_path = tmp_path / "games.jsonl"
    records = [
        {"checkpoint": 60, "opponent": "heuristic_200", "game_index": 41},
        {"checkpoint": 210, "opponent": "heuristic_200", "game_index": 2},
        {"checkpoint": 210, "opponent": "random", "game_index": 0},
    ]
    games_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    keys = evaluate.parse_retry_game_keys(
        ["60:heuristic_200:41", "210:heuristic_200:2"]
    )
    removed = evaluate.remove_selected_game_records(games_path, keys)
    remaining = [json.loads(line) for line in games_path.read_text().splitlines()]

    assert removed == [(60, "heuristic_200", 41), (210, "heuristic_200", 2)]
    assert remaining == [
        {
            **records[2],
            "model_fallback_count": 0,
            "model_fallback_events": [],
        }
    ]
