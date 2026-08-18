from __future__ import annotations

import csv
import importlib.util
import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_replay.py"
SPEC = importlib.util.spec_from_file_location("summarize_replay", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)


def sample(value: int, step: int, game_length: int) -> tuple:
    board = np.full((4, 3, 3), value, dtype=np.uint8)
    return (
        board,
        np.array([1.0], dtype=np.float32),
        1,
        np.array([1], dtype=np.uint8),
        step,
        game_length,
    )


def write_inputs(root: Path) -> tuple[Path, Path, Path]:
    buckets = [
        deque([sample(1, 1, 2), sample(1, 2, 2), sample(2, 1, 1)]),
        deque([sample(2, 1, 2), sample(3, 2, 2), sample(4, 1, 1)]),
        deque([sample(4, 1, 2), sample(5, 2, 2), sample(6, 1, 1)]),
    ]
    replay_path = root / "latest.examples"
    with replay_path.open("wb") as destination:
        pickle.dump({"iteration": 4, "examples": buckets}, destination)

    metrics_path = root / "metrics.jsonl"
    metrics = []
    for iteration in range(1, 5):
        metrics.append(
            {
                "iteration": iteration,
                "positions_generated": 3,
                "games_completed": 2,
                "min_game_length": 1,
                "max_game_length": 2,
                "mean_game_length": 1.5,
            }
        )
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metrics),
        encoding="utf-8",
    )

    config_path = root / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"board_size": 2},
                "replay": {"history_iterations": 3},
                "self_play": {"games_per_iteration": 2},
            }
        ),
        encoding="utf-8",
    )
    return replay_path, metrics_path, config_path


def test_summarizes_states_trajectories_and_limitations(tmp_path: Path) -> None:
    replay_path, metrics_path, config_path = write_inputs(tmp_path)
    output_dir = tmp_path / "output"

    summary = summarize.summarize_replay(
        replay_path,
        metrics_path,
        config_path,
        output_dir,
        expected_iteration=4,
        expected_history_buckets=3,
        expected_total_states=9,
    )

    assert summary["status"] == "completed"
    assert summary["replay"] == {
        "iteration": 4,
        "history_buckets": 3,
        "recovered_iteration_start": 2,
        "recovered_iteration_end": 4,
        "total_states": 9,
        "empty_buckets": 0,
    }
    assert summary["outputs"]["raw_states_exported"] is False
    assert summary["outputs"]["state_hashes_exported"] is False
    assert summary["limitations"] == list(summarize.LIMITATIONS)
    assert summary["trajectories"]["empty_games"] == 0
    assert summary["anomaly_counts"] == {}

    with (output_dir / "replay_iteration_stats.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        iteration_rows = list(csv.DictReader(source))
    assert len(iteration_rows) == 3
    assert int(iteration_rows[0]["unique_canonical_states"]) == 2
    assert float(iteration_rows[0]["duplicate_rate"]) == 1 / 3
    assert float(iteration_rows[0]["state_effective_count"]) == 1.8
    assert iteration_rows[0]["incoming_ratio_left_censored"] == "True"
    assert int(iteration_rows[1]["incoming_unique_states"]) == 2

    with (output_dir / "trajectory_stats.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        trajectory_rows = list(csv.DictReader(source))
    assert len(trajectory_rows) == 3
    assert all(int(row["recovered_games"]) == 2 for row in trajectory_rows)
    assert all(float(row["median_game_length"]) == 1.5 for row in trajectory_rows)
    assert all(int(row["p90_game_length"]) == 2 for row in trajectory_rows)

    exported_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".csv", ".json"}
    )
    assert "state_hash" not in exported_text or '"state_hashes_exported": false' in exported_text
    assert "[[[" not in exported_text


def test_cli_prints_required_acceptance_lines(tmp_path: Path, capsys) -> None:
    replay_path, metrics_path, config_path = write_inputs(tmp_path)
    output_dir = tmp_path / "output"

    result = summarize.main(
        [
            "--replay",
            str(replay_path),
            "--metrics",
            str(metrics_path),
            "--resolved-config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--expected-iteration",
            "4",
            "--expected-history-buckets",
            "3",
            "--expected-total-states",
            "9",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out.splitlines() == [
        "Replay iteration: 4",
        "History buckets: 3",
        "Recovered range: 2-4",
        "Total states: 9",
        "Empty buckets: 0",
        "Count matches metrics: yes",
        "Output status: completed",
    ]
