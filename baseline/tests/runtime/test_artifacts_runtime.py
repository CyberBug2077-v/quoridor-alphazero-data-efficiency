from __future__ import annotations

import json
from pathlib import Path

import yaml

from runtime.artifacts import (
    JsonlWriter,
    atomic_write_json,
    atomic_write_yaml,
    find_existing_run_artifacts,
    write_summary,
)


def test_atomic_json_and_yaml_replace_without_temp_files(tmp_path: Path) -> None:
    json_path = atomic_write_json(tmp_path / "payload.json", {"value": "棋"})
    yaml_path = atomic_write_yaml(tmp_path / "payload.yaml", {"value": "棋"})

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": "棋"}
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8")) == {"value": "棋"}
    assert not list(tmp_path.glob("*.tmp"))


def test_find_artifacts_and_write_summary(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "checkpoint_1.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    with JsonlWriter(tmp_path / "metrics.jsonl") as writer:
        writer({"iteration": 1, "optimizer_steps": 1, "checkpoint_path": checkpoint.as_posix()})

    resolved = {
        "_output_path": tmp_path,
        "self_play": {"iterations": 2},
        "logging": {"metrics_file": "metrics.jsonl", "summary_file": "summary.json"},
    }
    summary_path = write_summary(
        resolved,
        mode="fresh",
        status="stopped",
        evaluation_path=None,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed_iterations"] == [1]
    assert summary["target_iterations"] == 2
    assert {path.name for path in find_existing_run_artifacts(tmp_path)} >= {
        "metrics.jsonl",
        "summary.json",
        "checkpoint_1.pth.tar",
    }

