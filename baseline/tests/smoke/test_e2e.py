from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[2]
RUN_SMOKE = BASELINE_ROOT / "scripts" / "run_smoke.py"
VERIFY_SMOKE = BASELINE_ROOT / "scripts" / "verify_smoke.py"
SMOKE_CONFIG = BASELINE_ROOT / "configs" / "smoke_gpu.yaml"


def assert_all_numbers_finite(value, path: str = "root") -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(value), f"non-finite number at {path}: {value}"
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_all_numbers_finite(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            assert_all_numbers_finite(child, f"{path}[{index}]")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
def test_complete_smoke_pipeline_in_subprocess(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("complete smoke pipeline requires CUDA")

    output_dir = tmp_path / "smoke-e2e"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SMOKE),
            "fresh",
            "--config",
            str(SMOKE_CONFIG),
            "--output-dir",
            str(output_dir),
        ],
        cwd=BASELINE_ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"run_smoke failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    resolved = yaml.safe_load(
        (output_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["mode"] == "smoke"
    assert resolved["self_play"]["iterations"] == 4

    metrics_lines = (output_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(metrics_lines) == 4
    metrics = [json.loads(line) for line in metrics_lines]
    assert [record["iteration"] for record in metrics] == [1, 2, 3, 4]
    assert sum(record["games_completed"] for record in metrics) == 8
    assert all(record["positions_generated"] > 0 for record in metrics)
    assert all(record["illegal_action_count"] == 0 for record in metrics)
    assert all(record["optimizer_steps"] > 0 for record in metrics)

    checkpoint_dir = output_dir / "checkpoints"
    numbered_checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.pth.tar"),
        key=lambda path: int(path.name.split("_")[1].split(".")[0]),
    )
    assert [path.name for path in numbered_checkpoints] == [
        "checkpoint_1.pth.tar",
        "checkpoint_2.pth.tar",
        "checkpoint_3.pth.tar",
        "checkpoint_4.pth.tar",
    ]
    assert (checkpoint_dir / "best.pth.tar").is_file()
    assert (checkpoint_dir / "latest.examples").is_file()
    assert not (checkpoint_dir / "temp.pth.tar").exists()

    evaluation = json.loads(
        (output_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert set(evaluation["opponents"]) == {"random", "greedy"}
    for opponent in ("random", "greedy"):
        opponent_result = evaluation["opponents"][opponent]
        assert len(opponent_result["games"]) == 2
        assert (
            opponent_result["wins"]
            + opponent_result["draws"]
            + opponent_result["losses"]
            == 2
        )

    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "[run] mode=fresh" in run_log
    assert "mode=fresh status=finished" in run_log
    assert_all_numbers_finite(metrics, "metrics")
    assert_all_numbers_finite(evaluation, "evaluation")
    assert_all_numbers_finite(summary, "summary")

    verification = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SMOKE),
            "--run-dir",
            str(output_dir),
        ],
        cwd=BASELINE_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert verification.returncode == 0, (
        f"verify_smoke failed with exit code {verification.returncode}\n"
        f"stdout:\n{verification.stdout}\n"
        f"stderr:\n{verification.stderr}"
    )
    verification_result = json.loads(verification.stdout)
    assert verification_result["status"] == "verified"
    assert verification_result["self_play_games"] == 8
