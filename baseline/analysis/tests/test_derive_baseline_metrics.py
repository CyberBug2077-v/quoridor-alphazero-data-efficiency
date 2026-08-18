from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_baseline_metrics.py"
SPEC = importlib.util.spec_from_file_location("derive_baseline_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
derive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(derive)


def write_inputs(root: Path, *, bad_buffer: bool = False) -> tuple[Path, Path]:
    config_path = root / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"id": "synthetic_baseline"},
                "self_play": {"iterations": 4},
                "replay": {"history_iterations": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    positions = [10, 20, 30, 40]
    buffers = [10, 30, 50, 70]
    if bad_buffer:
        buffers[2] += 1
    metrics_path = root / "metrics.jsonl"
    records = []
    for index, (position_count, buffer_size) in enumerate(
        zip(positions, buffers), start=1
    ):
        records.append(
            {
                "iteration": index,
                "positions_generated": position_count,
                "games_completed": 5,
                "optimizer_steps": 2,
                "replay_buffer_size": buffer_size,
                "available_examples": buffer_size,
                "examples_used": min(buffer_size, 40),
                "samples_seen": min(buffer_size, 40) * 2,
                "iteration_seconds": 100.0,
                "self_play_seconds": 70.0,
                "training_seconds": 20.0,
                "peak_gpu_memory_mb": 1024.0,
            }
        )
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return metrics_path, config_path


def test_derives_metrics_ages_eviction_and_quality_report(tmp_path: Path) -> None:
    metrics_path, config_path = write_inputs(tmp_path)
    output_dir = tmp_path / "derived"

    csv_path, summary_path, quality_path = derive.derive_baseline_metrics(
        metrics_path, config_path, output_dir
    )

    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert float(rows[0]["fresh_states_per_update"]) == 5.0
    assert float(rows[1]["mean_sample_age"]) == pytest.approx(10 / 30)
    assert int(rows[1]["median_sample_age"]) == 0
    assert int(rows[1]["p90_sample_age"]) == 1
    assert int(rows[2]["evicted_states"]) == 10
    assert float(rows[2]["turnover_fraction"]) == pytest.approx(10 / 30)
    assert int(rows[3]["reconstructed_buffer_size"]) == 70

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["status"] == "passed"
    assert quality["iterations_passed"] == 4
    assert quality["iterations_failed"] == 0
    assert all(quality["checks"].values())

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["totals"]["positions_generated"] == 100
    assert summary["replay"]["final_buffer_size"] == 70
    assert summary["replay"]["total_evicted_states"] == 30


def test_rejects_a_replay_buffer_reconstruction_mismatch(tmp_path: Path) -> None:
    metrics_path, config_path = write_inputs(tmp_path, bad_buffer=True)
    output_dir = tmp_path / "derived"

    with pytest.raises(derive.DataQualityError, match="derived metric validation"):
        derive.derive_baseline_metrics(metrics_path, config_path, output_dir)

    quality = json.loads(
        (output_dir / "data_quality_report.json").read_text(encoding="utf-8")
    )
    assert quality["status"] == "failed"
    assert quality["iterations_failed"] == 1
    assert quality["failed_iterations"][0]["iteration"] == 3
    assert not quality["checks"][
        "replay_buffer_exactly_reconstructed_from_positions"
    ]
    assert not (output_dir / "derived_metrics.csv").exists()
    assert not (output_dir / "baseline_resource_summary.json").exists()
