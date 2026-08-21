from __future__ import annotations

import argparse
import csv
import hashlib
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

import compare_h2 as compare


def _metric_row(
    iteration: int,
    gpu_hours: float,
    numerator: float,
    denominator: float,
    *,
    censored: bool = False,
) -> dict:
    return {
        "iteration": iteration,
        "cumulative_gpu_hours": gpu_hours,
        "iteration_seconds": 3600.0,
        "iteration_gpu_hours": 1.0,
        "numerator": numerator,
        "denominator": denominator,
        "incoming_ratio_left_censored": censored,
    }


def test_horizon_is_read_from_matched_compute_and_excludes_iteration_180() -> None:
    matched = yaml.safe_load(
        (EXPERIMENTS / "configs" / "matched_compute_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    metrics = [
        json.loads(line)
        for line in (
            EXPERIMENTS
            / "outputs"
            / "adaptive_formal_seed1001_4090_v2"
            / "metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]

    horizon, endpoint_start, grid = compare.common_horizon_from_matched_compute(
        matched
    )
    selected = max(
        (row for row in metrics if row["cumulative_gpu_hours"] <= horizon),
        key=lambda row: row["iteration"],
    )

    assert horizon == 20.004163361943395
    assert endpoint_start == 19.115693432471787
    assert len(grid) == 12
    assert selected["iteration"] == 179
    assert metrics[179]["iteration"] == 180
    assert metrics[179]["cumulative_gpu_hours"] > horizon


def test_ratio_of_sums_is_not_the_mean_of_iteration_ratios() -> None:
    rows = [
        _metric_row(1, 1.0, 1.0, 1.0),
        _metric_row(2, 2.0, 99.0, 9.0),
    ]
    condition = {"metrics": rows, "derived": [], "replay": []}
    specification = {
        "source": "metrics",
        "aggregation": "ratio_of_sums",
        "numerator": "numerator",
        "denominator": "denominator",
    }

    result = compare._aggregate_condition(
        condition,
        specification,
        lower_bound=0.0,
        upper_bound=2.0,
        minimum_points=1,
    )

    assert result["value"] == pytest.approx(10.0)
    assert result["value"] != pytest.approx((1.0 + 11.0) / 2.0)
    assert result["aggregation_numerator"] == 100.0
    assert result["aggregation_denominator"] == 10.0


def test_left_censored_replay_row_is_excluded_from_ratio() -> None:
    rows = [
        _metric_row(66, 7.0, 100.0, 100.0, censored=True),
        _metric_row(67, 8.0, 50.0, 100.0),
    ]
    condition = {"metrics": [], "derived": [], "replay": rows}
    specification = {
        "source": "replay",
        "aggregation": "ratio_of_sums",
        "numerator": "numerator",
        "denominator": "denominator",
        "exclude_when_true": "incoming_ratio_left_censored",
    }

    result = compare._aggregate_condition(
        condition,
        specification,
        lower_bound=0.0,
        upper_bound=10.0,
        minimum_points=1,
    )

    assert result["value"] == pytest.approx(0.5)
    assert result["valid_points"] == 1
    assert result["excluded_rows"] == [
        {
            "iteration": 66,
            "cumulative_gpu_hours": 7.0,
            "reason": "incoming_ratio_left_censored=true",
        }
    ]


def test_decision_is_recomputed_from_effect_rows_only() -> None:
    effects_path = (
        EXPERIMENTS
        / "outputs"
        / "adaptive_seed1001_4090_v2_analysis"
        / "h2_v1"
        / "effects.csv"
    )
    with effects_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    recomputed = compare.decision_from_effects(rows)
    decision = json.loads(
        effects_path.with_name("decision.json").read_text(encoding="utf-8")
    )

    assert recomputed["status"] == decision["status"]
    assert recomputed["primary_improved"] == decision["primary_improved"]
    assert recomputed["improved_complementary_metrics"] == decision[
        "improved_complementary_metrics"
    ]
    assert hashlib.sha256(effects_path.read_bytes()).hexdigest() == decision[
        "effects_sha256"
    ]


def test_formal_effect_output_meets_primary_acceptance_conditions() -> None:
    effects_path = (
        EXPERIMENTS
        / "outputs"
        / "adaptive_seed1001_4090_v2_analysis"
        / "h2_v1"
        / "effects.csv"
    )
    with effects_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    primary = next(
        row
        for row in rows
        if row["metric"] == "fresh_states_per_update"
        and row["scope"] == "full_interval"
    )

    assert len(rows) == 18
    assert int(primary["baseline_valid_points"]) >= 4
    assert int(primary["adaptive_valid_points"]) >= 4
    assert float(primary["baseline_aggregation_denominator"]) > 0.0
    assert float(primary["adaptive_aggregation_denominator"]) > 0.0
    assert float(primary["baseline_value"]) == pytest.approx(
        float(primary["baseline_aggregation_numerator"])
        / float(primary["baseline_aggregation_denominator"])
    )
    assert float(primary["adaptive_value"]) == pytest.approx(
        float(primary["adaptive_aggregation_numerator"])
        / float(primary["adaptive_aggregation_denominator"])
    )


def test_failed_input_hash_writes_not_assessable_result(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "h2_v2.yaml").read_text(encoding="utf-8")
    )
    config["inputs"]["baseline_metrics"]["sha256"] = "0" * 64
    config_path = tmp_path / "h2_bad_hash.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "h2_output"

    compare.compare(
        argparse.Namespace(config=config_path, output_dir=output_dir)
    )

    audit = json.loads((output_dir / "input_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((output_dir / "decision.json").read_text(encoding="utf-8"))
    with (output_dir / "effects.csv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        effects = list(csv.DictReader(source))

    assert audit["status"] == "failed"
    assert decision["status"] == "not_assessable"
    assert len(effects) == 18
    assert {row["audit_status"] for row in effects} == {"failed"}
    assert compare.decision_from_effects(effects)["status"] == "not_assessable"
