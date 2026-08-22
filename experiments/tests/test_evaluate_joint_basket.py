from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = SOURCE_ROOT / "experiments"
EXPERIMENT_SCRIPTS = EXPERIMENTS / "scripts"
BASELINE_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for script_root in (EXPERIMENT_SCRIPTS, BASELINE_SCRIPTS):
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))

import joint_elo as joint


def test_v2_protocol_freezes_requested_output_and_counts() -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "adaptive_fixed_basket_v2.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["config_id"] == "adaptive_fixed_basket_v2"
    assert config["outputs"]["root"] == (
        "outputs/adaptive_seed1001_4090_v2_analysis/fixed_basket_v2"
    )
    assert config["checkpoint_alignment"]["required_target_count"] == 12
    assert config["pairing"]["expected_formal_games"] == 2400
    assert config["pairing"]["games_per_colour"] == 25
    assert config["joint_elo"]["uncertainty"]["resamples"] == 10000
    assert config["joint_elo"]["uncertainty"]["strata"] == [
        "condition",
        "target",
        "opponent",
        "model_colour",
    ]


def test_reuses_baseline_arena_and_summary_implementations() -> None:
    source = (EXPERIMENT_SCRIPTS / "evaluate_joint_basket.py").read_text(
        encoding="utf-8"
    )
    assert "from evaluate_fixed_basket import" in source
    assert "evaluate_matchups(" in source
    assert "from summarize_fixed_basket import" in source
    assert "summarize_results(" in source
    assert "from joint_elo import" in source


def test_real_registry_selects_twelve_targets_without_overshoot() -> None:
    matched = yaml.safe_load(
        (EXPERIMENTS / "configs" / "matched_compute_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (
            EXPERIMENTS
            / "outputs"
            / "adaptive_formal_seed1001_4090_v2"
            / "checkpoint_manifest.json"
        ).read_text(encoding="utf-8")
    )
    targets = matched["pairing_and_randomness"]["checkpoint_grid"]["targets"]
    selected = []
    for target in targets:
        eligible = [
            row
            for row in manifest["checkpoints"]
            if row["actual_gpu_hours"] <= target["gpu_hours"]
        ]
        selected.append(
            {
                **max(eligible, key=lambda row: row["iteration"]),
                "target_gpu_hours": target["gpu_hours"],
            }
        )

    assert len(selected) == 12
    assert [row["iteration"] for row in selected] == [
        0,
        22,
        41,
        58,
        75,
        91,
        108,
        124,
        140,
        156,
        172,
        179,
    ]
    assert all(
        row["actual_gpu_hours"] <= row["target_gpu_hours"] for row in selected
    )


def _synthetic_strata() -> tuple[list[str], list[str], list[dict]]:
    participants = [
        "baseline_checkpoint_0",
        "adaptive_checkpoint_0",
        "opponent_a",
        "opponent_b",
    ]
    anchors = ["opponent_a", "opponent_b"]
    specifications = [
        ("baseline_checkpoint_0", "opponent_a", 13),
        ("baseline_checkpoint_0", "opponent_b", 11),
        ("adaptive_checkpoint_0", "opponent_a", 18),
        ("adaptive_checkpoint_0", "opponent_b", 16),
    ]
    strata = []
    for index, (bot1, bot2, wins) in enumerate(specifications):
        strata.append(
            {
                "key": ("synthetic", index),
                "bot1": bot1,
                "bot2": bot2,
                "scores": [1.0] * wins + [0.0] * (25 - wins),
            }
        )
    return participants, anchors, strata


def _fit_config() -> dict:
    return {
        "elo_scale": 400.0,
        "ridge_precision": 1.0e-6,
        "max_iterations": 100,
        "tolerance": 1.0e-9,
    }


def test_joint_fit_uses_shared_anchors_and_one_location_constraint() -> None:
    participants, anchors, strata = _synthetic_strata()
    fitted = joint.fit_joint_elo(participants, anchors, strata, _fit_config())

    assert fitted["converged"] is True
    assert fitted["ratings"]["adaptive_checkpoint_0"] > fitted["ratings"]["baseline_checkpoint_0"]
    assert np.isclose(
        sum(fitted["ratings"][anchor] for anchor in anchors), 0.0, atol=1.0e-8
    )


def test_joint_bootstrap_is_stratified_and_deterministic() -> None:
    participants, anchors, strata = _synthetic_strata()
    fitted = joint.fit_joint_elo(participants, anchors, strata, _fit_config())
    first = joint.bootstrap_joint_elo(
        participants,
        anchors,
        strata,
        _fit_config(),
        resamples=12,
        seed=94001,
        initial_theta=fitted["theta"],
    )
    second = joint.bootstrap_joint_elo(
        participants,
        anchors,
        strata,
        _fit_config(),
        resamples=12,
        seed=94001,
        initial_theta=fitted["theta"],
    )

    assert first.shape == (12, 4)
    assert np.array_equal(first, second)
