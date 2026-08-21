from __future__ import annotations

import sys
from pathlib import Path

import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = SOURCE_ROOT / "experiments"
EXPERIMENT_SCRIPTS = EXPERIMENTS / "scripts"
BASELINE_SCRIPTS = SOURCE_ROOT / "baseline" / "analysis" / "scripts"
for script_root in (EXPERIMENT_SCRIPTS, BASELINE_SCRIPTS):
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))

import evaluate_adaptive_holdout as adaptive
import evaluate_holdout as baseline


def test_reuses_baseline_loss_and_cluster_bootstrap() -> None:
    assert adaptive.loss_arrays is baseline.loss_arrays
    assert adaptive.cluster_bootstrap_intervals is baseline.cluster_bootstrap_intervals


def test_v2_protocol_freezes_requested_output_and_acceptance_counts() -> None:
    config = yaml.safe_load(
        (EXPERIMENTS / "configs" / "adaptive_holdout_v2.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["config_id"] == "adaptive_holdout_v2"
    assert config["outputs"]["root"] == (
        "outputs/adaptive_seed1001_4090_v2_analysis/holdout_v2"
    )
    assert config["holdout"]["expected_states"] == 9259
    assert config["holdout"]["expected_games"] == 200
    assert config["checkpoint_alignment"]["expected_target_count"] == 12
    assert config["evaluation"]["bootstrap_resamples"] == 10000


def test_real_registry_maps_all_twelve_targets_without_overshoot() -> None:
    matched = adaptive._load_yaml(
        EXPERIMENTS / "configs" / "matched_compute_v1.yaml", "matched compute"
    )
    manifest = adaptive._load_json(
        EXPERIMENTS
        / "outputs"
        / "adaptive_formal_seed1001_4090_v2"
        / "checkpoint_manifest.json",
        "checkpoint manifest",
    )

    targets = adaptive._matched_targets(matched)
    registry = adaptive._checkpoint_registry(manifest)
    selected = adaptive.select_checkpoints(targets, registry)

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
    assert selected[-1]["actual_gpu_hours"] == 19.90316950874274
    assert selected[-1]["target_gpu_hours"] == 20.004163361943395


def test_checkpoint_output_schema_contains_all_requested_metrics() -> None:
    required = {
        "target_baseline_iteration",
        "target_gpu_hours",
        "selected_adaptive_iteration",
        "selected_adaptive_gpu_hours",
        "holdout_policy_loss",
        "holdout_policy_loss_ci_low",
        "holdout_policy_loss_ci_high",
        "holdout_value_loss",
        "holdout_value_loss_ci_low",
        "holdout_value_loss_ci_high",
        "holdout_total_loss",
        "holdout_total_loss_ci_low",
        "holdout_total_loss_ci_high",
        "logged_train_policy_loss",
        "logged_train_value_loss",
        "approx_policy_gap",
        "approx_value_gap",
        "checkpoint_sha256",
    }

    assert required <= set(adaptive.CHECKPOINT_FIELDS)
