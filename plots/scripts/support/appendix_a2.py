"""Extract the frozen training configuration reported in Appendix A.2.

The output contains the four groups used by the appendix table.  Paths are
resolved from the repository layout, so the script can be run from any working
directory::

    python plots/scripts/support/appendix_a2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PLOTS_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PLOTS_ROOT.parent

DEFAULT_PRETRAINING_CONFIG = (
    REPOSITORY_ROOT / "baseline" / "configs" / "pretraining_reproduction.yaml"
)
DEFAULT_BASELINE_CONFIG = (
    REPOSITORY_ROOT / "baseline" / "configs" / "baseline_reproduction.yaml"
)
DEFAULT_ADAPTIVE_CONFIG = (
    REPOSITORY_ROOT / "experiments" / "configs" / "adaptive_formal_v2.yaml"
)
DEFAULT_OUTPUT = PLOTS_ROOT / "tables" / "appendix_a2.json"


class AppendixConfigError(ValueError):
    """Raised when an Appendix A.2 source configuration is incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Appendix A.2 configuration summary as JSON."
    )
    parser.add_argument(
        "--pretraining-config",
        type=Path,
        default=DEFAULT_PRETRAINING_CONFIG,
        help="Pretraining YAML configuration.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG,
        help="Online Baseline YAML configuration.",
    )
    parser.add_argument(
        "--adaptive-config",
        type=Path,
        default=DEFAULT_ADAPTIVE_CONFIG,
        help="Formal Adaptive YAML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AppendixConfigError(f"Configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise AppendixConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AppendixConfigError(f"Configuration must be a mapping: {path}")
    return payload


def require_mapping(
    mapping: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise AppendixConfigError(f"{context}.{key} must be a mapping")
    return value


def require_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise AppendixConfigError(f"{context}.{key} is missing")
    return mapping[key]


def select(
    mapping: Mapping[str, Any], fields: tuple[str, ...], context: str
) -> dict[str, Any]:
    return {field: require_value(mapping, field, context) for field in fields}


def precision(training: Mapping[str, Any], context: str) -> dict[str, Any]:
    amp_enabled = bool(require_value(training, "amp", context))
    dtype = require_value(training, "amp_dtype", context) if amp_enabled else "fp32"
    return {"amp_enabled": amp_enabled, "dtype": dtype}


def extract_scheduler(
    adaptive: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    scheduler = require_mapping(adaptive, "adaptive_scheduler", "adaptive")
    estimator = require_mapping(scheduler, "length_estimator", "adaptive_scheduler")
    warm_start = require_mapping(scheduler, "warm_start", "adaptive_scheduler")
    bounds = require_mapping(scheduler, "bounds", "adaptive_scheduler")
    calibration = require_mapping(
        scheduler, "target_calibration", "adaptive_scheduler"
    )
    baseline_self_play = require_mapping(baseline, "self_play", "baseline")

    return {
        "method": require_value(scheduler, "method", "adaptive_scheduler"),
        "baseline_games": require_value(
            scheduler, "baseline_games", "adaptive_scheduler"
        ),
        "target_states": require_value(
            scheduler, "target_states", "adaptive_scheduler"
        ),
        "target_calibration": select(
            calibration,
            ("window_iterations", "statistic"),
            "adaptive_scheduler.target_calibration",
        ),
        "length_estimator": select(
            estimator,
            ("method", "alpha", "minimum_observations", "observation_statistic"),
            "adaptive_scheduler.length_estimator",
        ),
        "warm_start": select(
            warm_start,
            ("first_iteration_games", "initial_length_value"),
            "adaptive_scheduler.warm_start",
        ),
        "bounds": select(
            bounds,
            ("min_games", "max_games"),
            "adaptive_scheduler.bounds",
        ),
        "max_valid_game_length": require_value(
            baseline_self_play, "max_game_length", "baseline.self_play"
        ),
        "rounding": require_value(scheduler, "rounding", "adaptive_scheduler"),
        "update_frequency": require_value(
            scheduler, "update_frequency", "adaptive_scheduler"
        ),
        "update_timing": require_value(
            scheduler, "update_timing", "adaptive_scheduler"
        ),
        "clipping": require_value(scheduler, "clipping", "adaptive_scheduler"),
        "observation_policy": dict(
            require_mapping(scheduler, "observation_policy", "adaptive_scheduler")
        ),
        "estimator_guard_policy": dict(
            require_mapping(
                scheduler, "estimator_guard_policy", "adaptive_scheduler"
            )
        ),
    }


def build_summary(
    pretraining: Mapping[str, Any],
    baseline: Mapping[str, Any],
    adaptive: Mapping[str, Any],
) -> dict[str, Any]:
    pretraining_model = require_mapping(pretraining, "model", "pretraining")
    baseline_model = require_mapping(baseline, "model", "baseline")
    architecture_fields = (
        "board_size",
        "num_channels",
        "num_res_blocks",
        "attn_depth",
        "num_heads",
        "dropout",
        "se_enabled",
    )
    model_architecture = select(
        baseline_model, architecture_fields, "baseline.model"
    )
    if select(pretraining_model, architecture_fields, "pretraining.model") != (
        model_architecture
    ):
        raise AppendixConfigError(
            "Pretraining and Online Baseline model architectures differ"
        )

    pretraining_run = require_mapping(pretraining, "run", "pretraining")
    pretraining_data = require_mapping(pretraining, "data", "pretraining")
    pretraining_train = require_mapping(
        pretraining, "pretraining", "pretraining"
    )

    baseline_run = require_mapping(baseline, "run", "baseline")
    self_play = require_mapping(baseline, "self_play", "baseline")
    online_train = require_mapping(baseline, "training", "baseline")
    replay = require_mapping(baseline, "replay", "baseline")
    checkpoint = require_mapping(baseline, "checkpoint", "baseline")
    adaptive_checkpoint = require_mapping(adaptive, "checkpoint", "adaptive")

    return {
        "model_architecture": model_architecture,
        "pretraining": {
            "dataset_sha256": require_value(
                pretraining_data, "expected_sha256", "pretraining.data"
            ),
            "seed": require_value(pretraining_run, "seed", "pretraining.run"),
            **select(
                pretraining_train,
                (
                    "epochs",
                    "batch_size",
                    "micro_batch_size",
                    "learning_rate",
                    "weight_decay",
                    "gradient_clip",
                ),
                "pretraining.pretraining",
            ),
            "precision": precision(
                pretraining_train, "pretraining.pretraining"
            ),
        },
        "online_baseline": {
            "seed": require_value(baseline_run, "seed", "baseline.run"),
            "self_play": select(
                self_play,
                (
                    "iterations",
                    "games_per_iteration",
                    "mcts_simulations",
                    "eval_mcts_in_batch",
                    "cpuct",
                    "temperature_threshold",
                    "dirichlet_noise",
                    "dirichlet_alpha",
                    "dirichlet_epsilon",
                    "max_game_length",
                ),
                "baseline.self_play",
            ),
            "training": {
                **select(
                    online_train,
                    (
                        "epochs",
                        "batch_size",
                        "micro_batch_size",
                        "optimizer",
                        "learning_rate",
                        "weight_decay",
                        "gradient_clip",
                    ),
                    "baseline.training",
                ),
                "precision": precision(online_train, "baseline.training"),
            },
            "adaptive_scheduler": extract_scheduler(adaptive, baseline),
        },
        "replay_and_checkpointing": {
            "replay": select(
                replay,
                ("max_queue_size", "max_train_samples", "history_iterations"),
                "baseline.replay",
            ),
            "checkpoint": {
                **select(
                    checkpoint,
                    (
                        "save_every_iterations",
                        "save_replay_state",
                        "save_instrumentation_state",
                        "save_rng_state",
                    ),
                    "baseline.checkpoint",
                ),
                "save_scheduler_state": require_value(
                    adaptive_checkpoint,
                    "save_scheduler_state",
                    "adaptive.checkpoint",
                ),
            },
        },
    }


def write_summary(summary: Mapping[str, Any], output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    args = parse_args()
    summary = build_summary(
        load_yaml(args.pretraining_config.expanduser().resolve()),
        load_yaml(args.baseline_config.expanduser().resolve()),
        load_yaml(args.adaptive_config.expanduser().resolve()),
    )
    output = write_summary(summary, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
