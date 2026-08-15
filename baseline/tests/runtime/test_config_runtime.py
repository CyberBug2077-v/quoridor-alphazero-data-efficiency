from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import (
    ConfigError,
    load_yaml,
    map_baseline_to_train_args,
    map_model_to_nn_args,
    merge_defaults,
    resolve_baseline_config,
    resolve_pretraining_config,
)


BASELINE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = BASELINE_ROOT / "configs"


def test_reproduction_configs_resolve_and_map() -> None:
    pretraining_path = CONFIG_ROOT / "pretraining_reproduction.yaml"
    pilot_path = CONFIG_ROOT / "baseline_pilot.yaml"
    baseline_path = CONFIG_ROOT / "baseline_reproduction.yaml"

    pretraining = resolve_pretraining_config(load_yaml(pretraining_path), pretraining_path)
    pilot = resolve_baseline_config(load_yaml(pilot_path), pilot_path)
    baseline = resolve_baseline_config(load_yaml(baseline_path), baseline_path)

    assert pretraining["mapped_args"]["nn_args"] == map_model_to_nn_args(pretraining)
    assert pilot["mapped_args"]["train_args"] == map_baseline_to_train_args(pilot)
    assert pilot["mapped_args"]["train_args"]["numIters"] == 7
    assert baseline["mapped_args"]["train_args"]["numIters"] is None
    assert Path(pilot["mapped_args"]["train_args"]["checkpoint"]).is_absolute()
    assert pilot["model"] == baseline["model"] == pretraining["model"]


def test_merge_defaults_is_recursive_and_does_not_mutate_inputs() -> None:
    defaults = {"run": {"device": "cuda", "seed": 1}, "enabled": False}
    supplied = {"run": {"seed": 2}}

    merged = merge_defaults(defaults, supplied)

    assert merged == {"run": {"device": "cuda", "seed": 2}, "enabled": False}
    assert defaults["run"]["seed"] == 1
    assert supplied == {"run": {"seed": 2}}


def test_pretraining_rejects_online_sections(tmp_path: Path) -> None:
    source = load_yaml(CONFIG_ROOT / "pretraining_reproduction.yaml")
    source["self_play"] = {"games_per_iteration": 1}
    path = tmp_path / "pretraining.yaml"
    import yaml

    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="forbidden section"):
        resolve_pretraining_config(load_yaml(path), path)


def test_baseline_rejects_expert_top_up(tmp_path: Path) -> None:
    source = load_yaml(CONFIG_ROOT / "baseline_pilot.yaml")
    source["training"]["expert_top_up"] = False
    path = tmp_path / "baseline.yaml"
    import yaml

    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="expert_top_up"):
        resolve_baseline_config(load_yaml(path), path)
