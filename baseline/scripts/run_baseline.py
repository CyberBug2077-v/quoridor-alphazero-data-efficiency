#!/usr/bin/env python3
"""Run the fixed-games AlphaZero baseline from a frozen checkpoint_0."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from runtime.artifacts import (
    JsonlWriter,
    atomic_write_yaml,
    find_existing_run_artifacts,
    write_summary,
)
from runtime.checkpointing import (
    capture_rng_state,
    load_run_state,
    restore_rng_state,
    save_run_state,
    validate_checkpoint_hash,
)
from runtime.config import ConfigError, load_yaml, resolve_baseline_config
from runtime.metadata import collect_git_metadata, sha256_file, write_run_metadata


BASELINE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = BASELINE_ROOT.parent
DEFAULT_CONFIG = BASELINE_ROOT / "configs" / "baseline_pilot.yaml"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")


class _Tee:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def isatty(self) -> bool:
        method = getattr(self.stream, "isatty", None)
        return bool(method and method())


class RunLog:
    def __init__(self, path: Path, *, append: bool):
        self.path = path
        self.append = append

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open(
            "a" if self.append else "w", encoding="utf-8", newline="\n"
        )
        self.stdout = _Tee(sys.stdout, self.file)
        self.stderr = _Tee(sys.stderr, self.file)
        self.redirect_stdout = redirect_stdout(self.stdout)
        self.redirect_stderr = redirect_stderr(self.stderr)
        self.redirect_stdout.__enter__()
        self.redirect_stderr.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.redirect_stderr.__exit__(exc_type, exc_value, traceback)
        self.redirect_stdout.__exit__(exc_type, exc_value, traceback)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "fresh", "resume", "evaluate-only"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after-iteration", type=int)
    args = parser.parse_args()
    if args.mode in {"dry-run", "fresh"}:
        args.config = args.config or DEFAULT_CONFIG
        if args.run_dir is not None:
            parser.error("dry-run/fresh do not accept --run-dir")
    else:
        if args.run_dir is None:
            parser.error("resume/evaluate-only require --run-dir")
        if args.config is not None:
            parser.error("resume/evaluate-only restore the original resolved config")
    if args.checkpoint is not None and args.mode != "evaluate-only":
        parser.error("--checkpoint is only valid for evaluate-only")
    if args.stop_after_iteration is not None:
        if args.mode not in {"fresh", "resume"}:
            parser.error("--stop-after-iteration is only valid for fresh/resume")
        if args.stop_after_iteration < 1:
            parser.error("--stop-after-iteration must be >= 1")
    return args


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(BASELINE_ROOT.rglob("*.py")):
        if any(part in {"outputs", "data", "tests", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        digest.update(path.relative_to(BASELINE_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def initialize_cuda(resolved: dict[str, Any]) -> None:
    if resolved["run"]["device"] != "cuda" or not torch.cuda.is_available():
        raise ConfigError("baseline requires CUDA")
    gpu_index = int(resolved["run"]["gpu_index"])
    if gpu_index >= torch.cuda.device_count():
        raise ConfigError(
            f"run.gpu_index {gpu_index} is invalid; detected "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )
    torch.cuda.set_device(gpu_index)


def require_inside_baseline(resolved: dict[str, Any]) -> None:
    paths = {
        "run.output_dir": resolved["_output_path"],
        "checkpoint.directory": resolved["checkpoint"]["_directory_path"],
        "initialization.checkpoint_path": resolved["initialization"]["_checkpoint_path"],
    }
    for label, path in paths.items():
        if not Path(path).is_relative_to(BASELINE_ROOT):
            raise ConfigError(f"{label} must be inside {BASELINE_ROOT}")
    if resolved["checkpoint"]["_directory_path"] != resolved["_output_path"] / "checkpoints":
        raise ConfigError("checkpoint.directory must equal run.output_dir/checkpoints")
    for field, filename in resolved["logging"].items():
        requested = Path(filename)
        if requested.is_absolute() or requested.name != filename:
            raise ConfigError(f"logging.{field} must be a filename, not a path")


def validate_fixed_baseline_conditions(resolved: dict[str, Any]) -> None:
    self_play = resolved["self_play"]
    training = resolved["training"]
    replay = resolved["replay"]
    initialization = resolved["initialization"]
    if self_play["games_per_iteration"] != 75:
        raise ConfigError("fixed baseline requires 75 games per iteration")
    if self_play["mcts_simulations"] != 200:
        raise ConfigError("fixed baseline requires 200 MCTS simulations")
    if training["epochs"] != 4:
        raise ConfigError("fixed baseline requires 4 training epochs")
    if replay["history_iterations"] != 150:
        raise ConfigError("fixed baseline requires replay history 150")
    if initialization["load_replay"]:
        raise ConfigError("fresh pretrained initialization must not restore replay")
    train_args = resolved["mapped_args"]["train_args"]
    if train_args["fill_with_expert_data"] or train_args["expert_examples_data"]:
        raise ConfigError("expert top-up must be disabled")
    if train_args["heuristic_alpha"] != 0:
        raise ConfigError("heuristic guidance must be disabled")
    unsupported = sorted(set(resolved["evaluation"]["opponents"]) - {"random", "greedy"})
    if unsupported:
        raise ConfigError(f"unsupported evaluation opponent(s): {', '.join(unsupported)}")


def ensure_runnable_values(resolved: dict[str, Any]) -> None:
    if resolved["self_play"].get("iterations") is None:
        raise ConfigError("self_play.iterations must be set before launching")
    if resolved["checkpoint"]["save_every_iterations"] is None:
        raise ConfigError("checkpoint.save_every_iterations must be set before launching")
    if resolved["evaluation"]["enabled"]:
        if resolved["evaluation"]["games_per_opponent"] is None:
            raise ConfigError("evaluation.games_per_opponent must be set before launching")
        if resolved["evaluation"]["evaluate_every_iterations"] is None:
            raise ConfigError("evaluation.evaluate_every_iterations must be set before launching")


def ensure_output_is_unused(output_dir: Path) -> None:
    conflicts = find_existing_run_artifacts(output_dir)
    if output_dir.is_dir():
        conflicts.extend(path for path in output_dir.iterdir() if path not in conflicts)
    if conflicts:
        joined = "\n  - ".join(str(path) for path in sorted(set(conflicts)))
        raise ConfigError(f"fresh run would overwrite existing output:\n  - {joined}")


def create_runtime(resolved: dict[str, Any]):
    alphazero_root = BASELINE_ROOT / "external" / "alphazero"
    pathfinder_root = alphazero_root / "quoridor" / "pathFinder-module"
    for path in (alphazero_root, pathfinder_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from Coach import Coach
    from quoridor.QuoridorGame import QuoridorGame
    from quoridor.pytorch import NNet as nnet_module
    from quoridor.pytorch.NNet import NNetWrapper
    from utils import dotdict

    nn_args = copy.deepcopy(resolved["mapped_args"]["nn_args"])
    nnet_module.args.update(nn_args)
    game = QuoridorGame(resolved["model"]["board_size"])
    network = NNetWrapper(game, custom_args=dotdict(nn_args))
    train_args = dotdict(copy.deepcopy(resolved["mapped_args"]["train_args"]))
    return game, network, train_args, Coach


def strict_load_initial_weights(
    network,
    checkpoint_path: Path,
    expected_sha256: str | None,
) -> str:
    if expected_sha256 is None:
        raise ConfigError("initialization.expected_sha256 must be filled before baseline launch")
    digest = validate_checkpoint_hash(checkpoint_path, expected_sha256)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConfigError(f"checkpoint cannot be loaded: {exc}") from exc
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise ConfigError("checkpoint must contain a non-empty state_dict")
    current = network.nnet.state_dict()
    if set(state_dict) != set(current):
        missing = sorted(set(current) - set(state_dict))
        extra = sorted(set(state_dict) - set(current))
        raise ConfigError(f"checkpoint parameter names differ: missing={missing} extra={extra}")
    mismatched = [
        name for name in current if tuple(current[name].shape) != tuple(state_dict[name].shape)
    ]
    if mismatched:
        raise ConfigError(f"checkpoint parameter shapes differ: {mismatched}")
    network.nnet.load_state_dict(state_dict, strict=True)
    for name, parameter in network.nnet.state_dict().items():
        if not torch.equal(parameter.detach().cpu(), state_dict[name].detach().cpu()):
            raise ConfigError(f"loaded parameter differs from checkpoint: {name}")
    return digest


def serializable_config(resolved: dict[str, Any]) -> dict[str, Any]:
    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(nested)
                for key, nested in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    payload = clean(resolved)
    payload["schema_version"] = 1
    payload["mode"] = "baseline"
    return payload


def load_saved_resolved(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    saved_path = run_dir / "resolved_config.yaml"
    saved = load_yaml(saved_path)
    if saved.get("mode") != "baseline":
        raise ConfigError("resolved configuration mode must be baseline")
    config = {
        key: value
        for key, value in saved.items()
        if key not in {"schema_version", "mode", "mapped_args"}
    }
    placeholder = BASELINE_ROOT / "configs" / "_resolved_baseline.yaml"
    resolved = resolve_baseline_config(config, placeholder)
    if resolved["_output_path"] != run_dir:
        raise ConfigError("run directory does not match resolved run.output_dir")
    if run_dir.name != resolved["run"]["id"]:
        raise ConfigError("run ID does not match run directory name")
    return resolved


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigError(f"metrics file not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ConfigError(f"blank metrics line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid metrics JSON line {line_number}: {exc}") from exc
        records.append(record)
    iterations = [record.get("iteration") for record in records]
    if iterations != list(range(1, len(records) + 1)):
        raise ConfigError(f"metrics iterations are not contiguous: {iterations}")
    return records


def validate_resume_boundary(resolved: dict[str, Any]) -> tuple[int, Path, dict[str, Any]]:
    output_dir = resolved["_output_path"]
    checkpoint_dir = resolved["checkpoint"]["_directory_path"]
    metrics = read_metrics(output_dir / resolved["logging"]["metrics_file"])
    if not metrics:
        raise ConfigError("resume requires at least one completed iteration")
    iteration = int(metrics[-1]["iteration"])
    checkpoint = checkpoint_dir / f"checkpoint_{iteration}.pth.tar"
    if not checkpoint.is_file():
        raise ConfigError(f"resume checkpoint not found: {checkpoint}")
    if Path(metrics[-1].get("checkpoint_path", "")).resolve() != checkpoint.resolve():
        raise ConfigError("last metrics record does not match resume checkpoint")
    validate_checkpoint_hash(checkpoint, metrics[-1].get("checkpoint_sha256"))
    if (checkpoint_dir / f"checkpoint_{iteration + 1}.pth.tar").exists():
        raise ConfigError("next iteration checkpoint already exists")
    examples_path = checkpoint_dir / "latest.examples"
    try:
        with examples_path.open("rb") as source:
            replay = pickle.load(source)
    except Exception as exc:
        raise ConfigError(f"could not load latest.examples: {exc}") from exc
    if not isinstance(replay, dict) or replay.get("iteration") != iteration:
        raise ConfigError("replay iteration does not match resume checkpoint")
    history = replay.get("examples")
    if not isinstance(history, list) or not history:
        raise ConfigError("replay history is empty or invalid")
    if len(history) > int(resolved["replay"]["history_iterations"]):
        raise ConfigError("replay history exceeds configured window")
    if any(not hasattr(item, "__len__") or len(item) == 0 for item in history):
        raise ConfigError("replay contains an empty or corrupt iteration")
    state = load_run_state(checkpoint_dir / "latest.state.pt")
    if state.get("iteration") != iteration:
        raise ConfigError("run-state iteration does not match resume checkpoint")
    cumulative = state.get("cumulative_gpu_hours")
    if (
        isinstance(cumulative, bool)
        or not isinstance(cumulative, (int, float))
        or not np.isfinite(cumulative)
        or cumulative < 0
    ):
        raise ConfigError("run state has invalid cumulative GPU-hours")
    if not isinstance(state.get("instrumentation_state"), dict):
        raise ConfigError("run state has invalid instrumentation state")
    return iteration, checkpoint, state


def state_for_iteration(
    iteration: int,
    cumulative_gpu_hours: float,
    instrumentation_state: dict[str, Any],
) -> dict[str, Any]:
    rng = capture_rng_state()
    return {
        "schema_version": 1,
        "iteration": iteration,
        "python_rng_state": rng["python"],
        "numpy_rng_state": rng["numpy"],
        "torch_rng_state": rng["torch_cpu"],
        "cuda_rng_state": rng["torch_cuda"],
        "cumulative_gpu_hours": cumulative_gpu_hours,
        "instrumentation_state": instrumentation_state,
    }


def restore_iteration_state(state: dict[str, Any]) -> None:
    required = {
        "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state"
    }
    missing = sorted(required - set(state))
    if missing:
        raise ConfigError(f"run state missing RNG field(s): {', '.join(missing)}")
    restore_rng_state(
        {
            "python": state["python_rng_state"],
            "numpy": state["numpy_rng_state"],
            "torch_cpu": state["torch_rng_state"],
            "torch_cuda": state["cuda_rng_state"],
        }
    )


def run_evaluation(
    resolved: dict[str, Any], game, network, checkpoint: Path
) -> Path:
    from evaluate import evaluate_checkpoint

    match = CHECKPOINT_PATTERN.match(checkpoint.name)
    label = match.group(1) if match else checkpoint.stem.replace(".pth", "")
    output_path = (
        resolved["_output_path"]
        / "evaluations"
        / f"evaluation_checkpoint_{label}.json"
    )
    _, written = evaluate_checkpoint(
        resolved, game, network, checkpoint=checkpoint, output_path=output_path
    )
    return written


def train(
    resolved: dict[str, Any],
    *,
    mode: str,
    stop_after_iteration: int | None,
    resume_checkpoint: Path | None = None,
    resume_state: dict[str, Any] | None = None,
) -> tuple[Path, Path | None, Path]:
    game, network, train_args, Coach = create_runtime(resolved)
    initial_path = resolved["initialization"]["_checkpoint_path"]
    if mode == "fresh":
        strict_load_initial_weights(
            network, initial_path, resolved["initialization"]["expected_sha256"]
        )
    else:
        assert resume_checkpoint is not None and resume_state is not None
        strict_load_initial_weights(network, resume_checkpoint, sha256_file(resume_checkpoint))

    target = int(resolved["self_play"]["iterations"])
    invocation_target = min(stop_after_iteration or target, target)
    train_args.numIters = invocation_target
    checkpoint_dir = resolved["checkpoint"]["_directory_path"]
    metrics_path = resolved["_output_path"] / resolved["logging"]["metrics_file"]
    cumulative = {
        "gpu_hours": float(resume_state.get("cumulative_gpu_hours", 0.0))
        if resume_state
        else 0.0,
        "instrumentation": dict(resume_state.get("instrumentation_state", {}))
        if resume_state
        else {},
    }
    with JsonlWriter(metrics_path, append=(mode == "resume")) as writer:
        def on_iteration(metrics: dict[str, Any]) -> None:
            cumulative["gpu_hours"] += float(metrics["iteration_seconds"]) / 3600.0
            metrics["cumulative_gpu_hours"] = cumulative["gpu_hours"]
            checkpoint_value = metrics.get("checkpoint_path")
            metrics["checkpoint_sha256"] = (
                sha256_file(checkpoint_value) if checkpoint_value else None
            )
            save_run_state(
                checkpoint_dir / "latest.state.pt",
                state_for_iteration(
                    int(metrics["iteration"]),
                    cumulative["gpu_hours"],
                    cumulative["instrumentation"],
                ),
            )
            writer(metrics)

        coach = Coach(game, network, train_args, iteration_callback=on_iteration)
        if mode == "fresh":
            if coach.trainExamplesHistory or coach.current_iteration != 0:
                raise RuntimeError("fresh baseline did not start with an empty replay")
        else:
            coach.loadTrainExamples()
            expected_iteration = int(resume_state["iteration"])
            if coach.current_iteration != expected_iteration:
                raise ConfigError("restored replay iteration does not match run state")
            restore_iteration_state(resume_state)
        coach.learn()

    scratch = checkpoint_dir / "temp.pth.tar"
    scratch.unlink(missing_ok=True)
    records = read_metrics(metrics_path)
    completed = int(records[-1]["iteration"])
    status = "completed" if completed == target else "stopped"
    evaluation_path = None
    if status == "completed" and resolved["evaluation"]["enabled"]:
        final_checkpoint = checkpoint_dir / f"checkpoint_{completed}.pth.tar"
        evaluation_path = run_evaluation(resolved, game, network, final_checkpoint)
    summary_path = write_summary(
        resolved,
        mode=mode,
        status=status,
        evaluation_path=evaluation_path,
        resume_semantics=(
            "Iteration-boundary resume restores model, replay history, RNG state, "
            "cumulative GPU-hours, and instrumentation state."
        ),
    )
    return metrics_path, evaluation_path, summary_path


def prepare_fresh(config_path: Path) -> tuple[dict[str, Any], str]:
    resolved = resolve_baseline_config(load_yaml(config_path), config_path)
    require_inside_baseline(resolved)
    validate_fixed_baseline_conditions(resolved)
    ensure_runnable_values(resolved)
    ensure_output_is_unused(resolved["_output_path"])
    initialize_cuda(resolved)
    set_seed(resolved["run"]["seed"], resolved["run"]["deterministic"])
    _, network, _, _ = create_runtime(resolved)
    digest = strict_load_initial_weights(
        network,
        resolved["initialization"]["_checkpoint_path"],
        resolved["initialization"]["expected_sha256"],
    )
    del network
    torch.cuda.empty_cache()
    return resolved, digest


def evaluate_only(resolved: dict[str, Any], checkpoint: Path | None) -> Path:
    metrics_path = resolved["_output_path"] / resolved["logging"]["metrics_file"]
    replay_path = resolved["checkpoint"]["_directory_path"] / "latest.examples"
    state_path = resolved["checkpoint"]["_directory_path"] / "latest.state.pt"
    protected = {path: sha256_file(path) for path in (metrics_path, replay_path, state_path)}
    checkpoints_before = {
        path: sha256_file(path)
        for path in resolved["checkpoint"]["_directory_path"].glob("*.pth.tar")
    }
    records = read_metrics(metrics_path)
    selected = checkpoint or (
        resolved["checkpoint"]["_directory_path"]
        / f"checkpoint_{records[-1]['iteration']}.pth.tar"
    )
    selected = selected.expanduser().resolve()
    game, network, _, _ = create_runtime(resolved)
    strict_load_initial_weights(network, selected, sha256_file(selected))
    output_path = run_evaluation(resolved, game, network, selected)
    for path, digest in protected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"evaluate-only modified protected artifact: {path}")
    checkpoints_after = {
        path: sha256_file(path)
        for path in resolved["checkpoint"]["_directory_path"].glob("*.pth.tar")
    }
    if checkpoints_after != checkpoints_before:
        raise RuntimeError("evaluate-only modified training checkpoints")
    return output_path


def main() -> int:
    args = parse_args()
    try:
        if args.mode in {"dry-run", "fresh"}:
            resolved, initial_hash = prepare_fresh(args.config)
            printable = serializable_config(resolved)
            if args.mode == "dry-run":
                print(yaml.safe_dump(printable, sort_keys=False, allow_unicode=True))
                print("Mapped Coach arguments:")
                print(yaml.safe_dump(resolved["mapped_args"]["train_args"], sort_keys=False))
                print("Mapped NNet arguments:")
                print(yaml.safe_dump(resolved["mapped_args"]["nn_args"], sort_keys=False))
                print("Baseline dry-run validation passed; no files were written.")
                return 0
            target = int(resolved["self_play"]["iterations"])
            if args.stop_after_iteration is not None and args.stop_after_iteration > target:
                raise ConfigError("--stop-after-iteration exceeds configured iterations")
            output_dir = resolved["_output_path"]
            output_dir.mkdir(parents=True, exist_ok=True)
            resolved["checkpoint"]["_directory_path"].mkdir(parents=True, exist_ok=True)
            atomic_write_yaml(output_dir / "resolved_config.yaml", printable)
            write_run_metadata(
                output_dir / resolved["logging"]["metadata_file"],
                project_root=SOURCE_ROOT,
                resolved_config=printable,
                input_hashes={"initial_checkpoint": initial_hash},
                extra={
                    "run_type": "baseline",
                    "initialization_type": "pretrained",
                    "initial_checkpoint_path": resolved["initialization"]["_checkpoint_path"].as_posix(),
                    "initial_checkpoint_sha256": initial_hash,
                    "initial_replay_size": 0,
                    "starting_iteration": 1,
                    "code_fingerprint": code_fingerprint(),
                },
            )
            with RunLog(output_dir / "run.log", append=False):
                # prepare_fresh constructs a network for strict compatibility
                # validation. Reset before constructing the actual Coach so the
                # training RNG stream starts exactly from the configured seed.
                set_seed(resolved["run"]["seed"], resolved["run"]["deterministic"])
                train(
                    resolved,
                    mode="fresh",
                    stop_after_iteration=args.stop_after_iteration,
                )
            return 0

        resolved = load_saved_resolved(args.run_dir)
        require_inside_baseline(resolved)
        validate_fixed_baseline_conditions(resolved)
        ensure_runnable_values(resolved)
        initialize_cuda(resolved)
        metadata = json.loads(
            (resolved["_output_path"] / resolved["logging"]["metadata_file"]).read_text(
                encoding="utf-8"
            )
        )
        initial_hash = validate_checkpoint_hash(
            resolved["initialization"]["_checkpoint_path"],
            metadata.get("initial_checkpoint_sha256"),
        )
        if initial_hash != resolved["initialization"]["expected_sha256"]:
            raise ConfigError("initial checkpoint hash differs from resolved configuration")
        if args.mode == "evaluate-only":
            output = evaluate_only(resolved, args.checkpoint)
            print(f"Evaluation: {output}")
            return 0

        current_git = collect_git_metadata(SOURCE_ROOT).get("commit")
        original_git = metadata.get("git", {}).get("commit")
        if not current_git or current_git != original_git:
            raise ConfigError("current code commit does not match the original baseline run")
        if metadata.get("code_fingerprint") != code_fingerprint():
            raise ConfigError("current baseline code differs from the original run")
        iteration, checkpoint, state = validate_resume_boundary(resolved)
        target = int(resolved["self_play"]["iterations"])
        if iteration >= target:
            raise ConfigError("run already completed configured iterations")
        if args.stop_after_iteration is not None:
            if args.stop_after_iteration <= iteration:
                raise ConfigError("--stop-after-iteration must exceed the completed iteration")
            if args.stop_after_iteration > target:
                raise ConfigError("--stop-after-iteration exceeds configured iterations")
        with RunLog(resolved["_output_path"] / "run.log", append=True):
            train(
                resolved,
                mode="resume",
                stop_after_iteration=args.stop_after_iteration,
                resume_checkpoint=checkpoint,
                resume_state=state,
            )
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
        print(f"Baseline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
