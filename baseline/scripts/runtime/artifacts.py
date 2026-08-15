from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")


def _plain_data(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _plain_data(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(nested) for nested in value]
    return value


def _atomic_replace(path: Path, write: Callable[[Path], None]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        write(temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    destination = Path(path)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(
                _plain_data(payload),
                output,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(destination, write)


def atomic_write_yaml(path: Path | str, payload: Any) -> Path:
    destination = Path(path)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(
                _plain_data(payload),
                output,
                sort_keys=False,
                allow_unicode=True,
            )
            output.flush()
            os.fsync(output.fileno())

    return _atomic_replace(destination, write)


class JsonlWriter:
    """Durably append one JSON object per line.

    If a payload contains ``optimizer_steps``, the value must be positive. This
    preserves the smoke metrics contract while allowing other JSONL streams.
    """

    def __init__(self, path: Path | str, *, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open(
            "a" if append else "w",
            encoding="utf-8",
            newline="\n",
        )

    def __call__(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("JSONL payload must be a mapping")
        if "optimizer_steps" in payload and payload["optimizer_steps"] <= 0:
            raise ValueError("metrics.optimizer_steps must be greater than zero")
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self._file.write(serialized + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def find_existing_run_artifacts(output_dir: Path | str) -> list[Path]:
    output = Path(output_dir)
    candidates = [
        output / "metrics.jsonl",
        output / "run_metadata.json",
        output / "metadata.json",  # legacy smoke artifact name
        output / "evaluation.json",
        output / "summary.json",
        output / "run.log",
        output / "tracker.json",
        output / "checkpoints" / "latest.examples",
        output / "checkpoints" / "run_state.pth.tar",
    ]
    artifacts = [path for path in candidates if path.exists()]
    checkpoint_dir = output / "checkpoints"
    if checkpoint_dir.is_dir():
        artifacts.extend(
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file() and CHECKPOINT_PATTERN.match(path.name)
        )
    return sorted(set(artifacts), key=lambda path: str(path).lower())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {line_number}: {path}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} is not an object: {path}")
        records.append(payload)
    return records


def write_summary(
    resolved: dict[str, Any],
    *,
    mode: str,
    status: str,
    evaluation_path: Path | None,
    resume_semantics: str | None = None,
) -> Path:
    output_dir = Path(resolved["_output_path"])
    logging = resolved.get("logging", {})
    metrics_path = output_dir / logging.get("metrics_file", "metrics.jsonl")
    metrics = _read_jsonl(metrics_path) if metrics_path.is_file() else []
    self_play = resolved.get("self_play", {})
    target_iterations = self_play.get("iterations")
    if target_iterations is None:
        target_iterations = resolved.get("budget", {}).get("max_iterations")
    summary = {
        "schema_version": 1,
        "mode": mode,
        "status": status,
        "completed_iterations": [record["iteration"] for record in metrics],
        "target_iterations": target_iterations,
        "final_checkpoint": metrics[-1].get("checkpoint_path") if metrics else None,
        "metrics_path": metrics_path.as_posix() if metrics else None,
        "evaluation_path": evaluation_path.as_posix() if evaluation_path else None,
        "evaluation": (
            json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation_path and evaluation_path.is_file()
            else None
        ),
        "resume_semantics": resume_semantics or (
            "Iteration-boundary resume: model weights and replay history are restored; "
            "optimizer, scheduler, and RNG state are not restored."
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return atomic_write_json(
        output_dir / logging.get("summary_file", "summary.json"),
        summary,
    )
