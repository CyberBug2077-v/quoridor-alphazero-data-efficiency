from __future__ import annotations

import json
from pathlib import Path

from runtime.metadata import (
    collect_cuda_environment,
    collect_hardware_metadata,
    collect_python_environment,
    sha256_file,
    write_run_metadata,
)


def test_hash_and_environment_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")

    assert sha256_file(source) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    assert collect_python_environment()["executable"]
    assert "available" in collect_cuda_environment()
    assert collect_hardware_metadata()["logical_cpu_count"]


def test_write_run_metadata_serializes_paths(tmp_path: Path) -> None:
    path = write_run_metadata(
        tmp_path / "run_metadata.json",
        project_root=Path(__file__).resolve().parents[3],
        resolved_config={"_output_path": tmp_path},
        input_hashes={"checkpoint": None},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["resolved_config"]["_output_path"] == tmp_path.as_posix()
    assert set(payload) >= {"git", "python", "cuda", "hardware", "input_hashes"}

