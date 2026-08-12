from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from run_smoke import MetricsJsonlWriter


def metric_record(iteration: int) -> dict:
    return {
        "schema_version": 1,
        "iteration": iteration,
        "optimizer_steps": 1,
        "policy_loss": 0.25,
        "label": "迭代",
    }


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_one_record_produces_one_json_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with MetricsJsonlWriter(path) as writer:
        writer(metric_record(1))

    lines = read_lines(path)
    assert len(lines) == 1
    assert json.loads(lines[0])["iteration"] == 1


def test_two_records_produce_independent_ordered_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with MetricsJsonlWriter(path) as writer:
        writer(metric_record(1))
        writer(metric_record(2))

    lines = read_lines(path)
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["iteration"] for record in records] == [1, 2]
    assert not path.read_text(encoding="utf-8").lstrip().startswith("[")


@pytest.mark.parametrize("invalid_number", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(
    tmp_path: Path,
    invalid_number: float,
) -> None:
    path = tmp_path / "metrics.jsonl"
    record = metric_record(1)
    record["policy_loss"] = invalid_number

    with MetricsJsonlWriter(path) as writer:
        with pytest.raises(ValueError, match="JSON compliant"):
            writer(record)

    assert path.read_bytes() == b""


def test_zero_optimizer_steps_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    record = metric_record(1)
    record["optimizer_steps"] = 0

    with MetricsJsonlWriter(path) as writer:
        with pytest.raises(ValueError, match="optimizer_steps"):
            writer(record)

    assert path.read_bytes() == b""


def test_windows_compatible_utf8_and_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with MetricsJsonlWriter(path) as writer:
        writer(metric_record(1))
        writer(metric_record(2))

    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "迭代" in raw.decode("utf-8")
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2
    assert b"\r\r\n" not in raw
