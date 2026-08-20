#!/usr/bin/env python3
"""CLI entry point for Adaptive dry-run, fresh, and resume operations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = SOURCE_ROOT / "experiments"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from experiments.Adaptive.experiment_runtime import (  # noqa: E402
    ExperimentRuntimeError,
    ProtocolValidationError,
    ResumeStateError,
    RuntimeRequest,
    configured_run_dir,
    run_experiment,
)


EXIT_OK = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_PROTOCOL_FAILURE = 2
EXIT_RESUME_INCOMPLETE = 3


def _run_dir_argument(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    working_candidate = (Path.cwd() / expanded).resolve()
    experiments_candidate = (EXPERIMENTS_ROOT / expanded).resolve()
    if working_candidate.exists() or not experiments_candidate.exists():
        return working_candidate
    return experiments_candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "fresh", "resume"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    if args.mode in {"dry-run", "fresh"} and args.config is None:
        parser.error(f"{args.mode} requires --config")
    if args.mode == "resume" and args.run_dir is None:
        parser.error("resume requires --run-dir")
    return args


def _request_from_args(args: argparse.Namespace) -> RuntimeRequest:
    config_path = args.config.expanduser().resolve() if args.config else None
    if args.run_dir is not None:
        run_dir = _run_dir_argument(args.run_dir)
    else:
        assert config_path is not None
        run_dir = configured_run_dir(config_path)
    return RuntimeRequest(args.mode, config_path, run_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        request = _request_from_args(args)
        logging.info(
            "Adaptive request mode=%s config=%s run_dir=%s",
            request.mode,
            request.config_path,
            request.run_dir,
        )
        result = run_experiment(request)
    except ResumeStateError as exc:
        logging.error("Adaptive resume state is incomplete: %s", exc)
        return EXIT_RESUME_INCOMPLETE
    except ProtocolValidationError as exc:
        logging.error("Adaptive protocol validation failed: %s", exc)
        return EXIT_PROTOCOL_FAILURE
    except ExperimentRuntimeError as exc:
        logging.exception("Adaptive runtime failed: %s", exc)
        return EXIT_RUNTIME_FAILURE
    except (OSError, RuntimeError, ValueError) as exc:
        logging.exception("Adaptive technical failure: %s", exc)
        return EXIT_RUNTIME_FAILURE
    except KeyboardInterrupt:
        logging.error("Adaptive run interrupted by user")
        return EXIT_RUNTIME_FAILURE

    status = result.get("status", "unknown")
    if request.mode == "dry-run":
        print(f"Adaptive dry-run status: {status}")
        print(f"Run directory: {request.run_dir}")
    else:
        print(f"Adaptive run status: {status}")
        print(f"Run directory: {result['run_dir']}")
        print(f"Summary: {result['summary_path']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
