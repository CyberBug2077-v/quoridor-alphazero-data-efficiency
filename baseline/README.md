# Adapted Quoridor AlphaZero Baseline

This directory contains the inherited Quoridor AlphaZero system used to
establish the baseline for the MSc dissertation project. It is derived from
[Victor Baeza's Quoridor AlphaZero Bot][upstream], but it is **not** a verbatim
mirror of the upstream repository.

The baseline is intentionally allowed to contain the minimum changes needed
to make the inherited system reproducible on the current Windows workstation
and the school's Linux GPU cluster. New research mechanisms belong in
`../extension/`, not here.

## Purpose and boundary

The baseline is responsible for:

- validating the inherited game, MCTS, network, and training pipeline;
- running the end-to-end GPU smoke test;
- reproducing the inherited pretraining and baseline training conditions;
- producing a frozen pretrained checkpoint and its SHA-256 digest; and
- recording the resolved configuration, environment, seed, and input hashes.

The baseline may receive compatibility fixes, configuration plumbing, tests,
and reproducibility instrumentation. It must not contain the dissertation's
adaptive self-play intervention or other new experimental mechanisms. After
baseline validation, its code commit and checkpoint are frozen inputs to the
extension experiments.

## Repository layout

```text
baseline/
|-- arena/                 Fixed opponents, adapters, Elo, and match utilities
|-- configs/               Smoke-test and baseline configuration files
|-- data/                  Local pretraining artifacts (not normal Git content)
|-- external/
|   |-- alphazero/         Inherited AlphaZero and Quoridor implementation
|   |-- js-mcts/           Heuristic JavaScript MCTS opponent
|   `-- quoridor-server/   Arena rules and move-legality implementation
|-- scripts/
|   |-- run_smoke.py       Baseline GPU smoke-test entry point
|   `-- original/          Preserved upstream/legacy launch scripts for reference
|-- tests/                 Baseline compatibility and regression tests
`-- outputs/               Generated local run artifacts; not source code
```

Legacy entry points and shell launchers are retained only as references until
their required behaviour has been extracted into the new configuration-driven
scripts. They are not authoritative experiment launchers.

## Environment

The current Windows development environment uses a Conda environment named
`quoridor-az`, PyTorch with CUDA support, Node.js for the heuristic bot, and
MSVC for the native PathFinder extension. The formal cluster environment may
use different Python, PyTorch, CUDA, and compiler versions; every run must
record the versions actually used.

Install Python dependencies and the local arena package with the selected
environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e external/quoridor-server
```

Build PathFinder from source on the target platform:

```bash
cd external/alphazero/quoridor/pathFinder-module
python setup.py build_ext --inplace
```

This produces a `.pyd` on Windows or a `.so` on Linux. Compiled binaries are
platform-specific and must not be treated as portable source artifacts.

## Smoke test

The smoke-test configuration is `configs/smoke_gpu.yaml`. The runner exposes
four explicit modes:

```bash
python scripts/run_smoke.py dry-run
python scripts/run_smoke.py fresh
python scripts/run_smoke.py resume --run-dir outputs/smoke_gpu
python scripts/run_smoke.py evaluate-only --run-dir outputs/smoke_gpu
python scripts/verify_smoke.py --run-dir outputs/smoke_gpu
```

`dry-run` validates dependencies, CUDA, paths, fields, and the complete mapped
configuration without constructing a model. `fresh` refuses to overwrite any
existing training artifact. `resume` restores a complete iteration boundary
from the run directory and appends metrics; optimizer, scheduler, and RNG state
are not restored. `evaluate-only` loads a checkpoint and writes evaluation
results without training or changing `metrics.jsonl`.

For bounded validation runs, `fresh` and `resume` accept
`--stop-after-iteration N`. Tests may redirect a run with `--output-dir PATH`,
and `evaluate-only` accepts `--checkpoint PATH` for an explicit model file.
`verify_smoke.py` is read-only: it does not construct or run the model, and
validates the completed artifact tree, metrics, replay history, reloadable
checkpoint state dictionaries, evaluation results, and run identity.

## Data and generated artifacts

Large datasets, extracted pickle files, checkpoints, compiled extensions, and
training outputs are local artifacts rather than ordinary Git source files.
Each artifact consumed by an experiment must be identified by path, byte size,
and SHA-256 in the run configuration or data manifest.

The currently inherited heuristic dataset archive is
`data/heuristic_games.zip`. Its extracted form is a generated training input
and should not be duplicated into `../extension/`.

## Reproducibility rule

A formal baseline run is valid only when it records:

- a real local Git commit and clean/dirty status;
- the complete resolved configuration;
- the random seed;
- Python, PyTorch, CUDA, Node, compiler, GPU, and memory information;
- all input dataset and starting-checkpoint SHA-256 digests; and
- the output checkpoint SHA-256 digest.

Any change to a frozen core setting requires a new experiment ID.

## Attribution and licences

Upstream provenance and the local modification boundary are documented in
[UPSTREAM.md](UPSTREAM.md). Third-party licence notices are summarized in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the original licence
files remain beside their respective components.

[upstream]: https://gitlab.com/victorbaeza_h/quoridor-alphazero-bot
