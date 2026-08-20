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
|-- analysis/              Offline metrics, replay, fixed-basket, and hold-out evaluation
|   |-- configs/
|   |   |-- baseline_gate2.yaml
|   |   |-- fixed_basket_v1.yaml
|   |   |-- h1_v1.yaml
|   |   |-- h1_v1_1.yaml
|   |   `-- holdout_v1.yaml
|   |-- js/                Seeded bridge for the JavaScript MCTS opponent
|   |-- scripts/
|   |   |-- derive_baseline_metrics.py
|   |   |-- summarize_replay.py
|   |   |-- evaluate_fixed_basket.py
|   |   |-- summarize_fixed_basket.py
|   |   |-- detect_plateau.py
|   |   |-- merge_h1_evidence.py
|   |   |-- generate_holdout.py
|   |   |-- evaluate_holdout.py
|   |   |-- verify_holdout.py
|   |   `-- holdout_common.py
|   |-- tests/
|   |   |-- test_derive_baseline_metrics.py
|   |   |-- test_summarize_replay.py
|   |   |-- test_fixed_basket.py
|   |   |-- test_h1_evidence.py
|   |   `-- test_holdout.py
|   `-- README.md          Reproducible analysis operations manual
|-- arena/                 Fixed opponents, adapters, Elo, and match utilities
|-- configs/               Smoke-test and baseline configuration files
|-- data/                  Local pretraining artifacts (not normal Git content)
|-- external/
|   |-- alphazero/         Inherited AlphaZero and Quoridor implementation
|   |-- js-mcts/           Heuristic JavaScript MCTS opponent
|   `-- quoridor-server/   Arena rules and move-legality implementation
|-- scripts/
|   |-- runtime/           Shared config, artifact, metadata, and checkpoint code
|   |-- probe_pretraining_batch.py
|   |-- run_pretraining.py
|   |-- verify_pretraining.py
|   |-- run_baseline.py
|   |-- verify_baseline.py
|   |-- summarize_pilot.py
|   |-- run_smoke.py       Baseline GPU smoke-test entry point
|   `-- original/          Preserved upstream/legacy launch scripts for reference
|-- tests/
|   |-- pretraining/
|   |   |-- test_pretraining_entrypoint.py
|   |   |-- test_pretraining_probe.py
|   |   `-- test_verify_pretraining.py
|   |-- baseline/
|   |   |-- test_baseline_lifecycle.py
|   |   `-- test_summarize_pilot.py
|   |-- runtime/           Shared runtime unit tests
|   `-- smoke/             Game, MCTS, network, modes, and smoke E2E tests
`-- outputs/               Generated local run artifacts; not source code
```

Legacy entry points and shell launchers are retained only as references until
their required behaviour has been extracted into the new configuration-driven
scripts. They are not authoritative experiment launchers.

The post-run analysis workflow is documented separately in
[`analysis/README.md`](analysis/README.md). Its fixed-basket evaluation is kept
separate from the existing random + greedy sanity evaluation, and the frozen
MCTS policy/value hold-out is never admitted to training replay.

## Formal reproduction entry points

The formal workflow is ordered. First probe the configured batch with the
formal model and dataset, then validate the full configuration, run fresh
pretraining, and finally verify the frozen result:

```bash
python scripts/probe_pretraining_batch.py --config configs/pretraining_reproduction.yaml --effective-batch-size 2048 --micro-batch-size 1024 --steps 20 --trial 1
python scripts/run_pretraining.py dry-run --config configs/pretraining_reproduction.yaml
python scripts/run_pretraining.py fresh --config configs/pretraining_reproduction.yaml
python scripts/verify_pretraining.py --run-dir outputs/pretraining_reproduction_seed1001
```

The batch probe uses the formal network and dataset but writes only to
`outputs/pretraining_probe/`; it never creates or modifies the formal
pretraining run directory or a checkpoint. It returns zero only when the
optimizer step succeeds, all recorded values are finite, no formal artifact is
changed, and the conservative memory margin is at least 1 GiB and 10% of total
GPU memory. OOM, other runtime failures, or insufficient margin return status
2 while still preserving a diagnostic JSON whenever execution reaches the
probe result stage.

`batch_size` is the effective optimizer batch, while `micro_batch_size` is the
number of samples placed on the GPU for one forward/backward pass. Formal
pretraining and baseline training both retain effective batch 2048. The frozen
pretraining run used micro-batch 1024 and therefore two backward passes per
optimizer step. Baseline pilot, formal baseline reproduction, and subsequent
experiments target the RTX 4090 with micro-batch 2048, so each effective batch
uses one forward/backward pass followed by one gradient clip and optimizer
step. The effective batch must be divisible by the micro-batch.

The completed 20-step pretraining trial on the RTX 4070 Laptop GPU passed: 20
optimizer steps, 40 micro-batches, and 40,960 samples were recorded; peak
reserved memory was 3,846 MiB and the conservative remaining margin was 3,172
MiB (38.7%). The result is stored as
`outputs/pretraining_probe/effective_2048_micro_1024_steps_20_trial_1.json`.

`run_pretraining.py` supports only `dry-run` and `fresh`. It does not resume,
self-play, evaluate opponents, or create replay state. `fresh` refuses any
existing content in its output directory and produces:

```text
outputs/pretraining_reproduction_seed1001/
|-- resolved_config.yaml
|-- run_metadata.json
|-- pretraining_metrics.jsonl
|-- summary.json
|-- run.log
`-- checkpoints/
    |-- checkpoint_0.pth.tar
    `-- best.pth.tar
```

JSON, YAML, and checkpoint commits use temporary files followed by atomic
replacement; a failed write must not leave a `.tmp` artifact behind.

`verify_pretraining.py` is read-only. It re-hashes the dataset and both
checkpoints, loads checkpoints on CPU, reconstructs the configured model to
compare every state-dict name and shape, checks finite metrics and weights, and
rejects incomplete `.tmp` artifacts.

After copying the resulting `checkpoint_0_sha256` from the pretraining summary
into `initialization.expected_sha256`, validate or launch the fixed-games
baseline:

```bash
python scripts/run_baseline.py dry-run --config configs/baseline_pilot.yaml
python scripts/run_baseline.py fresh --config configs/baseline_pilot.yaml --stop-after-iteration 5
python scripts/run_baseline.py resume --run-dir outputs/baseline_pilot_seed1001_4090
python scripts/run_baseline.py evaluate-only --run-dir outputs/baseline_pilot_seed1001_4090
python scripts/verify_baseline.py --run-dir outputs/baseline_pilot_seed1001_4090
```

The RTX 4090 pilot uses run ID and output directory
`baseline_pilot_seed1001_4090`. The completed RTX 4070 pilot is retained
separately as `outputs/baseline_pilot_seed1001_4070`.

To improve RTX 4090 utilization, pilot and formal baseline self-play batch 10
MCTS neural-network evaluations at a time (`self_play.eval_mcts_in_batch: 10`).
The separate random/greedy evaluation pipeline remains at
`evaluation.eval_mcts_in_batch: 4` so its measurement protocol is unchanged.

`fresh` initializes only model weights from the frozen pretrained checkpoint
and always creates an empty online replay. `resume` restores a numbered model
checkpoint, replay history, RNG state, cumulative GPU-hours, and the reserved
instrumentation state from the same run.

The pilot configuration targets seven iterations. Its intended recovery test
is a fresh run stopped after iteration 5 followed by `resume` to iteration 7;
`--stop-after-iteration` is a normal stopped state, not a failure.

The RTX 4090 formal baseline is frozen as
`baseline_reproduction_seed1001_4090` with seed 1001, a target of 210
iterations and a 24 GPU-hour budget, 75 self-play games per iteration, 200 MCTS
simulations, self-play inference batch 10, four training epochs, effective and
micro-batch 2048, learning rate 0.0002, and replay history 150. Numbered
checkpoints are written every 10 iterations and evaluation is scheduled every
20 iterations. All other algorithmic, initialization, instrumentation, and
evaluation settings are inherited from the accepted pilot and frozen.

The formal evaluation set is the pretrained iteration 0 checkpoint, iterations
20, 40, 60, 80, 100, 120, 140, 160, 180, and 200, plus final checkpoint 210.
Iteration 0 is evaluated explicitly with `evaluate-only`; cadence 20 covers
20–200, and the completed training entry point evaluates checkpoint 210.

After the pilot and its evaluations finish, generate the read-only timing and
capacity analysis with:

```bash
python scripts/summarize_pilot.py --run-dir outputs/baseline_pilot_seed1001_4090
python scripts/summarize_pilot.py --run-dir outputs/baseline_pilot_seed1001_4090 --gpu-hours 24
```

The command writes `pilot_report.json`. It reports per-iteration self-play,
network-training, evaluation, replay, optimizer, GPU-memory, per-game, and
per-position costs; compares iteration 1 with later iterations; and projects
capacity from the median after excluding iteration 1 as warm-up. Repeating
`--gpu-hours` adds more projection budgets. It never edits the resolved pilot
configuration or the formal reproduction configuration. Evaluation artifacts
created by the current code record measured evaluation duration; older files
without timing remain readable and are explicitly marked as unmeasured.

## Environment

The current Windows development environment uses a Conda environment named
`quoridor-az`, PyTorch with CUDA support, Node.js for the heuristic bot, and
MSVC for the native PathFinder extension. The formal cluster environment may
use different Python, PyTorch, CUDA, and compiler versions; every run must
record the versions actually used.

Install Python dependencies and the local arena package with the selected
environment:

```bash
python -m pip install -r ../requirements.txt
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

## Test suite

Pytest configuration lives in `pytest.ini`, uses `tests/` as its only discovery
root, and rejects unregistered markers. Run the complete suite from this
directory with:

```bash
python -m pytest
```

The pretraining lifecycle tests use temporary directories, synthetic data,
and a mock network; they do not construct the formal model or write formal
outputs:

```bash
python -m pytest tests/pretraining/test_pretraining_entrypoint.py
python -m pytest tests/pretraining/test_pretraining_probe.py -m "not gpu"
python -m pytest tests/pretraining/test_verify_pretraining.py
python -m pytest tests/baseline/test_baseline_lifecycle.py
python -m pytest tests/baseline/test_summarize_pilot.py
```

The registered markers are:

- `gpu`: allocates or otherwise requires a CUDA-capable GPU;
- `slow`: materially slower than ordinary unit and artifact tests;
- `integration`: crosses process, filesystem, dataset, model, or GPU boundaries;
- `e2e`: runs the complete smoke train/resume/evaluate/verify pipeline.

The real formal-scale batch probe is opt-in even on GPU machines. On
PowerShell, enable and run it with:

```powershell
$env:RUN_FORMAL_PRETRAINING_PROBE = "1"
python -m pytest tests/pretraining/test_pretraining_probe.py -m "gpu and slow and integration"
```

Without that environment variable, the formal probe test is skipped. The
ordinary probe tests still cover argument validation, output isolation,
OOM/runtime failure handling, valid JSON output, input immutability, and the
absence of formal checkpoints.

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
