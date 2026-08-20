# Data-Efficient Quoridor AlphaZero Experiments

This directory contains the new research code for the MSc dissertation on
data-efficient post-pretraining self-play for 9x9 Quoridor AlphaZero under
compute constraints.

The experiments build on the adapted and validated system in `../baseline/`.
It does not copy or silently replace the inherited AlphaZero, Quoridor, Arena,
or heuristic-bot implementations. Instead, it consumes a frozen baseline code
commit, pretrained checkpoint, configuration, and artifact hashes.

## Research scope

New work developed here includes:

- replay-buffer and generalisation instrumentation;
- fresh-state, reuse, turnover, age, and coverage metrics;
- adaptive self-play scheduling;
- matched-compute Baseline versus Adaptive orchestration;
- experiment IDs and immutable resolved configurations;
- GPU-hour and resource accounting; and
- dissertation-specific evaluation and analysis.

## Contribution boundary

| Area | Location | Status |
|---|---|---|
| Quoridor rules, state/action representation | `../baseline/external/alphazero/quoridor/` | Inherited, with documented compatibility fixes |
| AlphaZero training, MCTS, and network | `../baseline/external/alphazero/` | Inherited baseline |
| Arena and fixed opponents | `../baseline/arena/` | Inherited/adapted baseline |
| Heuristic JS MCTS opponent | `../baseline/external/js-mcts/` | Inherited dependency |
| Baseline smoke, pretraining, and reproduction | `../baseline/configs/`, `../baseline/scripts/` | Reproduction infrastructure |
| Replay instrumentation and adaptive scheduling | `experiments/Adaptive/` | New dissertation work |
| Matched-compute experiments and analysis | `experiments/` | New dissertation work |

The baseline reproduction and H1 analysis are frozen. Adaptive runtime code,
instrumentation, resource accounting, and scheduling belong in
`experiments/Adaptive/`; protocols, generated artifacts, and tests remain in
their dedicated directories under `experiments/`.

## Planned layout

```text
experiments/
|-- Adaptive/             Adaptive scheduler, instrumentation, accounting, and runtime
|-- configs/              Instrumented and adaptive experiment configurations
|-- outputs/              Generated experiment artifacts
|-- scripts/              Adaptive run and verification CLI entry points
`-- tests/                Unit, integration, and regression tests for new work
```

## Adaptive command-line entry points

Validate or start the Pilot protocol from the repository root:

```powershell
python experiments/scripts/run_adaptive.py dry-run --config experiments/configs/adaptive_pilot_v2.yaml
python experiments/scripts/run_adaptive.py fresh --config experiments/configs/adaptive_pilot_v2.yaml
python experiments/scripts/run_adaptive.py resume --run-dir outputs/adaptive_pilot_seed2001_4090_v2
```

Verify a completed run and evaluate the Pilot gate:

```powershell
python experiments/scripts/verify_adaptive.py --run-dir outputs/adaptive_pilot_seed2001_4090_v2
```

The resume-equivalence gate additionally requires an independently completed
uninterrupted reference run, supplied with `--resume-reference-run-dir`. The
verifier writes `pilot_gate_summary.json` only for Pilot protocols.

## Frozen protocol lifecycle

The protocol configurations live in `configs/`. A run writes a complete
`resolved_config.yaml` before work begins and records the source configuration
SHA-256, checkpoint SHA-256 values, dataset SHA-256 values, and other important
input hashes in `input_manifest.json`. A configuration never writes its own
hash back into its YAML file, and result values belong only in JSON, JSONL, or
CSV outputs.

Once a `config_id` has started a run it is immutable. A frozen-parameter change
requires a new version such as `*_v2.yaml`; an existing v1 file must not be
overwritten or reused for a different run.

The activation order is:

1. `matched_compute_v1.yaml` defines the Baseline--Adaptive fairness contract.
2. `adaptive_preflight_seed1001_v2.yaml` runs the frozen Scheduler for five
   production iterations, including an interruption/resume boundary and an
   online evaluation at iteration 5.
3. After the production preflight completes, `adaptive_pilot_v2.yaml` and the
   frozen `adaptive_formal_v2.yaml` may start in parallel. Fresh runs require a
   clean worktree and record the actual Git HEAD in their resolved artifacts.
4. `adaptive_holdout_v1.yaml` and `adaptive_fixed_basket_v1.yaml` are frozen
   before the formal run.
5. `h2_v1.yaml`, `h3_v1.yaml`, and `head_to_head_v1.yaml` are frozen before any
   formal result is inspected.

The v1 Adaptive Scheduler protocols and `outputs/adaptive_short` remain immutable
evidence of the original run. Version 2 changes only the observation semantics:
structurally valid all-zero games are recorded as truncated and remain in the
Scheduler length estimate; they are not excluded observations.

## Baseline contract

Every extension run must identify its baseline inputs by content rather than
by filename alone:

- baseline Git commit;
- pretrained checkpoint SHA-256;
- dataset SHA-256;
- baseline resolved configuration SHA-256; and
- fixed network, MCTS, optimizer, and evaluation settings.

Baseline and Adaptive conditions must use the same checkpoint, paired random
seeds, evaluation schedule, and compute-accounting rules. A change to any
frozen setting creates a new experiment ID.

## Provenance

The original codebase is [Victor Baeza's Quoridor AlphaZero Bot][upstream].
The complete provenance chain, component-level licences, and local adaptation
notice are maintained in
[`../baseline/UPSTREAM.md`](../baseline/UPSTREAM.md) and
[`../baseline/THIRD_PARTY_NOTICES.md`](../baseline/THIRD_PARTY_NOTICES.md).

[upstream]: https://gitlab.com/victorbaeza_h/quoridor-alphazero-bot
