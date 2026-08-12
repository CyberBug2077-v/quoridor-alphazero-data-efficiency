# Data-Efficient Quoridor AlphaZero Extension

This directory contains the new research code for the MSc dissertation on
data-efficient post-pretraining self-play for 9x9 Quoridor AlphaZero under
compute constraints.

The extension builds on the adapted and validated system in `../baseline/`.
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
| Replay instrumentation and adaptive scheduling | `extension/` | New dissertation work |
| Matched-compute experiments and analysis | `extension/` | New dissertation work |

The baseline is not permanently read-only during the reproduction phase. It
may receive minimal portability, testing, and reproducibility changes. It is
frozen only after the baseline acceptance tests and pretrained checkpoint have
been completed. Subsequent research changes belong here.

## Planned layout

```text
extension/
|-- configs/              Instrumented and adaptive experiment configurations
|-- outputs/              Generated experiment artifacts
|-- quoridor_project/     New instrumentation and adaptive implementation
|-- scripts/              Extension experiment launchers
`-- tests/                Unit, integration, and regression tests for new work
```

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
