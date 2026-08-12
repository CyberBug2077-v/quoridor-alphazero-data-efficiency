# Upstream provenance and modification notice

## Primary upstream project

- **Project:** Quoridor AlphaZero Bot
- **Author/namespace:** Victor Baeza (`victorbaeza_h`)
- **Repository:** <https://gitlab.com/victorbaeza_h/quoridor-alphazero-bot>
- **Accessed for this record:** 11 August 2026
- **Exact imported commit:** not yet verified

This repository is the declared original source for the inherited Quoridor
AlphaZero code. The exact imported commit must be resolved and recorded before
formal baseline experiments begin. Until then, no local commit should be
presented as an unmodified copy of a specific upstream revision.

Suggested citation:

> Victor Baeza. *Quoridor AlphaZero Bot*. GitLab repository,
> https://gitlab.com/victorbaeza_h/quoridor-alphazero-bot.

## Inherited fork/snapshot context

The current `src/.git/config` points to the following development repository:

<https://github.com/CyberBug2077-v/quoridor-alphazero-data-efficiency.git>

A public short reference `2274923` was supplied during project setup. The
local Git repository currently has no commits, so this reference is recorded
only as provenance context. It has not yet been demonstrated to be the parent
or exact source of the local `baseline/` tree.

Before the baseline is frozen, record all three identifiers separately:

1. the exact Victor Baeza upstream commit, if it can be recovered;
2. the exact inherited fork/snapshot commit, if applicable; and
3. the new local commit containing the adapted baseline.

## Status of this baseline

`baseline/` is an **adapted reproduction baseline**, not a verbatim archive.
It retains the inherited algorithms while allowing the minimum changes needed
for reproducibility, portability, and controlled experiments.

Documented local adaptations currently include:

- Windows/MSVC support for the native PathFinder extension;
- replacement of GCC-only variable-length arrays with standard C++ storage;
- correct Python buffer release in the PathFinder binding;
- Windows and Linux build-argument selection;
- YAML-based smoke-test and baseline configuration;
- CUDA, seed, environment, and resolved-configuration checks; and
- reorganisation of legacy launch scripts for reference.

Additional changes must be added to this list when they are introduced.

## Component boundary

| Component | Provenance status | Permitted baseline changes |
|---|---|---|
| AlphaZero Coach, MCTS, Arena, and network | Inherited | Compatibility fixes, configuration mapping, tests, reproducibility logging |
| Quoridor rules and state/action representation | Inherited | Platform fixes and verified bug fixes only |
| Native PathFinder | Inherited and ported | Cross-platform build and standards-compliance changes |
| JS MCTS heuristic bot | Inherited dependency | Adapter and portability changes only |
| pyquoridor server | Inherited dependency | Packaging and compatibility changes only |
| Baseline smoke/pretraining launchers | New reproduction infrastructure | May be developed in `baseline/scripts/` |
| Replay instrumentation and adaptive scheduling | New dissertation contribution | Must be developed in `../extension/` |

## Reproducibility and attribution rule

Every formal run must reference the adapted baseline's real local commit. The
primary upstream URL and upstream commit remain separate provenance fields;
they must not be substituted for the local code commit.

The original licence files must remain with their components. This notice
describes provenance and modifications; it does not replace those licences.
