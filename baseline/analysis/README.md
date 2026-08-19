# Baseline Analysis Reproduction Manual

This directory contains the offline analyses and fixed-opponent evaluation for the completed baseline reproduction. It does not retrain the model or change the training protocol. This manual fixes the analysis protocol, commands, input and output contracts, and acceptance criteria; it intentionally does not record experimental scores.

All commands below assume that the working directory is `baseline/`:

```powershell
cd I:\dissertation\src\baseline
```

Use the same Python environment as the baseline reproduction. The JavaScript heuristic opponents also require `node` to be available on `PATH`.

## What the analysis covers

The current workflow has four parts:

1. **Training-efficiency and replay-derived metrics.** Derive generation efficiency, GPU-time fractions, replay inflow and consumption, sample exposure, eviction, turnover, and sample-age metrics from the per-iteration training log.
2. **Final replay snapshot analysis.** Analyse the 150-iteration rolling window saved at iteration 210, including canonical-state diversity, duplicate rate, state-effective count, per-iteration game counts and length distributions, and game boundaries recovered from `step` resets.
3. **Fixed-basket evaluation.** Evaluate selected checkpoints against the same four opponents under fixed seeds, side assignments, MCTS settings, temperature schedule, and 150-turn limit. Produce per-game records, checkpoint summaries, opponent-stratified summaries, and provisional Elo ratings.
4. **Frozen policy/value hold-out.** Generate 200 independent checkpoint-0 MCTS self-play games once, preserve complete `(canonical_board, policy_target, value_target, valid_moves, step, game_length)` samples, verify their integrity, and evaluate every registered checkpoint without placing any hold-out state in replay.

The workflow does not make the plateau decision or publication figures automatically. The corresponding registered paths in `baseline_gate2.yaml` define provenance and intended downstream use; their presence does not mean that an analysis has been run.

### Hold-out design correction

Random-greedy games do not contain the MCTS policy targets used by AlphaZero training. Treating their selected actions as one-hot labels would define a different policy loss, so it cannot support a rigorous train–hold-out policy-loss gap. Random-greedy data may still be used for position coverage or out-of-distribution value tests, while the 2,400 fixed-basket games remain exclusively a playing-strength evaluation.

The formal hold-out therefore uses both sides of the shared checkpoint 0 under the baseline MCTS self-play protocol. It records the root visit distribution `pi`, value outcome `z`, and valid-action mask for every state. Baseline and Adaptive must read the exact same frozen dataset. A canonical view means the view of the player whose turn it is: white-to-move states use the white view and black-to-move states use the black view. A state is not duplicated from the non-acting player's view because that view has no corresponding policy target.

## Authoritative configurations

| Configuration | Authority |
|---|---|
| `analysis/configs/baseline_gate2.yaml` | Baseline run ID, replay and state-hash definitions, derived-metric definitions, input provenance, and registered analysis output paths |
| `analysis/configs/fixed_basket_v1.yaml` | The authoritative fixed-basket protocol: checkpoints, opponents, game counts, side schedule, MCTS settings, temperature schedule, and seed |
| `analysis/configs/holdout_v1.yaml` | The authoritative frozen hold-out protocol: checkpoint-0 source hash, model, self-play, storage, evaluation checkpoints, and trajectory-bootstrap settings |
| `analysis/configs/h1_v1.yaml` | The authoritative H1 merge grid, interval aggregation, trend directions, evidence-stage classification, missing-data handling, and output paths |

There is deliberately no fixed-basket protocol section in `baseline_gate2.yaml`.
Fixed-basket protocol settings are read only from `fixed_basket_v1.yaml`; the
`plateau` section in `baseline_gate2.yaml` remains authoritative for the plateau
decision.

The state hash is fixed as:

```text
SHA256(dtype + shape + contiguous canonical-board bytes)
```

No additional symmetry reduction is applied. Replay summaries export neither raw states nor state hashes.

## Input provenance

The default source run is:

```text
outputs/baseline_reproduction_seed1001_4090/
```

| Input | Source | Used by |
|---|---|---|
| `metrics.jsonl` | Per-iteration baseline training records | `derive_baseline_metrics.py`, `summarize_replay.py` |
| `resolved_config.yaml` | Configuration resolved by the baseline run | `derive_baseline_metrics.py`, `summarize_replay.py` |
| `run_metadata.json` | Run ID, model configuration, and checkpoint provenance | `evaluate_fixed_basket.py` |
| `checkpoints/latest.examples` | Rolling replay snapshot saved at iteration 210 | `summarize_replay.py` |
| Initial checkpoint | Pretrained or initial checkpoint path recorded in `run_metadata.json` | Checkpoint 0 in `evaluate_fixed_basket.py` |
| Checkpoints 20–210 | Checkpoint manifest and checkpoint directory in the baseline run | `evaluate_fixed_basket.py` |
| Frozen checkpoint 0 | `outputs/pretraining_reproduction_seed1001/checkpoints/checkpoint_0.pth.tar`, pinned by SHA-256 in `holdout_v1.yaml` | `generate_holdout.py` |
| Per-game hold-out shards and `states.npz` | Generated once from checkpoint 0, never admitted to replay | `verify_holdout.py`, `evaluate_holdout.py` |
| Baseline `metrics.jsonl` | Same-iteration replay-sampled training losses and cumulative GPU hours | `evaluate_holdout.py` gap columns |

Before playing games, the evaluator resolves every checkpoint path and records its SHA-256 in the evaluation manifest. Checkpoint 0 is not assumed to live in the baseline checkpoint directory; its path comes from `run_metadata.json`.

## Does the analysis modify the source results?

The scripts do not modify `metrics.jsonl`, `resolved_config.yaml`, `run_metadata.json`, `latest.examples`, or checkpoint contents.

- By default, `derive_baseline_metrics.py` creates or atomically updates its three derived artifacts under the source run's new `gate2/` subdirectory. It does not overwrite a training artifact.
- Replay and fixed-basket outputs are written under the separate `outputs/baseline_seed1001_4090_analysis/` root.
- Hold-out generation writes only under `outputs/baseline_seed1001_4090_analysis/holdout_v1/`. Neither `Coach.learn()` nor a training entry point imports or reads the hold-out scripts or artifacts.
- `verify_holdout.py` is read-only. `evaluate_holdout.py` reads frozen states and checkpoints, then atomically writes only its metrics CSV files and summary.
- The fixed-basket evaluator appends and flushes one record to its analysis `games.jsonl` immediately after each completed game.
- Fixed-basket `--resume` reads the existing game records and skips completed stable game keys, so an interrupted run does not duplicate games. Hold-out resume uses the separate shard-first contract described below.
- `--verify-source-integrity` compares the source-run file inventory before and after evaluation. Any change fails the evaluation.
- The existing random + greedy sanity evaluation is outside this protocol and is not overwritten.

Re-running a summary script updates that script's own CSV or manifest summary. If formal per-game output already exists, the evaluator requires an explicit `--resume`.

## 1. Derive baseline metrics

Run:

```powershell
python analysis/scripts/derive_baseline_metrics.py `
  --metrics outputs/baseline_reproduction_seed1001_4090/metrics.jsonl `
  --resolved-config outputs/baseline_reproduction_seed1001_4090/resolved_config.yaml `
  --output-dir outputs/baseline_reproduction_seed1001_4090/gate2
```

Outputs:

```text
outputs/baseline_reproduction_seed1001_4090/gate2/
|-- derived_metrics.csv
|-- baseline_resource_summary.json
`-- data_quality_report.json
```

`derived_metrics.csv` contains at least the following fields for every iteration:

- `fresh_states_per_update`
- `states_per_gpu_hour`
- `games_per_gpu_hour`
- `self_play_fraction`
- `training_fraction`
- `buffer_inflow_fraction`
- `buffer_fraction_consumed`
- `mean_sample_exposure`
- `selected_sample_reuse`
- `evicted_states`
- `turnover_fraction`
- `mean_sample_age`
- `median_sample_age`
- `p90_sample_age`

Success requires exit code 0, `data_quality_report.json.status == "passed"`, all 210 iterations passing, all four entries in `checks` being `true`, and every derived value being present and finite. The replay size reconstructed from the most recent 150 iterations of `positions_generated` must exactly equal the recorded `replay_buffer_size` at every iteration.

## 2. Summarise the final replay

The defaults already target the formal baseline run, so the short command is:

```powershell
python analysis/scripts/summarize_replay.py
```

The equivalent explicit command is:

```powershell
python analysis/scripts/summarize_replay.py `
  --replay outputs/baseline_reproduction_seed1001_4090/checkpoints/latest.examples `
  --metrics outputs/baseline_reproduction_seed1001_4090/metrics.jsonl `
  --resolved-config outputs/baseline_reproduction_seed1001_4090/resolved_config.yaml `
  --output-dir outputs/baseline_seed1001_4090_analysis/replay `
  --expected-iteration 210 `
  --expected-history-buckets 150 `
  --expected-total-states 284234
```

Outputs:

```text
outputs/baseline_seed1001_4090_analysis/replay/
|-- replay_iteration_stats.csv
|-- replay_final_summary.json
`-- trajectory_stats.csv
```

A successful run prints at least:

```text
Replay iteration: 210
History buckets: 150
Recovered range: 61-210
Total states: 284234
Empty buckets: 0
Count matches metrics: yes
Output status: completed
```

Every entry in `replay_final_summary.json.validations` must also be `true`. The snapshot retains only iterations 61–210, so it cannot recover buffer-level diversity for iterations 1–60 or the number of times each state was actually sampled during training.

## 3. Fixed-basket preflight and pilot

Resolve the protocol and checkpoints and test JS determinism without playing games:

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode pilot `
  --checkpoints 0 210 `
  --games-per-opponent 2 `
  --prepare-only
```

Run the 16-game pilot and summarise it:

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode pilot `
  --checkpoints 0 210 `
  --games-per-opponent 2 `
  --verify-source-integrity

python analysis/scripts/summarize_fixed_basket.py --mode pilot
```

Pilot outputs:

```text
outputs/baseline_seed1001_4090_analysis/fixed_basket_v1_pilot/
|-- protocol.resolved.yaml
|-- evaluation_manifest.json
|-- games.jsonl
|-- checkpoint_summary.csv
`-- evaluation.log
```

Pilot acceptance requires checkpoints 0 and 210 to load; one model-white and one model-black game against each opponent; clean JS process startup and shutdown; exactly 16 lines in `games.jsonl`; zero `invalid_move` and `bot_error` terminations; temperature 0 from the model's sixth move onward; no change to the source run; and no duplicated games after resuming with `--resume`.

## 4. Run the complete fixed-basket evaluation

First run:

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode formal `
  --verify-source-integrity
```

Resume an interrupted run:

```powershell
python analysis/scripts/evaluate_fixed_basket.py `
  --mode formal `
  --resume `
  --verify-source-integrity
```

The protocol fixes 12 checkpoints, four opponents, and 50 games per checkpoint–opponent pair, for 2,400 games. The model plays white in the first 25 games and black in the final 25 games of every pair. `max_turns` is 150. Each game seed is derived stably from the protocol ID, base seed, checkpoint, opponent ID, and game index.

Use the retry options only after identifying a record that genuinely needs replaying:

```powershell
# Replay every game with the selected error termination.
python analysis/scripts/evaluate_fixed_basket.py --mode formal --resume --retry-termination bot_error

# Replay one stable game key.
python analysis/scripts/evaluate_fixed_basket.py --mode formal --resume --retry-game 210:heuristic_200:2
```

Formal outputs:

```text
outputs/baseline_seed1001_4090_analysis/fixed_basket_v1/
|-- protocol.resolved.yaml
|-- evaluation_manifest.json
|-- games.jsonl
|-- checkpoint_summary.csv
|-- opponent_summary.csv
|-- elo_summary.csv
`-- evaluation.log
```

After evaluation, generate the summaries:

```powershell
python analysis/scripts/summarize_fixed_basket.py --mode formal
```

Formal acceptance requires:

- `evaluation_manifest.json.status == "completed"`;
- complete coverage of 12 checkpoints and four opponents;
- 50 games in each of the 48 checkpoint–opponent groups, split 25/25 by model colour;
- 2,400 unique stable game keys in `games.jsonl`;
- zero `invalid_move`, `bot_error`, and non-null `fault` records;
- games reaching the turn limit may exist, but must use `termination == "max_turns"` and be reported separately;
- every recorded temperature history must use 0.18 for the first five model moves and 0 from the sixth model move onward;
- when source-integrity checking is enabled, `source_integrity.status == "passed"` and `changed_paths` is empty;
- summary output containing `Observed games: 2400`, `Expected games: 2400`, and `Summary status: completed`;
- 12 rows in `checkpoint_summary.csv` and 48 rows in `opponent_summary.csv`;
- score rate equal to `(wins + 0.5 * draws) / games`, with the fixed-basket score equal to the equal-weight mean of the four opponent scores;
- 95% confidence intervals calculated by bootstrap stratified by opponent and model colour; and
- Elo marked `provisional` with a fixed random seed. After Adaptive evaluation, Baseline checkpoints, Adaptive checkpoints, and fixed opponents must be fitted together in one final Elo model.

## 5. Generate, freeze, verify, and evaluate the formal hold-out

The fixed protocol is `analysis/configs/holdout_v1.yaml`. It uses seed 71001, 200 games, checkpoint 0 with its registered SHA-256, 200 MCTS simulations, inference batch 10, `cpuct` 1.25, root Dirichlet noise with alpha 0.15 and epsilon 0.25, and a 150-turn limit. Its `temperature_threshold: 15` follows `Coach.executeEpisode()` exactly: `temp = int(step < 15)`.

Run the four-game pilot before formal generation:

```powershell
python analysis/scripts/generate_holdout.py `
  --config analysis/configs/holdout_v1.yaml `
  --output-dir outputs/baseline_seed1001_4090_analysis/holdout_v1_pilot `
  --games 4 `
  --verify-source-integrity

python analysis/scripts/verify_holdout.py `
  --config analysis/configs/holdout_v1.yaml `
  --output-dir outputs/baseline_seed1001_4090_analysis/holdout_v1_pilot
```

`--games` is a bounded pilot-only runtime override recorded in `protocol.resolved.yaml` and `manifest.json`; it does not alter the frozen `games: 200` protocol. Resume the pilot with the identical arguments plus `--resume`. A completed resume must report that all artifacts match and must leave `games.jsonl`, the shard set, and `states.npz` unchanged.

Start generation:

```powershell
python analysis/scripts/generate_holdout.py --device cuda
```

If generation is interrupted, resume it without duplicating completed games:

```powershell
python analysis/scripts/generate_holdout.py --device cuda --resume
```

Every game receives seed `71001 + game_id`. Before that game, Python, NumPy, Torch CPU, and all Torch CUDA RNGs are seeded. A new empty MCTS is constructed for every game. The generator first atomically commits `shards/game_XXXX.npz`, hashes that shard, and only then appends and synchronises its `games.jsonl` record. Resume validates `game_id`, seed, shard path, shard hash, and state count before skipping a completed game. Once all games are present, the shards are concatenated in `game_id` order into `states.npz`.

Verify the frozen dataset before evaluation or transfer:

```powershell
python analysis/scripts/verify_holdout.py `
  --config analysis/configs/holdout_v1.yaml `
  --dataset outputs/baseline_seed1001_4090_analysis/holdout_v1/states.npz
```

The manifest records both the SHA-256 of the compressed NPZ file and a logical content SHA-256 over the ordered array name, dtype, shape, and contiguous bytes. After a formal manifest reaches `status: completed`, do not regenerate it, manually delete states, select samples based on loss, or add any hold-out sample to Baseline or Adaptive replay. Adaptive consumes the read-only dataset contract in `../experiments/configs/adaptive_holdout_v1.yaml`.

Run all registered baseline checkpoints:

```powershell
python analysis/scripts/evaluate_holdout.py `
  --config analysis/configs/holdout_v1.yaml `
  --dataset outputs/baseline_seed1001_4090_analysis/holdout_v1/states.npz `
  --run-dir outputs/baseline_reproduction_seed1001_4090 `
  --output-dir outputs/baseline_seed1001_4090_analysis/holdout_v1 `
  --verify-dataset-hash
```

The evaluator resolves checkpoint 0 from `run_metadata.json` and checkpoints 20–210 from the existing manifest/directory discovery logic, then records every checkpoint SHA-256. It uses the exact training policy-loss definition: request logits, apply the saved valid-action mask, softmax, and compute cross entropy against the MCTS root target. Value loss is mean squared error. Confidence intervals resample complete game trajectories and aggregate their state-level losses.

Only the final `latest.examples` replay snapshot was retained; a complete replay snapshot for each checkpoint does not exist. Therefore `approx_policy_gap` and `approx_value_gap` subtract the same-iteration online loss logged during training and must be described in code, tables, figures, and dissertation prose as the **approximate online train–hold-out gap**. They are not posterior training losses recomputed on each checkpoint's full replay. Checkpoint 0 has no logged training loss, so its logged-loss and gap cells are intentionally blank.

To evaluate Adaptive later, point `--run-dir` and `--training-metrics` at its run while retaining the same baseline `--holdout-dir`. Use a separate `--output-dir` so the Baseline metrics are not overwritten.

Outputs:

```text
outputs/baseline_seed1001_4090_analysis/holdout_v1/
|-- protocol.resolved.yaml
|-- manifest.json
|-- games.jsonl
|-- shards/
|   |-- game_0000.npz
|   |-- ...
|   `-- game_0199.npz
|-- states.npz
|-- generation.log
|-- checkpoint_metrics.csv
|-- trajectory_checkpoint_metrics.csv
`-- summary.json
```

Each NPZ stores `boards [N,4,17,17] uint8`, `policies [N,action_size] float32`, `values [N] float32`, `valids [N,action_size] uint8`, `game_ids [N] int32`, `steps [N] int16`, and `game_lengths [N] int16`. The scripts always obtain `action_size` from `game.getActionSize()` and validate it dynamically. The present 9x9 implementation returns 136; the protocol does not hard-code the previously estimated value 209.

Generation is accepted only when `manifest.json.status == "completed"`, `games.jsonl` contains exactly 200 unique game IDs and deterministic seeds, all 200 shard hashes match, and `states.npz` is the exact ordered concatenation. Verification must exit zero and print `Output status: passed`. It also requires finite normalised policies supported only on legal moves, value labels in `{-1,0,1}` with the correct acting-player sign, sequential steps, consistent game lengths, and exact dtypes and shapes. Evaluation is accepted when `summary.json.status == "completed"`, all 12 checkpoints have finite losses and trajectory-bootstrap intervals, `checkpoint_metrics.csv` has 12 rows, and `trajectory_checkpoint_metrics.csv` has one row for each checkpoint-game pair.

## Tests

After changing analysis code, run:

```powershell
python -m pytest analysis/tests -q
```

Do not publish or cite regenerated summaries when these tests fail. Fix the affected script first, then regenerate only the affected analysis artifacts.

## Mapping outputs to dissertation figures and tables

The repository does not currently assign final dissertation figure numbers. During writing, use the following source mapping rather than terminal output or manually copied intermediate values.

| Dissertation content | Authoritative source | Intended use |
|---|---|---|
| Training-resource and efficiency table | `gate2/baseline_resource_summary.json` | Total GPU hours, generation/training fractions, and aggregate throughput |
| Per-iteration efficiency curves | `gate2/derived_metrics.csv` | States or games per GPU hour, fresh states per update, replay consumption, and sample exposure |
| Replay-dynamics figure | `gate2/derived_metrics.csv` | Turnover, eviction, and mean/median/p90 sample age over iteration |
| Final replay-diversity table or figure | `replay/replay_iteration_stats.csv`, `replay/replay_final_summary.json` | Incoming unique ratio, duplicate rate, state-effective count, and final unique ratio |
| Self-play trajectory-length distribution | `replay/trajectory_stats.csv` | Per-iteration game counts, game-length distributions, and trajectory-boundary checks |
| Baseline checkpoint learning curve | `fixed_basket_v1/checkpoint_summary.csv` | Fixed-basket score rate and stratified bootstrap confidence intervals |
| Per-opponent comparison | `fixed_basket_v1/opponent_summary.csv` | Checkpoint-by-opponent performance and side balance |
| Elo appendix or diagnostic figure | `fixed_basket_v1/elo_summary.csv` | Provisional baseline-only Elo; the final dissertation Elo must use the joint fit |
| Policy/value generalisation curve | `holdout_v1/checkpoint_metrics.csv` | Frozen checkpoint-0 self-play hold-out losses, approximate online train–hold-out gaps, and game-cluster bootstrap intervals |
| Hold-out trajectory sensitivity | `holdout_v1/trajectory_checkpoint_metrics.csv` | Checkpoint-by-game loss distributions without treating within-game states as independent trajectories |
| Hold-out provenance and integrity | `holdout_v1/protocol.resolved.yaml`, `holdout_v1/manifest.json`, `holdout_v1/summary.json` | Dataset source hash, exact self-play protocol, state schema, checkpoint hashes, and evaluation definitions |
| Methods and reproducibility appendix | `evaluation_manifest.json`, `protocol.resolved.yaml` | Seeds, checkpoint provenance, protocol, source-integrity status, and data-quality notes |
| Per-game audit | `fixed_basket_v1/games.jsonl` | Trace abnormal termination, turn-limit games, colours, and game seeds; not a main result table |

Every dissertation figure or table should retain its source CSV or JSON path, filtering rule, and statistical method. Fixed-basket scores must come from `checkpoint_summary.csv` and must not be mixed with the existing random + greedy sanity evaluation.
