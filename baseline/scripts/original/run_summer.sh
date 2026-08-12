#!/bin/bash
# Summer experiments: architecture + efficiency options on Bot 1 (no pretrain).
# Four SLURM jobs, one per configuration:
#
#   bot1_baseline   defaults: attn_depth 1, SE off
#   bot1_attn2      attn_depth=2
#   bot1_se         se_enabled=1
#   bot1_fast       baseline net + GPU speedups (cudnn.benchmark, channels_last, torch.compile)
#
# Fairness:
#   * All four stop on a wall-clock budget enforced by the shell `timeout`
#     (TRAINING_HOURS) in the worker, compared at equal time, not equal iterations.
#   * Same arena/eval settings, same MCTS_SIMS, same NUM_EPS, same checkpoint schedule
#     (the normal schedule in Coach.learn) for every run.
#   * A fixed seed shared across all configs for comparability.
#
# Each run copies its results/checkpoints/config back to RESULTS_ROOT via the worker's
# cleanup() (same mechanism as run_acceleration_more_self_play.sh): best.pth.tar,
# latest.examples, all checkpoint_*.pth.tar, a checkpoints.tar.gz backup, config.txt,
# and the authoritative run_config.json.
#
# Usage: bash scripts/run_summer.sh
#   DRY_RUN=1 bash scripts/run_summer.sh   # submit tiny --dry_run smoke jobs instead

set -euo pipefail

PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
WORKER="$PROJECT_ROOT/scripts/train_acceleration_helper.sh"
source "$PROJECT_ROOT/scripts/acceleration_submit_lib.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/external/alphazero/training_runs/summer_experiments}"
export RESULTS_ROOT

# Pin saxa (fast H200 node)
SBATCH_NODELIST="${SBATCH_NODELIST:-saxa}"
SBATCH_GRES="${SBATCH_GRES:-gpu:h200_1g.18gb:1}"
export SBATCH_NODELIST SBATCH_GRES

# Mixed precision: BF16 autocast
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

# Equal wall-clock comparison. The batch budget is ~48h; reuse that here.
# TRAINING_HOURS is the shell `timeout` that stops each run.
STD_HOURS="${STD_HOURS:-47}"

# Shared self-play / eval settings — identical across all runs.
MCTS_SIMS="${MCTS_SIMS:-200}"
LR="${LR:-0.0002}"

# Replay buffer + self-play volume — identical across all runs. A 40-iteration window
# with 250 eps gives ~300k positions so the 200k cap keeps a fresh-weighted subsample.
# Game cap is 100 moves: a longer Quoridor game is almost always a non-decisive stall.
TRAIN_HISTORY_ITERS="${TRAIN_HISTORY_ITERS:-40}"
MAX_GAME_LENGTH="${MAX_GAME_LENGTH:-100}"
NUM_EPS="${NUM_EPS:-250}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-200000}"

# One shared seed across ALL configs, so the only difference between e.g. bot1_baseline
# and bot1_attn2 is the config itself, not seed variance. (Trajectories still diverge once
# arch changes consume RNG differently, but this removes one
# avoidable source of difference. Override SEED= to sweep seeds for a variance estimate.)
SEED="${SEED:-1001}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    # Tiny smoke jobs: a few iterations end-to-end, no real budget.
    DRY_ARGS="DRY_RUN=1"
    STD_HOURS=1
    echo ">>> DRY_RUN mode: submitting smoke jobs (a few iterations each)."
else
    DRY_ARGS=""
fi

echo "========================================================"
echo "Summer experiments (Bot 1 only — 4 jobs)"
echo "Results root : $RESULTS_ROOT"
echo "Node request : ${SBATCH_NODELIST:-auto}   GPU: ${SBATCH_GRES:-auto}"
echo "Mixed prec.  : USE_AMP=$USE_AMP  AMP_DTYPE=$AMP_DTYPE"
echo "MCTS sims    : $MCTS_SIMS   LR: $LR   Seed (shared): $SEED"
echo "Self-play    : num_eps ${NUM_EPS} | train cap ${MAX_TRAIN_SIZE}"
echo "Replay buffer: window ${TRAIN_HISTORY_ITERS} iters | max game len ${MAX_GAME_LENGTH}"
echo "Budget       : timeout=${STD_HOURS}h (shell timeout)"
echo "========================================================"
echo ""

submit() {
    local name="$1"; shift
    local out
    out=$(accel_submit_job "$WORKER" "$RESULTS_ROOT" "$name" "qaz_${name}" "$@")
    echo "  $name  ->  $out"
}

# Shared args common to every run (equal eval/budget settings).
common_args() {
    echo "MCTS_SIMS=${MCTS_SIMS}" \
         "LR=${LR}" \
         "LR_DECAY_STEP=0" \
         "LR_DECAY_FACTOR=0.0" \
         "HEURISTIC_ALPHA=0.0" \
         "TRAIN_HISTORY_ITERS=${TRAIN_HISTORY_ITERS}" \
         "MAX_GAME_LENGTH=${MAX_GAME_LENGTH}" \
         "USE_AMP=${USE_AMP}" \
         "AMP_DTYPE=${AMP_DTYPE}" \
         ${DRY_ARGS}
}

# Per-config knobs. baseline => all defaults (off). All configs share NUM_EPS /
# MAX_TRAIN_SIZE so the only difference is the variable under test. 'fast' uses the
# baseline NETWORK plus GPU efficiency opts, so bot1_baseline vs bot1_fast isolates the
# speedup (same net, same hardware slice, same seed) — an efficiency A/B, not accuracy.
config_args() {
    case "$1" in
        baseline) echo "ATTN_DEPTH=1 SE_ENABLED=0 FAST_OPTS=0 NUM_EPS=${NUM_EPS} MAX_TRAIN_SIZE=${MAX_TRAIN_SIZE}" ;;
        attn2)    echo "ATTN_DEPTH=2 SE_ENABLED=0 FAST_OPTS=0 NUM_EPS=${NUM_EPS} MAX_TRAIN_SIZE=${MAX_TRAIN_SIZE}" ;;
        se)       echo "ATTN_DEPTH=1 SE_ENABLED=1 FAST_OPTS=0 NUM_EPS=${NUM_EPS} MAX_TRAIN_SIZE=${MAX_TRAIN_SIZE}" ;;
        fast)     echo "ATTN_DEPTH=1 SE_ENABLED=0 FAST_OPTS=1 NUM_EPS=${NUM_EPS} MAX_TRAIN_SIZE=${MAX_TRAIN_SIZE}" ;;
        *) echo "ERROR: unknown config $1" >&2; exit 1 ;;
    esac
}

for config in baseline attn2 se fast; do
    # Bot 1: fresh start, no pretrain, no expert fill.
    submit "bot1_${config}" \
        $(common_args) \
        $(config_args "$config") \
        SEED="${SEED}" \
        PRETRAIN_DATA= \
        TRAINING_HOURS="${STD_HOURS}"
done

echo ""
echo "All four summer jobs submitted."
echo "Monitor: squeue -u \$USER"
echo "Results: ${RESULTS_ROOT}/"
echo ""
echo "Head-to-head later: every run saves checkpoint_*.pth.tar on the normal schedule,"
echo "so any variant can be matched against its baseline at matching iterations."
