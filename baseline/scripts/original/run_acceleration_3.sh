#!/bin/bash
# Submit all 4 Bot 1 learning-rate variants as separate SLURM jobs.
# All variants are seeded from the shared lr_v2_seed directory (checkpoint 100).
# Usage: bash scripts/run_acceleration_3.sh

set -euo pipefail

PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
WORKER="$PROJECT_ROOT/scripts/train_acceleration_helper.sh"
source "$PROJECT_ROOT/scripts/acceleration_submit_lib.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/external/alphazero/training_runs/acceleration_experiments_lr_v2}"
TRAINING_HOURS="${TRAINING_HOURS:-25}"
SBATCH_NODELIST="${SBATCH_NODELIST:-saxa}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1g.18gb:1}"

# Shared seed: checkpoint_100 + trimmed examples (first 100 iters)
# Run scripts/prepare_lr_v2_seed.py first to create this directory.
SEED_DIR="$PROJECT_ROOT/external/alphazero/training_runs/lr_v2_seed"

submit() {
    local exp_name="$1"
    shift
    local out
    out=$(accel_submit_job "$WORKER" "$RESULTS_ROOT" "$exp_name" "qaz_${exp_name}" "$@")
    echo "  $exp_name -> $out"
}

echo "Submitting Bot 1 LR schedule experiments (lr_v2)..."
echo "Results root: $RESULTS_ROOT"
echo "Seed dir:     $SEED_DIR"
echo "Training time per job: ${TRAINING_HOURS}h"
echo "Node request: ${SBATCH_NODELIST}"
echo "GPU request:  ${SBATCH_GRES}"
echo ""

if [ ! -d "$SEED_DIR" ]; then
    echo "ERROR: Seed directory not found: $SEED_DIR"
    echo "Run scripts/prepare_lr_v2_seed.py first."
    exit 1
fi

# Variant 1: constant LR — no decay
submit bot1_lr_constant \
    MCTS_SIMS=200 \
    NUM_EPS=75 \
    LR=0.0002 \
    LR_DECAY_STEP=0 \
    LR_DECAY_FACTOR=0.0 \
    HEURISTIC_ALPHA=0.0 \
    PRETRAIN_DATA= \
    LOAD_CHECKPOINT_FROM="$SEED_DIR" \
    TRAINING_HOURS=${TRAINING_HOURS}

# Variant 2: single drop at iter 150 -> 0.00002 (divide by 10)
submit bot1_lr_drop150 \
    MCTS_SIMS=200 \
    NUM_EPS=75 \
    LR=0.0002 \
    LR_DECAY_STEP=150 \
    LR_DECAY_FACTOR=0.1 \
    LR_MAX_DECAYS=1 \
    HEURISTIC_ALPHA=0.0 \
    PRETRAIN_DATA= \
    LOAD_CHECKPOINT_FROM="$SEED_DIR" \
    TRAINING_HOURS=${TRAINING_HOURS}

# Variant 3: halve every 100 iterations
submit bot1_lr_halve100 \
    MCTS_SIMS=200 \
    NUM_EPS=75 \
    LR=0.0002 \
    LR_DECAY_STEP=100 \
    LR_DECAY_FACTOR=0.5 \
    HEURISTIC_ALPHA=0.0 \
    PRETRAIN_DATA= \
    LOAD_CHECKPOINT_FROM="$SEED_DIR" \
    TRAINING_HOURS=${TRAINING_HOURS}

# Variant 4: halve every 200 iterations
submit bot1_lr_halve200 \
    MCTS_SIMS=200 \
    NUM_EPS=75 \
    LR=0.0002 \
    LR_DECAY_STEP=200 \
    LR_DECAY_FACTOR=0.5 \
    HEURISTIC_ALPHA=0.0 \
    PRETRAIN_DATA= \
    LOAD_CHECKPOINT_FROM="$SEED_DIR" \
    TRAINING_HOURS=${TRAINING_HOURS}

echo ""
echo "All LR jobs submitted."
echo "Monitor: squeue -u \$USER"
echo "Results: ${RESULTS_ROOT}/"
