#!/bin/bash
# Launch bot3 from its pretrained checkpoint_0 with more self-play per iteration.
# Seeds weights from bot3_pretrain/checkpoint_0.pth.tar, starts with fresh self-play
# history, and tops up early training with expert examples until the self-play
# buffer is large enough on its own.
# Usage: bash scripts/run_acceleration_more_self_play.sh

set -euo pipefail

PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
WORKER="$PROJECT_ROOT/scripts/train_acceleration_helper.sh"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"
source "$PROJECT_ROOT/scripts/acceleration_submit_lib.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$ALPHAZERO_DIR/training_runs/acceleration_experiments_v6}"
SOURCE_ROOT="${SOURCE_ROOT:-$ALPHAZERO_DIR/training_runs/acceleration_experiments_v5}"
SOURCE_BOT="${SOURCE_BOT:-$SOURCE_ROOT/bot5_hybrid}"
EXPERT_DATA="${EXPERT_DATA:-$PROJECT_ROOT/data/heuristic_games.pkl}"
EXP_NAME="${EXP_NAME:-bot5_2_smooth}"

# Start from the pretrained bot5 snapshot with smooth buffer + heuristic prior.
TRAINING_HOURS="${TRAINING_HOURS:-26}"
SBATCH_NODELIST="${SBATCH_NODELIST:-saxa}"
FILL_WITH_EXPERT_DATA="${FILL_WITH_EXPERT_DATA:-1}"

if [ ! -f "$EXPERT_DATA" ]; then
    echo "ERROR: Expert data not found: $EXPERT_DATA"
    exit 1
fi

echo "Starting bot5 from pretrained checkpoint in $SOURCE_BOT"
echo "Results root: $RESULTS_ROOT"
echo "Experiment:   $EXP_NAME"
echo "Node request: $SBATCH_NODELIST"
echo "Self-play:    250 games/iteration"
echo "Expert fill:  $EXPERT_DATA"
echo "Fill enabled: ${FILL_WITH_EXPERT_DATA} (fills to max_train_size)"
echo "Training:     ${TRAINING_HOURS}h"
echo ""

out=$(accel_submit_job \
    "$WORKER" \
    "$RESULTS_ROOT" \
    "$EXP_NAME" \
    "qaz_bot5_2_smooth" \
    MCTS_SIMS=200 \
    NUM_EPS=250 \
    LR=0.0002 \
    LR_DECAY_STEP=0 \
    LR_DECAY_FACTOR=0.0 \
    HEURISTIC_ALPHA=0.3 \
    HEURISTIC_DECAY_ITERS=125 \
    PRETRAIN_DATA= \
    EXPERT_EXAMPLES_DATA="${EXPERT_DATA}" \
    FILL_WITH_EXPERT_DATA="${FILL_WITH_EXPERT_DATA}" \
    TRAINING_HOURS="${TRAINING_HOURS}" \
    LOAD_CHECKPOINT_FROM="${SOURCE_BOT}" \
    LOAD_CHECKPOINT_FILE=checkpoint_0.pth.tar \
    COPY_LATEST_EXAMPLES=0)

echo "bot5_2_smooth -> $out"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    $RESULTS_ROOT/$EXP_NAME/slurm-<jobid>.out"
