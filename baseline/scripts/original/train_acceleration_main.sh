#!/bin/bash
# Submit all 5 acceleration experiments as separate SLURM jobs.
# Usage: bash scripts/train_acceleration_main.sh

set -euo pipefail

PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
WORKER="$PROJECT_ROOT/scripts/train_acceleration_helper.sh"
source "$PROJECT_ROOT/scripts/acceleration_submit_lib.sh"
PRETRAIN_DATA="$PROJECT_ROOT/data/heuristic_games.pkl"
RESULTS_ROOT="$PROJECT_ROOT/external/alphazero/training_runs/acceleration_experiments_v7"
export RESULTS_ROOT
SBATCH_NODELIST="${SBATCH_NODELIST:-saxa}"

STD_HOURS=40
# Bots 3 and 5 spend time pretraining, so get 3h less AZ training
PRETRAIN_HOURS=37

echo "Submitting acceleration experiments..."
echo ""

submit() {
    local name="$1"
    shift
    local out
    out=$(accel_submit_job "$WORKER" "$RESULTS_ROOT" "$name" "qaz_${name}" "$@")
    echo "  $name  ->  $out"
}

# Bot 1 – baseline: 200 MCTS, standard AZ
submit bot1_baseline \
    MCTS_SIMS=200 NUM_EPS=250 LR=0.0002 LR_DECAY_STEP=0 LR_DECAY_FACTOR=0.0 HEURISTIC_ALPHA=0.0 PRETRAIN_DATA= TRAINING_HOURS=${STD_HOURS}

# Bot 2 – brute force: 800 MCTS, fewer episodes to keep iteration time comparable
submit bot2_brute \
    MCTS_SIMS=800 NUM_EPS=100 LR=0.0002 LR_DECAY_STEP=0 LR_DECAY_FACTOR=0.0 HEURISTIC_ALPHA=0.0 PRETRAIN_DATA= TRAINING_HOURS=${STD_HOURS}

# Bot 3 – offline pretrain on JS-bot games, then AZ loop
if [ ! -f "$PRETRAIN_DATA" ]; then
    echo ""
    echo "  WARNING: $PRETRAIN_DATA not found — Bot 3 will skip pretraining."
    PRETRAIN_ARG=""
else
    PRETRAIN_ARG="$PRETRAIN_DATA"
fi

submit bot3_pretrain \
    MCTS_SIMS=200 NUM_EPS=250 LR=0.0002 LR_DECAY_STEP=0 LR_DECAY_FACTOR=0.0 HEURISTIC_ALPHA=0.0 PRETRAIN_DATA=${PRETRAIN_ARG} EXPERT_EXAMPLES_DATA=${PRETRAIN_DATA} FILL_WITH_EXPERT_DATA=1 TRAINING_HOURS=${PRETRAIN_HOURS}

# Bot 4 – online heuristic prior: α decays 0.3→0 over 125 iterations
submit bot4_heuristic \
    MCTS_SIMS=200 NUM_EPS=250 LR=0.0002 LR_DECAY_STEP=0 LR_DECAY_FACTOR=0.0 HEURISTIC_ALPHA=0.3 HEURISTIC_DECAY_ITERS=125 PRETRAIN_DATA= TRAINING_HOURS=${STD_HOURS}

# Bot 5 – hybrid: pretrain + heuristic prior (α decays 0.3→0 over 125 iterations)
submit bot5_hybrid \
    MCTS_SIMS=200 NUM_EPS=250 LR=0.0002 LR_DECAY_STEP=0 LR_DECAY_FACTOR=0.0 HEURISTIC_ALPHA=0.3 HEURISTIC_DECAY_ITERS=125 PRETRAIN_DATA=${PRETRAIN_ARG} EXPERT_EXAMPLES_DATA=${PRETRAIN_DATA} FILL_WITH_EXPERT_DATA=1 TRAINING_HOURS=${PRETRAIN_HOURS}

echo ""
echo "All jobs submitted."
echo "Results: ${RESULTS_ROOT}/"
echo ""
echo "Monitor:  squeue -u \$USER"
