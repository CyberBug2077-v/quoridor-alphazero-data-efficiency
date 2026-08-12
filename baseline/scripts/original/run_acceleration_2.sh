#!/bin/bash
# Continue the 5 standard acceleration bots from acceleration_experiments_v6
# into acceleration_experiments_v6_2.
# All bots continue from their saved best checkpoint + latest examples.
# Heuristic guidance and supervised pretraining are not re-run on continuation.

set -euo pipefail

PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
WORKER="$PROJECT_ROOT/scripts/train_acceleration_helper.sh"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"

RESULTS_ROOT="$ALPHAZERO_DIR/training_runs/acceleration_experiments_v6_2"
SOURCE_ORIG="$ALPHAZERO_DIR/training_runs/acceleration_experiments_v6"

EXTRA_HOURS=20
SBATCH_NODELIST="saxa"

echo "Continuing all 5 bots → acceleration_experiments_v6_2  (${EXTRA_HOURS}h each)"
echo ""

for bot in bot1_baseline bot2_brute bot3_pretrain bot4_heuristic bot5_hybrid; do
    mkdir -p "$RESULTS_ROOT/$bot"
done

# bot1: baseline (200 MCTS, constant LR)
BOT1_JID=$(sbatch --parsable \
    --nodelist=saxa \
    --job-name=qaz_bot1_cont \
    --output="$RESULTS_ROOT/bot1_baseline/slurm-%j.out" \
    --export=ALL,\
EXP_NAME=bot1_baseline,\
MCTS_SIMS=200,\
NUM_EPS=75,\
LR=0.0002,\
LR_DECAY_STEP=0,\
LR_DECAY_FACTOR=0.0,\
HEURISTIC_ALPHA=0.0,\
PRETRAIN_DATA=,\
TRAINING_HOURS=${EXTRA_HOURS},\
RESULTS_ROOT=${RESULTS_ROOT},\
LOAD_CHECKPOINT_FROM=${SOURCE_ORIG}/bot1_baseline \
    "$WORKER")
echo "bot1_baseline  -> job $BOT1_JID"

# bot2: brute (800 MCTS, constant LR)
BOT2_JID=$(sbatch --parsable \
    --nodelist=saxa \
    --job-name=qaz_bot2_cont \
    --output="$RESULTS_ROOT/bot2_brute/slurm-%j.out" \
    --export=ALL,\
EXP_NAME=bot2_brute,\
MCTS_SIMS=800,\
NUM_EPS=30,\
LR=0.0002,\
LR_DECAY_STEP=0,\
LR_DECAY_FACTOR=0.0,\
HEURISTIC_ALPHA=0.0,\
PRETRAIN_DATA=,\
TRAINING_HOURS=${EXTRA_HOURS},\
RESULTS_ROOT=${RESULTS_ROOT},\
LOAD_CHECKPOINT_FROM=${SOURCE_ORIG}/bot2_brute \
    "$WORKER")
echo "bot2_brute     -> job $BOT2_JID"

# bot3: pretrain (already pretrained — pure AZ continuation, constant LR)
BOT3_JID=$(sbatch --parsable \
    --nodelist=saxa \
    --job-name=qaz_bot3_cont \
    --output="$RESULTS_ROOT/bot3_pretrain/slurm-%j.out" \
    --export=ALL,\
EXP_NAME=bot3_pretrain,\
MCTS_SIMS=200,\
NUM_EPS=75,\
LR=0.0002,\
LR_DECAY_STEP=0,\
LR_DECAY_FACTOR=0.0,\
HEURISTIC_ALPHA=0.0,\
PRETRAIN_DATA=,\
TRAINING_HOURS=${EXTRA_HOURS},\
RESULTS_ROOT=${RESULTS_ROOT},\
LOAD_CHECKPOINT_FROM=${SOURCE_ORIG}/bot3_pretrain \
    "$WORKER")
echo "bot3_pretrain  -> job $BOT3_JID"

# bot4: heuristic (pure AZ continuation, constant LR — heuristic already fully decayed)
BOT4_JID=$(sbatch --parsable \
    --nodelist=saxa \
    --job-name=qaz_bot4_cont \
    --output="$RESULTS_ROOT/bot4_heuristic/slurm-%j.out" \
    --export=ALL,\
EXP_NAME=bot4_heuristic,\
MCTS_SIMS=200,\
NUM_EPS=75,\
LR=0.0002,\
LR_DECAY_STEP=0,\
LR_DECAY_FACTOR=0.0,\
HEURISTIC_ALPHA=0.0,\
PRETRAIN_DATA=,\
TRAINING_HOURS=${EXTRA_HOURS},\
RESULTS_ROOT=${RESULTS_ROOT},\
LOAD_CHECKPOINT_FROM=${SOURCE_ORIG}/bot4_heuristic \
    "$WORKER")
echo "bot4_heuristic -> job $BOT4_JID"

# bot5: hybrid (pure AZ continuation, constant LR — heuristic already fully decayed)
BOT5_JID=$(sbatch --parsable \
    --nodelist=saxa \
    --job-name=qaz_bot5_cont \
    --output="$RESULTS_ROOT/bot5_hybrid/slurm-%j.out" \
    --export=ALL,\
EXP_NAME=bot5_hybrid,\
MCTS_SIMS=200,\
NUM_EPS=75,\
LR=0.0002,\
LR_DECAY_STEP=0,\
LR_DECAY_FACTOR=0.0,\
HEURISTIC_ALPHA=0.0,\
PRETRAIN_DATA=,\
TRAINING_HOURS=${EXTRA_HOURS},\
RESULTS_ROOT=${RESULTS_ROOT},\
LOAD_CHECKPOINT_FROM=${SOURCE_ORIG}/bot5_hybrid \
    "$WORKER")
echo "bot5_hybrid    -> job $BOT5_JID"

echo ""
echo "Watch with:  squeue -u \$USER"
echo "Training logs:"
for bot in bot1_baseline bot2_brute bot3_pretrain bot4_heuristic bot5_hybrid; do
    echo "  tail -f $RESULTS_ROOT/$bot/training.log"
done
