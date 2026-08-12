#!/bin/bash
# Evaluate all five acceleration bots and save results to acceleration_2/evaluation/.
# Launched by run_acceleration_2.sh after training jobs finish.

#SBATCH --job-name=qaz_eval_accel2
#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=/home/s2431521/slurm_eval-%j.out

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

export PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"

RESULTS_ROOT="${RESULTS_ROOT:-$ALPHAZERO_DIR/training_runs/acceleration_2}"
ACCEL_DIR="${ACCEL_DIR:-$ALPHAZERO_DIR/training_runs/acceleration_experiments}"
EVAL_SUBDIR="${EVAL_SUBDIR:-evaluation}"

EVAL_OUT="$RESULTS_ROOT/$EVAL_SUBDIR"
mkdir -p "$EVAL_OUT"

echo -e "${GREEN}=== Acceleration 2 Evaluation ===${NC}"
echo "RESULTS_ROOT : $RESULTS_ROOT"
echo "ACCEL_DIR    : $ACCEL_DIR"
echo "EVAL_OUT     : $EVAL_OUT"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. "$PROJECT_ROOT/venv/bin/activate"

export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Build a merged view: bot1/2/4 from original run, bot3/5 from resumed run
SCRATCH_DIR="/disk/scratch/$USER"
MERGED_DIR="$SCRATCH_DIR/accel2_merged_${SLURM_JOB_ID}"
mkdir -p "$MERGED_DIR"

for bot in bot1_baseline bot2_brute bot4_heuristic; do
    [ -d "$ACCEL_DIR/$bot" ] && ln -sf "$ACCEL_DIR/$bot" "$MERGED_DIR/$bot"
done

for bot in bot3_pretrain bot5_hybrid; do
    if [ -d "$RESULTS_ROOT/$bot" ]; then
        ln -sf "$RESULTS_ROOT/$bot" "$MERGED_DIR/$bot"
        echo "Using resumed $bot from $RESULTS_ROOT/$bot"
    elif [ -d "$ACCEL_DIR/$bot" ]; then
        ln -sf "$ACCEL_DIR/$bot" "$MERGED_DIR/$bot"
        echo "WARNING: resumed $bot not found, using $ACCEL_DIR/$bot"
    fi
done

echo -e "${YELLOW}Bot dirs:${NC}"
for bot in bot1_baseline bot2_brute bot3_pretrain bot4_heuristic bot5_hybrid; do
    echo "  $bot -> $(readlink -f $MERGED_DIR/$bot 2>/dev/null || echo MISSING)"
done

echo -e "${GREEN}Starting evaluation...${NC}"

EXPERIMENTS_DIR="$MERGED_DIR" python3 -u "$PROJECT_ROOT/examples/compare_learning_accelerations.py" \
    --games    25        \
    --every-n  20        \
    --out      "$EVAL_OUT" \
    2>&1 | tee "$EVAL_OUT/eval_run.log"

rm -rf "$MERGED_DIR" || true

echo -e "${GREEN}=== Done. Results in $EVAL_OUT ===${NC}"
