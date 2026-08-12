#!/bin/bash
# Resume training of one bot from its latest checkpoint.
# Launched by run_acceleration_2.sh — don't submit directly.
#
# Required env vars:
#   EXP_NAME, MCTS_SIMS, NUM_EPS, HEURISTIC_ALPHA, HEURISTIC_DECAY_ITERS
#   PRETRAIN_DATA, TRAINING_HOURS, RESULTS_ROOT, RESUME_FROM

#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --signal=B:USR1@1800

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

EXP_NAME="${EXP_NAME:-bot3_pretrain}"
MCTS_SIMS="${MCTS_SIMS:-200}"
NUM_EPS="${NUM_EPS:-75}"
HEURISTIC_ALPHA="${HEURISTIC_ALPHA:-0.0}"
HEURISTIC_DECAY_ITERS="${HEURISTIC_DECAY_ITERS:-150}"
PRETRAIN_DATA="${PRETRAIN_DATA:-}"
TRAINING_HOURS="${TRAINING_HOURS:-36}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/s2431521/quoridor-ml-bot/external/alphazero/training_runs/acceleration_2}"
RESUME_FROM="${RESUME_FROM:-}"
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"

export PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"
SCRATCH_DIR="/disk/scratch/$USER"
WORK_DIR="$SCRATCH_DIR/qaz_resume_${EXP_NAME}_${SLURM_JOB_ID}"
RESULTS_DIR="$RESULTS_ROOT/${EXP_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "Resuming    : ${BLUE}${EXP_NAME}${NC}"
echo -e "Resume from : ${BLUE}${RESUME_FROM}${NC}"
echo -e "Results dir : ${BLUE}${RESULTS_DIR}${NC}"
echo -e "Node        : $(hostname)   Job: ${SLURM_JOB_ID}"
echo -e "${GREEN}========================================${NC}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. "$PROJECT_ROOT/venv/bin/activate"

export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR"
rsync -a --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
         --exclude 'logs'  --exclude 'training_runs' \
         "$ALPHAZERO_DIR/" "$WORK_DIR/"

# Restore checkpoints from previous run
CKPT_DIR="$WORK_DIR/temp/${EXP_NAME}"
mkdir -p "$CKPT_DIR"

if [ -n "$RESUME_FROM" ] && [ -d "$RESUME_FROM" ]; then
    echo "Restoring checkpoints from $RESUME_FROM ..."
    for f in "$RESUME_FROM"/checkpoint_*.pth.tar; do
        [ -f "$f" ] && cp "$f" "$CKPT_DIR/" && echo "  copied $(basename $f)"
    done
    [ -f "$RESUME_FROM/best.pth.tar" ] && cp "$RESUME_FROM/best.pth.tar" "$CKPT_DIR/best.pth.tar" && echo "  copied best.pth.tar"
    [ -f "$RESUME_FROM/latest.examples"    ] && cp "$RESUME_FROM/latest.examples"    "$CKPT_DIR/"              && echo "  copied latest.examples"
    echo "Done."
else
    echo "WARNING: RESUME_FROM not set or missing — starting from scratch."
fi

# Don't pass pretrain data when resuming — pretrain already ran in the original job.
# Passing it again would overwrite the loaded checkpoint with a fresh pretrain.
PRETRAIN_FLAG=""

CLEANUP_DONE=0
cleanup() {
    [ "${CLEANUP_DONE}" -eq 1 ] && return
    CLEANUP_DONE=1
    set +e

    echo -e "${YELLOW}>>> Copying results to: ${RESULTS_DIR}${NC}"
    mkdir -p "$RESULTS_DIR"

    TEMP="$WORK_DIR/temp/${EXP_NAME}"

    [ -f "$WORK_DIR/logs/training.log" ] && cp "$WORK_DIR/logs/training.log" "$RESULTS_DIR/" || true
    [ -f "$TEMP/best.pth.tar"          ] && cp "$TEMP/best.pth.tar"          "$RESULTS_DIR/best.pth.tar" || true
    [ -f "$TEMP/latest.examples"       ] && cp "$TEMP/latest.examples"       "$RESULTS_DIR/" || true

    for f in "$TEMP"/checkpoint_*.pth.tar; do [ -f "$f" ] && cp "$f" "$RESULTS_DIR/" || true; done

    [ -d "$WORK_DIR/temp" ] && tar -czf "$RESULTS_DIR/checkpoints.tar.gz" -C "$WORK_DIR" temp 2>/dev/null || true

    cat > "$RESULTS_DIR/config.txt" << CFG
exp_name              = $EXP_NAME
mcts_sims             = $MCTS_SIMS
heuristic_alpha       = $HEURISTIC_ALPHA
heuristic_decay_iters = $HEURISTIC_DECAY_ITERS
resumed_from          = ${RESUME_FROM:-none}
slurm_job_id          = $SLURM_JOB_ID
hostname              = $(hostname)
CFG

    rm -rf "$WORK_DIR" || true
    echo -e "${GREEN}>>> Done.${NC}"
}

trap 'cleanup; exit 0' SIGTERM
trap 'echo "[USR1] Time limit warning."; cleanup; exit 0' SIGUSR1
trap 'cleanup' EXIT

cd "$WORK_DIR"
mkdir -p logs

export USE_AMP="$USE_AMP"
export AMP_DTYPE="$AMP_DTYPE"

TRAINING_SECONDS=$(( TRAINING_HOURS * 3600 ))

timeout "$TRAINING_SECONDS" python3 -u main-accelerate-learning.py \
    --exp_name              "$EXP_NAME"              \
    --mcts_sims             "$MCTS_SIMS"             \
    --num_eps               "$NUM_EPS"               \
    --heuristic_alpha       "$HEURISTIC_ALPHA"       \
    --heuristic_decay_iters "$HEURISTIC_DECAY_ITERS" \
    $PRETRAIN_FLAG                                   \
    2>&1 | tee logs/training.log || true
