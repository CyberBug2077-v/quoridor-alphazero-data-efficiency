#!/bin/bash
# Worker script for one acceleration experiment.
# Launched by train_acceleration_main.sh — do not submit directly.
#
# Env vars expected:
#   EXP_NAME, MCTS_SIMS, HEURISTIC_ALPHA, HEURISTIC_DECAY_ITERS,
#   PRETRAIN_DATA, TRAINING_HOURS
# Optional:
#   LOAD_CHECKPOINT_FROM  — path to a results dir containing best.pth.tar
#                           and latest.examples to resume from
#   LOAD_CHECKPOINT_FILE  — checkpoint filename inside LOAD_CHECKPOINT_FROM
#                           to seed as best.pth.tar (default: best.pth.tar)
#   COPY_LATEST_EXAMPLES  — set to 0 to start with fresh self-play history
#   RESULTS_ROOT          — override the default output root directory
#   LR, LR_DECAY_STEP, LR_DECAY_FACTOR, LR_MAX_DECAYS
#   EXPERT_EXAMPLES_DATA, FILL_WITH_EXPERT_DATA

#SBATCH --job-name=qaz_exp
#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --signal=B:USR1@1800

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

EXP_NAME="${EXP_NAME:-bot1_baseline}"
MCTS_SIMS="${MCTS_SIMS:-200}"
NUM_EPS="${NUM_EPS:-75}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
LR="${LR:-0.0002}"
LR_DECAY_STEP="${LR_DECAY_STEP:-0}"
LR_DECAY_FACTOR="${LR_DECAY_FACTOR:-0.0}"
LR_MAX_DECAYS="${LR_MAX_DECAYS:-}"
HEURISTIC_ALPHA="${HEURISTIC_ALPHA:-0.0}"
HEURISTIC_DECAY_ITERS="${HEURISTIC_DECAY_ITERS:-150}"
PRETRAIN_DATA="${PRETRAIN_DATA:-}"
TRAINING_HOURS="${TRAINING_HOURS:-36}"
LOAD_CHECKPOINT_FROM="${LOAD_CHECKPOINT_FROM:-}"
LOAD_CHECKPOINT_FILE="${LOAD_CHECKPOINT_FILE:-best.pth.tar}"
COPY_LATEST_EXAMPLES="${COPY_LATEST_EXAMPLES:-1}"
EXPERT_EXAMPLES_DATA="${EXPERT_EXAMPLES_DATA:-}"
FILL_WITH_EXPERT_DATA="${FILL_WITH_EXPERT_DATA:-0}"
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"

# --- Summer-experiment knobs (architecture options + fairness controls) ---
SEED="${SEED:-}"
ATTN_DEPTH="${ATTN_DEPTH:-1}"
SE_ENABLED="${SE_ENABLED:-0}"
FAST_OPTS="${FAST_OPTS:-0}"
TRAIN_HISTORY_ITERS="${TRAIN_HISTORY_ITERS:-}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-}"
MAX_GAME_LENGTH="${MAX_GAME_LENGTH:-}"

export PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"

choose_scratch_root() {
    local candidate

    if [ -n "${SCRATCH_DIR:-}" ]; then
        candidate="${SCRATCH_DIR%/}"
        if mkdir -p "$candidate" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    fi

    candidate="/disk/scratch/$USER"
    if mkdir -p "$candidate" 2>/dev/null; then
        echo "$candidate"
        return 0
    fi

    if [ -n "${TMPDIR:-}" ]; then
        candidate="${TMPDIR%/}"
        if mkdir -p "$candidate" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    fi

    candidate="$PROJECT_ROOT/.scratch/$USER"
    mkdir -p "$candidate"
    echo "$candidate"
}

SCRATCH_ROOT="$(choose_scratch_root)"
WORK_DIR="$SCRATCH_ROOT/qaz_${EXP_NAME}_${SLURM_JOB_ID}"

# All experiments save into a shared folder so results are easy to compare
RESULTS_ROOT="${RESULTS_ROOT:-$ALPHAZERO_DIR/training_runs/acceleration_experiments}"
RESULTS_DIR="$RESULTS_ROOT/${EXP_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "Experiment  : ${BLUE}${EXP_NAME}${NC}"
echo -e "MCTS sims   : ${BLUE}${MCTS_SIMS}${NC}"
echo -e "LR base     : ${BLUE}${LR}${NC}"
echo -e "LR decay    : ${BLUE}${LR_DECAY_STEP:-none}${NC} iters x ${BLUE}${LR_DECAY_FACTOR}${NC}  max drops: ${BLUE}${LR_MAX_DECAYS:-unlimited}${NC}"
echo -e "Heuristic α : ${BLUE}${HEURISTIC_ALPHA}${NC}  (decay over ${HEURISTIC_DECAY_ITERS} iters)"
echo -e "Pretrain    : ${BLUE}${PRETRAIN_DATA:-none}${NC}"
echo -e "Expert fill : ${BLUE}${EXPERT_EXAMPLES_DATA:-none}${NC}  enabled: ${BLUE}${FILL_WITH_EXPERT_DATA}${NC}"
echo -e "Resume from : ${BLUE}${LOAD_CHECKPOINT_FROM:-none}${NC}"
echo -e "Resume ckpt : ${BLUE}${LOAD_CHECKPOINT_FILE}${NC}  copy examples: ${BLUE}${COPY_LATEST_EXAMPLES}${NC}"
echo -e "Seed        : ${BLUE}${SEED:-none}${NC}"
echo -e "Arch        : attn_depth=${BLUE}${ATTN_DEPTH}${NC}  se_enabled=${BLUE}${SE_ENABLED}${NC}  fast_opts=${BLUE}${FAST_OPTS}${NC}"
echo -e "Node        : $(hostname)   Job: ${SLURM_JOB_ID}"
echo -e "Scratch dir : ${BLUE}${SCRATCH_ROOT}${NC}"
echo -e "Work dir    : ${WORK_DIR}"
echo -e "Results dir : ${RESULTS_DIR}"
echo -e "${GREEN}========================================${NC}"

# --- Environment ---
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. "$PROJECT_ROOT/venv/bin/activate"

export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# --- Copy to scratch ---
mkdir -p "$WORK_DIR"
rsync -a --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
         --exclude 'logs'  --exclude 'training_runs' \
         "$ALPHAZERO_DIR/" "$WORK_DIR/"

# --- Seed checkpoint from a previous run (for resuming) ---
if [ -n "$LOAD_CHECKPOINT_FROM" ] && [ -d "$LOAD_CHECKPOINT_FROM" ]; then
    SEED_DIR="$LOAD_CHECKPOINT_FROM"
    DEST_TEMP="$WORK_DIR/temp/${EXP_NAME}"
    mkdir -p "$DEST_TEMP"
    echo -e "${YELLOW}>>> Seeding checkpoint from: ${SEED_DIR}${NC}"
    if [ -f "$SEED_DIR/$LOAD_CHECKPOINT_FILE" ]; then
        cp "$SEED_DIR/$LOAD_CHECKPOINT_FILE" "$DEST_TEMP/best.pth.tar"
        echo "  Copied $LOAD_CHECKPOINT_FILE -> best.pth.tar"
        if [ "$LOAD_CHECKPOINT_FILE" != "best.pth.tar" ]; then
            cp "$SEED_DIR/$LOAD_CHECKPOINT_FILE" "$DEST_TEMP/$LOAD_CHECKPOINT_FILE"
            echo "  Preserved $LOAD_CHECKPOINT_FILE"
        fi
    else
        echo "  WARNING: checkpoint file not found: $SEED_DIR/$LOAD_CHECKPOINT_FILE"
    fi
    if [ "$COPY_LATEST_EXAMPLES" = "1" ] && [ -f "$SEED_DIR/latest.examples" ]; then
        cp "$SEED_DIR/latest.examples" "$DEST_TEMP/"
        echo "  Copied latest.examples"
    fi
    # Also copy all numbered checkpoints so training history is visible
    for f in "$SEED_DIR"/checkpoint_*.pth.tar; do [ -f "$f" ] && cp "$f" "$DEST_TEMP/" || true; done
fi

PRETRAIN_ARGS=()
if [ -n "$PRETRAIN_DATA" ] && [ -f "$PRETRAIN_DATA" ]; then
    cp "$PRETRAIN_DATA" "$WORK_DIR/heuristic_games.pkl"
    PRETRAIN_ARGS=(--load_pretrain_data "$WORK_DIR/heuristic_games.pkl")
fi

EXPERT_ARGS=()
if [ -n "$EXPERT_EXAMPLES_DATA" ] && [ -f "$EXPERT_EXAMPLES_DATA" ]; then
    cp "$EXPERT_EXAMPLES_DATA" "$WORK_DIR/expert_examples.pkl"
    EXPERT_ARGS=(--expert_examples_data "$WORK_DIR/expert_examples.pkl" --fill_with_expert_data "$FILL_WITH_EXPERT_DATA")
fi

PY_ARGS=(
    --exp_name "$EXP_NAME"
    --mcts_sims "$MCTS_SIMS"
    --num_eps "$NUM_EPS"
    --lr "$LR"
    --heuristic_alpha "$HEURISTIC_ALPHA"
    --heuristic_decay_iters "$HEURISTIC_DECAY_ITERS"
    --pretrain_epochs "$PRETRAIN_EPOCHS"
)

if [ -n "$LR_DECAY_STEP" ]; then
    PY_ARGS+=(--lr_decay_step "$LR_DECAY_STEP" --lr_decay_factor "$LR_DECAY_FACTOR")
fi

if [ -n "$LR_MAX_DECAYS" ]; then
    PY_ARGS+=(--lr_max_decays "$LR_MAX_DECAYS")
fi

# Reproducibility (wall-clock budget is enforced by the shell `timeout` below)
if [ -n "$SEED" ]; then
    PY_ARGS+=(--seed "$SEED")
fi

# Architecture + efficiency options (defaults reproduce baseline)
PY_ARGS+=(--attn_depth "$ATTN_DEPTH")
PY_ARGS+=(--se_enabled "$SE_ENABLED")
PY_ARGS+=(--fast_opts "$FAST_OPTS")

# Replay-buffer sizing (optional; main.py defaults apply when unset)
if [ -n "$TRAIN_HISTORY_ITERS" ]; then
    PY_ARGS+=(--train_history_iters "$TRAIN_HISTORY_ITERS")
fi
if [ -n "$MAX_TRAIN_SIZE" ]; then
    PY_ARGS+=(--max_train_size "$MAX_TRAIN_SIZE")
fi
if [ -n "$MAX_GAME_LENGTH" ]; then
    PY_ARGS+=(--max_game_length "$MAX_GAME_LENGTH")
fi

PY_ARGS+=("${PRETRAIN_ARGS[@]}")
PY_ARGS+=("${EXPERT_ARGS[@]}")

# --- Cleanup / result copy ---
CLEANUP_DONE=0
cleanup() {
    [ "${CLEANUP_DONE}" -eq 1 ] && return
    CLEANUP_DONE=1
    set +e

    echo -e "${YELLOW}>>> Copying results to NFS: ${RESULTS_DIR}${NC}"
    mkdir -p "$RESULTS_DIR"

    TEMP="$WORK_DIR/temp/${EXP_NAME}"

    [ -f "$WORK_DIR/logs/training.log" ] && cp "$WORK_DIR/logs/training.log" "$RESULTS_DIR/" || true
    [ -f "$TEMP/best.pth.tar" ]          && cp "$TEMP/best.pth.tar"          "$RESULTS_DIR/best.pth.tar" || true
    [ -f "$TEMP/latest.examples" ]       && cp "$TEMP/latest.examples"       "$RESULTS_DIR/" || true
    # Authoritative, fully-resolved run configuration written by main.py.
    [ -f "$TEMP/run_config.json" ]       && cp "$TEMP/run_config.json"       "$RESULTS_DIR/" || true

    for f in "$TEMP"/checkpoint_*.pth.tar; do [ -f "$f" ] && cp "$f" "$RESULTS_DIR/" || true; done

    [ -d "$WORK_DIR/temp" ] && tar -czf "$RESULTS_DIR/checkpoints.tar.gz" -C "$WORK_DIR" temp 2>/dev/null || true

    # Save a record of what was run
    cat > "$RESULTS_DIR/config.txt" << CFG
exp_name              = $EXP_NAME
mcts_sims             = $MCTS_SIMS
num_eps               = $NUM_EPS
lr                    = $LR
lr_decay_step         = ${LR_DECAY_STEP:-none}
lr_decay_factor       = $LR_DECAY_FACTOR
lr_max_decays         = ${LR_MAX_DECAYS:-none}
heuristic_alpha       = $HEURISTIC_ALPHA
heuristic_decay_iters = $HEURISTIC_DECAY_ITERS
pretrain_data         = ${PRETRAIN_DATA:-none}
expert_examples_data  = ${EXPERT_EXAMPLES_DATA:-none}
fill_with_expert_data = ${FILL_WITH_EXPERT_DATA}
load_checkpoint_from  = ${LOAD_CHECKPOINT_FROM:-none}
load_checkpoint_file  = ${LOAD_CHECKPOINT_FILE}
copy_latest_examples  = ${COPY_LATEST_EXAMPLES}
seed                  = ${SEED:-none}
attn_depth            = ${ATTN_DEPTH}
se_enabled            = ${SE_ENABLED}
fast_opts             = ${FAST_OPTS}
train_history_iters   = ${TRAIN_HISTORY_ITERS:-default}
max_train_size        = ${MAX_TRAIN_SIZE:-default}
max_game_length       = ${MAX_GAME_LENGTH:-default}
slurm_job_id          = $SLURM_JOB_ID
hostname              = $(hostname)
CFG

    rm -rf "$WORK_DIR" || true
    echo -e "${GREEN}>>> Done.${NC}"
}

trap 'cleanup; exit 0' SIGTERM
trap 'echo "[USR1] Time limit warning."; cleanup; exit 0' SIGUSR1
trap 'cleanup' EXIT

# --- Train ---
cd "$WORK_DIR"
mkdir -p logs

export USE_AMP="$USE_AMP"
export AMP_DTYPE="$AMP_DTYPE"

TRAINING_SECONDS=$(( TRAINING_HOURS * 3600 ))

timeout "$TRAINING_SECONDS" python3 -u main-accelerate-learning.py \
    "${PY_ARGS[@]}" \
    2>&1 | tee logs/training.log || true
