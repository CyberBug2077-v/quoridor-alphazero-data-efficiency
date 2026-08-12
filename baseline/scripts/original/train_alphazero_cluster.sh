#!/bin/bash

################################################################################
# AlphaZero Training
# Submit with: sbatch scripts/train_alphazero.sh
################################################################################

#SBATCH --job-name=quoridor_training
#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:USR1@3600

# --- CONFIGURATION ---
# Training name: used for organizing different model experiments
TRAINING_NAME="no_gating_fresh_2"
# Leave empty "" to start fresh, or specify job ID to resume from
RESUME_FROM_JOB_ID="2200151"
# Override the resume dir name if it differs from ${TRAINING_NAME}_job_${RESUME_FROM_JOB_ID}
RESUME_FROM_DIR_OVERRIDE="no_gating_fresh_job_2200151"
# AMP settings (can be overridden via env)
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- PATHS ---
PROJECT_ROOT="/home/s2431521/quoridor-ml-bot"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"

# Scratch space (local to compute node)
SCRATCH_DIR="/disk/scratch/$USER"
WORK_DIR="$SCRATCH_DIR/alphazero_${TRAINING_NAME}_$SLURM_JOB_ID"

# Persistent storage (home / NFS)
RESULTS_DIR="$ALPHAZERO_DIR/training_runs/${TRAINING_NAME}_job_$SLURM_JOB_ID"

echo -e "${GREEN}[1/7] Initializing...${NC}"
echo -e "Training Name: ${BLUE}$TRAINING_NAME${NC}"
echo -e "Node: $(hostname)"
echo -e "Job ID: $SLURM_JOB_ID"
echo -e "Work Dir: $WORK_DIR"
echo -e "Save Dir: $RESULTS_DIR"

# --- ENVIRONMENT ---
echo -e "${GREEN}[2/7] Environment setup...${NC}"
. /home/htang2/toolchain-20251006/toolchain.rc
. "$PROJECT_ROOT/venv/bin/activate"

# Critical torch / threading tweaks
export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- COPY TO SCRATCH ---
echo -e "${GREEN}[3/7] Copying project to scratch...${NC}"
mkdir -p "$WORK_DIR"
rsync -a --exclude 'venv' --exclude '.git' --exclude '__pycache__' --exclude 'logs' "$ALPHAZERO_DIR/" "$WORK_DIR/"

# Save the exact training config used for this job
mkdir -p "$RESULTS_DIR"
if [ -f "$WORK_DIR/main-cluster.py" ]; then
    cp "$WORK_DIR/main-cluster.py" "$RESULTS_DIR/main-cluster.py" || true
fi

# --- RESUME LOGIC ---
if [ ! -z "$RESUME_FROM_JOB_ID" ]; then
    echo -e "${YELLOW}[RESUME] Looking for checkpoints from Job $RESUME_FROM_JOB_ID (training: $TRAINING_NAME)...${NC}"
    if [ -n "${RESUME_FROM_DIR_OVERRIDE:-}" ]; then
        PREV_DIR="$ALPHAZERO_DIR/training_runs/$RESUME_FROM_DIR_OVERRIDE"
    else
        PREV_DIR="$ALPHAZERO_DIR/training_runs/${TRAINING_NAME}_job_$RESUME_FROM_JOB_ID"
    fi

    if [ -f "$PREV_DIR/checkpoints.tar.gz" ]; then
        echo "Found previous checkpoint history. Extracting..."
        # Extract the old temp folder into the new Work Dir
        tar -xzf "$PREV_DIR/checkpoints.tar.gz" -C "$WORK_DIR"
    fi
    
    # We look for the best model in the previous result folder
    if [ -f "$PREV_DIR/best.pth.tar" ]; then
        echo "Found best.pth.tar. Copying..."
        mkdir -p "$WORK_DIR/temp/$TRAINING_NAME"
        cp "$PREV_DIR/best.pth.tar" "$WORK_DIR/temp/$TRAINING_NAME/best.pth.tar"

        # Look for latest.examples file
        if [ -f "$PREV_DIR/latest.examples" ]; then
             cp "$PREV_DIR/latest.examples" "$WORK_DIR/temp/$TRAINING_NAME/latest.examples"
             echo "Found and copied latest.examples"
        fi

        # Also look for examples_best.examples as fallback
        if [ -f "$PREV_DIR/examples_best.examples" ]; then
             cp "$PREV_DIR/examples_best.examples" "$WORK_DIR/temp/$TRAINING_NAME/latest.examples"
             echo "Found and copied examples_best.examples"
        fi

        echo -e "${GREEN}Resume files prepared.${NC}"
    else
        echo -e "${RED}ERROR: Could not find best.pth.tar in $PREV_DIR${NC}"
        # Decide: exit 1, or continue from scratch? Let's exit to be safe.
        exit 1
    fi
fi

# --- BACKGROUND BACKUP DAEMON ---
backup_daemon() {
    while true; do
        sleep 7200 # 120 Minutes
        # echo "[Backup] Syncing data to NFS..."

        mkdir -p "$RESULTS_DIR"
        if [ -f "$WORK_DIR/main-cluster.py" ]; then
            cp "$WORK_DIR/main-cluster.py" "$RESULTS_DIR/main-cluster.py" || true
        fi

        # 1. Sync Logs
        if [ -d "$WORK_DIR/logs" ]; then
            rsync -a "$WORK_DIR/logs/" "$RESULTS_DIR/" || true
        fi

        # 2. Sync Best Model immediately (Small file)
        if [ -f "$WORK_DIR/temp/$TRAINING_NAME/best.pth.tar" ]; then
            cp "$WORK_DIR/temp/$TRAINING_NAME/best.pth.tar" "$RESULTS_DIR/best.pth.tar" || true
        fi

        # 3. Sync latest examples only
        if [ -f "$WORK_DIR/temp/$TRAINING_NAME/latest.examples" ]; then
            cp "$WORK_DIR/temp/$TRAINING_NAME/latest.examples" "$RESULTS_DIR/latest.examples" || true
        fi

        # 4. Create a checkpoint tarball
        # We tar to a temp file first so we don't corrupt the destination during write
        tar -czf "$WORK_DIR/checkpoints_temp.tar.gz" -C "$WORK_DIR" temp || true
        mv "$WORK_DIR/checkpoints_temp.tar.gz" "$RESULTS_DIR/checkpoints.tar.gz" || true

        echo "[Backup] Done."
    done
}

echo -e "${GREEN}[4/7] Starting Backup Daemon...${NC}"
backup_daemon &
DAEMON_PID=$!

# --- TRAP SIGNALS ---
# If the job hits the time limit (SIGTERM), kill daemon and save one last time
cleanup() {
    if [ "${CLEANUP_DONE:-0}" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1
    set +e

    echo -e "${YELLOW}!!! CAUGHT SIGNAL / FINISHING !!!${NC}"
    kill $DAEMON_PID 2>/dev/null || true

    echo "Final Save..."
    mkdir -p "$RESULTS_DIR"
    if [ -f "$WORK_DIR/main-cluster.py" ]; then
        cp "$WORK_DIR/main-cluster.py" "$RESULTS_DIR/main-cluster.py" || true
    fi
    if [ -d "$WORK_DIR/logs" ]; then
        rsync -a "$WORK_DIR/logs/" "$RESULTS_DIR/" || true
    fi

    # Save model
    if [ -f "$WORK_DIR/temp/$TRAINING_NAME/best.pth.tar" ]; then
        cp "$WORK_DIR/temp/$TRAINING_NAME/best.pth.tar" "$RESULTS_DIR/best.pth.tar" || true
    fi

    # Save all examples files
    if [ -f "$WORK_DIR/temp/$TRAINING_NAME/latest.examples" ]; then
        cp "$WORK_DIR/temp/$TRAINING_NAME/latest.examples" "$RESULTS_DIR/latest.examples" || true
    fi

    # Create final tarball
    if [ -d "$WORK_DIR/temp" ]; then
        tar -czf "$RESULTS_DIR/checkpoints.tar.gz" -C "$WORK_DIR" temp || true
    fi

    echo "Cleaning scratch..."
    rm -rf "$WORK_DIR"
    echo "Done."
}
trap 'cleanup; exit 0' SIGTERM
trap 'echo "[Signal] USR1 received, saving early and exiting."; cleanup; exit 0' SIGUSR1
trap cleanup EXIT

# --- START TRAINING ---
echo -e "${GREEN}[5/7] Starting training...${NC}"
cd "$WORK_DIR"
mkdir -p logs

# Export training name for Python to use
export TRAINING_NAME="$TRAINING_NAME"
export USE_AMP="$USE_AMP"
export AMP_DTYPE="$AMP_DTYPE"

python3 -u main-cluster.py 2>&1 | tee logs/training.log

# The 'trap' function handles the cleanup when this finishes.
