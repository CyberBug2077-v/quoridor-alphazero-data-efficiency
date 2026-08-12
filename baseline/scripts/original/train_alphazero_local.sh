#!/bin/bash

################################################################################
# AlphaZero Training — Local
# Usage: bash scripts/train_alphazero_local.sh
################################################################################

# --- CONFIGURATION ---
TRAINING_NAME="${TRAINING_NAME:-no_gating_fresh_2}"
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"

# Folder to resume from (copies best.pth.tar + latest.examples into temp/TRAINING_NAME).
# Leave empty to start fresh or resume from an existing temp/TRAINING_NAME folder.
RESUME_FROM="no_gating_fresh_1"

# Set to 1 to save checkpoints into temp/<TRAINING_NAME> directly (local mode).
# Set to 0 to keep the old cluster behaviour (scratch + results copy).
LOCAL_MODE=1

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- PATHS ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ALPHAZERO_DIR="$PROJECT_ROOT/external/alphazero"

echo -e "${GREEN}[1/4] Initializing...${NC}"
echo -e "Training Name: ${BLUE}$TRAINING_NAME${NC}"
echo -e "Resume From:   ${BLUE}${RESUME_FROM:-none}${NC}"
echo -e "AMP: $USE_AMP ($AMP_DTYPE)"

# --- ENVIRONMENT ---
echo -e "${GREEN}[2/4] Activating environment...${NC}"
source "$ALPHAZERO_DIR/.venv/bin/activate"

# --- RESUME LOGIC ---
if [ ! -z "$RESUME_FROM" ]; then
    RESUME_DIR="$ALPHAZERO_DIR/temp/$RESUME_FROM"
    DEST_DIR="$ALPHAZERO_DIR/temp/$TRAINING_NAME"
    mkdir -p "$DEST_DIR"

    if [ -f "$DEST_DIR/best.pth.tar" ]; then
        echo -e "${YELLOW}[Resume] $DEST_DIR already has a model — skipping copy.${NC}"
    elif [ -f "$RESUME_DIR/best.pth.tar" ]; then
        echo -e "${GREEN}[3/4] Copying checkpoint from $RESUME_FROM...${NC}"
        cp "$RESUME_DIR/best.pth.tar" "$DEST_DIR/best.pth.tar"
        if [ -f "$RESUME_DIR/latest.examples" ]; then
            cp "$RESUME_DIR/latest.examples" "$DEST_DIR/latest.examples"
            echo "Copied best.pth.tar + latest.examples"
        else
            echo "Copied best.pth.tar (no examples found)"
        fi
    else
        echo -e "${YELLOW}[Resume] No model found in $RESUME_DIR — starting fresh.${NC}"
    fi
else
    echo -e "${GREEN}[3/4] No resume source set — using existing temp/$TRAINING_NAME or starting fresh.${NC}"
fi

# --- START TRAINING ---
echo -e "${GREEN}[4/4] Starting training...${NC}"
cd "$ALPHAZERO_DIR"
mkdir -p logs

export TRAINING_NAME="$TRAINING_NAME"
export USE_AMP="$USE_AMP"
export AMP_DTYPE="$AMP_DTYPE"

mkdir -p "temp/$TRAINING_NAME"
python3 -u main.py 2>&1 | tee logs/training_${TRAINING_NAME}.log
