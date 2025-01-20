#!/bin/bash

ENV="/data/weka/ellisb/LLaVA-NeXT/.conda/ov"

# Example experiment logic
log() {
    # printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    # no colors
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}


log "HF_HOME: $HF_HOME"

log "Activating conda environment $ENV"
conda activate $ENV

DEEPSPEED=$ENV/bin/deepspeed

CMD="$DEEPSPEED llava/train/train_mem.py --help"
log "Running command: $CMD"
eval $CMD
log "Experiment completed."
