#!/bin/bash

# Function to print logs with timestamp
log() {
    # printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

TIMESTAMP=$(TZ="America/New_York" date "+%Y%m%d_%H%M%S")
log "Starting at $TIMESTAMP"

ENV="/data/weka/ellisb/LLaVA-NeXT/.conda/ov"

############### Params ################
LR=5e-6
VIS_LR=1e-6

FRAMES=32
PD_BS=2
GA_STEPS=4
EPOCHS=1
GLOBAL_BS=64

###############  Data ################
DATA_YAML_PATH="/data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_09_1k_mt5_mixed.yaml"

IMAGE_FOLDER="/data/weka/ellisb/datasets/video/all_images"
VIDEO_FOLDER="/data/weka/ellisb/datasets/video/all_videos"

###############  Model ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Run Settings ################
PROMPT_VERSION="qwen_1_5"
RUN_NAME="ft-llava-ov-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${FRAMES}F_${DESCRIPTION}"

PREV_STAGE_CHECKPOINT="/data/weka/ellisb/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov"
OUTPUT_CHECKPOINT="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/$RUN_NAME"

log "DESCRIPTION: ${DESCRIPTION}"
log "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
log "RUN_NAME: ${RUN_NAME}"

############### Verify ################

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS
log "NUM_GPUS: ${NUM_GPUS}"
log " - passed N_GPUS: $N_GPUS"
log " - passed N_NODES: $N_NODES"


eff_bs=$((PD_BS * GA_STEPS * NUM_GPUS * N_NODES))
log "Effective batch size: $eff_bs
    - accumulation steps: $GA_STEPS
    - per device batch size: $PD_BS
    - num gpus: $NUM_GPUS
    - num nodes: $N_NODES"

# verify that the effective batch size is equal to the global batch size
if [ $((PD_BS * GA_STEPS * NUM_GPUS * N_NODES)) -ne $GLOBAL_BS ]; then
    log "Effective batch size does not match global batch size"
    echo "  - effective batch size: $eff_bs"
    echo "  - global batch size: $GLOBAL_BS"
    exit 1
fi

export WANDB_NAME=$RUN_NAME

nvidia-smi

############### Run ################
DEEPSPEED=$ENV/bin/deepspeed
CMD="$DEEPSPEED \
    --master_port $PORT \
    --num_nodes $N_NODES \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts='mm_vision_tower,mm_mlp_adapter,mm_language_model' \
    --mm_vision_tower_lr=$VIS_LR \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints '(1x1),...,(6x6)' \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_CHECKPOINT \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $PD_BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GA_STEPS \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend 'inductor' \
    --dataloader_drop_last True \
    --frames_upbound $FRAMES"

echo ""
log "Running Command:"
echo $CMD
echo ""

eval $CMD

log "Finished at $(TZ="America/New_York" date "+%Y%m%d_%H%M%S")"
