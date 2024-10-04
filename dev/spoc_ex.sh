#! /bin/bash

ROOT_DIR="/nfs/ellisb/LLaVA-NeXT"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false


CKPT=lmms-lab/LLaVA-NeXT-Video-7B-DPO
CONV_MODE=vicuna_v1
FRAMES=32  # sampled frames
POOL_STRIDE=2  # temporal pooling stride
POOL_MODE=average  # average, max
NEWLINE_POSITION=no_token  # no_token, grid
OVERWRITE=True

# lmms-lab/LLaVA-NeXT-Video-32B-Qwen qwen_1_5 32 2 average after grid True playground/demo/xU25MMA2N4aVtYay.mp4

# CKPT=lmms-lab/LLaVA-NeXT-Video-32B-Qwen
# CONV_MODE=qwen_1_5
# FRAMES=32  # sampled frames

# CKPT=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2-Video-Only
CKPT=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
CONV_MODE=qwen_2
FRAMES=64  # sampled frames
POOL_STRIDE=2  # temporal pooling stride
POOL_MODE=average  # average, max
NEWLINE_POSITION=grid  # no_token, grid
OVERWRITE=True
VIDEO_PATH=playground/demo/xU25MMA2N4aVtYay.mp4

# VIDEO_PATH=playground/demo/raw_navigation_camera__0.mp4

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}
    echo "overwrite is False, saving to a new directory: $SAVE_DIR"
else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
    echo "overwrite is True, overwriting the existing directory: $SAVE_DIR"
fi


# bash scripts/video/demo/video_demo.sh \
#     $CKPT \
#     $CONV_MODE \
#     $FRAMES \
#     $POOL_STRIDE \
#     $POOL_MODE \
#     $NEWLINE_POSITION \
#     $OVERWRITE \
#     $VIDEO_PATH

python3 playground/demo/video_demo.py \
    --model-path $CKPT \
    --video_path ${VIDEO_PATH} \
    --output_dir ./work_dirs/video_demo/$SAVE_DIR \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --add_time_instruction True \
    --torch_dtype bfloat16 \
    --prompt "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."

    # --prompt "Did you see a yellow vespa or a bed first?"

