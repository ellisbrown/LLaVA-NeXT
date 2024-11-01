#! /bin/bash

# Set the root directory where your project resides
ROOT_DIR="/data/weka/ellisb/LLaVA-NeXT"

# Check if the root directory exists
if [ ! -e $ROOT_DIR ]; then
    echo "The root directory does not exist. Exiting the script."
    exit 1
fi

# Navigate to the root directory
cd $ROOT_DIR

# Suppress Python warnings and tokenizer parallelism warnings
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# # Set model and evaluation parameters
# CKPT=lmms-lab/LLaVA-NeXT-Video-7B-DPO
# CONV_MODE=vicuna_v1
# FRAMES=32          # Number of frames to sample from each video
# POOL_STRIDE=2      # Temporal pooling stride
# POOL_MODE=average  # Pooling mode: average or max
# NEWLINE_POSITION=no_token  # Position of newline tokens: no_token or grid
# OVERWRITE=True     # Whether to overwrite existing outputs

CKPT=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
CONV_MODE=qwen_2
FRAMES=64  # sampled frames
POOL_STRIDE=2  # temporal pooling stride
POOL_MODE=average  # average, max
NEWLINE_POSITION=grid  # no_token, grid
OVERWRITE=True


# VIDEO_DIR=playground/demo/           # Directory containing the videos
# INPUT_JSONL=data/example.jsonl  # Path to your input JSONL file
# SOURCE=example

# Set the base directory for videos and the path to the input JSONL file
VIDEO_DIR=data/oct23/           # Directory containing the videos
INPUT_JSONL=data/oct23/combined_qa_pairs.jsonl  # Path to your input JSONL file
SOURCE=oct23

# Determine the save directory based on the overwrite flag
SAVE_DIR="${SOURCE}_$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}"
if [ "$OVERWRITE" = False ]; then
    SAVE_DIR="${SAVE_DIR}_overwrite_${OVERWRITE}"
    echo "Overwrite is False, saving to a new directory: $SAVE_DIR"
else
    echo "Overwrite is True, overwriting the existing directory: $SAVE_DIR"
fi

# Run the evaluation script
python3 playground/video_eval.py \
    --model-path $CKPT \
    --video_dir ${VIDEO_DIR} \
    --input_jsonl ${INPUT_JSONL} \
    --output_dir ./work_dirs/video_eval/$SAVE_DIR \
    --output_name pred \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --add_time_instruction True \
    --torch_dtype bfloat16
