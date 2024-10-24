#! /bin/bash

# video tasks
accelerate launch --num_processes=4 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks videomme_w_subtitle \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/

# --tasks activitynetqa,videochatgpt,nextqa_mc_test,egoschema,video_dc499,videmme,videomme_w_subtitle,perceptiontest_val_mc \
