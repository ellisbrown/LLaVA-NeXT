# bash scripts/ai2/launch.sh --gpus 8 --clusters all --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_11__ov_ai2_SAT_10k.sh &
# bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_11__vid_ai2_SAT_10k.sh &

# bash scripts/ai2/launch.sh --gpus 8 --clusters all --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_11__ov_ai2_SAT_50k.sh &
# bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_11__vid_ai2_SAT_50k.sh &


# # replace "long_size_est" with "long_est_v2" in all files *mt1_long_size_est_v2_*
# # sed -i 's/long_size_est/long_size_est_v2/g' scripts/ai2/exp/*mt1_long_size_est_v2_*



# # DATE=$(date '+%Y_%m_%d')
# DATE=2025_02_11
# TASK1="house_size_est"
# TASK2="n_rooms"
# TASK2="temporal_order_2"
# TASK2="temporal_order_3"
# TASK2="temporal_order_4"
# TASK2="temporal_order_5"
# cp scripts/ai2/exp/${DATE}__vid_mt1_${TASK1}_mc.sh scripts/ai2/exp/${DATE}__vid_mt1_${TASK2}_mc.sh
# cp scripts/ai2/exp/${DATE}__vid_mt1_${TASK1}_oe.sh scripts/ai2/exp/${DATE}__vid_mt1_${TASK2}_oe.sh
# cp scripts/ai2/exp/${DATE}__ov_mt1_${TASK1}_oe.sh scripts/ai2/exp/${DATE}__ov_mt1_${TASK2}_oe.sh
# cp scripts/ai2/exp/${DATE}__ov_mt1_${TASK1}_mc.sh scripts/ai2/exp/${DATE}__ov_mt1_${TASK2}_mc.sh


# # replace "TASK1" with "TASK2" in all files *DATE*mt1_TASK2_*.sh
# sed -i "s/${TASK1}/${TASK2}/g" scripts/ai2/exp/${DATE}__*mt1_${TASK2}_*.sh

# # launch
# bash scripts/ai2/launch.sh --gpus 8 --clusters all --script scripts/ai2/exp/${DATE}__ov_mt1_${TASK2}_oe.sh &
# bash scripts/ai2/launch.sh --gpus 8 --clusters all --script scripts/ai2/exp/${DATE}__ov_mt1_${TASK2}_mc.sh &
# bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/${DATE}__vid_mt1_${TASK2}_oe.sh &
# bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/${DATE}__vid_mt1_${TASK2}_mc.sh &


bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_mt1_mix_abs_dist_rel_dir.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_mt1_mix_abs_dist_rel_dir_OE.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_mt1_mix_abs_dist_rel_dir_temp_ord.sh &




# ablation launch scripts

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_visual_ablation_2k_2qa_rgb.sh &

VERSIONS=(
    # "rgb"
    "depth" "edge" "colored_edge" "colored_edge_no" "semantic_seg" "instance_seg" "mean_mask" "masked_bg")
for VERSION in "${VERSIONS[@]}"
do
    # copy the rgb script, then replace "_rgb.yaml" with "_${VERSION}.yaml"
    cp /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_visual_ablation_2k_2qa_rgb.sh /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_visual_ablation_2k_2qa_${VERSION}.sh
    sed -i "s/_rgb.yaml/_${VERSION}.yaml/g" /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_visual_ablation_2k_2qa_${VERSION}.sh

    # copy the rgb yaml, then replace "/qas/rgb/" with "/qas/${VERSION}/"
    cp /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17_vid_visual_ablation_2k_2qa_rgb.yaml /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17_vid_visual_ablation_2k_2qa_${VERSION}.yaml
    sed -i "s/qas\/rgb\//qas\/${VERSION}\//g" /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17_vid_visual_ablation_2k_2qa_${VERSION}.yaml

    # launch
    bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script /data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_17__vid_visual_ablation_2k_2qa_${VERSION}.sh &
done
