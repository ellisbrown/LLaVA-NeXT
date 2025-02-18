# ablation launch scripts

SCRIPT_DIR=/data/weka/ellisb/LLaVA-NeXT/scripts/ai2/exp/2025_02_18_visual_ablation_3k_3qa

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script $SCRIPT_DIR/2025_02_18__vid_visual_ablation_3k_3qa_rgb.sh &

VERSIONS=(
    # "rgb"
    "depth" "edge" "colored_edge" "colored_edge_no" "semantic_seg" "instance_seg" "mean_mask" "masked_bg")
for VERSION in "${VERSIONS[@]}"
do
    # copy the rgb script, then replace "_rgb.yaml" with "_${VERSION}.yaml"
    cp $SCRIPT_DIR/2025_02_18__vid_visual_ablation_3k_3qa_rgb.sh $SCRIPT_DIR/2025_02_18__vid_visual_ablation_3k_3qa_${VERSION}.sh
    sed -i "s/_rgb.yaml/_${VERSION}.yaml/g" $SCRIPT_DIR/2025_02_18__vid_visual_ablation_3k_3qa_${VERSION}.sh

    # copy the rgb yaml, then replace "/qas/rgb/" with "/qas/${VERSION}/"
    cp $SCRIPT_DIR/2025_02_18_vid_visual_ablation_3k_3qa_rgb.yaml $SCRIPT_DIR/2025_02_18_vid_visual_ablation_3k_3qa_${VERSION}.yaml
    sed -i "s/qas\/rgb\//qas\/${VERSION}\//g" $SCRIPT_DIR/2025_02_18_vid_visual_ablation_3k_3qa_${VERSION}.yaml

    # launch
    bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script $SCRIPT_DIR/2025_02_18__vid_visual_ablation_3k_3qa_${VERSION}.sh &
done
