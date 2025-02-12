bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_abs_dist_oe.sh &
bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_abs_dist.sh &

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_abs_dist_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_abs_dist.sh &



bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_oe.sh &
bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_mc.sh &

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_mc.sh &


# cp scripts/ai2/exp/2025_02_10_mt1_long_size_est_oe.yaml scripts/ai2/exp/2025_02_10_mt1_short_size_est_oe.yaml
# cp scripts/ai2/exp/2025_02_10_mt1_long_size_est_mc.yaml scripts/ai2/exp/2025_02_10_mt1_short_size_est_mc.yaml
# cp scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_oe.sh scripts/ai2/exp/2025_02_10__vid_mt1_short_size_est_oe.sh
# cp scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_mc.sh scripts/ai2/exp/2025_02_10__vid_mt1_short_size_est_mc.sh
# cp scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_oe.sh scripts/ai2/exp/2025_02_10__ov_mt1_short_size_est_oe.sh
# cp scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_mc.sh scripts/ai2/exp/2025_02_10__ov_mt1_short_size_est_mc.sh

# # replace "long_size_est" with "short_size_est" in all files *mt1_short_size_est_*
# sed -i 's/long_size_est/short_size_est/g' scripts/ai2/exp/*mt1_short_size_est_*


bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_short_size_est_oe.sh &
bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_short_size_est_mc.sh &

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_short_size_est_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_short_size_est_mc.sh &


# long v2

cp scripts/ai2/exp/2025_02_10_mt1_long_size_est_oe.yaml scripts/ai2/exp/2025_02_10_mt1_long_size_est_v2_oe.yaml
cp scripts/ai2/exp/2025_02_10_mt1_long_size_est_mc.yaml scripts/ai2/exp/2025_02_10_mt1_long_size_est_v2_mc.yaml
cp scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_oe.sh scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_v2_oe.sh
cp scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_mc.sh scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_v2_mc.sh
cp scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_oe.sh scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_v2_oe.sh
cp scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_mc.sh scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_v2_mc.sh

# replace "long_size_est" with "long_est_v2" in all files *mt1_long_size_est_v2_*
sed -i 's/long_size_est/long_size_est_v2/g' scripts/ai2/exp/*mt1_long_size_est_v2_*


bash scripts/ai2/launch.sh --gpus 8 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_v2_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_v2_mc.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_v2_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_v2_mc.sh &