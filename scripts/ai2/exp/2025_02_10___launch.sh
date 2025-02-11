bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_abs_dist_oe.sh &
bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_abs_dist.sh &

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_abs_dist_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_abs_dist.sh &



bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_oe.sh &
bash scripts/ai2/launch.sh --gpus 4 --clusters all --script scripts/ai2/exp/2025_02_10__ov_mt1_long_size_est_mc.sh &

bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_oe.sh &
bash scripts/ai2/launch.sh --gpus 8 --clusters 80gb --script scripts/ai2/exp/2025_02_10__vid_mt1_long_size_est_mc.sh &