bash scripts/ai2/launch.sh --script scripts/ai2/exp/2025_02_09__vid_1k_mt1_grouped.sh
bash scripts/ai2/launch.sh --script scripts/ai2/exp/2025_02_09__vid_1k_mt1_mixed.sh
bash scripts/ai2/launch.sh --script scripts/ai2/exp/2025_02_09__vid_1k_mt3_grouped.sh
bash scripts/ai2/launch.sh --script scripts/ai2/exp/2025_02_09__vid_1k_mt3_mixed.sh
bash scripts/ai2/launch.sh --script scripts/ai2/exp/2025_02_09__vid_1k_mt5_grouped.sh
bash scripts/ai2/launch.sh --script scripts/ai2/exp/2025_02_09__vid_1k_mt5_mixed.sh


# one vision can fit on L40s, so use all clusters
bash scripts/ai2/launch.sh --clusters all --script scripts/ai2/exp/2025_02_09__ov_1k_mt5_grouped.sh &
bash scripts/ai2/launch.sh --clusters all --script scripts/ai2/exp/2025_02_09__ov_1k_mt5_mixed.sh &
bash scripts/ai2/launch.sh --clusters all --script scripts/ai2/exp/2025_02_09__ov_1k_mt3_grouped.sh &
bash scripts/ai2/launch.sh --clusters all --script scripts/ai2/exp/2025_02_09__ov_1k_mt3_mixed.sh &
bash scripts/ai2/launch.sh --clusters all --script scripts/ai2/exp/2025_02_09__ov_1k_mt1_grouped.sh &
bash scripts/ai2/launch.sh --clusters all --script scripts/ai2/exp/2025_02_09__ov_1k_mt1_mixed.sh &
