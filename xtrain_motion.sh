
# MambaVT-Small
## train for non-lasher evaluation
python tracking/train.py --script mambatrack_motion --config mambavt_motion_s256_ep10  --save_dir ./output --mode multiple --nproc_per_node 2
## train for lasher evaluation
python tracking/train.py --script mambatrack_motion --config mambavt_motion_s256_ep10_lasher  --save_dir ./output --mode multiple --nproc_per_node 2


# MambaVT-Middle
## train for non-lasher evaluation
python tracking/train.py --script mambatrack_motion --config mambavt_motion_m256_ep10  --save_dir ./output --mode multiple --nproc_per_node 2
## train for lasher evaluation
python tracking/train.py --script mambatrack_motion --config mambavt_motion_m256_ep10_lasher  --save_dir ./output --mode multiple --nproc_per_node 2
