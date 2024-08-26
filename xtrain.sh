
# MambaVT-Small
## train for non-lasher evaluation
python tracking/train.py --script mambatrack --config mambavt_s256_ep20  --save_dir ./output --mode multiple --nproc_per_node 2
## train for lasher evaluation
python tracking/train.py --script mambatrack --config mambavt_s256_ep20_lasher  --save_dir ./output --mode multiple --nproc_per_node 2


# MambaVT-Middle
## train for non-lasher evaluation
python tracking/train.py --script mambatrack --config mambavt_m256_ep20  --save_dir ./output --mode multiple --nproc_per_node 2
## train for lasher evaluation
python tracking/train.py --script mambatrack --config mambavt_m256_ep20_lasher  --save_dir ./output --mode multiple --nproc_per_node 2
