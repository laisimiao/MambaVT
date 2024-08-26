# test for MambaVT-Small
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_s256_ep10 --dataset_name GTOT --epoch 10
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_s256_ep10 --dataset_name RGBT234 --epoch 10
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_s256_ep10 --dataset_name RGBT210 --epoch 10
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_s256_ep10_lasher --dataset_name LasHeR --epoch 10

# test for MambaVT-Middle
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_m256_ep10 --dataset_name GTOT --epoch 10
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_m256_ep10 --dataset_name RGBT234 --epoch 10
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_m256_ep10 --dataset_name RGBT210 --epoch 10
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack_motion --yaml_name mambavt_motion_m256_ep10_lasher --dataset_name LasHeR --epoch 10