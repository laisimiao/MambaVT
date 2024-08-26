# test for MambaVT-Small
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_s256_ep20 --dataset_name GTOT --epoch 20
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_s256_ep20 --dataset_name RGBT234 --epoch 20
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_s256_ep20 --dataset_name RGBT210 --epoch 20
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_s256_ep20_lasher --dataset_name LasHeR --epoch 20

# test for MambaVT-Middle
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_m256_ep20 --dataset_name GTOT --epoch 20
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_m256_ep20 --dataset_name RGBT234 --epoch 20
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_m256_ep20 --dataset_name RGBT210 --epoch 20
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mambatrack --yaml_name mambavt_m256_ep20_lasher --dataset_name LasHeR --epoch 20