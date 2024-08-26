# MambaVT
The official implementation for the **\[AAAI 2025\]** paper: "MambaVT: Spatio-Temporal Contextual Modeling for robust RGB-T Tracking"

:rocket: Update Models and Results (2024/08/07)  
[Models & Raw Results](https://drive.google.com/drive/folders/1Ww-cMuzJ-6XcnTSsPuedlyR_LWSgwAaR?usp=sharing) [Google Driver]  
[Models & Raw Results](https://pan.baidu.com/s/1XaXsSrToDqbLLAStJcMr9g) [Baidu Driver: iiau]

<p align="center">
  <img width="85%" src="https://github.com/laisimiao/MambaVT/blob/main/assets/framework.png" alt="Framework"/>
</p>

## Highlights
### :star2: New Unified Mamba-based Tracking Framework  
MambaVT is a simple, neat, high-performance **unified Mamba-based tracking framework** for global long-range and local short-term spatio-temporal contextual modeling for robust RGB-T Tracking. MambaVT achieves SOTA performance on multiple RGB-T benchmarks with fewer FLOPs and Params. MambaVT can serve as a strong baseline for further research.

| Tracker     | GTOT (SR) | RGBT210 (SR) | RGBT234 (MSR) | LasHeR(SR) |
|:-----------:|:------------:|:-----------:|:-----------------:|:-----------:|
| MambaVT-S256 | 75.3         | 63.7        | 65.8              | 57.9        |
| MambaVT-M256 | 78.6         | 64.4        | 67.5              | 57.5        |

## Install the environment
We've tested the results on the PyTorch2.1.1+cuda11.8+Python3.9+causal-conv1d==1.1.0  

**Option1**: Use the Anaconda (CUDA 11.8)
```bash
conda create -n mambavt python=3.9
conda activate mambavt
bash install.sh
```
  
And we strongly recommend installing torch/torchvision/causal-conv1d manually by:  
```bash
# Download torch from: https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl
pip install torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl

# Download torchvision from: https://download.pytorch.org/whl/cu118/torchvision-0.16.1%2Bcu118-cp39-cp39-linux_x86_64.whl
pip install torchvision-0.16.1%2Bcu118-cp39-cp39-linux_x86_64.whl

# Download causal-conv1d from: https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install causal_conv1d-1.1.0+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
prepare the LasHeR dataset. It should look like:
   ```
   ${PROJECT_ROOT}
    -- LasHeR
        -- train/
            |-- trainingset
              |-- 1boygo
              |-- ...
              |-- trainingsetList.txt
            |-- tracker_predicted
            ...
        -- test
            |-- testingset
              |-- 1blackteacher
              |-- ...
              |-- testingsetList.txt
   ```

## Training
Download pre-trained `OSTrack_videomambas_ep300.pth.tar` or `OSTrack_videomambam_ep300.pth.tar` from above driver link and put it under `$PROJECT_ROOT$/pretrained_models` . Then  

```
bash xtrain.sh
bash xtrain_motion.sh
```

Replace `--config` with the desired model config under `experiments/mambatrack` or `experiments/mambatrack_motion`.


## Evaluation
Download the model weights from above driver link  

Put the downloaded weights on `$PROJECT_ROOT$/checkpoints/`  

Change the corresponding values of `lib/test/parameter/mambatrack_motion.py` to the actual checkpoint paths. Then  

```
bash ytest.sh
bash ytest_motion.sh
```

## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single 3090 GPU.

```bash
# Profiling mambavt_s256_ep20
python tracking/profile_model.py --script mambatrack --config mambavt_s256_ep20
# Profiling mambavt_m256_ep20
python tracking/profile_model.py --script mambatrack --config mambavt_m256_ep20
```

## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack), [Vim](https://github.com/hustvl/Vim) and [VideoMamba](https://github.com/OpenGVLab/VideoMamba) library, which helps us to quickly implement our ideas.
 

## Contact
If you have any question, feel free to email laisimiao@mail.dlut.edu.cn. ^_^
