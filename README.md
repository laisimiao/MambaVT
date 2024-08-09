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
prepare the LSOTB dataset. It should look like:
   ```
   ${PROJECT_ROOT}
    -- LSOTB
        -- Train
            |-- TrainingData
            |-- MyAnnotations
            ...
        -- Eval
            |-- aircraft_car
            |-- airplane_H_001
            |-- LSOTB-TIR-120.json
            |-- LSOTB-TIR-136.json
            |-- LSOTB-TIR-LT11.json
            |-- LSOTB-TIR-ST100.json
   ```

## Training
Download pre-trained `OSTrack_ep0300.pth.tar` from above driver link and put it under `$PROJECT_ROOT$/pretrained_models` . Then  

```
bash xtrain.sh
```

Replace `--config` with the desired model config under `experiments/refocus`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights `OSTrack_ep0060.pth.tar` from above driver link  

Put the downloaded weights on `$PROJECT_ROOT$/checkpoints/`  

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths. Then  

```
bash ytest.sh
```

## Acknowledgments
- This repo is based on [OSTrack](https://github.com/botaoye/OSTrack) which is an excellent work.  

## Contact
If you have any question, feel free to email laisimiao@mail.dlut.edu.cn. ^_^
