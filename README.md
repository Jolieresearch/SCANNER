# Shedding the Facades, Connecting the Domains: Detecting Shifting Multimodal Hate Video with Test-Time Adaptation
Official github repository for SCANNER：
Shedding the Facades, Connecting the Domains: Detecting Shifting Multimodal Hate Video with Test-Time Adaptation
## Dataset

You can obtain the datasets from the following project sites.

### HMM

Access the HMM dataset from [hate-alert/HateMM](https://github.com/hate-alert/HateMM).

### MHB and MHY

Access the MHY and MHB dataset from [Social-AI-Studio/MultiHateClip: Official repository for ACM Multimedia'24 paper "MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili"](https://github.com/social-ai-studio/multihateclip).

# Usage

## Requirements

To set up the environment, run the following commands:

```bash
conda create --name py312 python=3.12
pip install torch transformers tqdm loguru pandas torchmetrics scikit-learn colorama wandb hydra-core
```

## Run

```shell
# Run SCANNER for the MHB to MHY dataset
python src/train.py --config-name ZH2EN

# Run SCANNER for the MHY to MHB dataset
python src/train.py --config-name EN2ZH

```

