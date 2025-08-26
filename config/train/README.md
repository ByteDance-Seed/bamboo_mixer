# Training Configuration Guide

This directory contains the configuration file for training the predictive model. You can use you custom data with conductivity and anion ratio labels.

## Configuration File

### predictor_config.yaml
**Purpose**: Training the electrolyte property predictor  
**Key Parameters**:
```yaml
meta:
  work_folder: "<path_to_train_logs>"  # Training logs and checkpoints
  random_seed: 123                     # Reproducibility seed

dataset:
  config:  "<path to your train data folder>/dataset_config.yaml" # Training data path

model:
  pretrain:
    ckpt_path: "<path to your checkpoint folder>/pretrain.pt" # Checkpoint path for pretrained mono model
```
