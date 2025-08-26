# Test Results Configuration Guide

This directory contains configuration files and scripts for model inference tasks including property prediction and formulation generation.

## Key Tasks
1. **Molecular Property Prediction** (`mono_config.yaml`)
2. **Electrolyte Property Prediction** (`predict_config.yaml`)
3. **Formulation Generation** (`generate_config.yaml`)

---

## Configuration Files

### 1. mono_config.yaml - Molecular Property Prediction
```yaml
# Example configuration
ckpt_path: "<path to the checkpoint folder>/optimal.pt" # Model checkpoint path
save_dir: "<path to your data folder>" # Testing data path
output_path: "<path to your output folder>/output_mono.json" # Output data path
data_cls: "MonoData"
key_map:
  temperature: "temperature"

# Model hyperparameters
model: ...
```

### 2. predict_config.yaml - Electrolyte Property Prediction
```yaml
# Example configuration
ckpt_path: "<path to the checkpoint folder>" # Model checkpoint path
save_dir: "<path to your data folder>" # Testing data path
output_path: "<path to your output folder>"/output_test.json" # Output data path
data_cls: "FormulaData"
key_map:
  concentration: "salt_molar_ratio"
  temperature: "temperature"

# Model hyperparameters
model: ...
```

### 3. generate_config.yaml - Formulation Generation
```yaml
# Example configuration
ckpt_path: "<path to the checkpoint folder>" # Model checkpoint path
output_dir: "<path to your output folder>" # Output data path
num_batch: 2 # Number of batches to generate
batch_size: 128 # Batch size for each generation
temperature: 25.0 # Temperature for generation (frozen)
concentration: 0.1 # Concentration for generation (frozen)
conductivities: [5.0, 10.0, 30.0] # List of conductivities for generation 
anion_ratios: [0.3, 0.5, 0.7] # List of anion ratios for generation

# Model hyperparameters
model: ...
```