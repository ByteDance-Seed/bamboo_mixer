# Data Preparation Configuration Guide

This document explains the three YAML configuration files for dataset processing:

- `formula_config.yaml` - Electrolyte formulation datasets
- `mono_config.yaml` - Single-molecule property datasets
- `train_data_config.yaml` - Training data preprocessing

## Core Parameters

### Common Parameters (All Configs)
| Parameter    | Description                          | Example Value               |
|--------------|--------------------------------------|-----------------------------|
| `json_path`  | Path to raw JSON dataset             | `/data/data.json`     |
| `save_dir`    | Processed data output directory      | `/data`           |
| `data_cls`   | Dataset loader class                 | `FormulaData` or `MonoData`    |
| `key_map`     | Field mapping between raw data and model inputs | `{temperature: "temperature"}` |

---

## Configuration Details

### 1. mono_config.yaml
For single-molecule data
```yaml
# Example configuration
json_path: "<path to your data folder>/data.json" # path to your data folder
save_dir: "<path to your data folder>" # the processed data will be generally stored in the same folder

data_cls: MonoData
# input feature
key_map:  
  temperature: temperature # temperature of the molecular property
```

### 2. formula_config.yaml
For electrolyte formulation data (solution systems)
```yaml
# Example configuration
json_path: "<path to your data folder>/data.json" # path to your data folder
save_dir: "<path to your data folder>"  # the processed data will be generally stored in the same folder

data_cls: FormulaData
key_map:  
  concentration: salt_molar_ratio # concentration of the Li salt
  temperature: temperature # temperature of conductivity and anion ratio measurement

```

### 3. train_data_config.yaml
For preparing training data.
```yaml
# Example configuration
json_path: "<path to your data folder>/data.json" # path to your data folder
save_dir: "<path to your data folder>"  # the processed data will be generally stored in the same folder

data_cls: FormulaData
key_map:  
  concentration: salt_molar_ratio
  temperature: temperature
  # Property to predict
  conductivity: conductivity 
  conductivity_mask: conductivity_mask # whether the label exists or not. # True if label exists, False if missing
  anion_ratio: anion_ratio
  anion_ratio_mask: anion_ratio_mask
```

## Data entry example
### [`MonoData`](./formula_design/data/data.py#L243) 
```json
{
    "name": "PC",
    "smiles":"CC1COC(=O)O1", 
    "temperature": 25
}
```
### [`FormulaData`](./formula_design/data/data.py#L289)
- All the molar ratios below are discussed separately for solvents and salts (normalized within solvents and salts respectively).
- salt_molar_ratio = number of salts / (number of solvents + number of salts)
```json
{
    "solvents": [
        {
            "name": "PC",
            "smiles": "CC1COC(=O)O1",
            "molar_ratio": 0.5364247732959164
        },
        {
            "name": "DEC",
            "smiles": "CCOC(=O)OCC",
            "molar_ratio": 0.46357522670408363
        }
    ],
    "salts": [
        {
            "name": "LiPF6",
            "smiles": "F[P-](F)(F)(F)(F)F",
            "molar_ratio": 1.0
        }
    ],
    "temperature": -40.0,
    "salt_molar_ratio": 0.045560510565158335
}
```

### [`TrainData`](./formula_design/data/data.py#L289)
- All the molar ratios below are discussed separately for solvents and salts (normalized within solvents and salts respectively).
- salt_molar_ratio = number of salts / (number of solvents + number of salts)
- property mask (e.g. conductivity_mask): show whether the label exists or not (True if exists, False if missing).
```json
{
    "solvents": [
        {
            "name": "PC",
            "smiles": "CC1COC(=O)O1",
            "molar_ratio": 146.0,
            "coord_num": 2.2690946930280953
        },
        {
            "name": "DEC",
            "smiles": "CCOC(=O)OCC",
            "molar_ratio": 126.0,
            "coord_num": 1.9808532778355883
        }
    ],
    "salts": [
        {
            "name": "LiPF6",
            "smiles": "F[P-](F)(F)(F)(F)F",
            "molar_ratio": 13.0,
            "coord_num": 0.37315296566077005
        }
    ],
    "salt_molar_ratio": 0.0456140350877193,
    "anion_ratio": 0.08071486449986495,
    "temperature": -40.0,
    "conductivity": 4.413599968395286,
    "conductivity_mask": false,
    "anion_ratio_mask": true
}
```