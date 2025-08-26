# Data preparation

## Configuration file
- configuration files are stored as yaml files under [config/prepare_data](../../config/prepare_data/) 

## Single molecule dataset
```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/mono_config.yaml \
  --data_type mono
```

## Electrolyte dataset
```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/formula_config.yaml \
  --data_type formula
```

## Train dataset
```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/train_data_config.yaml \
  --data_type formula
```


