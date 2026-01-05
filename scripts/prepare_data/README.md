# Data preparation

- configuration files are stored as yaml files under [config/prepare_data](./config/prepare_data/)

## Training data configuration file

### Single molecule dataset

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/train_predictor_mono_config.yaml \
  --data_type mono
```

### Formula MD pretrain dataset

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/train_predictor_pretrain_config.yaml \
  --data_type formula
```

### Formula MD+Experimental dataset

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/train_predictor_finetune_config.yaml \
  --data_type formula
```

### Formula synthetic dataset

First preprocess the json so the embedding.h5 file can be generated

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/synthetic_preprocess_config.yaml \
  --data_type formula
```

Then run the following command to generate files necessary for diffusion and decoder training

```
python3 scripts/prepare_data/embedding.py \
  --conf config/prepare_data/synthetic_embedding_config.yaml
```

You get 3 files from the above command:

- `embedding.h5`
- `solv_embs.pt`
- `salt_embs.pt`

Then preprocess again with the embedding.h5 file for diffusion and decoder training

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/train_generator_diffusion_decoder_config.yaml \
  --data_type formula
```

## Inference data configuration file

### Single molecule dataset

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/inference_predictor_mono_config.yaml \
  --data_type mono
```

### Formula dataset

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/inference_predictor_formula_config.yaml \
  --data_type formula
```
