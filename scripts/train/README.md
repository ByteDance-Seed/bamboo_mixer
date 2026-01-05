# Training

## Configuration file

- configuration files are stored as yaml files under [config/train](config/train/)

## Molecular pretraining

```
python3 scripts/train/train.py \
  --conf config/train/predictor_mono_config.yaml \
  --task mono
```

## Computational pretraining

```
python3 scripts/train/train.py \
  --conf config/train/predictor_pretrain_config.yaml \
  --task formula
```

## Experimental fine-tuning

```
python3 scripts/train/train.py \
  --conf config/train/predictor_finetune_config.yaml \
  --task formula
```

## Conditional diffusion training

```
python3 scripts/train/train.py \
  --conf config/train/generator_diffusion_config.yaml \
  --task formula
```

## Decoder training

```
python3 scripts/train/train.py \
  --conf config/train/generator_decoder_config.yaml \
  --task formula
```
