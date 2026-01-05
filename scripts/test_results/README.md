# Inference

## Configuration file

- configuration files are stored as yaml files under [config/test_results](../../config/test_results/)

## Molecular property prediction

```
python3 scripts/test_results/predict_mono.py \
  --conf config/test_results/mono_config.yaml
```

## Electrolyte property prediction

```
python3 scripts/test_results/predict_formula.py \
  --conf config/test_results/predict_config.yaml
```

## Formulation generation and evaluation

```
python3 scripts/test_results/generate.py \
  --conf config/test_results/generate_config.yaml
```

This will generate three files.
The specific molecules corresponding to the index can be found in the volcabulary [salts.json](bamboo_mixer/generator/salts.json) and [solvs.json](bamboo_mixer/generator/solvs.json):

1. gen_emb.pt: generated electrolyte embedding.
2. bow.pt: converted BoM vectors.
3. output.json: predicted properties of generated formulation.
