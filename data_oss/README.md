This folder contains json files matching the NMI paper.

| Stage                        | Train                         | Inference              |
| ---------------------------- | ----------------------------- | ---------------------- |
| Prediction: Single Molecule  | mono_data.json                | mono_inference.json    |
| Prediction: Formula Pretrain | formula_pretrain_md_data.json | NA                     |
| Prediction: Formula Finetune | formula_finetune_data.json    | formula_inference.json |
| Design: Formula              | synthetic_data.json           | NA                     |

Note 1: for formula model finetuning, both MD pretrain data (for anion ratio) and experimental data (for conductivity) are used.
Therefore `formula_finetune_data.json = formula_pretrain_md_data.json + formula_finetune_exp_data.json`.

Note 2: for training the design models (diffusion and decoder), the files `salt_embs.pt` and `solv_embs.pt` are required. The two `_embs.pt` files provided here are specifically paired with the open-sourced model checkpoint available on HuggingFace and assume a fixed indexing and label mapping defined in `synthetic_data.json`.

If you retrain the model using your own molecular library or synthetic data, you must relabel `synthetic_data.json` to match your salt/solvent vocabulary and regenerate the corresponding `salt_embs.pt` and `solv_embs.pt`. Reusing the provided `_embs.pt` files with a relabeled or custom `synthetic_data.json` will lead to incorrect index–embedding alignment.
