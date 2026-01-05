# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import typing as T

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from bamboo_mixer.data import FormulaData, collate_data
from bamboo_mixer.generator import FormulaDecoder, FormulaDiffusion
from bamboo_mixer.mol import Molecule
from bamboo_mixer.predictor import FormulaPredictor, MonoPredictor
from bamboo_mixer.utils import get_data_file_path, get_default_device, setup_default_logging

logger = setup_default_logging()

device = get_default_device()

parser = argparse.ArgumentParser(description='test generator')
parser.add_argument('--conf', type=str, default='generator_config.yaml')
args = parser.parse_args()

outputs = {}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def bow2graph(bow_vec, solvents_map, salts_map, temperature, concentration):
    """
    bow_vec: torch.Tensor, shape (vocab_size)
    solvents_map: dict
    salts_map: dict
    """

    num_solvs = len(solvents_map)
    num_salts = len(salts_map)
    solv_index = [i for i in range(num_solvs) if bow_vec[i] > 0]
    salt_index = [i for i in range(num_salts) if bow_vec[i + num_solvs] > 0]
    solv_names = [list(solvents_map.keys())[i] for i in solv_index]
    salt_names = [list(salts_map.keys())[i] for i in salt_index]
    solv_mapped_smiles = [Molecule.from_smiles(list(solvents_map.values())[i]).get_mapped_smiles() for i in solv_index]
    salt_mapped_smiles = [Molecule.from_smiles(list(salts_map.values())[i]).get_mapped_smiles() for i in salt_index]
    solv_molar_ratios = [bow_vec[i].item() for i in solv_index]
    salt_molar_ratios = [bow_vec[i + num_solvs].item() for i in salt_index]

    return FormulaData(solv_names,
                       solv_mapped_smiles,
                       salt_names,
                       salt_mapped_smiles,
                       solv_molar_ratios,
                       salt_molar_ratios,
                       temperature=temperature,
                       concentration=concentration)


def cond_diffusion(conductivities: T.Iterable, anion_ratios: T.Iterable, diff_config: dict, output_dir: str,
                   num_batch: int, batch_size: int):

    ckpt_path = diff_config.pop("ckpt_path")
    model_config = diff_config.pop("model_config")

    model = FormulaDiffusion(**model_config)
    model.load_ckpt(ckpt_path, device=device)

    for c in conductivities:
        for a in anion_ratios:

            logger.info(f"Generating samples for conductivity={c}, anion ratio={a}")
            ratio = a * torch.ones((batch_size, 1)).to(device)
            cond = c * torch.ones((batch_size, 1)).to(device)
            cond = (torch.log(cond) + 4.6002) / 8.3344  # Normalize condition
            prop = torch.cat([cond, ratio], dim=1)

            batch_x0 = []

            for i in range(num_batch):
                set_seed(i)  # Ensure reproducibility
                x0, _ = model.sample(prop)  # Generate samples

                batch_x0.append(x0["frm_emb"])

            # Stack results across batches for this `c`
            batch_x0 = {'frm_emb': torch.cat(batch_x0, dim=0)}  # Concatenate along batch dimension

            res = batch_x0
            torch.save(res, os.path.join(output_dir, f"gen_emb_{c}_{a}.pt"))

    logger.info("################################")
    logger.info("Generation complete.")
    logger.info("################################")


def decode(decoder_config, output_dir):

    ckpt_path = decoder_config.pop("ckpt_path")
    model_config = decoder_config.pop("model_config")
    model = FormulaDecoder(**model_config)
    model.load_ckpt(ckpt_path, device=device)

    for file in os.listdir(output_dir):
        if file.startswith("gen_emb"):
            gen_emb_path = os.path.join(output_dir, file)

            output_path = os.path.join(output_dir, file.replace("gen_emb", "bow"))

            data = torch.load(gen_emb_path)["frm_emb"]

            data = data.view(-1, 384)

            bow_vecs = model.predict(data)["bow_vec"]

            torch.save(bow_vecs, output_path)

    logger.info("################################")
    logger.info("Decoding complete.")
    logger.info("################################")


def predict(pred_config, output_dir, batch_size, temperature, concentration):

    with open(get_data_file_path("solvs.json", "bamboo_mixer.generator"), "r") as f:
        solvents = json.load(f)
    with open(get_data_file_path("salts.json", "bamboo_mixer.generator"), "r") as f:
        salts = json.load(f)

    num_solvents = len(solvents)

    logger.info('loading mono model')
    mono_config = pred_config.pop('mono')
    mono_predictor = MonoPredictor(**mono_config['model_config'])
    mono_predictor.load_ckpt(mono_config['ckpt_path'], device=device)
    mono_predictor.eval()
    logger.info('loaded mono model')

    logger.info('loading formula model')
    formula_config = pred_config.pop('formula')
    formula_predictor = FormulaPredictor(mono_predictor, **formula_config, train_mode='finetune')
    formula_predictor.load_ckpt(pred_config['ckpt_path'], device)
    formula_predictor.eval()
    logger.info('loaded formula model')

    for file in os.listdir(output_dir):
        if file.startswith("bow"):

            logger.info(f"Predicting conductivities and anion ratio for generated samples for {file}")
            bow_path = os.path.join(output_dir, file)

            output_path = os.path.join(output_dir, file.replace("bow", "output").replace(".pt", ".json"))

            bow_vecs = torch.load(bow_path, map_location=device)
            bow_vecs = bow_vecs.squeeze()
            bow_vecs = torch.where(bow_vecs > 0.02, bow_vecs, 0)
            bow_left = F.normalize(bow_vecs[:, :num_solvents], p=1, dim=-1)
            bow_right = F.normalize(bow_vecs[:, num_solvents:], p=1, dim=-1)

            #Concatenate to avoid in-place operations
            bow_vecs = torch.cat([bow_left, bow_right], dim=-1)

            solvents_map = {solvents[i]['name']: solvents[i]['smiles'] for i in range(len(solvents))}
            salts_map = {salts[i]['name']: salts[i]['smiles'] for i in range(len(salts))}

            logger.info(f"Converting bow vectors to Formuladata")
            data_list = []
            for i in range(bow_vecs.shape[0]):
                data_list.append(bow2graph(bow_vecs[i], solvents_map, salts_map, temperature, concentration))

            dataset = DataLoader(data_list,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=collate_data)

            pred_conductivities = []
            pred_anion_ratios = []

            for batch_idx, batch_formula in enumerate(dataset):
                batch_formula = batch_formula.to(device)
                pred = formula_predictor(batch_formula)

                conductivity = torch.exp(pred["conductivity"].squeeze(-1) * 8.3344 - 4.6002)
                anion_ratio = pred["anion_ratio"].squeeze(-1)

                for c in conductivity:
                    pred_conductivities.append(c.item())
                for a in anion_ratio:
                    pred_anion_ratios.append(a.item())

                logger.info(f"Predictive model finishing prediction of batch {batch_idx}")

            entries = []
            for i in range(len(bow_vecs)):
                entry = {
                    "bow": bow_vecs[i].detach().cpu().numpy().tolist(),
                    "temperature": 25,
                    "conductivity": pred_conductivities[i],
                    "anion_ratio": pred_anion_ratios[i],
                }

                entries.append(entry)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as file:
                json.dump(entries, file, indent=4)

    logger.info("################################")
    logger.info("Prediction complete.")
    logger.info("################################")


def main():
    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'

    with open(args.conf, 'r') as f:
        config = yaml.safe_load(f)

    diff_config = config["diffusion_config"]
    decoder_config = config["decoder_config"]
    pred_config = config["predictor_config"]

    conductivities = config["conductivities"]
    anion_ratios = config["anion_ratios"]
    temperature = config["temperature"]
    temperature = (temperature + 273.15) / 373.15
    concentration = config["concentration"]
    output_dir = config["output_dir"]
    num_batch = config["num_batch"]
    batch_size = config["batch_size"]

    os.makedirs(output_dir, exist_ok=True)

    cond_diffusion(conductivities, anion_ratios, diff_config, output_dir, num_batch, batch_size)
    decode(decoder_config, output_dir)
    predict(pred_config, output_dir, batch_size, temperature, concentration)


if __name__ == "__main__":
    main()
