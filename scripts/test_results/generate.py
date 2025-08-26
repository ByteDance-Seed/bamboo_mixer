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
import copy
import json
import os
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from formula_design.data import FormulaData, collate_data
from formula_design.generator import FormulaDecoder, FormulaDiffusion
from formula_design.mol import Molecule
from formula_design.predictor import MolMix, Mono
from formula_design.utils import setup_default_logging, to_dense_batch

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='test local')
parser.add_argument('--conf', type=str, default='test_config.yaml')
args = parser.parse_args()

outputs = {}


def make_hook(output_store, layer_name):

    def hook_fn(module, input, output):
        # Convert to list for JSON serialization
        output_store["layer_outputs"][layer_name] = output

    return hook_fn


def use_one_layer(model, layer_name, emb, molar_ratios):
    submodule = dict(model.named_modules())[layer_name]
    if not isinstance(submodule, torch.nn.Module):
        raise ValueError(f"Layer {layer_name} not found in the model.")
    return submodule(emb, molar_ratios)


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
    solv_mapped_smiles = [
        Molecule.from_smiles(list(
            solvents_map.values())[i]).get_mapped_smiles() for i in solv_index
    ]
    salt_mapped_smiles = [
        Molecule.from_smiles(list(salts_map.values())[i]).get_mapped_smiles()
        for i in salt_index
    ]
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


def cond_diffusion(conductivities, anion_ratios, diff_config, output_dir,
                   num_batch, batch_size):

    ckpt_path = diff_config.pop("ckpt_path")
    model_config = diff_config.pop("model_config")

    model = FormulaDiffusion(**model_config)
    model.load_ckpt(os.path.join(ckpt_path, "diffusion.pt"))
    device = "cuda"
    model.to(device)

    for c in conductivities:
        for a in anion_ratios:

            logger.info(
                f"Generating samples for conductivity={c}, anion ratio={a}")
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
            batch_x0 = {
                'frm_emb': torch.cat(batch_x0, dim=0)
            }  # Concatenate along batch dimension

            res = batch_x0
            torch.save(res, os.path.join(output_dir, f"gen_emb_{c}_{a}.pt"))

    logger.info("################################")
    logger.info("Generation complete.")
    logger.info("################################")


def decode(decoder_config, output_dir):

    ckpt_path = decoder_config.pop("ckpt_path")
    model_config = decoder_config.pop("model_config")
    model = FormulaDecoder(**model_config)
    model.load_ckpt(os.path.join(ckpt_path, "decoder.pt"))
    model.to("cuda")

    for file in os.listdir(output_dir):
        if file.startswith("gen_emb"):
            gen_emb_path = os.path.join(output_dir, file)

            output_path = os.path.join(output_dir,
                                       file.replace("gen_emb", "bow"))

            data = torch.load(gen_emb_path)["frm_emb"]

            data = data.view(-1, 384)
            data = data.to("cuda")

            bow_vecs = model.predict(data)["bow_vec"]

            torch.save(bow_vecs, output_path)

    logger.info("################################")
    logger.info("Decoding complete.")
    logger.info("################################")


def predict(pred_config, output_dir, batch_size, temperature, concentration):

    dict_dir = pred_config.pop("dict_dir")
    with open(os.path.join(dict_dir, "solvents.json"), "r") as f:
        solvents = json.load(f)
    with open(os.path.join(dict_dir, "salts.json"), "r") as f:
        salts = json.load(f)

    num_solvents = len(solvents)

    ckpt_path = pred_config.pop("ckpt_path")
    model_config = pred_config.pop("model_config")
    pretrain_config = model_config.pop("pretrain")
    pretrained_model = Mono(**pretrain_config["model"])
    pretrained_model.load_ckpt(os.path.join(ckpt_path, "pretrain.pt"))
    model_config = model_config.pop("model")

    model = MolMix(pretrained_model, **model_config)
    model.load_ckpt(os.path.join(ckpt_path, "predictor.pt"))
    model.to("cuda")

    for file in os.listdir(output_dir):
        if file.startswith("bow"):

            logger.info(
                f"Predicting conductivities and anion ratio for generated samples for {file}"
            )
            bow_path = os.path.join(output_dir, file)

            output_path = os.path.join(
                output_dir,
                file.replace("bow", "output").replace(".pt", ".json"))

            bow_vecs = torch.load(bow_path)
            bow_vecs = bow_vecs.squeeze().to("cuda")
            bow_vecs = torch.where(bow_vecs > 0.02, bow_vecs, 0)
            bow_left = F.normalize(bow_vecs[:, :num_solvents], p=1, dim=-1)
            bow_right = F.normalize(bow_vecs[:, num_solvents:], p=1, dim=-1)

            #Concatenate to avoid in-place operations
            bow_vecs = torch.cat([bow_left, bow_right], dim=-1)

            solvents_map = {
                solvents[i]['name']: solvents[i]['smiles']
                for i in range(len(solvents))
            }
            salts_map = {
                salts[i]['name']: salts[i]['smiles']
                for i in range(len(salts))
            }

            logger.info(f"Converting bow vectors to Formuladata")
            data_list = []
            for i in range(bow_vecs.shape[0]):
                data_list.append(
                    bow2graph(bow_vecs[i], solvents_map, salts_map,
                              temperature, concentration))

            dataset = DataLoader(data_list,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=collate_data)

            pred_conductivities = []
            pred_anion_ratios = []

            for batch_idx, graph in enumerate(dataset):

                graph = graph.to("cuda")
                pred = model(graph)

                conductivity = torch.exp(pred["conductivity"].squeeze(-1) *
                                         8.3344 - 4.6002)
                anion_ratio = pred["anion_ratio"].squeeze(-1)

                for c in conductivity:
                    pred_conductivities.append(c.item())
                for a in anion_ratio:
                    pred_anion_ratios.append(a.item())

                logger.info(
                    f"Predictive model finishing prediction of batch {batch_idx}"
                )

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


if __name__ == "__main__":

    assert os.path.exists(args.conf) and args.conf.endswith(
        '.yaml'), f'yaml config {args.conf} not found.'

    with open(args.conf, 'r') as f:
        config = yaml.safe_load(f)

    ckpt_path = config["ckpt_path"]

    diff_config = config["diffusion_config"]
    diff_config["ckpt_path"] = ckpt_path
    decoder_config = config["decoder_config"]
    decoder_config["ckpt_path"] = ckpt_path
    pred_config = config["predictor_config"]
    pred_config["ckpt_path"] = ckpt_path

    conductivities = config["conductivities"]
    anion_ratios = config["anion_ratios"]
    temperature = config["temperature"]
    temperature = (temperature + 273.15) / 373.15
    concentration = config["concentration"]
    output_dir = config["output_dir"]
    num_batch = config["num_batch"]
    batch_size = config["batch_size"]

    os.makedirs(output_dir, exist_ok=True)

    cond_diffusion(conductivities, anion_ratios, diff_config, output_dir,
                   num_batch, batch_size)
    decode(decoder_config, output_dir)
    predict(pred_config, output_dir, batch_size, temperature, concentration)
