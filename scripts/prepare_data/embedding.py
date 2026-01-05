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

# this file requires these input:
# 1. trained mono model
# 2. trained formula model, which uses 1 as its molecule module
# 3. synthetic_data.json generated using the trained formula model
# 4. given vocabulary of solvents and salts, which must be consistent with 1\2\3
# this file then generates these outputs:
# 1. embedding.h5 file used to train the diffusion model
# 2. solv_emb.pt used to train the decoder
# 3. salt_emb.pt used to train the decoder

# the generated embedding.h5 file explained:
# 1. the h5 file contains a list of data groups,
#       `data1` corresponds to item 0 in synthetic_data.json,
#       `data2` corresppnds to item 1, and so on.
# 2. each `dataXXX` item contains 6 arrays:
#    - `mol_emb`:   molecule embedding of all solvents and salts, with fixed size [24,D].
#                   the first [12,D] are solvents and the last [12,D] are salts, 'unused' slots are filled with zero
#    - `mol_ratio`: NOT-NORMALIZED molar ratio of all solvents and salts.
#                   The sum of molar ratio of all solvents is 1,
#                   and the sum of molar ratio of all salts is 1, respectively.
#    - `frm_emb`: formula embedding (including solvents and salts)
#    - `bow`: bag-of-words vector
#
#   these two are stored in the h5 file for reference, but not loaded for training.
#    - `conductivity`: conductivity value, NOT NORMALIZED to [0,1]
#    - `anion_ratio`: anion ratio, naturally normalized [0,1]

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool

from bamboo_mixer.data import FormulaData, FormulaDataset, MonoData, MonoDataset, collate_data
from bamboo_mixer.data.data import _MixData
from bamboo_mixer.predictor import FormulaPredictor, MonoPredictor
from bamboo_mixer.utils import get_data_file_path, get_default_device, setup_default_logging, to_dense_batch

logger = setup_default_logging()

device = get_default_device()
logger.info(f'using device: {device}')

parser = argparse.ArgumentParser(description='load config file to generate {salt,solv}_emb.pt and embedding.h5 files')
parser.add_argument('--conf', type=str, default='config.yaml')
args = parser.parse_args()

salts_json = get_data_file_path('salts.json', 'bamboo_mixer.generator')
solvs_json = get_data_file_path('solvs.json', 'bamboo_mixer.generator')


def parse_config():
    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'
    with open(args.conf, 'r') as f:
        config = yaml.safe_load(f)

    output_folder = config['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(os.path.join(output_folder, 'embedding.h5')):
        logger.warning(f'embedding.h5 already exists in {output_folder}, will be overwritten.')
    if os.path.exists(os.path.join(output_folder, 'salt_embs.pt')):
        logger.warning(f'salt_embs.pt already exists in {output_folder}, will be overwritten.')
    if os.path.exists(os.path.join(output_folder, 'solv_embs.pt')):
        logger.warning(f'solv_embs.pt already exists in {output_folder}, will be overwritten.')

    return config


def load_models(model_config: dict) -> (MonoPredictor, FormulaPredictor):
    logger.info('loading mono model')
    mono_config = model_config.pop('mono')
    mono_predictor = MonoPredictor(**mono_config['model_config'])
    mono_predictor.load_ckpt(mono_config['ckpt_path'], device=device)
    mono_predictor.eval()
    logger.info('loaded mono model')

    logger.info('loading formula model')
    formula_predictor = FormulaPredictor(mono_predictor, **model_config.pop('formula'), train_mode='finetune')
    formula_predictor.load_ckpt(model_config['ckpt_path'], device)
    formula_predictor.eval()
    logger.info('loaded formula model')

    return mono_predictor, formula_predictor


def load_data(data_config: dict):
    logger.info('loading dataset')
    dataset = FormulaDataset(config=data_config['config'])
    logger.info(f'loaded dataset with {len(dataset)} items')

    dl = DataLoader(
        dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,  # must be False, so the order match original json file
        drop_last=False,
        collate_fn=collate_data,
        pin_memory=True,
        prefetch_factor=data_config['prefetch_factor'],
        num_workers=data_config['num_workers'])
    logger.info('prepared data loader')

    return dl


def generate_mol_embs(mono_model: MonoPredictor, json_path: str, pt_path: str):
    basename = Path(json_path).stem
    save_dir = os.path.join('run/' + basename, 'mol_emb_preprocess')
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(json_path, os.path.join(save_dir, 'processed_data.json'))
    config = {"json_path": json_path, "save_dir": save_dir, "key_map": {'temperature': 'temperature'}}

    # step 1 convert MonoDataset
    MonoDataset.process(config)
    dataset = MonoDataset(config, processing=False)
    # step 2 construct a monodataset dataloader
    dl = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_data)

    # step 3 loop over the monodataset dataloader to get the molecule embeddings
    with torch.no_grad():
        batch: MonoData = next(iter(dl))
        batch = batch.to(device)
        results = mono_model.predict(batch, batch.temperature)
        torch.save(results['h_all'], pt_path)
    logger.info(f'saved {results["h_all"].shape[0]} molecule embeddings to {pt_path}')

    return


def aggr_func(data: _MixData, node_h: torch.Tensor, edge_h: torch.Tensor):
    """
    Aggregates node and edge features for molecules in a formula.

    This function performs global mean pooling on node and edge features to get
    molecule-level representations. It then converts these representations into
    dense tensors, padding them to a fixed size (12). The molar ratios are
    also converted to a dense tensor.

    Args:
        data: A _MixData object containing information about the molecules,
              including counts and molar ratios.
        node_h: A tensor of node features for all molecules in the batch.
        edge_h: A tensor of edge features for all molecules in the batch.

    Returns:
        A tuple containing:
        - node_h_formula: Dense tensor of aggregated node features.
        - edge_h_formula: Dense tensor of aggregated edge features.
        - molar_ratios: Dense tensor of molar ratios.
    """

    node_count, edge_count = data.counts[:, 0], data.counts[:, 1]

    node_batch_mol = torch.arange(node_count.size()[0], dtype=torch.int64, device=device).repeat_interleave(node_count)

    edge_batch_mol = torch.arange(edge_count.size()[0], dtype=torch.int64, device=device).repeat_interleave(edge_count)

    node_h_mol = global_mean_pool(node_h, node_batch_mol)
    edge_h_mol = global_mean_pool(edge_h, edge_batch_mol)

    # unsqueeze the batch size dimension to account for "interaction" between different component molecules in the electrolyte system
    node_count_formula, edge_count_formula = data.counts_mol[:, 0], data.counts_mol[:, 1]
    node_batch_formula = torch.arange(node_count_formula.size()[0], dtype=torch.int64,
                                      device=device).repeat_interleave(node_count_formula)
    edge_batch_formula = torch.arange(edge_count_formula.size()[0], dtype=torch.int64,
                                      device=device).repeat_interleave(edge_count_formula)

    node_h_formula, _ = to_dense_batch(node_h_mol, node_batch_formula, fill_value=0.0, max_num_nodes=12)
    edge_h_formula, _ = to_dense_batch(edge_h_mol, edge_batch_formula, fill_value=0.0, max_num_nodes=12)

    molar_ratios, _ = to_dense_batch(data.molar_ratios, node_batch_formula, fill_value=0.0, max_num_nodes=12)

    return node_h_formula, edge_h_formula, molar_ratios


def formula2bow(formula_data: FormulaData, molar_ratios: torch.Tensor):
    """ 
    Convert FormulaData to bow vectors 
    """
    solv_names = formula_data.solv_graphs.mol_name
    salt_names = formula_data.salt_graphs.mol_name
    solv_molar_ratios = molar_ratios[:, :12, :]
    salt_molar_ratios = molar_ratios[:, 12:, :]

    # fixed, change this if your vocabulary is larger
    with open(salts_json, 'r') as f:
        salts = json.load(f)
        num_salt = len(salts)
        salt_map = {salt['name'].upper(): i for i, salt in enumerate(salts)}
    with open(solvs_json, 'r') as f:
        solvs = json.load(f)
        num_solv = len(solvs)
        solv_map = {solv['name'].upper(): i for i, solv in enumerate(solvs)}

    bow_vec = torch.zeros((solv_molar_ratios.size()[0], num_solv + num_salt))

    for i, (solvs, mrs) in enumerate(zip(solv_names, solv_molar_ratios)):
        for solv, mr in zip(solvs, mrs):
            bow_vec[i][solv_map[solv.upper()]] = mr[0]

    for i, (salts, mrs) in enumerate(zip(salt_names, salt_molar_ratios)):
        for salt, mr in zip(salts, mrs):
            bow_vec[i][num_solv + salt_map[salt.upper()]] = mr[0]

    return bow_vec


def generate_formula_embs(formula_predictor: FormulaPredictor, formula_dl: DataLoader, output_path: str) -> None:
    h_invs = []
    mol_embs = []
    bows = []
    mol_ratios = []
    conductivities = []
    anion_ratios = []

    for batch_idx, batch_formula_data in enumerate(formula_dl):

        batch_formula_data: FormulaData = batch_formula_data.to(device)
        pred = formula_predictor(batch_formula_data, get_mol_embs=True)
        # molecule embs of a formula
        # Note that the max number of allowed species is 12 salts and 12 solvents
        solv_node_h = pred["solv_node_h"]
        solv_edge_h = pred["solv_edge_h"]
        salt_node_h = pred["salt_node_h"]
        salt_edge_h = pred["salt_edge_h"]

        solv_mol_node_h, solv_mol_edge_h, solv_molar_ratios = aggr_func(batch_formula_data.solv_graphs, solv_node_h,
                                                                        solv_edge_h)
        solv_mol_emb = torch.cat([solv_mol_node_h, solv_mol_edge_h], dim=-1)

        salt_mol_node_h, salt_mol_edge_h, salt_molar_ratios = aggr_func(batch_formula_data.salt_graphs, salt_node_h,
                                                                        salt_edge_h)
        salt_mol_emb = torch.cat([salt_mol_node_h, salt_mol_edge_h], dim=-1)

        mol_emb = torch.cat([solv_mol_emb, salt_mol_emb], dim=-2)
        mol_ratio = torch.cat([solv_molar_ratios, salt_molar_ratios], dim=-2)

        bow = formula2bow(batch_formula_data, mol_ratio)

        for ratio, emb, b in zip(mol_ratio, mol_emb, bow):
            mol_embs.append(emb.detach().cpu())
            mol_ratios.append(ratio.detach().cpu())
            bows.append(b.detach().cpu())

        conductivity = torch.exp(pred["conductivity"].squeeze(-1) * 8.3344 - 4.6002).unsqueeze(-1)
        # use formula embedding from model prediction
        h_inv = pred["h_inv"]
        # use anion ratio from synthetic data, not current model prediction
        # anion_ratio = pred["anion_ratio"]
        anion_ratio = batch_formula_data.anion_ratio

        for c, a, h in zip(conductivity, anion_ratio, h_inv):
            if c > 100:
                logger.warning(f'abnormal pred conductivity detected {c}, anion_ratio {a}')
                conductivities.append([10.0])  # use fake value
                anion_ratios.append(a.detach().cpu())
                h_invs.append(h.detach().cpu())  # this is formula embedding
            else:
                conductivities.append(c.detach().cpu())
                anion_ratios.append(a.detach().cpu())
                h_invs.append(h.detach().cpu())  # this is formula embedding

        logger.info(f"Finishing prediction of batch {batch_idx}")

    h_invs = np.array(h_invs)
    mol_embs = np.array(mol_embs)
    bows = np.array(bows)
    mol_ratios = np.array(mol_ratios)
    conductivities = np.array(conductivities)
    anion_ratios = np.array(anion_ratios)

    print(h_invs.shape)
    print(mol_embs.shape)
    print(bows.shape)
    print(mol_ratios.shape)
    print(conductivities.shape)
    print(anion_ratios.shape)
    print(np.mean(conductivities))

    assert output_path.endswith('.h5'), f'output_file {output_path} must be h5'

    if os.path.exists(output_path):
        logger.warning(f'h5 file already exists in {output_path}, will be overwritten.')

    with h5py.File(output_path, 'w') as f:
        idx = 0
        for mol_emb, frm_emb, mr, bow, cond, anion_ratio in zip(
                mol_embs,
                h_invs,
                mol_ratios,
                bows,
                conductivities,
                anion_ratios,
        ):
            idx += 1
            mol_group = f.create_group(f'data{idx}')
            mol_group.create_dataset("mol_emb", data=mol_emb)
            mol_group.create_dataset("frm_emb", data=frm_emb)
            mol_group.create_dataset("mol_ratio", data=mr)
            mol_group.create_dataset("bow", data=bow)
            mol_group.create_dataset("conductivity", data=cond)
            mol_group.create_dataset("anion_ratio", data=anion_ratio)

    logger.info(f"All embedding tensors saved to {output_path}")

    return


def main():
    config = parse_config()
    mono_predictor, formula_predictor = load_models(config['model'])
    # generate molecule embeddings
    generate_mol_embs(mono_predictor, salts_json, os.path.join(config['output_folder'], 'salt_embs.pt'))
    generate_mol_embs(mono_predictor, solvs_json, os.path.join(config['output_folder'], 'solv_embs.pt'))

    # generate formula embeddings and bow vectors
    formula_dl = load_data(config['dataset'])
    generate_formula_embs(formula_predictor, formula_dl, os.path.join(config['output_folder'], 'embedding.h5'))


if __name__ == "__main__":

    main()
