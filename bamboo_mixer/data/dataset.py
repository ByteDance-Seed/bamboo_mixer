# The code is adapted from https://github.com/bytedance/byteff/blob/master/byteff/data/dataset.py

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

import copy
import json
import logging
import os
import random
import traceback
from typing import Any, Union

import h5py
import torch
import yaml
from torch.utils.data import Dataset

from bamboo_mixer.data.data import FormulaData, MonoData, _MixData
from bamboo_mixer.mol import Molecule
from bamboo_mixer.utils import get_timestamp

logger = logging.getLogger(__name__)


class DatasetConfig:

    def __init__(self, config: Union[str, dict, Any] = None):

        # default config
        self._config = {
            'json_path': '',
            'save_dir': '',  # path to json and h5 file
            # 'data_cls': 'Data',
            'key_map': {
                'temperature': 'temperature',
            },  # argument: dataset name
            'kwargs': {}  # other kwargs for Data
        }

        custom_config: dict[str, dict] = None

        if isinstance(config, dict):
            custom_config = config
        elif isinstance(config, str):
            with open(config) as file:
                custom_config = yaml.safe_load(file)
        elif config is not None:
            raise TypeError(f"Type {type(config)} is not allowed.")

        if custom_config is not None:
            for k in self._config:
                if k in custom_config:
                    self._config[k] = copy.deepcopy(custom_config[k])

    def __str__(self) -> str:
        return self._config.__str__()

    def get(self, name: str):
        return self._config[name]

    def to_yaml(self, save_path=None, timestamp=True):
        if save_path is None:
            save_path = os.path.join(self.get('save_dir'), 'dataset_config.yaml')
        else:
            assert save_path.endswith('.yaml')

        if timestamp:
            self._config['timestamp'] = get_timestamp()

        with open(save_path, 'w') as file:
            yaml.dump(self._config, file)


class MonoDataset(Dataset):

    def __init__(self, config: Union[str, dict], processing: bool = False):
        super().__init__()

        self.config = DatasetConfig(config)

        self.data_list: list = []

        if not processing:
            self.check_exist()
            self.load()

    def copy(self) -> 'MonoDataset':
        new_dataset = MonoDataset(config=self.config._config, processing=True)
        new_dataset.data_list = self.data_list.copy()
        return new_dataset

    def __len__(self):
        return len(self.data_list)

    @property
    def processed_names(self) -> list[str]:
        return [os.path.join(self.config.get('save_dir'), f'processed_data.pkl')]

    def check_exist(self):
        assert all([os.path.exists(name) for name in self.processed_names])

    def __getitem__(self, index: Union[int, slice, Any]) -> Union[MonoData, list[MonoData]]:
        if isinstance(index, int):
            return self.data_list[index]
        elif isinstance(index, slice):
            ret = self.copy()
            ret.data_list = ret.data_list[index]
            return ret
        else:
            raise TypeError(f'index of type {type(index)} is not allowed.')

    def shuffle(self):
        ret = self.copy()
        random.shuffle(ret.data_list)
        return ret

    def load(self):
        self.data_list = []
        # assert self.config.get('data_cls') == 'MonoData'
        for fname in self.processed_names:
            self.data_list += [MonoData.from_dict(d) for d in torch.load(fname)]

    @classmethod
    def process(cls, config: str):

        ds = cls(config, processing=True)
        logger.info(f'processing data')

        save_dir = ds.config.get('save_dir')
        # assert ds.config.get('data_cls') == 'MonoData'

        json_path = os.path.join(save_dir, 'processed_data.json')

        with open(json_path, 'r') as file:
            molecules = json.load(file)

        data_list = []
        mol_idx = 0
        for mol in molecules:
            if "." not in mol["smiles"]:
                mol["smiles"] = Molecule.from_smiles(
                    mol["smiles"]).get_mapped_smiles()  # convert smiles to mapped smiles
                mol_idx += 1
                if len(data_list) % 1000 == 0:
                    logger.info(f'finished molecules {len(data_list)}')

                data_dict = {k: mol[v] for k, v in ds.config.get('key_map').items()}

                try:

                    data = MonoData(mol["name"], mol["smiles"], **data_dict)
                except:  # pylint: disable=bare-except
                    logger.warning(f'failed molecule: {mol_idx}, skip!')
                    logger.warning(traceback.format_exc())
                    continue

                data_list.append(data)
        os.makedirs(os.path.dirname(ds.processed_names[0]), exist_ok=True)
        torch.save([dict(d) for d in data_list], ds.processed_names[0])
        ds.config.to_yaml()

        logger.info(f'finished data processing')


class FormulaDataset(Dataset):

    def __init__(self, config: Union[str, dict], processing: bool = False):
        super().__init__()

        self.config = DatasetConfig(config)

        self.data_list: list = []

        if not processing:
            self.check_exist()
            self.load()

    def copy(self) -> 'FormulaDataset':
        new_dataset = FormulaDataset(config=self.config._config, processing=True)
        new_dataset.data_list = self.data_list.copy()
        return new_dataset

    def __len__(self):
        return len(self.data_list)

    @property
    def processed_names(self) -> list[str]:
        return [os.path.join(self.config.get('save_dir'), f'processed_data.pkl')]

    def check_exist(self):
        assert all([os.path.exists(name) for name in self.processed_names])

    def __getitem__(self, index: Union[int, slice, Any]) -> Union[FormulaData, list[FormulaData]]:
        if isinstance(index, int):
            return self.data_list[index]
        elif isinstance(index, slice):
            ret = self.copy()
            ret.data_list = ret.data_list[index]
            return ret
        else:
            raise TypeError(f'index of type {type(index)} is not allowed.')

    def shuffle(self):
        ret = self.copy()
        random.shuffle(ret.data_list)
        return ret

    def load(self):
        self.data_list = []
        # assert self.config.get('data_cls') == 'FormulaData'
        with torch.serialization.safe_globals([_MixData, FormulaData]):  # this is necessary for torch >= 2.6
            for fname in self.processed_names:
                self.data_list += [FormulaData.from_dict(d) for d in torch.load(fname)]

    @classmethod
    def process(cls, config: str):

        ds = cls(config, processing=True)
        logger.info(f'processing data')

        save_dir = ds.config.get('save_dir')
        include_emb = ds.config.get("kwargs")["include_emb"]
        # data_cls = getattr(bdata, ds.config.get('data_cls'))
        # assert issubclass(data_cls, Data)

        json_path = os.path.join(save_dir, 'processed_data.json')

        with open(json_path, 'r') as file:
            formulas = json.load(file)

        if include_emb:
            h5_path = ds.config.get("kwargs")["emb_h5_path"]
            # h5_path = os.path.join(save_dir, 'embedding.h5')
            mol_embs = []
            molar_ratios = []
            frm_embs = []
            bow_vecs = []
            with h5py.File(h5_path, "r") as f:
                for i in range(len(formulas)):
                    key = f"data{i+1}"
                    mol_embs.append(f[key]["mol_emb"][:])  # molecule embedding of all solvents and salts
                    molar_ratios.append(f[key]["mol_ratio"][:])  # molar ratio of all solvents and salts
                    frm_embs.append(f[key]["frm_emb"][:])  # formula embedding (including solvents and salts)
                    bow_vecs.append(f[key]["bow"][:])  # bow vector

        data_list = []
        formula_idx = 0
        for formula in formulas:
            formula_idx += 1
            if len(data_list) % 100 == 0:
                logger.info(f'finished formula {len(data_list)}')

            solv_names = [f["name"] for f in formula["solvents"]]
            solv_mapped_smiles = [Molecule.from_smiles(f["smiles"]).get_mapped_smiles() for f in formula["solvents"]]
            solv_molar_ratios = [f["molar_ratio"] for f in formula["solvents"]]

            salt_names = [f["name"] for f in formula["salts"]]
            salt_mapped_smiles = [Molecule.from_smiles(f["smiles"]).get_mapped_smiles() for f in formula["salts"]]
            salt_molar_ratios = [f["molar_ratio"] for f in formula["salts"]]

            data_dict = {k: formula[v] for k, v in ds.config.get('key_map').items()}

            data = FormulaData(
                solv_names,
                solv_mapped_smiles,
                salt_names,
                salt_mapped_smiles,
                solv_molar_ratios,
                salt_molar_ratios,  # molecule info
                mol_emb=mol_embs[formula_idx - 1] if include_emb else None,
                frm_emb=frm_embs[formula_idx - 1] if include_emb else None,
                all_molar_ratio=molar_ratios[formula_idx - 1] if include_emb else None,
                bow_vec=bow_vecs[formula_idx - 1] if include_emb else None,  # embedding info
                **data_dict)

            data_list.append(data)
        os.makedirs(os.path.dirname(ds.processed_names[0]), exist_ok=True)
        torch.save([dict(d) for d in data_list], ds.processed_names[0])
        ds.config.to_yaml()

        logger.info(f'finished data processing')
