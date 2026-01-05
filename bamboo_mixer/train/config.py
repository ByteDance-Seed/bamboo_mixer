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

import os
from copy import deepcopy
from glob import glob
from typing import Any, Union

import yaml


class TrainConfig:

    def __init__(self, config: Union[str, dict, Any] = None, make_working_dir: bool = True):

        self._config = {
            'meta': {
                'work_folder': '',
                'random_seed': 42,
                'fp64': False
            },
            'dataset': [{
                'config': '',
                'batch_size': 10,
                'train_ratio': 0.9,
                'shuffle': True,
                'loss': {},
                'num_workers': 4,
                'prefetch_factor': 4
            }],
            'model': {},
            'training': {
                'max_epoch': 999,
                'valid_interval': 5,
                'ckpt_interval': 20,
                'optimizer': 'RAdam',
                'optimizer_params': {},
                'amp': False
            }
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
                    if k == 'dataset':
                        defaul_config: dict = self._config[k][0]
                        for i in range(len(custom_config[k])):
                            (new_config := deepcopy(defaul_config)).update(custom_config[k][i])
                            custom_config[k][i] = new_config
                        self._config[k] = custom_config[k]
                    else:
                        self._config[k].update(custom_config[k])

        self.meta: dict = self._config['meta']
        self.dataset: list[dict] = self._config['dataset']
        self.model = self._config['model']
        self.model_type = self._config['model'].pop('model_type', None)  # only used in formula training

        self.training: dict = self._config['training']

        self.work_folder = self.meta['work_folder']
        self.ckpt_folder = os.path.join(self.work_folder, "ckpt")
        if make_working_dir:
            assert not os.path.exists(self.work_folder), self.work_folder
            os.makedirs(self.ckpt_folder, exist_ok=True)
            self.to_yaml()

    def to_yaml(self, save_path: Union[str, None] = None):
        if save_path is None:
            save_path = os.path.join(self.work_folder, 'trainer_config_in_use.yaml')
        else:
            assert save_path.endswith('.yaml')
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file)

    def get_latest_ckpt(self) -> str:
        paths = glob(os.path.join(self.ckpt_folder, "ckpt_epoch_*.pt"))
        if not paths:
            return None
        else:
            epochs = [int(fp.split('_')[-1].split('.')[0]) for fp in paths]
            path = os.path.join(self.ckpt_folder, f"ckpt_epoch_{max(epochs)}.pt")
            return path

    @property
    def finish_flag(self):
        return os.path.join(self.work_folder, 'FINISHED')

    def optimal_path(self, label='') -> str:
        if label:
            return os.path.join(self.work_folder, f'optimal_{label}.pt')
        else:
            return os.path.join(self.work_folder, 'optimal.pt')
