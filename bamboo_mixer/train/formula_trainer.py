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

import torch
from torch.utils.data import DataLoader

from bamboo_mixer.data import FormulaDataset, collate_data
from bamboo_mixer.generator import FormulaDecoder, FormulaDiffusion
from bamboo_mixer.predictor import FormulaPredictor, MonoPredictor
from bamboo_mixer.train.trainer import BaseTrainer


class FormulaTrainer(BaseTrainer):

    def _init_model(self):

        ckpt = self.config.model.pop("ckpt_path", None)

        if self.config.model_type == 'predictor':
            mono_config = self.config.model.pop('mono')
            mono_gnn = MonoPredictor(**mono_config['model_config'])
            mono_gnn.load_ckpt(mono_config['ckpt_path'], device=self.device)
            self.model = FormulaPredictor(mono_gnn,
                                          **self.config.model['formula'],
                                          train_mode=self.config.model['train_mode'])
            if self.config.model['train_mode'] == 'finetune':
                assert ckpt is not None and os.path.exists(ckpt), f'{ckpt} is required for finetuning'
        elif self.config.model_type == 'diffusion':
            self.model = FormulaDiffusion(**self.config.model['diffusion_model_config'])
        elif self.config.model_type == 'decoder':
            self.model = FormulaDecoder(**self.config.model['decoder_model_config'])
        else:
            raise NotImplementedError

        self.model.to(self.device)
        return ckpt

    def _get_dataset_class(self):
        return FormulaDataset

    def _train_valid_split(self, dataset: FormulaDataset, config: dict):
        if data_nums := config.get('data_num', None):
            dataset = dataset[:data_nums]
        seed = self.config.meta.get('dataset_seed', self.config.meta['random_seed'])
        self._set_seed(seed)
        if shuffle := config.get('shuffle', True):
            dataset = dataset.shuffle()
        train_ratio = config['train_ratio']
        assert isinstance(train_ratio, float) and 0. < train_ratio < 1.
        train_num = round(len(dataset) * train_ratio)
        train_ds, valid_ds = dataset[:train_num], dataset[train_num:]
        train_dl = DataLoader(train_ds,
                              batch_size=config['batch_size'],
                              shuffle=shuffle,
                              drop_last=True,
                              collate_fn=collate_data,
                              pin_memory=True,
                              prefetch_factor=config['prefetch_factor'],
                              num_workers=config['num_workers'])
        valid_dl = DataLoader(valid_ds,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              drop_last=False,
                              collate_fn=collate_data,
                              pin_memory=True,
                              prefetch_factor=config['prefetch_factor'],
                              num_workers=config['num_workers'])
        return train_dl, valid_dl, int(len(train_ds) / config['batch_size'])

    def _load_data(self, dataset_config: list[dict]) -> tuple[list[FormulaDataset], list[DataLoader], list[DataLoader]]:
        datasets = []
        train_dls, valid_dls, epoch_steps = [], [], []
        for i, config in enumerate(dataset_config):
            dataset = FormulaDataset(config=config['config'])
            datasets.append(dataset)
            train_dl, valid_dl, train_step = self._train_valid_split(dataset, config)
            train_dls.append(train_dl)
            valid_dls.append(valid_dl)
            epoch_steps.append(train_step)
            self.logger.info(f"dataset {i}, num data: {len(dataset)}")
        min_step = torch.tensor(min(epoch_steps), dtype=torch.int32, device=self.device)
        self.epoch_step_num = min_step.item()
        return datasets, train_dls, valid_dls
