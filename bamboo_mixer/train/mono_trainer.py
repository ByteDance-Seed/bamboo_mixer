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

import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from bamboo_mixer.data import MonoDataset, collate_data
from bamboo_mixer.predictor import MonoPredictor
from bamboo_mixer.train.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MonoTrainer(BaseTrainer):

    def _init_model(self):

        ckpt = self.config.model.pop("ckpt_path", None)
        self.model = MonoPredictor(**self.config.model)
        self.model.to(self.device)
        return ckpt

    def _get_dataset_class(self):
        return MonoDataset

    def _train_valid_split(self, dataset: MonoDataset, config: dict):
        if data_nums := config.get('data_num', None):
            dataset = dataset[:data_nums]
            logger.info(f'pruned data, used the first {data_nums} items')

        # this process takes a lot of memory
        # reason for grouping:
        # if one molecule has labels at different temperatures,
        # these are split as several items in the dataset sharing the same 'mol_name'
        # simple random_split would put some into train and some into valid
        # here we first group by names and then split according to names

        logger.info('started grouping dataset by "mol_name"')
        # Step 1: Group dataset by "name"
        grouped_data = defaultdict(list)
        for data in dataset:
            grouped_data[data["mol_name"]].append(data)

        # Convert the grouped data back into a list of lists
        grouped_dataset = list(grouped_data.values())

        # Step 2: Set the seed for reproducibility
        seed = self.config.meta.get('dataset_seed', self.config.meta['random_seed'])
        self._set_seed(seed)

        # Step 3: Shuffle the grouped dataset if specified
        if shuffle := config.get('shuffle', True):
            import random
            random.shuffle(grouped_dataset)

        # Step 4: Split the dataset into training and validation sets
        train_ratio = config['train_ratio']
        assert isinstance(train_ratio, float) and 0. < train_ratio < 1.

        # Calculate the number of groups for training
        train_num = round(len(grouped_dataset) * train_ratio)
        train_groups = grouped_dataset[:train_num]
        valid_groups = grouped_dataset[train_num:]

        # Flatten the grouped lists back into a single dataset
        train_ds = [item for group in train_groups for item in group]
        valid_ds = [item for group in valid_groups for item in group]

        # Step 5: Create DataLoaders
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

        logger.info('completed splitting')

        return train_dl, valid_dl, int(len(train_ds) / config['batch_size'])

    def _load_data(self, dataset_config: list[dict]) -> tuple[list[MonoDataset], list[DataLoader], list[DataLoader]]:
        datasets = []
        train_dls, valid_dls, epoch_steps = [], [], []
        for i, config in enumerate(dataset_config):
            dataset = MonoDataset(config=config['config'])
            datasets.append(dataset)
            train_dl, valid_dl, train_step = self._train_valid_split(dataset, config)
            train_dls.append(train_dl)
            valid_dls.append(valid_dl)
            epoch_steps.append(train_step)
            self.logger.info(f"dataset {i}, num data: {len(dataset)}")
        min_step = torch.tensor(min(epoch_steps), dtype=torch.int32, device=self.device)
        self.epoch_step_num = min_step.item()
        return datasets, train_dls, valid_dls
