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

import numpy as np
import yaml
from torch.utils.data import DataLoader

from bamboo_mixer.data import MonoData, MonoDataset, collate_data
from bamboo_mixer.predictor import MonoPredictor
from bamboo_mixer.utils import get_default_device, setup_default_logging

logger = setup_default_logging()

device = get_default_device()

parser = argparse.ArgumentParser(description='test local')
parser.add_argument('--conf', type=str, default='mono_config.yaml')
args = parser.parse_args()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'

    with open(args.conf, 'r') as f:
        test_conf = yaml.safe_load(f)

    output_path = test_conf.pop("output_path")
    ckpt_path = test_conf.pop("ckpt_path")
    model_config = test_conf.pop("model")
    model = MonoPredictor(**model_config)
    model.load_ckpt(ckpt_path, device=device)
    model.eval()

    dataset = MonoDataset(test_conf)
    valid_dl = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_data)

    properties = ['Tm', 'bp', 'nD', 'nD_liquid', 'pka_a', 'pka_b', 'density', 'dc', 'vis', 'ST', 'vapP']

    preds_list = {prop: [] for prop in properties}

    for _batch_idx, mono in enumerate(valid_dl):
        mono: MonoData = mono.to(device)

        preds = model.predict(mono, mono.temperature)

        for prop in properties:
            for p in preds[prop]:
                preds_list[prop].append(p.item())

    with open(os.path.join(test_conf["save_dir"], "processed_data.json"), "r") as file:
        test_data = json.load(file)

    # load min_values and max_values from test_conf['min_max_values']
    with open(test_conf["min_max_values"], "r") as file:
        min_max_values = json.load(file)

    min_values = min_max_values["min_values"]
    max_values = min_max_values["max_values"]

    for i, entry in enumerate(test_data):

        entry["temperature"] = entry["temperature"] * 373.15 - 273.15

        for prop in properties:
            entry[prop] = preds_list[prop][i] * (max_values[prop] - min_values[prop]) + min_values[prop]
            if prop in ["dc", "vis", "vapP"]:
                entry[prop] = np.exp(entry[prop])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    main()
