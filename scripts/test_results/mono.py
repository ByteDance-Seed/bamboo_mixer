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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from torch.utils.data import DataLoader

from formula_design.data import GraphData, MonoData, MonoDataset, collate_data
from formula_design.predictor import Mono
from formula_design.utils import setup_default_logging, to_dense_batch

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='test local')
parser.add_argument('--conf', type=str, default='mono_config.yaml')
args = parser.parse_args()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith(
        '.yaml'), f'yaml config {args.conf} not found.'

    with open(args.conf, 'r') as f:
        test_conf = yaml.safe_load(f)

    output_path = test_conf.pop("output_path")
    ckpt_path = test_conf.pop("ckpt_path")
    model_config = test_conf.pop("model")
    model = Mono(**model_config)
    model.load_ckpt(ckpt_path)
    model.to("cuda")

    dataset = MonoDataset(test_conf)
    valid_dl = DataLoader(dataset,
                          batch_size=256,
                          shuffle=False,
                          drop_last=False,
                          collate_fn=collate_data)

    properties = [
        'Tm', 'bp', 'nD', 'nD_liquid', 'pka_a', 'pka_b', 'density', 'dc',
        'vis', 'ST', 'vapP'
    ]

    preds_list = {prop: [] for prop in properties}

    for batch_idx, graph in enumerate(valid_dl):
        graph: GraphData = graph.to("cuda")

        preds = model.predict(graph, graph.temperature)

        for prop in properties:
            for p in preds[prop]:
                preds_list[prop].append(p.item())

    with open(os.path.join(test_conf["save_dir"], "processed_data.json"),
              "r") as file:
        test_data = json.load(file)

    min_values = {
        "Tm": 3.1499999999999773,
        "bp": 81.64999999999998,
        "pka_a": -7.15,
        "pka_b": -8.0,
        "nD": 1.1674,
        "nD_T": 0,
        "nD_liquid": 1.1674,
        "nD_liquid_T": 0,
        "ST": 0.0,
        "ST_T": 0,
        "density": 0.377,
        "density_T": 0,
        "vis": -2.3025850929940455,
        "vis_T": 0,
        "vapP": 0.013160223137316041,
        "vapP_T": 0,
        "dc": 0.5423242908253615,
        "dc_T": 0
    }
    max_values = {
        "Tm": 823.15,
        "bp": 1273.15,
        "pka_a": 18.46,
        "pka_b": 30.9,
        "nD": 1.9377,
        "nD_T": 407.15,
        "nD_liquid": 1.8225,
        "nD_liquid_T": 407.15,
        "ST": 78.73,
        "ST_T": 473.15,
        "density": 3.034,
        "density_T": 548.15,
        "vis": 3.637586159726385,
        "vis_T": 404.1,
        "vapP": 16.203447318057822,
        "vapP_T": 473.1499939,
        "dc": 5.210960475329785,
        "dc_T": 502.15
    }

    for i, entry in enumerate(test_data):

        entry["temperature"] = entry["temperature"] * 373.15 - 273.15

        for prop in properties:
            entry[prop] = preds_list[prop][i] * (
                max_values[prop] - min_values[prop]) + min_values[prop]
            if prop in ["dc", "vis", "vapP"]:
                entry[prop] = np.exp(entry[prop])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    main()
