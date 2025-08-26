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
import torch
import yaml
from torch.utils.data import DataLoader

from formula_design.data import FormulaData, GraphData, MXDataset, collate_data
from formula_design.predictor import MolMix, Mono
from formula_design.train import FormulaTrainer, TrainConfig
from formula_design.utils import setup_default_logging, to_dense_batch

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='test local')
parser.add_argument('--conf', type=str, default='test_config.yaml')
args = parser.parse_args()

outputs = {}


def make_hook(output_store, layer_name):

    def hook_fn(module, input, output):
        # Convert to list for JSON serialization
        output_store["layer_outputs"][layer_name] = output.detach().cpu(
        ).tolist()

    return hook_fn


def main():

    assert os.path.exists(args.conf) and args.conf.endswith(
        '.yaml'), f'yaml config {args.conf} not found.'

    with open(args.conf, 'r') as f:
        test_conf = yaml.safe_load(f)

    output_path = test_conf.pop("output_path")
    ckpt_path = test_conf.pop("ckpt_path")
    model_config = test_conf.pop("model")
    pretrain_config = model_config.pop("pretrain")
    pretrained_model = Mono(**pretrain_config["model_config"])
    pretrained_model.load_ckpt(os.path.join(ckpt_path, "pretrain.pt"))
    model = MolMix(pretrained_model, **model_config["model"])
    model.load_ckpt(os.path.join(ckpt_path, "optimal.pt"))
    model.to("cuda")

    dataset = MXDataset(test_conf)
    valid_dl = DataLoader(dataset,
                          batch_size=256,
                          shuffle=False,
                          drop_last=False,
                          collate_fn=collate_data)

    conductivities_pred = []
    anion_ratios_pred = []
    n1s = []
    n2s = []
    T0s = []
    As = []
    Bs = []
    sigma0s = []
    viss = []

    for batch_idx, graph in enumerate(valid_dl):

        graph: GraphData = graph.to("cuda")

        batch_outputs = {"layer_outputs": {}}
        hook1 = model.readout_conductivity.sigma.register_forward_hook(
            make_hook(batch_outputs, "readout_conductivity.sigma"))
        hook2 = model.readout_conductivity.n1.register_forward_hook(
            make_hook(batch_outputs, "readout_conductivity.n1"))
        hook3 = model.readout_conductivity.n2.register_forward_hook(
            make_hook(batch_outputs, "readout_conductivity.n2"))
        hook4 = model.readout_conductivity.T0.register_forward_hook(
            make_hook(batch_outputs, "readout_conductivity.T0"))
        hook5 = model.readout_conductivity.A.register_forward_hook(
            make_hook(batch_outputs, "readout_conductivity.A"))
        hook6 = model.readout_conductivity.B.register_forward_hook(
            make_hook(batch_outputs, "readout_conductivity.B"))
        hook7 = model.aggr_vis.register_forward_hook(
            make_hook(batch_outputs, "aggr_vis"))

        pred = model(graph)

        hook1.remove()
        hook2.remove()
        hook3.remove()
        hook4.remove()
        hook5.remove()
        hook6.remove()
        hook7.remove()

        pred_conductivities = torch.exp(pred["conductivity"].squeeze(-1) *
                                        8.3344 - 4.6002)
        pred_anion_ratios = pred["anion_ratio"].squeeze(-1)

        for c in pred_conductivities:
            conductivities_pred.append(c.item())

        for a in pred_anion_ratios:
            anion_ratios_pred.append(a.item())

        for sigma0 in batch_outputs["layer_outputs"][
                "readout_conductivity.sigma"]:
            if isinstance(sigma0, (float, int)):
                sigma0s.append(sigma0)
            else:
                sigma0s.append(sigma0[0])
        for n1 in batch_outputs["layer_outputs"]["readout_conductivity.n1"]:
            if isinstance(n1, (float, int)):
                n1s.append(n1)
            else:
                n1s.append(n1[0])
        for n2 in batch_outputs["layer_outputs"]["readout_conductivity.n2"]:
            if isinstance(n2, (float, int)):
                n2s.append(n2)
            else:
                n2s.append(n2[0])
        for A in batch_outputs["layer_outputs"]["readout_conductivity.A"]:
            if isinstance(A, (float, int)):
                As.append(A)
            else:
                As.append(A[0])
        for B in batch_outputs["layer_outputs"]["readout_conductivity.B"]:
            if isinstance(B, (float, int)):
                Bs.append(B)
            else:
                Bs.append(B[0])
        for T0 in batch_outputs["layer_outputs"]["readout_conductivity.T0"]:
            if isinstance(T0, (float, int)):
                T0s.append(T0)
            else:
                T0s.append(T0[0])

        if isinstance(batch_outputs["layer_outputs"]["aggr_vis"],
                      (float, int)):
            viss.append(batch_outputs["layer_outputs"]["aggr_vis"])
        else:
            for vis in batch_outputs["layer_outputs"]["aggr_vis"]:
                viss.append(vis)

        logger.info(f"Finishing prediction of batch {batch_idx}")

    with open(os.path.join(test_conf["save_dir"], "processed_data.json"),
              "r") as file:
        test_data = json.load(file)

    for i, entry in enumerate(test_data):
        entry["temperature"] = entry["temperature"] * 373.15 - 273.15
        entry["conductivity_pred"] = conductivities_pred[i]
        entry["anion_ratio_pred"] = anion_ratios_pred[i]
        entry["sigma0"] = sigma0s[i]
        entry["n1"] = n1s[i]
        entry["n2"] = n2s[i]
        entry["A"] = As[i]
        entry["B"] = Bs[i]
        entry["T0"] = T0s[i]
        entry["vis"] = viss[i]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    main()
