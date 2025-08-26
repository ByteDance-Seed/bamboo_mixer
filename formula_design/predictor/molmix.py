# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
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

import json

import torch

from formula_design.data import Data, FormulaData
from formula_design.predictor.aggr import AttentionAggr, VisAggr
from formula_design.predictor.graph_block import Graph2DBlock
from formula_design.predictor.tdep import TMDepBlock
from formula_design.utils import ActivationFunction

from .mono import Mono


class ReadoutBlock(torch.nn.Module):

    def __init__(self, act=None, **config):
        super(ReadoutBlock, self).__init__()

        # Dynamically retrieve values from config
        self.act = ActivationFunction.get_activation(act)
        self.input_dim = config.get('input_dim')
        self.hidden_dims = config.get('hidden_dims', [])
        self.output_dim = config.get('output_dim')

        # Create the readout layer dynamically
        self.readout_block = self.create_readout_block()

    def create_readout_block(self):
        layers = []
        input_dim = self.input_dim

        # Create layers based on hidden dimensions
        for hidden_dim in self.hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            input_dim = hidden_dim  # Update input dim for the next layer

        # Add the final output layer
        layers.append(torch.nn.Linear(input_dim, self.output_dim))

        if self.act:
            layers.append(self.act)

        # Return the layers as a Sequential block
        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.readout_block:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.readout_block(x)


class MolMix(torch.nn.Module):

    def __init__(self, pretrained_model, aggr_block: dict, readout_block: dict,
                 anion_block: dict):
        super().__init__()

        self.graph_embed_block = pretrained_model

        self.aggr_block_solv = AttentionAggr(**aggr_block)
        self.aggr_block_salt = AttentionAggr(**aggr_block, num_heads=2)
        self.aggr_vis = VisAggr()

        # Define grouped readout heads for NE and Mistry
        def make_heads():
            return torch.nn.ModuleDict({
                "sigma":
                ReadoutBlock(**readout_block, act="softplus"),
                "n1":
                ReadoutBlock(**readout_block, act="softplus"),
                "n2":
                ReadoutBlock(**readout_block, act="softplus"),
                "T0":
                ReadoutBlock(**readout_block, act="sigmoid"),
                "A":
                ReadoutBlock(**readout_block, act="softplus"),
                "B":
                ReadoutBlock(**readout_block, act="softplus"),
            })

        self.readout_conductivity = make_heads()

        self.readout_anion_ratio = ReadoutBlock(**anion_block, act="sigmoid")

        self.empirical_formula_block = TMDepBlock()

    def _get_device(self):
        return next(self.parameters()).device

    def reset_parameters(self):
        self.aggr_block_solv.reset_parameters()
        self.aggr_block_salt.reset_parameters()
        for layer in self.readout_conductivity.values():
            layer.reset_parameters()
        self.readout_anion_ratio.reset_parameters()

    def get_parameters(self, name=None):
        return self.parameters() if name is None else getattr(
            self, name).parameters()

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def forward(self, data: Data):

        with torch.no_grad():
            solv_node_h, solv_edge_h = self.graph_embed_block._compute_embedding(
                data.solv_graphs)
            solv_vis = self.graph_embed_block.predict(data.solv_graphs,
                                                      data.temperature,
                                                      formula_vis=True)["vis"]
            salt_node_h, salt_edge_h = self.graph_embed_block._compute_embedding(
                data.salt_graphs)

        solv_node_h_inv, solv_edge_h_inv = self.aggr_block_solv(
            data.solv_graphs, solv_node_h, solv_edge_h)
        salt_node_h_inv, salt_edge_h_inv = self.aggr_block_salt(
            data.salt_graphs, salt_node_h, salt_edge_h)

        h_inv = torch.cat([
            solv_node_h_inv, solv_edge_h_inv, salt_node_h_inv, salt_edge_h_inv
        ],
                          dim=-1)
        vis = self.aggr_vis(data.solv_graphs, solv_vis)
        vis = vis.to(h_inv)
        T = data.temperature
        c = data.concentration

        # Empirical block inputs
        def get_inputs(readout):
            return (
                readout["sigma"](h_inv),
                readout["n1"](h_inv),
                readout["n2"](h_inv),
                readout["T0"](h_inv),
                readout["A"](h_inv),
                readout["B"](h_inv),
            )

        sigma_T = self.empirical_formula_block(
            data, *get_inputs(self.readout_conductivity), vis)
        h_all = torch.cat([
            h_inv,
            c.unsqueeze(-1).expand(-1, 8),
            T.unsqueeze(-1).expand(-1, 8)
        ],
                          dim=-1)
        anion_ratio = self.readout_anion_ratio(h_all)

        return {
            "h_inv": h_inv,
            "conductivity": sigma_T,
            "anion_ratio": anion_ratio
        }
