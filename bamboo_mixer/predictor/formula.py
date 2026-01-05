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

import torch

from bamboo_mixer.data import FormulaData
from bamboo_mixer.predictor.aggr import AttentionAggr, VisAggr
from bamboo_mixer.predictor.mono import MonoPredictor
from bamboo_mixer.predictor.readout import ReadoutBlock
from bamboo_mixer.predictor.tdep import TMDepBlock

logger = logging.getLogger(__name__)


class FormulaPredictor(torch.nn.Module):

    def __init__(self,
                 mono_model: MonoPredictor,
                 aggr_block: dict,
                 readout_block: dict,
                 anion_block: dict,
                 train_mode: str = 'pretrain'):
        super().__init__()

        self.graph_embed_block: MonoPredictor = mono_model
        self.graph_embed_block.eval()

        # Override the .train() method of child_a so it does nothing.
        # Even if you call model_b.train(), child_a will ignore it.
        self.graph_embed_block.train = lambda mode=True: self.graph_embed_block

        self.aggr_block_solv = AttentionAggr(**aggr_block)
        self.aggr_block_salt = AttentionAggr(**aggr_block, num_heads=2)
        self.aggr_vis = VisAggr()

        assert train_mode in ['pretrain', 'finetune'], f'unsupported train_mode {train_mode}'
        self.train_mode: str = train_mode

        # Define grouped readout heads for NE and pred
        def make_heads():
            return torch.nn.ModuleDict({
                "Asigma0": ReadoutBlock(**readout_block, act="softplus"),
                "n1": ReadoutBlock(**readout_block, act="softplus"),
                "n2": ReadoutBlock(**readout_block, act="softplus"),
                "T0": ReadoutBlock(**readout_block, act="sigmoid"),
                "B": ReadoutBlock(**readout_block, act="softplus"),
                "D": ReadoutBlock(**readout_block, act="softplus"),
            })

        self.readout_ne = make_heads()  # this is not trained when train_mode == finetune
        self.readout_pred = make_heads()

        self.readout_anion_ratio = ReadoutBlock(**anion_block, act="sigmoid")

        self.empirical_formula_block = TMDepBlock()

    def reset_parameters(self):
        self.aggr_block_solv.reset_parameters()
        self.aggr_block_salt.reset_parameters()
        for block in [self.readout_ne, self.readout_pred]:
            for layer in block.values():
                layer.reset_parameters()
        self.readout_anion_ratio.reset_parameters()

    def load_ckpt(self, ckpt_path: str, device: torch.device = None):
        if device is None:
            logger.info('loading trained Formula ckpt to device of model')
            self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        else:
            logger.info(f'loading trained Formula ckpt to device {device}')
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict']
            self.load_state_dict(state_dict)
            self.to(device)

    def forward(self, data: FormulaData, get_mol_embs: bool = False):

        with torch.no_grad():
            solv_node_h, solv_edge_h = self.graph_embed_block.compute_embedding(data.solv_graphs)
            solv_vis = self.graph_embed_block.predict(data.solv_graphs, data.temperature, formula_vis=True)["vis"]
            salt_node_h, salt_edge_h = self.graph_embed_block.compute_embedding(data.salt_graphs)

        solv_node_h_inv, solv_edge_h_inv = self.aggr_block_solv(data.solv_graphs, solv_node_h, solv_edge_h)
        salt_node_h_inv, salt_edge_h_inv = self.aggr_block_salt(data.salt_graphs, salt_node_h, salt_edge_h)

        h_inv = torch.cat([solv_node_h_inv, solv_edge_h_inv, salt_node_h_inv, salt_edge_h_inv], dim=-1)
        vis = self.aggr_vis(data.solv_graphs, solv_vis)
        vis = vis.to(h_inv)
        T = data.temperature
        c = data.concentration

        # Empirical block inputs
        def get_inputs(readout):
            return (
                readout["Asigma0"](h_inv),
                readout["n1"](h_inv),
                readout["n2"](h_inv),
                readout["T0"](h_inv),
                readout["B"](h_inv),
                readout["D"](h_inv),
            )

        if self.train_mode == "pretrain":
            sigma_T_ne = self.empirical_formula_block(data, *get_inputs(self.readout_ne), vis)
            sigma_T_pred = self.empirical_formula_block(data, *get_inputs(self.readout_pred), vis)
            h_all = torch.cat([h_inv, c.unsqueeze(-1).expand(-1, 8), T.unsqueeze(-1).expand(-1, 8)], dim=-1)
            anion_ratio = self.readout_anion_ratio(h_all)
            result = {
                "h_inv": h_inv,
                "conductivity_ne": sigma_T_ne,
                "conductivity_pred": sigma_T_pred,
                "anion_ratio": anion_ratio,
            }

        elif self.train_mode == "finetune":
            sigma_T = self.empirical_formula_block(data, *get_inputs(self.readout_pred), vis)
            h_all = torch.cat([h_inv, c.unsqueeze(-1).expand(-1, 8), T.unsqueeze(-1).expand(-1, 8)], dim=-1)
            anion_ratio = self.readout_anion_ratio(h_all)
            result = {
                "h_inv": h_inv,
                "conductivity": sigma_T,
                "anion_ratio": anion_ratio,
            }

        else:
            raise ValueError(f"Invalid mode: {self.train_mode}")

        if get_mol_embs:
            result["solv_node_h"] = solv_node_h
            result["solv_edge_h"] = solv_edge_h
            result["salt_node_h"] = salt_node_h
            result["salt_edge_h"] = salt_edge_h

        return result
