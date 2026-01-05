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
import logging

import torch
from torch_geometric.nn import global_mean_pool

from bamboo_mixer.data import Data
from bamboo_mixer.predictor.graph_block import Graph2DBlock
from bamboo_mixer.predictor.readout import ReadoutBlock
from bamboo_mixer.predictor.tdep import LinearBlock, VFTBlock, VFTNoShiftBlock

logger = logging.getLogger(__name__)


class MonoPredictor(torch.nn.Module):

    def __init__(self, graph_block: dict, readout_block: dict):
        """

        readout_config example:
        config = {
            'input_dim': 64,
            'hidden_dims': [128, 64],
            'output_dim': 10
        }
        """
        super(MonoPredictor, self).__init__()
        self.graph_embed_block = Graph2DBlock(**graph_block)
        temp_readout_block = copy.deepcopy(readout_block)
        temp_readout_block['input_dim'] += 8  # cat temperature
        self.readout_Tm = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_bp = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_pka_a = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_pka_b = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_nD = ReadoutBlock(**temp_readout_block, act="sigmoid")
        self.readout_nD_liquid = ReadoutBlock(**temp_readout_block, act="sigmoid")
        self.readout_density = ReadoutBlock(**temp_readout_block, act="sigmoid")
        self.readout_dc = ReadoutBlock(**temp_readout_block, act="sigmoid")

        self.readout_vis_vis0 = ReadoutBlock(**readout_block, act="softplus")
        self.readout_vis_A = ReadoutBlock(**readout_block, act="softplus")
        self.readout_vis_T0 = ReadoutBlock(**readout_block, act="softplus")
        self.temp_vis = VFTBlock()

        self.readout_ST_A = ReadoutBlock(**readout_block, act="softplus")
        self.readout_ST_B = ReadoutBlock(**readout_block, act="softplus")
        self.temp_ST = LinearBlock()

        self.readout_vapP_A = ReadoutBlock(**readout_block, act="softplus")
        self.readout_vapP_B = ReadoutBlock(**readout_block, act="softplus")
        self.temp_vapP = VFTNoShiftBlock()

    def reset_parameters(self):
        self.graph_embed_block.reset_parameters()
        self.readout_Tm.reset_parameters()
        self.readout_bp.reset_parameters()
        self.readout_nD.reset_parameters()
        self.readout_nD_liquid.reset_parameters()
        self.readout_pka_a.reset_parameters()
        self.readout_pka_b.reset_parameters()
        self.readout_density.reset_parameters()
        self.readout_dc.reset_parameters()
        self.readout_vis_vis0.reset_parameters()
        self.readout_vis_A.reset_parameters()
        self.readout_vis_T0.reset_parameters()
        self.readout_ST_A.reset_parameters()
        self.readout_ST_B.reset_parameters()
        self.readout_vapP_A.reset_parameters()
        self.readout_vapP_B.reset_parameters()

    def load_ckpt(self, ckpt_path: str, device: torch.device = None):
        if device is None:
            logger.info(f'loading trained Mono ckpt to device of model')
            self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        else:
            logger.info(f'loading trained Mono ckpt to device {device}')
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict']
            self.load_state_dict(state_dict)
            self.to(device)

    def compute_embedding(self, data: Data):
        """
        """
        node_h, edge_h = self.graph_embed_block(data)  # graph information and weigth ratio information

        return node_h, edge_h

    def forward(self, data: Data):
        """
        """

        node_h, edge_h = self.compute_embedding(data)  # graph information and weigth ratio information

        device = node_h.device

        node_count, edge_count = data.counts[:, 0], data.counts[:, 1]

        node_batch_mol = torch.arange(node_count.size()[0], dtype=torch.int64,
                                      device=device).repeat_interleave(node_count)

        edge_batch_mol = torch.arange(edge_count.size()[0], dtype=torch.int64,
                                      device=device).repeat_interleave(edge_count)

        node_h_mean = global_mean_pool(node_h, node_batch_mol)
        edge_h_mean = global_mean_pool(edge_h, edge_batch_mol)

        h_all = torch.cat([node_h_mean, edge_h_mean], dim=-1)

        # Properties without temperature input
        Tm = self.readout_Tm(h_all)
        bp = self.readout_bp(h_all)
        pka_a = self.readout_pka_a(h_all)
        pka_b = self.readout_pka_b(h_all)

        # Properties with temperature

        h_nD = torch.cat([h_all, data.nD_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        nD = self.readout_nD(h_nD)

        h_nD_liquid = torch.cat([h_all, data.nD_liquid_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        nD_liquid = self.readout_nD_liquid(h_nD_liquid)

        h_density = torch.cat([h_all, data.density_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        density = self.readout_density(h_density)

        h_dc = torch.cat([h_all, data.dc_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        dc = self.readout_dc(h_dc)

        # Properties with temperature relation
        # viscosity (VFT)
        vis_vis0 = self.readout_vis_vis0(h_all)
        vis_A = self.readout_vis_A(h_all)
        vis_T0 = self.readout_vis_T0(h_all)
        vis_T = self.temp_vis(vis_vis0, vis_A, data.vis_T.unsqueeze(-1), vis_T0)

        # ST (linear)
        ST_A = self.readout_ST_A(h_all)
        ST_B = self.readout_ST_B(h_all)
        ST_T = self.temp_ST(ST_A, ST_B, data.ST_T.unsqueeze(-1))

        # vapour pressure (VFTnonshift)
        vapP_A = self.readout_vapP_A(h_all)
        vapP_B = self.readout_vapP_B(h_all)
        vapP_T = self.temp_vapP(vapP_A, vapP_B, data.vapP_T.unsqueeze(-1))

        return {
            "Tm": Tm,
            "bp": bp,
            "nD": nD,
            "nD_liquid": nD_liquid,
            "pka_a": pka_a,
            "pka_b": pka_b,
            "density": density,
            "dc": dc,
            "vis": vis_T,
            "ST": ST_T,
            "vapP": vapP_T
        }

    def predict(self, data: Data, temperature: torch.Tensor, formula_vis: bool = False):
        """
        Default normalization ranges used in this function. 
        You may need to modify them on your own data.
        """

        node_h, edge_h = self.graph_embed_block(data)  # graph information and weigth ratio information

        device = node_h.device

        node_count, edge_count = data.counts[:, 0], data.counts[:, 1]

        node_batch_mol = torch.arange(node_count.size()[0], dtype=torch.int64,
                                      device=device).repeat_interleave(node_count)

        edge_batch_mol = torch.arange(edge_count.size()[0], dtype=torch.int64,
                                      device=device).repeat_interleave(edge_count)

        if formula_vis:
            node_count_formula = data.counts_mol[:, 0]
            temperature = temperature.repeat_interleave(node_count_formula, dim=0)
            vis_temperature = 0.8 * torch.ones_like(temperature).unsqueeze(-1)
        else:
            vis_temperature = temperature.unsqueeze(-1) * 373.15 / 404.1

        node_h_mean = global_mean_pool(node_h, node_batch_mol)
        edge_h_mean = global_mean_pool(edge_h, edge_batch_mol)

        h_all = torch.cat([node_h_mean, edge_h_mean], dim=-1)

        # Properties without temperature input
        Tm = self.readout_Tm(h_all)
        bp = self.readout_bp(h_all)
        pka_a = self.readout_pka_a(h_all)
        pka_b = self.readout_pka_b(h_all)

        nD_T = temperature.unsqueeze(-1) * 373.15 / 407.15
        h_nD = torch.cat([h_all, nD_T.repeat(1, 8)], dim=-1)
        nD = self.readout_nD(h_nD)

        nD_liquid_T = temperature.unsqueeze(-1) * 373.15 / 407.15
        h_nD_liquid = torch.cat([h_all, nD_liquid_T.repeat(1, 8)], dim=-1)
        nD_liquid = self.readout_nD_liquid(h_nD_liquid)

        density_T = temperature.unsqueeze(-1) * 373.15 / 548.15
        h_density = torch.cat([h_all, density_T.repeat(1, 8)], dim=-1)
        density = self.readout_density(h_density)

        dc_T = temperature.unsqueeze(-1) * 373.15 / 502.15
        h_dc = torch.cat([h_all, dc_T.repeat(1, 8)], dim=-1)
        dc = self.readout_dc(h_dc)

        # Properties with temperature relation
        # viscosity (VFT)
        vis_vis0 = self.readout_vis_vis0(h_all)
        vis_A = self.readout_vis_A(h_all)
        vis_T0 = self.readout_vis_T0(h_all)
        vis_T = self.temp_vis(vis_vis0, vis_A, vis_temperature, vis_T0)

        # ST (linear)
        ST_A = self.readout_ST_A(h_all)
        ST_B = self.readout_ST_B(h_all)
        ST_temperature = temperature.unsqueeze(-1) * 373.15 / 473.15  # convert to the scale for pretrained model
        ST_T = self.temp_ST(ST_A, ST_B, ST_temperature)

        # vapour pressure (VFTnonshift)
        vapP_A = self.readout_vapP_A(h_all)
        vapP_B = self.readout_vapP_B(h_all)
        vapP_temperature = temperature.unsqueeze(-1) * 373.15 / 473.15  # convert to the scale for pretrained model
        vapP_T = self.temp_vapP(vapP_A, vapP_B, vapP_temperature)

        return {
            "h_all": h_all,  # not used for training, readout only
            "Tm": Tm,
            "bp": bp,
            "nD": nD,
            "nD_liquid": nD_liquid,
            "pka_a": pka_a,
            "pka_b": pka_b,
            "density": density,
            "dc": dc,
            "vis": vis_T,
            "ST": ST_T,
            "vapP": vapP_T
        }
