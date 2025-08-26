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

import h5py
import torch

from formula_design.generator.aggr import AttentionAggr


class AggrBlock(torch.nn.Module):

    def __init__(self, pretrained_ckpt, aggr_block: dict):
        super(AggrBlock, self).__init__()

        if pretrained_ckpt:
            self.ckpt = torch.load(pretrained_ckpt)
            self.model_state_dict = self.ckpt['model_state_dict']

        self.emb_dim = aggr_block['node_emb_dim'] + aggr_block["edge_emb_dim"]

        self.aggr_block_solv = AttentionAggr(**aggr_block, num_heads=4)
        self.aggr_block_salt = AttentionAggr(**aggr_block, num_heads=2)

        if pretrained_ckpt:
            relevant_weights = {
                key: self.model_state_dict[key]
                for key in self.model_state_dict
                if key.startswith("aggr_block_solv")
            }
            state_dict_solv = {
                key.replace("aggr_block_solv.", ""): val
                for key, val in relevant_weights.items()
            }
            self.aggr_block_solv.load_state_dict(state_dict_solv, strict=False)

            relevant_weights = {
                key: self.model_state_dict[key]
                for key in self.model_state_dict
                if key.startswith("aggr_block_salt")
            }
            state_dict_solv = {
                key.replace("aggr_block_salt.", ""): val
                for key, val in relevant_weights.items()
            }
            self.aggr_block_salt.load_state_dict(state_dict_solv, strict=False)

    def forward(self, emb, molar_ratios):

        solv_emb = emb[:, :emb.size(1) // 2, :]  # solvent embedding
        salt_emb = emb[:, emb.size(1) // 2:, :]  # salt embedding
        solv_molar_ratios = molar_ratios[:, :emb.size(1) //
                                         2, :]  # solvent molar ratio
        salt_molar_ratios = molar_ratios[:, emb.size(1) //
                                         2:, :]  # salt molar ratio

        solv_node_h = solv_emb[:, :, :emb.size(2) //
                               2]  # solvent node embedding
        solv_edge_h = solv_emb[:, :,
                               emb.size(2) // 2:]  # solvent edge embedding

        salt_node_h = salt_emb[:, :, :emb.size(2) // 2]  # salt node embedding
        salt_edge_h = salt_emb[:, :, emb.size(2) // 2:]  # salt edge embedding

        solv_node_h_inv, solv_edge_h_inv = self.aggr_block_solv(
            solv_molar_ratios, solv_node_h, solv_edge_h)
        salt_node_h_inv, salt_edge_h_inv = self.aggr_block_salt(
            salt_molar_ratios, salt_node_h, salt_edge_h)

        h_inv = torch.cat([
            solv_node_h_inv, solv_edge_h_inv, salt_node_h_inv, salt_edge_h_inv
        ],
                          dim=-1)

        return h_inv
