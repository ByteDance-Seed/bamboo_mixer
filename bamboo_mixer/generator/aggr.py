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

import torch

from bamboo_mixer.generator.aggr import MultiHeadAttentionBlock


class AttentionAggr(torch.nn.Module):

    def __init__(self, node_emb_dim=32, node_att_dim=32, edge_emb_dim=32, edge_att_dim=32, num_heads=4):
        super().__init__()

        self.node_attention_block = MultiHeadAttentionBlock(node_emb_dim, node_att_dim, num_heads)
        self.edge_attention_block = MultiHeadAttentionBlock(edge_emb_dim, edge_att_dim, num_heads)
        self.node_norm = torch.nn.LayerNorm(node_att_dim * num_heads)
        self.edge_norm = torch.nn.LayerNorm(edge_att_dim * num_heads)

    def reset_parameters(self):
        self.node_attention_block.reset_parameters()
        self.edge_attention_block.reset_parameters()

    def forward(self, molar_ratios, node_h, edge_h):

        node_h = molar_ratios * node_h
        edge_h = molar_ratios * edge_h

        node_outputs = self.node_attention_block(node_h)
        edge_outputs = self.edge_attention_block(edge_h)

        node_embedding_inv = torch.bmm(molar_ratios.transpose(1, 2), node_outputs).squeeze()
        edge_embedding_inv = torch.bmm(molar_ratios.transpose(1, 2), edge_outputs).squeeze()

        node_embedding_inv = self.node_norm(node_embedding_inv)
        edge_embedding_inv = self.edge_norm(edge_embedding_inv)

        return node_embedding_inv, edge_embedding_inv
