# The code is adapted from https://github.com/bytedance/byteff/blob/master/byteff/model/gnn.py

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

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch_geometric.nn import MLP, Linear
from torch_geometric.utils import scatter

from formula_design.data import Data
from formula_design.predictor.gnn import EGT, GAT, GINE, BasicGNN
from formula_design.utils.definitions import (MAX_CONNECTIVITY,
                                              MAX_FORMAL_CHARGE, MAX_RING_SIZE,
                                              SUPPORTED_ELEMENTS, BondOrder)


def equi_features(features: Tensor, equi_index: Tensor) -> Tensor:
    equi_value = scatter(features, equi_index, 0, reduce='mean')
    return equi_value[equi_index]


class FeatureLayer(nn.Module):
    """ atom features and/or bond features
    """

    def __init__(
            self,
            atom_embedding_dim=16,
            connectivity_embedding_dim=8,
            ring_con_embedding_dim=8,
            min_ring_size_embedding_dim=8,
            fm_chg_embedding_dim=8,
            bond_ring_embedding_dim=8,
            bond_order_embedding_dim=8,
            scale_grad_by_freq=False,
            node_mlp_dims=(32, 32, 3),  # (hidden_dim, out_dim, layers)
            edge_mlp_dims=(32, 32, 3),  # (hidden_dim, out_dim, layers)
            act='gelu'):
        super().__init__()

        self.atom_embedding_dim = atom_embedding_dim
        self.atom_embedding = nn.Embedding(
            num_embeddings=len(SUPPORTED_ELEMENTS),
            embedding_dim=atom_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)
        self.connectivity_embedding_dim = connectivity_embedding_dim
        self.connectivity_embedding = nn.Embedding(
            num_embeddings=MAX_CONNECTIVITY + 1,
            embedding_dim=connectivity_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)
        self.fm_chg_embedding_dim = fm_chg_embedding_dim
        self.fm_chg_embedding = nn.Embedding(
            num_embeddings=MAX_FORMAL_CHARGE * 2 + 1,
            embedding_dim=fm_chg_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)
        self.ring_con_embedding_dim = ring_con_embedding_dim
        self.ring_con_embedding = nn.Embedding(
            num_embeddings=MAX_CONNECTIVITY + 1,
            embedding_dim=ring_con_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)
        self.min_ring_size_embedding_dim = min_ring_size_embedding_dim
        self.min_ring_size_embedding = nn.Embedding(
            num_embeddings=MAX_RING_SIZE + 1,
            embedding_dim=min_ring_size_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)

        self.bond_ring_embedding_dim = bond_ring_embedding_dim
        self.bond_ring_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=bond_ring_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)
        self.bond_order_embedding_dim = bond_order_embedding_dim
        self.bond_order_embedding = nn.Embedding(
            num_embeddings=len(BondOrder),
            embedding_dim=bond_order_embedding_dim,
            scale_grad_by_freq=scale_grad_by_freq)

        # node mlp
        self.node_mlp = MLP(in_channels=self.raw_node_dim,
                            hidden_channels=node_mlp_dims[0],
                            out_channels=node_mlp_dims[1],
                            num_layers=node_mlp_dims[2],
                            norm=None,
                            act=act,
                            plain_last=False)
        # edge mlp
        self.edge_mlp = MLP(in_channels=self.raw_edge_dim,
                            hidden_channels=edge_mlp_dims[0],
                            out_channels=edge_mlp_dims[1],
                            num_layers=edge_mlp_dims[2],
                            norm=None,
                            act=act,
                            plain_last=False)

        self.node_out_dim = node_mlp_dims[1]
        self.edge_out_dim = edge_mlp_dims[1]

    @property
    def raw_node_dim(self) -> int:
        """dim for raw node feature"""
        return self.atom_embedding_dim + self.connectivity_embedding_dim + self.fm_chg_embedding_dim \
                + self.ring_con_embedding_dim + self.min_ring_size_embedding_dim

    @property
    def raw_edge_dim(self) -> int:
        """dim for raw edge feature"""
        return self.bond_order_embedding_dim + self.bond_ring_embedding_dim

    def reset_parameters(self):
        '''Reset parameters using kaiming_uniform (default)'''
        self.atom_embedding.reset_parameters()
        self.connectivity_embedding.reset_parameters()
        self.fm_chg_embedding.reset_parameters()
        self.ring_con_embedding.reset_parameters()
        self.min_ring_size_embedding.reset_parameters()
        self.bond_ring_embedding.reset_parameters()
        self.bond_order_embedding.reset_parameters()

        self.node_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()

    def get_node_features(self, graph: Data):
        x = graph.node_features
        x = x.long()
        embeddings = []
        embeddings.append(self.atom_embedding(
            x[:, 0]))  # [natoms, atom_embedding_dim]
        embeddings.append(self.connectivity_embedding(
            x[:, 1]))  # [natoms, connectivity_embedding_dim]
        embeddings.append(
            self.fm_chg_embedding(
                torch.clamp(
                    x[:, 2] + MAX_FORMAL_CHARGE,
                    min=0,
                    max=2 *
                    MAX_FORMAL_CHARGE)))  # [natoms, fm_chg_embedding_dim]
        embeddings.append(self.ring_con_embedding(
            x[:, 3]))  # [natoms, ring_con_embedding_dim]
        embeddings.append(self.min_ring_size_embedding(
            x[:, 4]))  # [natoms, min_ring_size_embedding_dim]
        node_features = torch.concat(embeddings, dim=-1)
        node_features = self.node_mlp(node_features)
        node_features = equi_features(node_features,
                                      graph.inc_node_equiv.long())
        return node_features

    def get_edge_features(self, graph: Data):
        x = graph.edge_features
        edge_ring = self.bond_ring_embedding(x[:, 0])
        edge_order = self.bond_order_embedding(x[:, 1])
        edge_features = torch.concat([edge_ring, edge_order], dim=-1)
        edge_features = self.edge_mlp(edge_features)
        edge_features = equi_features(edge_features,
                                      graph.inc_edge_equiv.long())
        return edge_features

    def forward(self, graph: Data) -> tuple[Tensor, Tensor]:
        """ return node and edge features
        """
        x_h = self.get_node_features(graph)
        e_h = self.get_edge_features(graph)
        return x_h, e_h


class GNNLayer(Module):

    gnn_map: dict[str, BasicGNN] = {'EGT': EGT, 'GINE': GINE, 'GAT': GAT}

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        gnn_type='EGT',
        gnn_dims=(32, 32, 3),  #   # (hidden_dim, out_dim, layers)
        act='gelu',
        jk=None,
        **kwargs,
    ):
        super().__init__()

        self.node_out_dim = gnn_dims[1]
        self.edge_out_dim = edge_in_dim
        self.gnn: BasicGNN = self.gnn_map[gnn_type](
            in_channels=node_in_dim,
            hidden_channels=gnn_dims[0],
            out_channels=gnn_dims[1],
            num_layers=gnn_dims[2],
            act=act,
            jk=jk,
            **kwargs)
        self.edge_in_lin = Linear(edge_in_dim, node_in_dim)
        self.edge_out_lin = Linear(self.node_out_dim, edge_in_dim)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.edge_in_lin.reset_parameters()
        self.edge_out_lin.reset_parameters()

    def forward(self, graph: Data, x_h: Tensor,
                e_h: Tensor) -> tuple[Tensor, Tensor]:
        edge_index = graph.inc_node_edge.long().T  # [2, n_edge]
        e_h = self.edge_in_lin(e_h)

        if self.gnn.supports_edge_update:
            x_h, e_h = self.gnn(x_h, edge_index, e_h)
        else:
            x_h = self.gnn(x_h, edge_index, e_h)

        e_h = self.edge_out_lin(e_h)

        # average bidirectional edge
        n_edge = e_h.shape[0]
        e_h = (e_h[::2] + e_h[1::2]) / 2
        e_h = e_h.repeat(1, 2).reshape(n_edge, -1)

        return x_h, e_h


class Graph2DBlock(Module):
    """
    Extract node/edge features and update them via GNN.
    """

    def __init__(self,
                 feature_layer: dict = None,
                 gnn_layer: dict = None) -> None:
        super().__init__()
        feature_layer = {} if feature_layer is None else feature_layer
        gnn_layer = {} if gnn_layer is None else gnn_layer

        self.feature_layer = FeatureLayer(**feature_layer)
        self.gnn_layer = GNNLayer(self.feature_layer.node_out_dim,
                                  self.feature_layer.edge_out_dim, **gnn_layer)

        self.node_out_dim = self.gnn_layer.node_out_dim
        self.edge_out_dim = self.gnn_layer.edge_out_dim

    def reset_parameters(self):
        self.feature_layer.reset_parameters()
        self.gnn_layer.reset_parameters()

    def forward(self, graph: Data):
        x_h, e_h = self.feature_layer(graph)
        x_h, e_h = self.gnn_layer(graph, x_h, e_h)
        return x_h, e_h
