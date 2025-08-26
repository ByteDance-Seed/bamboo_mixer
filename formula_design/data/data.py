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

import functools
import json
import logging
from operator import itemgetter
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from formula_design.mol import Molecule, MoleculeGraph
from formula_design.utils.definitions import (ELEMENT_MAP, MAX_RING_SIZE,
                                              BondOrder)
from formula_design.utils.mol_utils import find_equivalent_index, get_ring_info

_count_names = ['node', 'edge']
_count_idx = {name: idx for idx, name in enumerate(_count_names)}


class Data:
    """
    Data is a dict containing features, most of which are Tensors.
    
    If one key is start by 'inc_{name}_', its value contains {name} indices and
      should be increased by the cummulation of counts[COUNT_IDX[name]] when collating.
    """

    def __setattr__(self, name: str, value: Tensor):
        if name == 'dict':
            super().__setattr__(name, value)
        else:
            self.dict[name] = value

    def __getattr__(self, name: str):
        if name == 'dict':
            object.__setattr__(self, 'dict', {})
            return self.dict

        if 'dict' in self.__dict__ and name in self.dict:
            return self.dict[name]

        raise AttributeError(f"'Data' object has no attribute '{name}'")

    def __delattr__(self, name):
        del self.dict[name]

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        if not hasattr(self, 'dict'):
            object.__setattr__(self, 'dict', {})
        self.dict[key] = value

    def __contains__(self, key):
        return key in self.dict

    def __init__(self,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32,
                 **kwargs):
        super().__init__()
        self.dict = {}
        self.counts = torch.tensor([0] * len(_count_names),
                                   dtype=int_dtype).reshape(1, -1)
        self.counts_cluster = torch.tensor([0] * len(_count_names),
                                           dtype=int_dtype).reshape(1, -1)

    def get_count(self, name: str, idx=0, cluster=False):
        counts = self.counts_cluster[:, _count_idx[
            name]] if cluster else self.counts[:, _count_idx[name]]
        if idx is not None:
            return counts[idx]
        else:
            return counts

    def set_count(self, name: str, count: int, idx: int = 0, cluster=False):
        if cluster:
            self.counts_cluster[idx, _count_idx[name]] = count
        else:
            self.counts[idx, _count_idx[name]] = count

    @classmethod
    def from_dict(cls, data_dict: dict):
        data = cls()
        for k, v in data_dict.items():
            data[k] = v
        return data

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()

    def to(self, device: str, non_blocking=False):
        ret = {}
        for k, v in self.items():
            if isinstance(v, Tensor):
                ret[k] = v.to(device, non_blocking=non_blocking)
            elif isinstance(v, Data):
                ret[k] = v.to(device)
            else:
                ret[k] = v
        return Data.from_dict(ret)

    def pin_memory(self):
        for k, v in self.items():
            if isinstance(v, Tensor):
                self[k] = v.pin_memory()
            elif isinstance(v, Data):
                self[k] = v.pin_memory()
            else:
                self[k] = v
        return self


def collate_data(data_list: list[Data]):
    """
    Collate a list of data, according to the increasing rule.
    """

    def collate_tensor(k: str, vs: list[Tensor], incs: dict[str, Tensor],
                       cluster: bool):

        if "_emb" in k or k == "all_molar_ratio" or k == "bow_vec":
            cat_vs = torch.stack(vs, dim=0)
        else:
            cat_vs = torch.concat(vs, dim=0)
        if k.startswith('inc_'):
            inc_name = k.split('_')[1]
            if inc_name in incs:
                inc = incs[inc_name]
            else:
                inc = torch.tensor([
                    data.get_count(inc_name, cluster=cluster)
                    for data in data_list
                ],
                                   device=vs[0].device,
                                   dtype=vs[0].dtype)
                inc = torch.concat(
                    (torch.tensor([0], device=vs[0].device, dtype=vs[0].dtype),
                     torch.cumsum(inc, 0)[:-1]),
                    dim=0)
                incs[inc_name] = inc
            nums = torch.tensor([v.shape[0] for v in vs],
                                dtype=vs[0].dtype,
                                device=vs[0].device)
            size = (-1, ) + (1, ) * (vs[0].dim() - 1)
            cat_vs += torch.repeat_interleave(inc, nums).view(size)
        return cat_vs

    cat_data = Data()
    data0 = data_list[0]
    cluster = 'cluster_flag' in data0
    incs = {}
    for k in data0.keys():
        vs = [data[k] for data in data_list]
        if isinstance(vs[0], Data):
            vs = collate_data(vs)
        elif isinstance(vs[0], Tensor):
            try:
                vs = collate_tensor(k, vs, incs, cluster)
            except Exception as e:
                print(f"An error occurred: {e}")
                print(k)
                print(data_list[0]["mol_name"])
                print(data_list[0]["smiles"])
                print(vs[0])
        cat_data[k] = vs
    return cat_data


class GraphData(Data):
    """
    GraphData extract graph and mol features from a molecule.

    Mol features:
    - mol_name: str
    - mapped_smiles: str
    - counts: IntTensor (COUNT_NAMES)  # [1, 9]

    Graph features:
    - node_features: IntTensor (atom_type, connectivity, formal_charge, ring_con, min_ring_size)  # [n_node, 5]
    - edge_features: IntTensor (bond_ring, bond_order)  # [n_edge, 2]

    - inc_node_edge: IntTensor  # [n_edge, 2]
    - inc_node_equiv: IntTensor  # [n_node, 1]
    - inc_edge_equiv: IntTensor  # [n_edge, 1]
    """

    def __init__(self,
                 name: str = '',
                 mapped_smiles: str = '',
                 int_dtype=torch.int32,
                 float_dtype=torch.float32):
        super().__init__(int_dtype, float_dtype)

        if not mapped_smiles:
            return

        mol = Molecule.from_mapped_smiles(mapped_smiles, name=name)
        self.mol_name = mol.name
        self.mapped_smiles = mol.get_mapped_smiles()

        graph = MoleculeGraph(mol, max_include_ring=MAX_RING_SIZE)
        topos = graph.get_intra_topo()

        # node features
        atom_type = torch.tensor([ELEMENT_MAP[i] for i in mol.atomic_numbers],
                                 dtype=int_dtype)
        connectivity = torch.tensor(
            [atom.connectivity for atom in graph.get_atoms()], dtype=int_dtype)
        formal_charge_vec = torch.tensor(mol.formal_charges, dtype=int_dtype)
        ring_con, min_ring_size = get_ring_info(graph)
        ring_con = torch.tensor(ring_con, dtype=int_dtype)
        min_ring_size = torch.tensor(min_ring_size, dtype=int_dtype)
        features = torch.vstack([
            atom_type, connectivity, formal_charge_vec, ring_con, min_ring_size
        ]).T
        self.node_features = features  # [n_node, 5]
        self.set_count('node', mol.natoms)
        assert self.get_count('node') == self.node_features.shape[0]

        # edge features
        bond_orders = list(BondOrder)
        edge_idx_dict = {}
        edge_features = []
        inc_node_edge = []
        for i, atomidx in enumerate(topos['Bond']):
            edge_idx_dict[atomidx] = 2 * i
            edge_idx_dict[atomidx[::-1]] = 2 * i + 1
            bond = graph.get_bond(*atomidx)
            # duplicate for bidirectional edge
            edge_features.append(
                (int(bond.in_ring), bond_orders.index(BondOrder(bond.order))))
            edge_features.append(
                (int(bond.in_ring), bond_orders.index(BondOrder(bond.order))))
            inc_node_edge.append((int(bond.end_idx), int(bond.begin_idx)))
            inc_node_edge.append((int(bond.begin_idx), int(bond.end_idx)))

        self.edge_features = torch.tensor(edge_features, dtype=int_dtype)
        self.inc_node_edge = torch.tensor(inc_node_edge, dtype=int_dtype)
        self.set_count('edge', self.inc_node_edge.shape[0])

        # equivalent
        atom_equi_index, edge_equi_index = find_equivalent_index(
            mol, self.inc_node_edge.tolist())
        self.inc_node_equiv = torch.tensor(atom_equi_index, dtype=int_dtype)
        self.inc_edge_equiv = torch.tensor(edge_equi_index, dtype=int_dtype)


class MonoData(GraphData):
    """"
    Data containing one molecule

    properties: 


    - inc_node_edge3d: IntFloat  # [n_edge3d, 2]
    - conf_mask: FloatTensor  # [1, n_conf]
    """

    def __init__(self,
                 name: str = '',
                 mapped_smiles: str = '',
                 properties: dict = {},
                 int_dtype=torch.int32,
                 float_dtype=torch.float32,
                 **kwargs):
        super().__init__(name=name,
                         mapped_smiles=mapped_smiles,
                         int_dtype=int_dtype,
                         float_dtype=float_dtype)

        if not mapped_smiles:
            return

        for k, v in properties.items():
            if len(v) > 0:
                self[k] = v

        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                self[k] = v
            if isinstance(v, float):
                self[k] = torch.tensor(v, dtype=float_dtype).unsqueeze(-1)
            elif isinstance(v, int):
                self[k] = torch.tensor(v, dtype=float_dtype).unsqueeze(-1)
            elif isinstance(v, bool):
                self[k] = torch.tensor(v, dtype=int_dtype).unsqueeze(-1)
            elif isinstance(v, np.ndarray):
                self[k] = torch.tensor(v, dtype=float_dtype)
            elif isinstance(v, list):
                self[k] = torch.tensor(v, dtype=float_dtype)
            else:
                raise ValueError(f'unknown kwargs {k}, {v}')


class MixData(GraphData):
    """
    Data class for a mixture of Li salt and solvent molecules

    - molar_ratio: FloatTensor # [n_mol, 1]
    - temperature: FloatTensor # [1, ]

    """

    def __init__(self,
                 names: list[str] = None,
                 mapped_smiles: list[str] = None,
                 molar_ratios: np.ndarray = None,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32):
        super().__init__(int_dtype=int_dtype, float_dtype=float_dtype)

        if not mapped_smiles:
            return

        graph_list = [
            GraphData(name, mps, int_dtype, float_dtype)
            for name, mps in zip(names, mapped_smiles)
        ]

        graphs = collate_data(graph_list)

        for k, v in graphs.items():
            self[k] = v

        self.counts_cluster = torch.sum(self.counts, dim=0, keepdim=True)
        # counts_mol: count the number of molecules in each formula
        self.counts_mol = torch.sum(torch.ones_like(self.counts),
                                    dim=0,
                                    keepdim=True).to(int_dtype)

        self.cluster_flag = True

        if molar_ratios is not None:
            self.molar_ratios = torch.tensor(molar_ratios,
                                             dtype=float_dtype).unsqueeze(-1)


class FormulaData(MixData):
    """
    Data class for a mixture of Li salt and solvent molecules.

    Attributes:
        - mol_emb: molecular embeddings [n_mol, d]
        - frm_emb: formula embeddings [1, d]
        - all_molar_ratio: full molar ratios [n_mol]
        - bow_vec: bag-of-words representation
        - concentration, temperature, conductivity, etc.: physical properties as FloatTensors
    """

    def __init__(self,
                 solv_names: Optional[list[str]] = None,
                 solv_mapped_smiles: Optional[list[str]] = None,
                 salt_names: Optional[list[str]] = None,
                 salt_mapped_smiles: Optional[list[str]] = None,
                 solv_molar_ratios: Optional[np.ndarray] = None,
                 salt_molar_ratios: Optional[np.ndarray] = None,
                 mol_emb: Optional[np.ndarray] = None,
                 frm_emb: Optional[np.ndarray] = None,
                 all_molar_ratio: Optional[np.ndarray] = None,
                 bow_vec: Optional[np.ndarray] = None,
                 concentration: Optional[np.ndarray] = None,
                 temperature: Optional[np.ndarray] = None,
                 viscosity: Optional[np.ndarray] = None,
                 conductivity: Optional[np.ndarray] = None,
                 conductivity_mask: Optional[np.ndarray] = None,
                 conductivity_NE: Optional[np.ndarray] = None,
                 conductivity_mistry: Optional[np.ndarray] = None,
                 anion_ratio: Optional[np.ndarray] = None,
                 anion_ratio_mask: Optional[np.ndarray] = None,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32):

        super().__init__(int_dtype=int_dtype, float_dtype=float_dtype)

        # Early exit if SMILES are not provided
        if not solv_mapped_smiles or not salt_mapped_smiles:
            return

        self.solv_graphs = MixData(solv_names, solv_mapped_smiles,
                                   solv_molar_ratios)
        self.salt_graphs = MixData(salt_names, salt_mapped_smiles,
                                   salt_molar_ratios)

        def set_tensor(attr_name, array, unsqueeze=False, squeeze=False):
            if array is not None:
                t = torch.tensor(array, dtype=float_dtype)
                if squeeze:
                    t = t.squeeze()
                if unsqueeze:
                    t = t.unsqueeze(-1)
                setattr(self, attr_name, t)

        set_tensor("mol_emb", mol_emb, squeeze=True)
        set_tensor("frm_emb", frm_emb, squeeze=True)
        set_tensor("all_molar_ratio", all_molar_ratio, squeeze=True)
        set_tensor("bow_vec", bow_vec, squeeze=True)

        set_tensor("concentration", concentration, unsqueeze=True)
        set_tensor("temperature", temperature, unsqueeze=True)
        set_tensor("viscosity", viscosity, unsqueeze=True)
        set_tensor("conductivity", conductivity, unsqueeze=True)
        set_tensor("conductivity_mask", conductivity_mask, unsqueeze=True)
        set_tensor("conductivity_NE", conductivity_NE, unsqueeze=True)
        set_tensor("conductivity_mistry", conductivity_mistry, unsqueeze=True)
        set_tensor("anion_ratio", anion_ratio, unsqueeze=True)
        set_tensor("anion_ratio_mask", anion_ratio_mask, unsqueeze=True)
