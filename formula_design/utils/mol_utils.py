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

from formula_design.mol import Molecule, MoleculeGraph, rkutil


def get_ring_info(mol_graph: MoleculeGraph):
    ring_con, min_ring_size = [], []
    for atom in mol_graph.get_atoms():
        ring_con.append(atom.ring_connectivity)
        min_ring_size.append(atom.min_ring_size)
    return ring_con, min_ring_size


def find_equivalent_index(
        mol: Molecule,
        bond_index: list[tuple[int]]) -> tuple[list[int], list[int]]:
    atom_ranks = rkutil.find_symmetry_rank(mol.get_rkmol())
    atom_rec = {}
    atom_equi_index = []
    for i, rank in enumerate(atom_ranks):
        if rank in atom_rec:
            atom_equi_index.append(atom_rec[rank])
        else:
            atom_rec[rank] = i
            atom_equi_index.append(i)

    bond_rec = {}
    bond_equi_index = []
    for i, bond in enumerate(bond_index):
        bond_rank = tuple(sorted([atom_ranks[b] for b in bond]))
        if bond_rank in bond_rec:
            bond_equi_index.append(bond_rec[bond_rank])
        else:
            bond_rec[bond_rank] = i
            bond_equi_index.append(i)
    return atom_equi_index, bond_equi_index


def match_linear_proper(mol: Molecule):
    linear_patterns = [
        "[*:1]~[#6X2;!r5;!r6:2]~[*:3]~[*:4]",
        "[!$([#7H1]):1]~[#7X2+:2]~[!$([#7H1]):3]~[*:4]"
    ]
    rkmol = mol.get_rkmol()
    match_results = set()
    for pattern in linear_patterns:
        matches = rkutil.find_mapped_smarts_matches(rkmol, pattern)
        for atomidxs in matches:
            ordered_atomidxs = rkutil.sorted_atomids(atomidxs)
            match_results.add(
                ordered_atomidxs)  # update, overwrite previous match
    return match_results
