# Some modules used in this code are adapted from: https://github.com/juho-lee/set_transformer/blob/master/modules.py
# Copyright (c) 2020 Juho Lee
# Copyright (c) 2025 ByteDance Ltd. 
# SPDX-License-Identifier: MIT license
#
# This file has been modified by ByteDance Ltd. on 08/26/2025
#
# Original file was released under MIT license, with the full license text
# available at https://github.com/juho-lee/set_transformer/blob/master/LICENSE.
#
# This modified file is released under Apache License 2.0.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from formula_design.data import Data
from formula_design.generator.encoder import AggrBlock
from formula_design.utils.definitions import SALTS_MAP, SOLVENTS_MAP


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MultiHeadAttention, self).__init__()
        """
        Args:
            dim_Q: dimension of query
            dim_K: dimension of key
            dim_V: dimension of value
            num_heads: number of heads
            ln: normalize the output
        """
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln = ln
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        """
        Args:
            Q: (batch_size, num_mol, dim_Q)
            K: (batch_size, num_mol, dim_K) 
        Returns:
            O: (batch_size, num_mol, dim_V) 
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(
            Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.ln:
            O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
            O = O + F.relu(self.fc_o(O))
            O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        else:
            O = self.fc_o(O)

        return O


class SetTransformerBlock(nn.Module):

    def __init__(self, num_heads, emb_dim, **config):
        super(SetTransformerBlock, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.hidden_dim = config.get("hidden_dim")
        self.max_num_mol = config.get("max_num_mol")
        self.num_layers = config.get("num_layers")

        # Projection layer to convert the shape of input embedding to hidden dim

        self.proj = nn.Sequential(nn.Linear(self.emb_dim, self.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_dim, self.hidden_dim))

        # Learnable Queries
        self.query = nn.Parameter(
            torch.randn(self.max_num_mol, self.hidden_dim))

        self.mab = nn.ModuleList([
            MultiHeadAttention(self.hidden_dim,
                               self.hidden_dim,
                               self.hidden_dim,
                               self.num_heads,
                               ln=False) for _ in range(self.num_layers)
        ])

        # Molar ratio predictor
        self.molar_ratio_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, emb):
        """
        Args:
            emb: (batch_size, emb_dim)
        Returns:
            h: (batch_size, num_mol, hidden_dim)
        """
        batch_size = emb.size(0)

        # obtain context/memory from input embedding
        context = self.proj(emb).unsqueeze(1).repeat(1, self.max_num_mol, 1)

        # Reshape learnable query for the whole batch
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)

        # Set transformer
        for layer in self.mab:
            queries = layer(queries, context)

        # Molar ratio prediction
        molar_ratio_logits = self.molar_ratio_predictor(
            torch.cat((queries, context),
                      dim=-1)).squeeze(-1)  # Shape: (batch_size, num_mol)

        # Normalization
        molar_ratios = F.softmax(molar_ratio_logits, dim=-1).unsqueeze(
            -1)  # Shape: (batch_size, num_mol)

        return queries, molar_ratios


class BowDecoder(nn.Module):

    def __init__(self, formula_emb_dim: int, hidden_dim: int,
                 solv_bow_dim: int, salt_bow_dim: int):
        """
        Args:
            emb_dim: dimension of the input embedding
            mol_dict: dictionary of molecules
        """
        super(BowDecoder, self).__init__()
        self.frm_emb_dim = formula_emb_dim
        self.hidden_dim = hidden_dim
        self.solv_bow_dim = solv_bow_dim
        self.salt_bow_dim = salt_bow_dim

        self.proj_solv = nn.Sequential(
            nn.Linear(self.frm_emb_dim * 2 // 3, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.solv_bow_dim), nn.Softmax(dim=-1))

        self.proj_salt = nn.Sequential(
            nn.Linear(self.frm_emb_dim // 3, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.salt_bow_dim), nn.Softmax(dim=-1))

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def forward(self, data: Data):
        """
        Args:
            data: Data object
        Returns:
            bow: (batch_size, bow_dim)
        """
        device = next(self.proj_solv.parameters()).device

        emb = data.frm_emb.to(device)

        solv_emb = emb[:, :emb.size(1) * 2 //
                       3]  # Shape: (batch_size, hidden_dim)
        salt_emb = emb[:, emb.size(1) * 2 //
                       3:]  # Shape: (batch_size, hidden_dim // 2)

        solv_bow = self.proj_solv(solv_emb)
        salt_bow = self.proj_salt(salt_emb)
        bow_vec = torch.cat([solv_bow, salt_bow], dim=-1)

        return {"bow_vec": bow_vec}


class FormulaDecoder(nn.Module):

    def __init__(self, formula_emb_dim: int, set_transformer_block: dict,
                 aggr_block: dict, mol_dict: dict):
        """
        Args:
            emb_dim: dimension of input formula
            hidden_dim: hidden dimension of each molecule
            max_num_mol: maximum number of molecules (similar as the maximum sequence length in LM)
            num_heads: number of heads in self-attention
        """
        super(FormulaDecoder, self).__init__()

        self.frm_emb_dim = formula_emb_dim

        self.solv_dict = torch.load(mol_dict["solv_dict_path"])
        self.salt_dict = torch.load(mol_dict["salt_dict_path"])

        # Number of heads used for solvent is 2 times of salt, as a result, the split emb dim for solvent and salt are emb_dim * 2 // 3 and emb_dim // 3
        # Projection layer to convert the shape of input embedding to hidden dim
        # Number of heads used for solvent is 2 times of salt, as a result, the split emb dim for solvent and salt are emb_dim * 2 // 3 and emb_dim // 3
        self.solv_sab = SetTransformerBlock(num_heads=4,
                                            emb_dim=formula_emb_dim * 2 // 3,
                                            **set_transformer_block)
        self.salt_sab = SetTransformerBlock(num_heads=4,
                                            emb_dim=formula_emb_dim // 3,
                                            **set_transformer_block)

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def predict(self, emb: torch.Tensor):
        """
        Args:
            emb: (batch_size, emb_dim)
        """

        device = emb.device

        solv_emb = emb[:, :emb.size(1) * 2 //
                       3]  # Shape: (batch_size, hidden_dim)
        salt_emb = emb[:, emb.size(1) * 2 //
                       3:]  # Shape: (batch_size, hidden_dim // 2)

        # obtain molecular embedding for solvents and salts
        solv_mol_emb, solv_molar_ratios = self.solv_sab(solv_emb)
        norm_solv_mol_emb = F.normalize(solv_mol_emb, dim=-1, p=2)
        salt_mol_emb, salt_molar_ratios = self.salt_sab(salt_emb)
        norm_salt_mol_emb = F.normalize(salt_mol_emb, dim=-1, p=2)

        solv_dict = F.normalize(self.solv_dict, dim=-1,
                                p=2).to(device).unsqueeze(0).expand(
                                    solv_mol_emb.size()[0], -1, -1)
        salt_dict = F.normalize(self.salt_dict, dim=-1,
                                p=2).to(device).unsqueeze(0).expand(
                                    salt_mol_emb.size()[0], -1, -1)

        solv_dist = torch.cdist(norm_solv_mol_emb, solv_dict, p=2)
        salt_dist = torch.cdist(norm_salt_mol_emb, salt_dict, p=2)

        # 1000 is used as a constant to adjust the probability, higer scaler will make the probability closer to 1, like 1/temperature in language model
        solv_prob_matrix = F.softmax(-1000 * solv_dist, dim=-1)
        salt_prob_matrix = F.softmax(-1000 * salt_dist, dim=-1)

        # Weighted sum to obtain BoW representation
        solv_bow_vectors = torch.bmm(solv_prob_matrix.transpose(1, 2),
                                     solv_molar_ratios).squeeze(-1)
        solv_bow_vectors = F.normalize(solv_bow_vectors, dim=-1,
                                       p=1)  # Normalize by the sum
        salt_bow_vectors = torch.bmm(salt_prob_matrix.transpose(1, 2),
                                     salt_molar_ratios).squeeze(-1)
        salt_bow_vectors = F.normalize(salt_bow_vectors, dim=-1,
                                       p=1)  # Normalize by the sum

        bow_vec = torch.cat([solv_bow_vectors, salt_bow_vectors], dim=-1)

        return {"bow_vec": bow_vec}

    def forward(self, data: Data):
        """
        Args:
            data: FormulaData
        Returns:
            molecule_embeddings: (batch_size, num_mol, hidden_dim)
            mass_fractions: (batch_size, num_mol, 1)
        """

        device = next(self.solv_sab.parameters()).device

        emb = data.frm_emb.to(device)

        solv_emb = emb[:, :emb.size(1) * 2 //
                       3]  # Shape: (batch_size, hidden_dim)
        salt_emb = emb[:, emb.size(1) * 2 //
                       3:]  # Shape: (batch_size, hidden_dim // 2)

        # obtain molecular embedding for solvents and salts
        solv_mol_emb, solv_molar_ratios = self.solv_sab(solv_emb)
        norm_solv_mol_emb = F.normalize(solv_mol_emb, dim=-1, p=2)
        salt_mol_emb, salt_molar_ratios = self.salt_sab(salt_emb)
        norm_salt_mol_emb = F.normalize(salt_mol_emb, dim=-1, p=2)

        solv_dict = F.normalize(self.solv_dict, dim=-1,
                                p=2).to(device).unsqueeze(0).expand(
                                    solv_mol_emb.size()[0], -1, -1)
        salt_dict = F.normalize(self.salt_dict, dim=-1,
                                p=2).to(device).unsqueeze(0).expand(
                                    salt_mol_emb.size()[0], -1, -1)

        solv_dist = torch.cdist(norm_solv_mol_emb, solv_dict, p=2)
        salt_dist = torch.cdist(norm_salt_mol_emb, salt_dict, p=2)

        # 1000 is used as a constant to adjust the probability, higer scaler will make the probability closer to 1, like 1/temperature in language model
        solv_prob_matrix = F.softmax(-1000 * solv_dist, dim=-1)
        salt_prob_matrix = F.softmax(-1000 * salt_dist, dim=-1)

        # Weighted sum to obtain BoW representation
        solv_bow_vectors = torch.bmm(solv_prob_matrix.transpose(1, 2),
                                     solv_molar_ratios).squeeze(-1)
        solv_bow_vectors = F.normalize(solv_bow_vectors, dim=-1,
                                       p=1)  # Normalize by the sum
        salt_bow_vectors = torch.bmm(salt_prob_matrix.transpose(1, 2),
                                     salt_molar_ratios).squeeze(-1)
        salt_bow_vectors = F.normalize(salt_bow_vectors, dim=-1,
                                       p=1)  # Normalize by the sum

        bow_vec = torch.cat([solv_bow_vectors, salt_bow_vectors], dim=-1)

        # molecular embedding all the whole formula
        mol_emb = torch.cat([solv_mol_emb, salt_mol_emb],
                            dim=-2)  # Shape: (batch_size, num_mol, hidden_dim)

        return {"bow_vec": bow_vec, "mol_emb": mol_emb}
