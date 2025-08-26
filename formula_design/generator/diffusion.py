# Some modules used in this code are adapted from: https://github.com/hspark1212/chemeleon/blob/df4c67449265f5dfad66b1300ba438b35b9fcb7c/chemeleon/modules/cspnet.py
# Copyright (c) 2024 Hyunsoo Park
# Copyright (c) 2025 ByteDance Ltd. 
# SPDX-License-Identifier: MIT license
#
# This file has been modified by ByteDance Ltd. on 08/26/2025
#
# Original file was released under MIT license, with the full license text
# available at https://github.com/hspark1212/chemeleon/blob/df4c67449265f5dfad66b1300ba438b35b9fcb7c/LICENSE.
#
# This modified file is released under Apache License 2.0.

import math
import os
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from formula_design.data import Data
from formula_design.utils.diff_utils import BetaScheduler, SigmaScheduler

from .decoder import FormulaDecoder
from .unet1d import Unet1D


### Model definition
class ScalarEmbed(nn.Module):

    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        # x shape: [batch_size] or [batch_size, 1]
        return self.net(x).unsqueeze(1)  # 输出 [batch_size, 1, embed_dim]


class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FormulaDiffusion(nn.Module):

    def __init__(self, unet1d_config: dict, beta_scheduler_config: dict,
                 sigma_scheduler_config: dict, diff_config: dict) -> None:
        super().__init__()

        self.noise_predictor = Unet1D(**unet1d_config)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_config)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_config)
        self.time_dim = diff_config.get("time_dim")

        # self.prop_embedding = ScalarEmbed(diff_config.get("prop_embed_dim"))

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def forward(self, data: Data):

        batch_size = data.frm_emb.size()[0]

        device = next(self.noise_predictor.parameters()).device

        times = self.beta_scheduler.uniform_sample_t(batch_size, device)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        frm_emb = data["frm_emb"].unsqueeze(1)
        prop = torch.cat([
            data['conductivity'].unsqueeze(-1),
            data['anion_ratio'].unsqueeze(-1)
        ],
                         dim=-1)

        #rand_e, rand_p = torch.randn_like(frm_emb), torch.randn_like(p_embed)
        rand_e = torch.randn_like(frm_emb)

        input_e = c0[:, None, None] * frm_emb + c1[:, None, None] * rand_e

        pred_e = self.noise_predictor(times, input_e, prop)

        return {"pred_e": pred_e, "rand_e": rand_e}

    @torch.no_grad()
    def sample(self, prop):

        batch_size = prop.size()[0]
        device = next(self.noise_predictor.parameters()).device

        e_T = torch.randn([batch_size, 1, 384]).to(device)

        time_start = self.beta_scheduler.timesteps

        traj = {
            time_start: {
                'frm_emb': e_T,
            }
        }

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device=device)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            e_t = traj[t]['frm_emb']

            rand_e = torch.randn_like(e_T) if t > 1 else torch.zeros_like(e_T)

            pred_e = self.noise_predictor(times, e_t, prop)
            e_t_minus_1 = c0 * (e_t - c1 * pred_e) + sigmas * rand_e

            traj[t - 1] = {
                'frm_emb': e_t_minus_1,
            }

        traj_stack = {
            'all_frm_emb':
            torch.stack(
                [traj[i]['frm_emb'] for i in range(time_start, -1, -1)]),
        }

        return traj[0], traj_stack

    def compute_stats(self, output_dict, prefix):

        loss_emb = output_dict['loss_emb']
        loss_prop = output_dict['loss_prop']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_emb_loss': loss_emb,
            f'{prefix}_prop_loss': loss_prop
        }

        return log_dict, loss
