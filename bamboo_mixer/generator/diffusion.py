# Some modules are adapted from https://github.com/hspark1212/chemeleon/blob/df4c67449265f5dfad66b1300ba438b35b9fcb7c/chemeleon/modules/cspnet.py
# Licensed under the MIT License.

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
import torch.nn as nn
from tqdm import tqdm

from bamboo_mixer.data import FormulaData
from bamboo_mixer.generator.unet1d import Unet1D
from bamboo_mixer.utils.diff_utils import BetaScheduler

logger = logging.getLogger(__name__)


class FormulaDiffusion(nn.Module):

    def __init__(self, unet1d_config: dict, beta_scheduler_config: dict) -> None:
        super().__init__()

        assert unet1d_config['prop_dim'] == 2
        self.noise_predictor = Unet1D(**unet1d_config)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_config)

    def load_ckpt(self, ckpt_path: str, device: torch.device = None):
        if device is None:
            logger.info(f'loading trained FormulaDiffusion ckpt to device of model')
            self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        else:
            logger.info(f'loading trained FormulaDiffusion ckpt to device {device}')
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict']
            self.load_state_dict(state_dict)
            self.to(device)

    def forward(self, data: FormulaData):

        batch_size = data.frm_emb.size()[0]

        device = next(self.noise_predictor.parameters()).device

        times = self.beta_scheduler.uniform_sample_t(batch_size, device)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        frm_emb = data["frm_emb"].unsqueeze(1)
        prop = torch.cat([data['conductivity'].unsqueeze(-1), data['anion_ratio'].unsqueeze(-1)], dim=-1)

        rand_e = torch.randn_like(frm_emb)

        input_e = c0[:, None, None] * frm_emb + c1[:, None, None] * rand_e

        pred_e = self.noise_predictor(times, input_e, prop)

        return {"pred_e": pred_e, "rand_e": rand_e}

    @torch.no_grad()
    def sample(self, prop: torch.Tensor):

        batch_size = prop.size()[0]
        device = next(self.noise_predictor.parameters()).device

        e_T = torch.randn([batch_size, 1, 384]).to(device)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start: {'frm_emb': e_T,}}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size,), t, device=device)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            e_t = traj[t]['frm_emb']

            rand_e = torch.randn_like(e_T) if t > 1 else torch.zeros_like(e_T)

            pred_e = self.noise_predictor(times, e_t, prop)
            e_t_minus_1 = c0 * (e_t - c1 * pred_e) + sigmas * rand_e

            traj[t - 1] = {'frm_emb': e_t_minus_1}

        traj_stack = {'all_frm_emb': torch.stack([traj[i]['frm_emb'] for i in range(time_start, -1, -1)])}

        return traj[0], traj_stack

    def compute_stats(self, output_dict: dict, prefix: str):

        loss_emb = output_dict['loss_emb']
        loss_prop = output_dict['loss_prop']
        loss = output_dict['loss']

        log_dict = {f'{prefix}_loss': loss, f'{prefix}_emb_loss': loss_emb, f'{prefix}_prop_loss': loss_prop}

        return log_dict, loss
