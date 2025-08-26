# Some modules used in this code are adapted from: https://github.com/hspark1212/chemeleon/blob/main/chemeleon/utils/diff_utils.py
# Copyright (c) 2024 Hyunsoo Park
# Copyright (c) 2025 ByteDance Ltd. 
# SPDX-License-Identifier: MIT license
#
# This file has been modified by ByteDance Ltd. on 08/26/2025
#
# Original file was released under MIT license, with the full license text
# available at https://github.com/hspark1212/chemeleon/blob/main/LICENSE.
#
# This modified file is released under Apache License 2.0.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)**2


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i)**2 / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma**2 * torch.exp(
            -(x + T * i)**2 / 2 / sigma**2)
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(dim=0)


class BetaScheduler(nn.Module):

    def __init__(self,
                 timesteps,
                 scheduler_mode,
                 beta_start=0.0001,
                 beta_end=0.02):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown scheduler_mode: {scheduler_mode}")

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)

        sigmas[1:] = betas[1:] * (1. - alphas_cumprod[:-1]) / (
            1. - alphas_cumprod[1:])

        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sigmas', sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)


class SigmaScheduler(nn.Module):

    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(
            np.exp(
                np.linspace(np.log(sigma_begin), np.log(sigma_end),
                            timesteps)))

        sigmas_norm_ = sigma_norm(sigmas)

        self.register_buffer('sigmas',
                             torch.cat([torch.zeros([1]), sigmas], dim=0))
        self.register_buffer('sigmas_norm',
                             torch.cat([torch.ones([1]), sigmas_norm_], dim=0))

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)
