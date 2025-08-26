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


class TMDepBlock(torch.nn.Module):

    def __init__(self):
        """
        Implement a VFT fitting for conductivity of the electrolyte formula based on temperature
        """
        super(TMDepBlock, self).__init__()

    def forward(self, data, sigma0, n1, n2, T0, A, B, vis):
        """
        sigma0 and sigmaT are both in log conductivity. 
        """

        sigma_T = sigma0 - vis.unsqueeze(-1) + n1 * torch.log(
            data.concentration.unsqueeze(-1)) - (
                A * data.concentration.unsqueeze(-1).pow(n2) +
                B) / (data.temperature.unsqueeze(-1) - T0)
        return sigma_T


class VFTBlock(torch.nn.Module):

    def __init__(self):
        """
        Implement a VFT fitting for conductivity of the electrolyte formula based on temperature
        """
        super(VFTBlock, self).__init__()

    def forward(self, sigma0, A, T, T0):
        """
        sigma0 and sigmaT are both in log conductivity. 
        """
        eps = 1e-3
        sigma_T = sigma0 + A / (T - T0 + eps)
        return sigma_T


class VFTNoShiftBlock(torch.nn.Module):

    def __init__(self):
        """
        Implement a VFT fitting for conductivity of the electrolyte formula based on temperature
        """
        super(VFTNoShiftBlock, self).__init__()

    def forward(self, sigma0, A, T):
        """
        sigma0 and sigmaT are both in log conductivity. 
        """
        eps = 1e-3
        sigma_T = sigma0 - A / (T + eps)
        return sigma_T


class LinearBlock(torch.nn.Module):

    def __init__(self):
        """
        Implement a VFT fitting for conductivity of the electrolyte formula based on temperature
        """
        super(LinearBlock, self).__init__()

    def forward(self, sigma0, A, T):
        """
        sigma0 and sigmaT are both in log conductivity. 
        """

        sigma_T = sigma0 - A * T
        return sigma_T
