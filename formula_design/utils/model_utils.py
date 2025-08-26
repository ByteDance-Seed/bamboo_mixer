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

from enum import Enum

import torch


class ActivationFunction(Enum):
    ELU = torch.nn.ELU
    SIGMOID = torch.nn.Sigmoid
    GELU = torch.nn.GELU
    TANH = torch.nn.Tanh
    SOFTPLUS = torch.nn.Softplus
    NONE = None

    @staticmethod
    def get_activation(name):
        if name is None:
            return None
        try:
            # Convert the name to uppercase to match the Enum members
            return ActivationFunction[name.upper()].value()
        except KeyError:
            raise NotImplementedError(
                f"Activation '{name}' is not implemented")