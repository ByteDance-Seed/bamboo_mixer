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

import errno
import logging
import os
import sys
from datetime import datetime
from typing import Union

import torch
from torch import LongTensor, Tensor
from torch_geometric.utils import cumsum, scatter


def get_timestamp():
    return datetime.now().strftime("%y_%m_%d_%H_%M_%S")


def _timestamp_formatter(with_lineno: bool = False) -> logging.Formatter:
    return logging.Formatter(
        fmt='[%(asctime)s PID %(process)d] %(levelname)s %(message)s'
        if not with_lineno else
        '[%(asctime)s PID %(process)d] %(levelname)s %(message)s \t(%(pathname)s:%(lineno)d)',
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_default_logging(stdout: bool = True,
                          *,
                          file_path=None,
                          file_mode='a',
                          level=logging.INFO,
                          formatter: Union[str, logging.Formatter] = 'time'):
    if formatter == 'time':
        formatter = _timestamp_formatter()
    elif formatter == 'lineno':
        formatter = _timestamp_formatter(with_lineno=True)
    elif not isinstance(formatter, logging.Formatter):
        raise ValueError(f'invalid formatter {formatter}')

    logger = logging.getLogger()
    logger.setLevel(level)

    # clear old handlers
    logger.handlers = []

    if stdout:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file_path != None:
        file_handler = logging.FileHandler(file_path, mode=file_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_data_file_path(relative_path: str, package_name: str) -> str:

    from importlib.resources import files

    file_path = files(package_name) / relative_path

    if not file_path.is_file():
        try_path = files(package_name) / f"data/{relative_path}"
        if try_path.is_file():
            file_path = try_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    file_path)

    return file_path.as_posix()  # type: ignore


def to_dense_batch(x: Tensor,
                   batch: Tensor,
                   fill_value=0.,
                   fill_rand=False,
                   need_mask=False,
                   max_num_nodes=None):
    """
    modified from https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/utils/_to_dense_batch.py

    x: [n_atom, ...]
    batch: [n_atom]
    fill_rand: fill with random values
    """

    batch_size = int(batch.max()) + 1
    num_nodes = scatter(batch.new_ones(x.size(0)),
                        batch,
                        dim=0,
                        dim_size=batch_size,
                        reduce='sum')  # [n_batch]
    cum_nodes = cumsum(num_nodes)

    if max_num_nodes is not None:
        max_num_nodes = max(max_num_nodes, int(num_nodes.max()))
    else:
        max_num_nodes = int(num_nodes.max())
    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]

    if fill_rand:
        out = torch.rand(size, device=x.device, dtype=x.dtype)
    else:
        out = torch.as_tensor(fill_value, device=x.device, dtype=x.dtype)
        out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    if need_mask:
        mask = torch.zeros(batch_size * max_num_nodes,
                           dtype=torch.bool,
                           device=x.device)
        mask[idx] = 1
        mask = mask.view(batch_size, max_num_nodes)
    else:
        mask = None

    return out, mask
