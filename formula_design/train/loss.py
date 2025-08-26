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

import logging
from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor

from formula_design.data import MixData

logger = logging.getLogger(__name__)


class LossType(Enum):

    Tm_MSE = 1
    bp_MSE = 2
    nD_MSE = 3
    nD_liquid_MSE = 4
    pka_a_MSE = 5
    pka_b_MSE = 6
    dc_MSE = 7
    ST_MSE = 8
    density_MSE = 9
    vis_MSE = 10
    vapP_MSE = 11
    conductivity_MAE = 12
    conductivity_NE_MAE = 13
    conductivity_mistry_MAE = 14
    anion_ratio_MAE = 15
    emb_inv_MSE = 16
    emb_inv_MAE = 17
    dist_MAE = 18
    dist_MSE = 19
    num_mol_penalty = 20
    bow_MAE = 21
    bow_MSE = 22
    bow_BCE = 23
    mol_emb_MAE = 24
    mol_emb_MSE = 25
    diff_MSE = 26


def loss_func(preds: dict, data: MixData, loss_type: LossType, **kwargs):

    if loss_type is LossType.Tm_MSE:

        label = data.Tm
        loss = (preds["Tm"].squeeze() - label)**2
        assert loss.shape == data.Tm_mask.squeeze().shape

        loss = loss * data.Tm_mask.squeeze()

        loss = torch.sum(loss)
        num_data = torch.sum(data.Tm_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.bp_MSE:

        label = data.bp

        loss = (preds["bp"].squeeze() - label)**2
        assert loss.shape == data.bp_mask.squeeze().shape
        loss = loss * data.bp_mask.squeeze()
        loss = torch.sum(loss)

        num_data = torch.sum(data.bp_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.nD_MSE:

        label = data.nD

        loss = (preds["nD"].squeeze() - label)**2
        assert loss.shape == data.nD_mask.squeeze().shape
        loss = loss * data.nD_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.nD_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.nD_liquid_MSE:

        label = data.nD_liquid

        loss = (preds["nD_liquid"].squeeze() - label)**2
        assert loss.shape == data.nD_liquid_mask.squeeze().shape
        loss = loss * data.nD_liquid_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.nD_liquid_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.pka_a_MSE:

        label = data.pka_a

        loss = (preds["pka_a"].squeeze() - label)**2
        assert loss.shape == data.pka_a_mask.squeeze().shape
        loss = loss * data.pka_a_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.pka_a_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.pka_b_MSE:

        label = data.pka_b

        loss = (preds["pka_b"].squeeze() - label)**2
        assert loss.shape == data.pka_b_mask.squeeze().shape
        loss = loss * data.pka_b_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.pka_b_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.dc_MSE:

        label = data.dc

        loss = (preds["dc"].squeeze() - label)**2
        assert loss.shape == data.dc_mask.squeeze().shape
        loss = loss * data.dc_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.dc_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.ST_MSE:

        label = data.ST

        loss = (preds["ST"].squeeze() - label)**2
        assert loss.shape == data.ST_mask.squeeze().shape
        loss = loss * data.ST_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.ST_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.density_MSE:

        label = data.density

        loss = (preds["density"].squeeze() - label)**2
        assert loss.shape == data.density_mask.squeeze().shape
        loss = loss * data.density_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.density_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.vis_MSE:

        label = data.vis

        loss = (preds["vis"].squeeze() - label)**2
        assert loss.shape == data.vis_mask.squeeze().shape
        loss = loss * data.vis_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.vis_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.vapP_MSE:

        label = data.vapP

        loss = (preds["vapP"].squeeze() - label)**2
        assert loss.shape == data.vapP_mask.squeeze().shape
        loss = loss * data.vapP_mask.squeeze()
        loss = torch.sum(loss)
        num_data = torch.sum(data.vapP_mask.squeeze())
        if num_data == 0:
            loss *= 0
        else:
            loss /= num_data

    elif loss_type is LossType.conductivity_MAE:

        label = data.conductivity
        loss = torch.abs(preds["conductivity"].squeeze() - label)

        if hasattr(data, 'conductivity_mask'):
            assert loss.shape == data.conductivity_mask.squeeze().shape
            loss = loss * data.conductivity_mask.squeeze()
            loss = torch.sum(loss)
            num_data = torch.sum(data.conductivity_mask.squeeze())
            if num_data == 0:
                loss *= 0
            else:
                loss /= num_data

        else:
            loss = torch.mean(loss)

    elif loss_type is LossType.conductivity_NE_MAE:
        label = data.conductivity_NE
        loss = torch.abs(preds["conductivity_NE"].squeeze() - label)
        loss = torch.mean(loss)

    elif loss_type is LossType.conductivity_mistry_MAE:
        label = data.conductivity_mistry
        loss = torch.abs(preds["conductivity_mistry"].squeeze() - label)
        loss = torch.mean(loss)

    elif loss_type is LossType.anion_ratio_MAE:

        label = data.anion_ratio
        loss = torch.abs(preds["anion_ratio"].squeeze() - label)

        if hasattr(data, 'anion_ratio_mask'):
            assert loss.shape == data.anion_ratio_mask.squeeze().shape
            loss = loss * data.anion_ratio_mask.squeeze()
            loss = torch.sum(loss)
            num_data = torch.sum(data.anion_ratio_mask.squeeze())
            if num_data == 0:
                loss *= 0
            else:
                loss /= num_data

        else:
            loss = torch.mean(loss)

    elif loss_type is LossType.emb_inv_MSE:

        label = data.frm_emb
        loss = (preds["frm_emb"].squeeze() - label)**2
        loss = torch.mean(loss)

    elif loss_type is LossType.emb_inv_MAE:

        label = data.frm_emb
        loss = torch.abs(preds["frm_emb"].squeeze() - label)
        loss = torch.mean(loss)

    elif loss_type is LossType.dist_MSE:

        loss = preds["min_dist"].squeeze()**2
        loss = torch.mean(loss)

    elif loss_type is LossType.dist_MAE:

        loss = torch.abs(preds["min_dist"].squeeze())
        loss = torch.mean(loss)

    elif loss_type is LossType.num_mol_penalty:

        loss = torch.mean(preds["num_mol"].squeeze())

    elif loss_type is LossType.bow_MAE:

        label = data.bow_vec

        loss = torch.abs(preds["bow_vec"].squeeze() - label)
        loss = torch.mean(loss)

    elif loss_type is LossType.bow_MSE:

        label = data.bow_vec
        loss = (preds["bow_vec"].squeeze() - label)**2
        loss = torch.mean(loss)

    elif loss_type is LossType.bow_BCE:

        label = data.bow_vec
        pred = preds["bow_vec"].squeeze()  # normalization
        loss = F.binary_cross_entropy(pred, label)

    elif loss_type is LossType.mol_emb_MAE:

        label = data.mol_emb
        loss = torch.abs(preds["mol_emb"].squeeze() - label)
        loss = torch.mean(loss)

    elif loss_type is LossType.mol_emb_MSE:

        label = data.mol_emb
        loss = (preds["mol_emb"].squeeze() - label)**2
        loss = torch.mean(loss)

    elif loss_type is LossType.diff_MSE:
        loss = F.mse_loss(preds["pred_e"].squeeze(), preds["rand_e"].squeeze())

    else:
        raise NotImplementedError(loss_type)

    return loss
