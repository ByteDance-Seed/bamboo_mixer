# The code is adapted from https://github.com/bytedance/byteff/blob/master/byteff/train/trainer.py

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

import json
import os
import random
import textwrap
from collections import OrderedDict
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from bamboo_mixer.data import Data
from bamboo_mixer.train.config import TrainConfig
from bamboo_mixer.train.loss import LossType, loss_func
from bamboo_mixer.utils import get_default_device, setup_default_logging


class BaseTrainer:

    def __init__(
        self,
        config: Union[str, dict],
        load_data: bool = True,
        make_working_dir: bool = True,
    ) -> None:

        self.config = TrainConfig(config, make_working_dir=make_working_dir)
        self.logger = self._init_logger()

        self.model: torch.nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None
        self.optimal_state_dict: dict[str, torch.Tensor] = None
        self.early_stop_count: int = 0

        self.load_data: bool = load_data
        self.datasets, self.train_dls, self.valid_dls = [], [], []

        # training states
        self.epoch: int = 0
        self.best_valid_loss: float = torch.finfo().max
        self.early_stop_count: int = 0
        self.train_history: list[float] = [[] for _ in self.config.dataset]
        self.valid_history: list[float] = [[] for _ in self.config.dataset]
        self.loss_term_history: list[float] = [[] for _ in self.config.dataset]
        self.trainer_state_variables = [
            'epoch', 'best_valid_loss', 'early_stop_count', 'train_history', 'valid_history', 'loss_term_history'
        ]

        self.device: torch.device = get_default_device()
        self.logger.info(f'using device {self.device}')
        self.start()

    def start(self):
        self._set_seed(self.config.meta['random_seed'])

        # dtype
        dtype = torch.float64 if self.config.meta['fp64'] else torch.float32
        torch.set_default_dtype(dtype)
        self.logger.info('set default dtype to %s', dtype)

        if self.load_data:
            self.logger.info("loading dataset")
            self.datasets, self.train_dls, self.valid_dls = self._load_data(self.config.dataset)
        else:
            self.datasets, self.train_dls, self.valid_dls = [], [], []

        self.logger.info("loading model")
        ckpt = self._init_model()

        if ckpt is not None:
            self.logger.info("loading check point from %s", ckpt)
            # all models should implement this load_ckpt function
            self.model.load_ckpt(ckpt, device=self.device)

    def _init_model(self):
        raise NotImplementedError

    def _load_data(self, dataset_config):
        raise NotImplementedError

    def _train_valid_split(self, dataset, config):
        raise NotImplementedError

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_optimizer_scheduler(self):
        optim_config = self.config.training['optimizer'].copy()

        optim_type = optim_config.pop('type')

        lr = optim_config.pop('lr')
        self.logger.info(f"initiating optimizer, lr: {lr}")

        parameters = self.model.parameters()

        self.optimizer = getattr(optim, optim_type)(parameters, lr=lr)

        sched_config = self.config.training.get('scheduler', None)

        if sched_config is not None:
            sched_config = sched_config.copy()
            sched_type = sched_config.pop('type')
            self.scheduler = getattr(optim.lr_scheduler, sched_type)(self.optimizer, **sched_config)

    def _init_logger(self):
        """set logging config for a logger"""
        log_path = os.path.join(self.config.work_folder, 'trainer.log')
        logger = setup_default_logging(stdout=True, file_path=log_path)
        logger.info(f"writing logs to {log_path}")
        return logger

    def save_ckpt(self, save_path: Union[str, None] = None):
        if save_path is None:
            ckpt_savepath = os.path.join(self.config.ckpt_folder, f"ckpt_epoch_{self.epoch}.pt")
        else:
            ckpt_savepath = save_path

        self.logger.info(f'saving ckpt to: {ckpt_savepath}')
        sd = {
            'model_state_dict': self.model.state_dict(),
            'optimal_state_dict': self.optimal_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'trainer_state_dict': {
                name: getattr(self, name) for name in self.trainer_state_variables
            },
        }
        torch.save(sd, ckpt_savepath)

    # def load_ckpt(self, ckpt_path: str, model_only: bool = True):
    #     sd = torch.load(ckpt_path, map_location=self.device)
    #     model = self.model
    #     try:
    #         model.load_state_dict(sd['model_state_dict'])
    #     except RuntimeError as e:
    #         self.logger.warning("%s", e)
    #         self.logger.warning("load state dict failed, use strict=False")
    #         new_sd = self.model.state_dict()
    #         old_sd = sd['model_state_dict']
    #         for k in list(old_sd.keys()):
    #             if k in new_sd and new_sd[k].shape != old_sd[k].shape:
    #                 old_sd.pop(k)
    #         model.load_state_dict(old_sd, strict=False)
    #     if model_only:
    #         return

    #     self.optimal_state_dict = sd['optimal_state_dict']
    #     self.optimizer.load_state_dict(sd['optimizer_state_dict'])
    #     if self.scheduler is not None:
    #         self.scheduler.load_state_dict(sd['scheduler_state_dict'])

    #     for k in self.trainer_state_variables:
    #         setattr(self, k, sd['trainer_state_dict'][k])

    def save_optimal(self):
        optimal_state_dict: dict[str, torch.Tensor] = self.model.state_dict().copy()
        self.optimal_state_dict = OrderedDict()
        for k, v in optimal_state_dict.items():
            self.optimal_state_dict[k] = v.clone().detach()
        torch.save({'model_state_dict': self.optimal_state_dict}, self.config.optimal_path())

    def load_optimal(self):
        if self.optimal_state_dict is not None:
            self.model.load_state_dict(self.optimal_state_dict)

    def calc_loss(self,
                  pred: dict,
                  data: Data,
                  dataset_index: int,
                  is_valid=True) -> Tuple[list[torch.FloatTensor], dict[str, float]]:

        loss_types: list[dict] = self.config.dataset[dataset_index]['loss']

        losses = [0.]
        loss_terms = {}

        for dct in loss_types:
            name = dct['loss_type']
            kwargs = dct.get('kwargs', {})
            loss_type = getattr(LossType, name)
            loss = loss_func(pred, data, loss_type=loss_type, **kwargs)
            weight = dct.get('valid_weight', dct['weight']) if is_valid else dct['weight']

            loss_terms[name] = loss.detach().item()
            losses[0] += loss * weight

        return losses, loss_terms

    def valid_epoch(self):

        self.model.eval()
        self.logger.debug('starting validation epoch')

        averaged_loss = 0.
        for ids, valid_dl in enumerate(self.valid_dls):

            tot_loss = 0.
            nbatch = torch.tensor(0, dtype=torch.int64, device=self.device)
            ds_conf = self.config.dataset[ids]
            for vdata in valid_dl:

                vdata: Data = vdata.to(self.device)
                nbatch += vdata.counts.shape[0]
                pred = self.model(vdata)  # pylint: disable=not-callable
                with torch.no_grad():
                    losses, loss_terms = self.calc_loss(pred, vdata, ids, is_valid=True)
                    losses = torch.tensor([l.item() for l in losses], dtype=torch.float64, device=self.device)

                tot_loss += losses * vdata.counts.shape[0]
                del pred, losses

            losses = tot_loss / nbatch

            averaged_loss += losses[0] * ds_conf['loss_weight']

            losses = losses.detach().tolist()
            self.valid_history[ids].append([self.epoch, 0, losses[0]])

            loss_str = ", ".join([f"{name}: {value:.4f}" for name, value in loss_terms.items()])

            self.logger.info(f'valid epoch {self.epoch}, dataset {ids}, loss: {losses[0]}, loss_term: {loss_str}')

        self.logger.info(f'valid epoch {self.epoch} combined loss {averaged_loss}')

        return averaged_loss

    def train_epoch(self):
        # train goes over same steps on each dataloader_impl
        # the train loop takes one minibatch from each train_dataset,
        # calculates the grad, saves the grad and continues to next minibatch from next dataloader_impl

        self.model.train()
        self.logger.debug('starting training epoch')
        self.logger.debug(f'train steps: {self.epoch_step_num}')
        try:
            loaders = [iter(l) for l in self.train_dls]

            for step in range(self.epoch_step_num):

                self.optimizer.zero_grad()

                for ids, loader in enumerate(loaders):
                    ds_conf = self.config.dataset[ids]
                    tdata = next(loader).to(self.device)

                    pred = self.model(tdata)  # pylint: disable=not-callable

                    losses, loss_terms = self.calc_loss(pred, tdata, ids, is_valid=False)
                    loss = losses[0] * ds_conf['loss_weight']
                    cur_loss = loss.detach().item()  # number
                    self.train_history[ids].append([
                        self.epoch,
                        step,
                        cur_loss,
                    ])
                    self.loss_term_history[ids].append([
                        self.epoch,
                        step,
                        loss_terms,
                    ])
                    loss_str = ", ".join([f"{name}: {value:.4f}" for name, value in loss_terms.items()])
                    self.logger.info(
                        f'Train epoch {self.epoch}, step {step}, dataset {ids}, loss: {cur_loss}, loss_terms: {loss_str}'
                    )

                    if self.config.meta.get('save_debug_data', False):
                        if len(self.train_history[ids]) > 1 and cur_loss > 1.5 * self.train_history[ids][-2][-1]:
                            save_dir = os.path.join(self.config.work_folder, "debug")
                            os.makedirs(save_dir, exist_ok=True)
                            self.save_ckpt(os.path.join(save_dir, f"ckpt_{self.epoch}_{step}.pt"))
                            torch.save(tdata, os.path.join(save_dir, f"data_{self.epoch}_{step}.pt"))
                            self.logger.info("debug data saved!")

                    loss.backward()

                    if grad_clip := self.config.training.get('grad_clip', None):
                        clip_grad_norm_(self.model.parameters(), grad_clip)

                    self.optimizer.step()

                    del pred, losses

        except KeyboardInterrupt:
            self.logger.info('stopped by KeyboardInterrupt')
            exit(-1)

    def train_loop(self):

        self._init_optimizer_scheduler()
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f'total number of parameters of the model: {total_params}')

        while True:
            if self.epoch % self.config.training['ckpt_interval'] == 0:
                self.save_ckpt()

            # reach max epochs for training iteration
            if self.epoch >= self.config.training['max_epoch']:
                break

            if self.epoch % self.config.training['valid_interval'] == 0:
                averaged_loss = self.valid_epoch()
                if self.scheduler is not None:
                    self.scheduler.step(averaged_loss)
                    self.logger.info(f"current learning rate: {self.optimizer.param_groups[0]['lr']}")

                if averaged_loss > self.best_valid_loss - self.config.training['ignore_tolerance']:
                    self.early_stop_count += 1
                else:
                    self.early_stop_count = 0
                    self.best_valid_loss = averaged_loss

                self.logger.info(f'early_stop_count: {self.early_stop_count}')

                if averaged_loss <= self.best_valid_loss:
                    self.save_optimal()

                if self.epoch % 10 == 0:
                    self.plot_history()

                # early stop:
                if self.config.training['early_stop_patience'] <= self.early_stop_count:
                    self.logger.info(f"Early stop! Best combined loss: {self.best_valid_loss}")
                    break

            self.train_epoch()
            if self.epoch % 10 == 0:
                self.plot_history()

            self.epoch += 1

        with open(self.config.finish_flag, "w"):
            pass

        return self.epoch

    def plot_history(self):

        def convert_to_serializable(obj):
            """
            Helper function to convert any non-serializable data (like tensors) to serializable types.
            """
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(x) for x in obj]
            else:
                return obj

        # Prepare the history dictionary
        history = {
            "train": convert_to_serializable(self.train_history),
            "valid": convert_to_serializable(self.valid_history),
            "loss_terms": convert_to_serializable(self.loss_term_history)
        }
        # history = {"train": self.train_history, "valid": self.valid_history, "loss_terms": self.loss_term_history}
        with open(os.path.join(self.config.work_folder, 'history.json'), "w") as file:
            json.dump(history, file)

        plt.cla()
        plt.clf()
        nds = len(self.train_dls)
        fig, axes = plt.subplots(1, nds, figsize=(4 * nds, 3), constrained_layout=True)
        axes = [axes] if nds == 1 else axes.flat

        for ids, ax in enumerate(axes):
            epoch_to_step = OrderedDict()
            epoch_to_step[0] = 0
            for i, res in enumerate(self.train_history[ids]):
                epoch = round(res[0])
                if epoch not in epoch_to_step:
                    epoch_to_step[epoch] = i
            epoch_to_step[max(epoch_to_step) + 1] = len(self.train_history[ids])

            step_to_epoch = OrderedDict()
            for epoch, step in epoch_to_step.items():
                step_to_epoch[step] = epoch

            train_rmse = []
            for i, (epoch, _, rmse) in enumerate(self.train_history[ids]):
                train_rmse.append([i, rmse])
            train_rmse = np.asarray([[0, np.nan]]) if len(train_rmse) == 0 else np.asarray(train_rmse)
            ax.plot(train_rmse[:, 0], train_rmse[:, 1], label='train')
            if train_rmse[:, 1].max() / train_rmse[:, 1].min() > 20.:
                ax.semilogy()

            valid_rmse = []
            for epoch, _, rmse in self.valid_history[ids]:
                valid_rmse.append([epoch_to_step[epoch], rmse])
            valid_rmse = np.asarray(valid_rmse)
            ax.plot(valid_rmse[:, 0], valid_rmse[:, 1], '.-', label='valid')

            ax.set_xlabel('step')
            ax.grid(visible=True, zorder=1)
            secax = ax.secondary_xaxis('top')
            secax.set_xlabel('epoch')
            epoch_ticks = np.asarray(sorted([(k, v) for k, v in step_to_epoch.items()]))
            if len(epoch_ticks) < 10:
                secax.set_xticks(ticks=epoch_ticks[:, 0], labels=epoch_ticks[:, 1])
            else:
                skip = len(epoch_ticks) // 10 + 1
                secax.set_xticks(ticks=epoch_ticks[::skip, 0], labels=epoch_ticks[::skip, 1])
            loss_str = ' '.join([f"{l['loss_type']}: {l['weight']}" for l in self.config.dataset[ids]['loss']])
            loss_str = '\n'.join(textwrap.wrap(loss_str, width=75))
            ax.set_title(loss_str, fontdict={"fontsize": 8})
            ax.legend(frameon=False, fontsize="small")

        fig.savefig(os.path.join(self.config.work_folder, 'history.jpg'), dpi=200)
        plt.close(fig)

        if len(self.loss_term_history[0]) > 0:
            fig, axes = plt.subplots(1, nds, figsize=(4 * nds, 3), constrained_layout=True)
            axes = [axes] if nds == 1 else axes.flat

            for ids, ax in enumerate(axes):
                epoch_to_step = OrderedDict()
                epoch_to_step[0] = 0
                for i, res in enumerate(self.loss_term_history[ids]):
                    epoch = round(res[0])
                    if epoch not in epoch_to_step:
                        epoch_to_step[epoch] = i
                epoch_to_step[max(epoch_to_step) + 1] = len(self.loss_term_history[ids])

                step_to_epoch = OrderedDict()
                for epoch, step in epoch_to_step.items():
                    step_to_epoch[step] = epoch

                epochs = [entry[0] for entry in self.loss_term_history[ids]]
                loss_terms = {}

                for name, loss in self.loss_term_history[ids][0][-1].items():
                    loss_terms[name] = []
                    for entry in self.loss_term_history[ids]:
                        loss_terms[name].append(entry[-1][name])

                for name, loss in loss_terms.items():
                    ax.plot(epochs, loss, label=name)

                ax.semilogy()

                ax.set_xlabel('step')
                ax.grid(visible=True, zorder=1)
                secax = ax.secondary_xaxis('top')
                secax.set_xlabel('epoch')
                epoch_ticks = np.asarray(sorted([(k, v) for k, v in step_to_epoch.items()]))
                if len(epoch_ticks) < 10:
                    secax.set_xticks(ticks=epoch_ticks[:, 0], labels=epoch_ticks[:, 1])
                else:
                    skip = len(epoch_ticks) // 10 + 1
                    secax.set_xticks(ticks=epoch_ticks[::skip, 0], labels=epoch_ticks[::skip, 1])
                loss_str = ' '.join([f"{l['loss_type']}: {l['weight']}" for l in self.config.dataset[ids]['loss']])
                loss_str = '\n'.join(textwrap.wrap(loss_str, width=75))
                ax.set_title(loss_str, fontdict={"fontsize": 8})
                ax.legend(frameon=False, fontsize="small")

            fig.savefig(os.path.join(self.config.work_folder, 'loss_term_history.jpg'), dpi=200)
            plt.close(fig)
