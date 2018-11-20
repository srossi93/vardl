# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

try:
    from tqdm import tqdm
except:
    def tqdm(f):
        return f

from . import BaseInitializer
from ..layers import BayesianLinear, BayesianConv2d
from ..likelihoods import Softmax


class BLMInitializer(BaseInitializer):

    def __init__(self, model, train_dataloader: DataLoader,
                 device: torch.device,
                 lognoise: float = -2,
                 logalpha: float = -4):
        super(BLMInitializer, self).__init__(model)
        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = iter(self.train_dataloader)
        self.device = device
        self.lognoise = lognoise * torch.ones(1, device=self.device)
        self.log_alpha = logalpha * torch.ones(1, device=self.device)
        self._logger = logging.getLogger(__name__)

        self._logger.info('Initialization with BayesianLinear Model')

    def _initialize_layer(self, layer: BayesianLinear, layer_index: int):

        #print('INFO - Initialization of layer %d' % layer_index)

        in_features = layer.q_posterior_W.n
        out_features = layer.q_posterior_W.m


        try:
            X, Y = next(self.train_dataloader_iterator)
        except:
            self.train_dataloader_iterator = iter(self.train_dataloader)
            X, Y = next(self.train_dataloader_iterator)

        for out_index in tqdm(range(out_features)):

            if not out_index % Y.size(1):   # TODO: test if it's correct with multiple output dimensions
                try:
                    X, Y = next(self.train_dataloader_iterator)
                except:
                    self.train_dataloader_iterator = iter(self.train_dataloader)
                    X, Y = next(self.train_dataloader_iterator)

            X = X.to(self.device)
            Y = Y.to(self.device)
            #print(Y)
#            print(out_index % Y.size(1))

            if layer_index == len(self.model.architecture) - 1 \
                    and type(self.model.likelihood) == Softmax:
                # - If we are in the last layer and it is a classification problem,
                # - the labels are transformed as Dirichlet variables
                vv = torch.log(1.0 + 1.0 / (Y + torch.exp(self.log_alpha)))
                mm = torch.log(Y + torch.exp(self.log_alpha)) - vv / 2.0
                #print('Classification Layer')
            else:
                vv = torch.ones_like(Y) * self.lognoise  # -2
                mm = Y
                #mm = (2.0 * Y - 1.0)


            if layer_index == 0:
                index = np.random.random_integers(0, Y.size(1) - 1)

                if isinstance(layer, BayesianConv2d):
                    #print('INFO - Extracting patches...')
                    patches = layer.extract_patches(X).mean(dim=0)


                    n_patches = patches.size(-1)
                    n_weights = patches.size(1)

                    patches = patches.permute(2, 0, 1).contiguous().view(-1, n_weights)

                    labels = (mm[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)

                    log_noise = (vv[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)


                    blm_W_m, blm_W_cov = bayesian_linear_model(patches, labels, log_noise)

                else:
                    blm_W_m, blm_W_cov = bayesian_linear_model(X, mm[:, index],
                                                               vv[:, index])

            else:
                stop_l = -len(list(self.model.architecture.children())) + layer_index
                new_in_data = nn.Sequential(
                        *list(self.model.architecture)[:stop_l])(X).mean(dim=0)

                index = np.random.random_integers(0, Y.size(1) - 1)

                if isinstance(layer, BayesianConv2d):
                    #print('INFO - Extracting patches...')
                    patches = layer.extract_patches(new_in_data).mean(dim=0)

                    #print('INFO - Done (patches size =', *patches.size(), ')')

                    n_patches = patches.size(-1)
                    n_weights = patches.size(1)

                    patches = patches.permute(2, 0, 1).contiguous().view(-1, n_weights)

                    labels = (mm[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)

                    log_noise = (vv[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)

                    blm_W_m, blm_W_cov = bayesian_linear_model(patches, labels, log_noise)

                else:
                    blm_W_m, blm_W_cov = bayesian_linear_model(
                    new_in_data,
                    mm[:, index],
                    vv[:, index])

            if layer.approx == 'factorized':
                if layer_index != 0:
                    blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                else:
                    blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()

                layer.q_posterior_W.mean.data[:, out_index] = blm_W_m
                layer.q_posterior_W.logvars.data[:, out_index] = blm_W_logv

            if layer.approx == 'full':
                #raise NotImplementedError()
                layer.q_posterior_W.mean.data[:, out_index] = blm_W_m
                q_chol_L = torch.potrf(blm_W_cov, upper=True).t()
                blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                # layer.q_W_dial_log_L_chol.data.copy_(q_chol_L.diag().log())
                layer.q_posterior_W.cov_lower_triangular[out_index] = q_chol_L
                layer.q_posterior_W.logvars.data[:, out_index] = blm_W_logv

            if layer.approx == 'low-rank':
                if layer_index != 0:
                    blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                else:
                    blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()

                layer.q_posterior_W.mean.data[:, out_index] = blm_W_m
                layer.q_posterior_W.logvars.data[:, out_index] = blm_W_logv


def bayesian_linear_model(X: torch.Tensor, Y: torch.Tensor, log_noise: torch.Tensor) -> torch.Tensor:

    noise_var = torch.exp(log_noise)

    #print('X', X.size())
    #print('Y:', Y.size())
    #print('lognoise:', log_noise.size())

    identity = torch.eye(X.size(1), device=X.device)

    W_true_post_cov = torch.inverse(
        identity +
        torch.matmul(
            X.transpose(-1, -2),
            torch.mul(
                noise_var.unsqueeze(1),
                X)))
    W_true_post_mean = torch.matmul(torch.matmul(W_true_post_cov, X.transpose(-1, -2)), Y / noise_var)

    return W_true_post_mean, W_true_post_cov
