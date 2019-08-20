#  Copyright (c) 2019
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Authors:
#      Simone Rossi <simone.rossi@eurecom.fr>
#      Maurizio Filippone <maurizio.filippone@eurecom.fr>
#

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(f):
        return f

from ..likelihoods import Softmax

from torch.utils import hooks as hooks
from torch.utils.data import DataLoader

import numpy as np
import torch
from typing import Union

from . import BaseInitializer
from ..layers import VariationalLinear, VariationalConv2d
from ..distributions import FullyFactorizedMatrixGaussian, FullCovarianceMatrixGaussian

import logging
logger = logging.getLogger(__name__)


def save_input_to_layer(self, input, output):
    self.input_to_layer = input[0]
    self.output_to_layer = output[0]


class IBLMInitializer(BaseInitializer):

    def __init__(self, model, train_dataloader, device, lognoise=-2, logalpha=-4):
        """
        Implements a I-BLM initializer
        Args:
            model:
            train_dataloader (DataLoader):
            device:
            lognoise:
            logalpha:
        """
        super(IBLMInitializer, self).__init__(model)

        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = iter(self.train_dataloader)
        self.device = device
        self.lognoise = lognoise * torch.ones(1, device=self.device)
        self.log_alpha = logalpha * torch.ones(1, device=self.device)

        logger.info('Initialization with I-BLM')

    def _initialize_layer(self, layer, layer_index=None):

        hook_hadler = layer.register_forward_hook(save_input_to_layer)  # type: hooks.RemovableHandle

        in_features = layer.posterior_weights.n
        out_features = layer.posterior_weights.m

        try:
            X, Y = next(self.train_dataloader_iterator)
        except StopIteration:
            self.train_dataloader_iterator = iter(self.train_dataloader)
            X, Y = next(self.train_dataloader_iterator)

        for out_index in tqdm(range(out_features)):

            if not out_index % Y.size(1):
                try:
                    X, Y = next(self.train_dataloader_iterator)
                except StopIteration:
                    self.train_dataloader_iterator = iter(self.train_dataloader)
                    X, Y = next(self.train_dataloader_iterator)

            X = X.to(self.device)
            Y = Y.to(self.device)

#            if layer_index == len(self.model.architecture) - 1 and type(self.model.likelihood) == Softmax:
            if isinstance(self.model.likelihood, Softmax):
                # - If we are in the last layer and it is a classification problem,
                # - the labels are transformed as Dirichlet variables
                vv = torch.log(1.0 + 1.0 / (Y + torch.exp(self.log_alpha)))
                mm = torch.log(Y + torch.exp(self.log_alpha)) - vv / 2.0

            else:
                vv = torch.ones_like(Y) * self.lognoise  # -2
                mm = Y
                # mm = (2.0 * Y - 1.0)  # Scale the mean to be symmetrical

            if layer_index == 0:
                index = np.random.random_integers(0, Y.size(1) - 1)

                if isinstance(layer, VariationalConv2d):
                    patches = layer.extract_patches(X).mean(dim=0)
                    n_patches = patches.size(-1)
                    n_weights = patches.size(1)

                    patches = patches.permute(2, 0, 1).contiguous().view(-1, n_weights)
                    labels = (mm[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)
                    log_noise = (vv[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)

                    blm_W_m, blm_W_cov = bayesian_linear_model(patches, labels, log_noise)
                else:
                    blm_W_m, blm_W_cov = bayesian_linear_model(X, mm[:, index],  vv[:, index])

            else:
                # Run a forward pass (the hook will save the input to the layer)
                self.model(X)
                new_in_data = layer.input_to_layer.mean(0)

                index = np.random.random_integers(0, Y.size(1) - 1)
                if isinstance(layer, VariationalConv2d):
                    patches = layer.extract_patches(new_in_data).mean(dim=0)
                    n_patches = patches.size(-1)
                    n_weights = patches.size(1)

                    patches = patches.permute(2, 0, 1).contiguous().view(-1, n_weights)
                    labels = (mm[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)
                    log_noise = (vv[:, index] * torch.ones(n_patches, 1, device=X.device)).view(-1)

                    blm_W_m, blm_W_cov = bayesian_linear_model(patches, labels, log_noise)
                else:
                    blm_W_m, blm_W_cov = bayesian_linear_model(new_in_data, mm[:, index], vv[:, index])

            if isinstance(layer.posterior_weights, FullyFactorizedMatrixGaussian):
                blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                layer.posterior_weights.mean.data[:, out_index] = blm_W_m
                layer.posterior_weights.logvars.data[:, out_index] = blm_W_logv

            elif isinstance(layer.posterior_weights, FullCovarianceMatrixGaussian):
                layer.posterior_weights.mean.data[:, out_index] = blm_W_m
                q_chol_L = torch.potrf(blm_W_cov, upper=True).t()
                blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                layer.posterior_weights.cov_lower_triangular.data[out_index] = q_chol_L
                layer.posterior_weights.logvars.data[:, out_index] = blm_W_logv

            else:
                logger.warning('Distribution not available yet for this type of initialization. Skipping')

        hook_hadler.remove()


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
