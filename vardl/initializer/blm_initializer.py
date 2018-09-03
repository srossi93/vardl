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

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader


from . import BaseInitializer
from ..layers import BayesianLinear
from ..likelihoods import Softmax


class BLMInitializer(BaseInitializer):

    def __init__(self, model, train_dataloader: DataLoader):
        super(BLMInitializer, self).__init__(model)
        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = iter(self.train_dataloader)
        self.lognoise = -2 * torch.ones(1)
        self.log_alpha = -4 * torch.ones(1)

    def _initialize_layer(self, layer: BayesianLinear, layer_index: int):

        try:
            X, Y = next(self.train_dataloader_iterator)
        except BaseException:
            self.train_dataloader_iterator = iter(self.train_dataloader)
            X, Y = next(self.train_dataloader_iterator)

        for out_index in range(layer.out_features):

            if out_index % Y.size(1):
                try:
                    X, Y = next(self.train_dataloader_iterator)
                except BaseException:
                    self.train_dataloader_iterator = iter(self.train_dataloader)
                    X, Y = next(self.train_dataloader_iterator)

            if layer_index == len(self.model.architecture) - \
                    1 and type(self.model.likelihood == Softmax):
                # - If we are in the last layer and it is a classification problem,
                # - the labels are transformed as Dirichlet variables
                vv = torch.log(1.0 + 1.0 / (Y + torch.exp(self.log_alpha)))
                mm = torch.log(Y + torch.exp(self.log_alpha)) - vv / 2.0
            else:
                vv = self.lognoise * torch.ones_like(Y)  # -2
                mm = Y

            if layer_index == 0:
                blm_W_m, blm_W_cov = bayesian_linear_model(X, mm[:, np.random.random_integers(0, Y.size(1) - 1)],
                                                           vv[:, np.random.random_integers(0, Y.size(1) - 1)])

            else:
                stop_l = -len(list(self.model.architecture.children())) + layer_index
                blm_W_m, blm_W_cov = bayesian_linear_model(
                    nn.Sequential(
                        *list(self.model.architecture.children())[:stop_l])(X).mean(dim=0),
                    mm[:, np.random.random_integers(0, Y.size(1) - 1)],
                    vv[:, np.random.random_integers(0, Y.size(1) - 1)])

            if layer.approx == 'factorized':
                blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                # print(layer.q_W_logv.size(), blm_W_logv.size())
                # blm_W_logv = blm_W_cov.diag().log()
                layer.q_posterior_W.mean[:, out_index] = blm_W_m
                layer.q_posterior_W.logvars[:, out_index] = blm_W_logv

            elif layer.approx == 'full':
                raise NotImplementedError()
                layer.q_W_m[:, out_index].data.copy_(blm_W_m)
                q_chol_L = torch.potrf(blm_W_cov, upper=True).t()
                blm_W_logv = (1. / torch.inverse(blm_W_cov).diag()).log()
                # layer.q_W_dial_log_L_chol.data.copy_(q_chol_L.diag().log())
                layer.q_W_dial_log_L_chol.data.copy_(blm_W_logv)
                layer.q_W_L_chol_Sigma.data.copy_(q_chol_L)


def bayesian_linear_model(X, Y, log_noise):

    noise_var = torch.exp(log_noise * torch.ones(1))

    W_true_post_cov = torch.inverse(
        torch.eye(
            X.size(1)) +
        torch.matmul(
            X.t(),
            torch.mul(
                noise_var.unsqueeze(1),
                X)))
    W_true_post_mean = torch.matmul(torch.matmul(W_true_post_cov, X.t()), Y / noise_var)

    return W_true_post_mean, W_true_post_cov
