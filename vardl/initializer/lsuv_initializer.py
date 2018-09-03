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


class LSUVInitializer(BaseInitializer):

    def __init__(self, model, train_dataloader: DataLoader,
                 tollerance: float, max_iter: int):
        super(LSUVInitializer, self).__init__(model)
        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = iter(self.train_dataloader)
        self.tollerance = tollerance
        self.max_iter = max_iter

    def _initialize_layer(self, layer: BayesianLinear, layer_index: int):

        torch.nn.init.orthogonal_(layer.q_posterior_W.mean)

        try:
            data, target = next(self.train_dataloader_iterator)
        except BaseException:
            self.train_dataloader_iterator = iter(self.train_dataloader)
            data, target = next(self.train_dataloader_iterator)

        last_idx = -len(list(self.model.architecture.children())) + layer_index + 1

        layer_output = nn.Sequential(
            *
            list(
                self.model.architecture.children())[
                :last_idx])(data)

        current_output_variance = layer_output.var()

        step = 0

        while torch.abs(current_output_variance -
                        1.) > self.tollerance and step < self.max_iter:
            step += 1

            layer.q_posterior_W.mean = layer.q_posterior_W.mean / \
                (torch.sqrt(current_output_variance))

            try:
                data, target = next(self.train_dataloader_iterator)
            except BaseException:
                self.train_dataloader_iterator = iter(self.train_dataloader)
                data, target = next(self.train_dataloader_iterator)

            layer_output = nn.Sequential(
                *list(self.model.architecture.children())[:last_idx])(data)
            current_output_variance = layer_output.var()

        print('INFO - Variance at layer %d (iter #%d): %.3f' %
              (layer_index, step, current_output_variance))

        if layer.q_posterior_W.approx == 'factorized':
            var = (2. * torch.ones(1)) / (layer.in_features)
            layer.q_posterior_W.logvars = (
                np.log(var) *
                torch.ones_like(
                    layer.q_posterior_W.logvars))

        elif layer.approx == 'full':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
