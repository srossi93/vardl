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
import numpy as np

from . import BaseInitializer
from ..layers import BayesianLinear


class HeuristicInitializer(BaseInitializer):

    def __init__(self, model):
        super(HeuristicInitializer, self).__init__(model)

    def _initialize_layer(self, layer: BayesianLinear):

        stdv = float(1. / torch.sqrt(torch.ones(1) * layer.in_features))
        layer.q_posterior_W.mean = torch.zeros_like(layer.q_posterior_W.mean)

        if layer.q_posterior_W.approx == 'factorized':
            layer.q_posterior_W.logvars = torch.ones_like(
                layer.q_posterior_W.logvars) * 2 * np.log(stdv)
        elif layer.approx == 'full':
            layer.q_posterior_W.logvars = np.log(1. / layer.in_features) * torch.ones(layer.out_features,
                                                                                      layer.in_features)

            layer.q_posterior_W.cov_lower_triangular = np.log(1. / layer.in_features) * torch.eye(layer.in_features,
                                                                                                  layer.in_features) *\
                torch.ones(layer.out_features, 1, 1)

        else:
            raise NotImplementedError()
