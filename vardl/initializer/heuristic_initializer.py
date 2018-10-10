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

from . import BaseInitializer
from ..layers import BayesianLinear


class HeuristicInitializer(BaseInitializer):

    def __init__(self, model):
        super(HeuristicInitializer, self).__init__(model)

        print('INFO - Initialization with Heuristic')

    def _initialize_layer(self, layer: BayesianLinear, layer_index: int = None):

        in_features = layer.q_posterior_W.n
        out_features = layer.q_posterior_W.m

        stdv = float(1. / torch.sqrt(torch.ones(1) * in_features))
        layer.q_posterior_W.mean = torch.zeros_like(layer.q_posterior_W.mean)

        if layer.q_posterior_W.approx == 'factorized':
            layer.q_posterior_W.logvars = torch.ones_like(
                layer.q_posterior_W.logvars) * 2 * np.log(stdv)
        elif layer.approx == 'full':



            layer.q_posterior_W.logvars = np.log(1. / in_features) * torch.ones(out_features,
                                                                                      in_features)

            layer.q_posterior_W.cov_lower_triangular = np.log(1. / in_features) * torch.eye(in_features,
                                                                                                  in_features) *\
                torch.ones(out_features, 1, 1)

        else:
            raise NotImplementedError()
