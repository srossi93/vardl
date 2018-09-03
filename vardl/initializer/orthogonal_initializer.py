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

from torch.utils.data import DataLoader


from . import BaseInitializer
from ..layers import BayesianLinear


class OrthogonalInitializer(BaseInitializer):

    def __init__(self, model, ):
        super(OrthogonalInitializer, self).__init__(model)

    def _initialize_layer(self, layer: BayesianLinear):

        torch.nn.init.orthogonal_(layer.q_posterior_W.mean)

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
