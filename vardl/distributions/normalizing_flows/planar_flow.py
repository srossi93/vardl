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

import torch
import torch.nn as nn
import torch.nn.functional as f

from .. import FullyFactorizedMultivariateGaussian
from .. import BaseDistribution


class SingleFlow(nn.Module):
    def __init__(self, d):
        super(SingleFlow, self).__init__()
        self.d = d
        self.u = nn.Parameter(torch.randn(1, d))
        self.w = nn.Parameter(torch.randn(d, 1))
        self.b = nn.Parameter(torch.zeros(1))
        self.activation_function = lambda x: torch.tanh(x)
        self.d_activation_function = lambda x: 1 - (torch.tanh(x) ** 2)
        self.log_det_jacobian = torch.randn(1)

    def forward(self, input):
        lin = torch.matmul(input, self.w) + self.b
        out = self.activation_function(lin)
        out = (self.u * out) + input

        self.log_det_jacobian = (1 + torch.matmul(self.u, self.w) * self.d_activation_function(lin)).abs().log()
        return out


class PlanarFlow(BaseDistribution):
    def __init__(self, d, n_transformations=5, **kwargs):
        super(PlanarFlow, self).__init__()
        self.n_transformations = n_transformations
        self.d = d
        self.initial_distribution = FullyFactorizedMultivariateGaussian(d)
        self.initial_distribution.logvars.fill_(-8)
        self.flows = nn.Sequential(*[SingleFlow(d) for _ in range(n_transformations)])

    def sample(self, n_samples):
        q0 = self.initial_distribution.sample(n_samples)
        qk = self.flows(q0)
        return qk

    def sample_local_reparam_conv2d(self, *args, **kargs):
        raise ValueError('Local reparameterization not available')

    def sample_local_reparam_linear(self, *args, **kargs):
        raise ValueError('Local reparameterization not available')

