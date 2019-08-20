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

from . import BaseVariationalLayer

from ..distributions import available_distributions, kl_divergence
from ..distributions import FullyFactorizedMultivariateGaussian

class VariationalLinear(BaseVariationalLayer):
    def __init__(self, in_features, out_features, bias=True, prior='fully_factorized_matrix_gaussian',
                 posterior='fully_factorized_matrix_gaussian', local_reparameterization=True, **kwargs):
        super(VariationalLinear, self).__init__(**kwargs)
        self.prior_weights = available_distributions[prior](in_features, out_features, self.dtype)
        self.posterior_weights = available_distributions[posterior](in_features, out_features, self.dtype)
        self.local_reparameterization = local_reparameterization
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias  # type: bool

        if self.bias:
            self.prior_bias = FullyFactorizedMultivariateGaussian(out_features)
            self.posterior_bias = FullyFactorizedMultivariateGaussian(out_features)
            self.posterior_bias.optimize(True)

        self.posterior_weights.optimize(True)
        self.reset_parameters()


    def kl_divergence(self):
        divergence = kl_divergence(self.posterior_weights, self.prior_weights)
        if self.bias:
            # should be added in the divergence amount
            divergence += kl_divergence(self.posterior_bias, self.prior_bias)
        return divergence

    def reset_parameters(self):
        self.posterior_weights.mean.data.fill_(0)
        self.posterior_weights.logvars.data.fill_(-4)
        if self.bias:
            self.posterior_bias.mean.data.fill_(0)
            self.posterior_bias.logvars.data.fill_(-4)


    def forward(self, input):
        nmc = input.shape[0]
        if self.local_reparameterization:
            output = self.posterior_weights.sample_local_reparam_linear(nmc, input)
        else:
            weights = self.posterior_weights.sample(nmc)
            output = torch.matmul(input, weights)

        if self.bias:
            bias = self.posterior_bias.sample(nmc).unsqueeze(1)
            output += bias

        return output
