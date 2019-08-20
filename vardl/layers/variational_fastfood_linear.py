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
import numpy as np
import logging
import torch.nn

from . import BaseVariationalLayer
from .. import functional

from ..distributions import available_distributions, kl_divergence

logger = logging.getLogger(__name__)


class VariationalFastfoodLinear(BaseVariationalLayer):
    """
    Implements a Variational Fastfood linear layer layer using Fast Hadamard transform
    output = V @ input + c
    where V = S H q_G P H B
        P: permutation matrix
        H: Hadamard matrix (implicit thanks to FHT)
        B: Binary scaling (+-1)
        q_G: Gaussian vector
        S: Scaling vector
    """

    def __init__(self, in_features, out_features, bias=True, prior='fully_factorized_multivariate_gaussian',
                 posterior='fully_factorized_multivariate_gaussian', local_reparameterization=True, **kwargs):
        super(VariationalFastfoodLinear, self).__init__(**kwargs)
        self.local_reparameterization = local_reparameterization

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias  # type: bool

        self.in_features_pad, self.out_features_pad, \
        self.times_to_stack_v, self.features_to_pad = self._setup_dimensions(in_features, out_features)

        self.prior_weights = available_distributions[prior](self.in_features_pad, self.dtype)
        self.prior_weights.logvars.data.fill_(-10)

        self.posterior_weights = available_distributions[posterior](self.in_features_pad, **kwargs)
        # torch.nn.init.normal_(self.posterior_weights.mean.data, 0, 0.1)
        # self.posterior_weights.logvars.data.fill_(-9)
        self.posterior_weights.optimize()

        #self.B = torch.nn.Parameter(torch.from_numpy(np.random.choice((-1, 1), size=[self.times_to_stack_v,
        #                                                                         self.in_features_pad])).float(),
        #                            requires_grad=True)  # is false
        self.B = torch.nn.Parameter(torch.randn(self.times_to_stack_v, self.in_features_pad)*0.01)
        torch.nn.init.kaiming_uniform_(self.B)

        self.P = torch.nn.Parameter(
            torch.from_numpy(
                np.hstack([np.random.permutation(self.in_features_pad) for i in range(self.times_to_stack_v)])),
            requires_grad=False)

        self.S = torch.nn.Parameter(torch.randn(self.times_to_stack_v, self.in_features_pad)*0.01)
        torch.nn.init.kaiming_uniform_(self.S)

        if bias:
            # self.posterior_bias = available_distributions[posterior](self.out_features)
            self.posterior_bias = available_distributions['fully_factorized_multivariate_gaussian'](self.out_features)
            self.posterior_bias.optimize()
            self.prior_bias = available_distributions[prior](self.out_features)
            self.prior_bias.logvars.fill_(np.log(0.01))


    def _setup_dimensions(self, in_features, out_features):
        # Perform sanity checks on dimensions to match the requirements from fastfood
        next_power_of_two = int(np.power(2, np.floor(np.log2(in_features)) + 1))
        if next_power_of_two == in_features * 2:
            features_to_pad = 0
        else:
            features_to_pad = next_power_of_two - in_features
            in_features = next_power_of_two
            logger.warning('Input space is not a power of 2. Zero padding of %d input features' % features_to_pad)

        divisor, remainder = divmod(out_features, in_features)
        times_to_stack_v = int(divisor)
        if remainder != 0:
            original_out_features = out_features
            out_features = (divisor + 1) * in_features
            logger.warning('Output space is not a power of 2. Discarding %d output features' % (out_features-original_out_features))
            times_to_stack_v = int(divisor + 1)
        return int(in_features), int(out_features), times_to_stack_v, features_to_pad

    def forward(self, input: torch.Tensor):

        batch_size = input.size(1)
        if self.local_reparameterization:
            output = self.forward_local_reparam(input)
        else:
            output = self.forward_no_local_reparam(input)

        if self.bias:
            return output + self.posterior_bias.sample(input.shape[0] * batch_size).view(input.shape[0], batch_size, -1)

        return output

    def kl_divergence(self):
        kl = kl_divergence(self.posterior_weights, self.prior_weights)
        if self.bias:
            kl += kl_divergence(self.posterior_bias, self.prior_bias)
        kl = kl + 0.005 * (torch.norm(self.B, 2) + torch.norm(self.S, 2))
        return kl

    def extra_repr(self):
        string = r"""in_features={}, out_features={}, bias={}""".format(
            self.in_features, self.out_features, self.bias)
        return string

    def forward_local_reparam(self, input: torch.Tensor):
        batch_size = input.size(1)

        mean_G = torch.stack([self.posterior_weights.mean for _ in range(self.times_to_stack_v)]).view(-1)
        mean_F = fastfood(input, self.S, self.P, mean_G, self.B, self.times_to_stack_v, self.features_to_pad)
        mean_F = mean_F.view(input.shape[0], batch_size, -1)[..., :self.out_features]

        std_G = torch.stack([self.posterior_weights.logvars for _ in range(self.times_to_stack_v)]).view(-1).exp()
    #    G_eps = std_G * torch.randn_like(std_G)
        G_eps = std_G * torch.randn(input.shape[0], input.shape[1], std_G.shape[0], requires_grad=False, device=std_G.device)
        G_eps = G_eps.view(input.shape[0], input.shape[1],  self.times_to_stack_v * self.in_features_pad)
        std_F = fastfood(input, self.S, self.P, G_eps, self.B, self.times_to_stack_v, self.features_to_pad)
        std_F = std_F.view(input.shape[0], batch_size, -1)[..., :self.out_features]

        return mean_F + std_F

    def forward_no_local_reparam(self, input: torch.Tensor):
        nmc = input.shape[0]
        batch_size = input.size(1)

        G = self.posterior_weights.sample(nmc * self.times_to_stack_v).view(nmc, 1, -1)
        output = fastfood(input, self.S, self.P, G, self.B, self.times_to_stack_v, self.features_to_pad)
        output = output.view(nmc, batch_size, -1)[..., :self.out_features]
        return output


def fastfood(inputs, S, P, G, B, times_to_stack_v=1, features_to_pad=0) -> torch.Tensor:
    if len(inputs.shape) != 3 and len(inputs.shape) != 4:
        logger.error('Expecting 3D or 4D input tensor but got %dD' % len(inputs.shape))
        raise RuntimeError('Expecting 3D or 4D input tensor but got %dD' % len(inputs.shape))

    nmc = inputs.shape[0]
    batch_size = inputs.shape[1]

    input = torch.nn.functional.pad(inputs, (0, features_to_pad), 'constant', 0).unsqueeze(-2)

    out = (B * input)
    out = functional.FastWalshHadamardTransform.apply(out).view(nmc, batch_size, -1)
  #  out = out[..., P.long()]
 #   out = out
    out = (G * out).view(nmc, batch_size, times_to_stack_v, -1)
    out = functional.FastWalshHadamardTransform.apply(out)
    out = (S * out)
    return out
