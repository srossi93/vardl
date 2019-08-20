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

from . import BaseVariationalLayer,compute_output_shape_conv2d
from .. import functional

from ..distributions import available_distributions_multivariate_gaussian as available_distributions, kl_divergence

logger = logging.getLogger(__name__)


class VariationalFastfoodConv2d(BaseVariationalLayer):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size, prior='fully_factorized_multivariate_gaussian',
                 posterior='fully_factorized_multivariate_gaussian', stride=1,
                 padding=0, dilation=1, bias=True, local_reparameterization=True, **kwargs):
        super(VariationalFastfoodConv2d, self).__init__(**kwargs)
        self.local_reparameterization = local_reparameterization

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_features = self.in_channels * kernel_size * kernel_size
        out_features = self.out_channels
        self.bias = bias  # type: bool

        self.in_features_pad, self.out_features_pad, self.times_to_stack_v, self.features_to_pad = self._setup_dimensions(self.in_features, out_features)

        self.prior_weights = available_distributions[prior](self.in_features_pad * self.times_to_stack_v, self.dtype)
        self.prior_weights.logvars.data.fill_(-11)
        self.posterior_weights = available_distributions[posterior](self.in_features_pad * self.times_to_stack_v, self.dtype)
        torch.nn.init.normal_(self.posterior_weights.mean.data, 0, 0.1)
        self.posterior_weights.logvars.data.fill_(-4)
        self.posterior_weights.optimize()

        self.B = torch.nn.Parameter(torch.randn(self.times_to_stack_v, self.in_features_pad)*0.01)
        #self.B = torch.nn.Parameter(torch.from_numpy(np.random.choice((-1, 1), size=[self.times_to_stack_v,
        #                                                                             self.in_features_pad])).float(),
        #                            requires_grad=True)  # is false
        torch.nn.init.kaiming_uniform_(self.B)

        self.P = torch.nn.Parameter(
            torch.from_numpy(
                np.hstack([np.random.permutation(self.in_features_pad) for i in range(self.times_to_stack_v)])),
            requires_grad=False)

        self.S = torch.nn.Parameter(torch.randn(self.times_to_stack_v, self.in_features_pad)*0.01)
        torch.nn.init.kaiming_uniform_(self.S)

      #  self.S_v = torch.nn.Parameter(torch.randn(self.times_to_stack_v, self.in_features_pad))
      #  torch.nn.init.kaiming_uniform_(self.S_v)

        if bias:
            self.posterior_bias = available_distributions[posterior](self.out_channels)
            self.posterior_bias.optimize()
            self.prior_bias = available_distributions[prior](self.out_channels)
            self.prior_bias.logvars.fill_(np.log(0.01))


    def _setup_dimensions(self, in_features, out_features):
        # Perform sanity checks on dimensions to match the requirements from fastfood
        next_power_of_two = int(np.power(2, np.floor(np.log2(in_features)) + 1))
        original_in_features = in_features
        if next_power_of_two == in_features * 2:
            features_to_pad = 0
        else:
            features_to_pad = next_power_of_two - in_features
            in_features = next_power_of_two
            logger.warning('Input space [%d] is not a power of 2. Zero padding of %d input features' % (
                original_in_features, features_to_pad))

        divisor, remainder = divmod(out_features, in_features)
        times_to_stack_v = int(divisor)
        if remainder != 0:
            original_out_features = out_features
            out_features = (divisor + 1) * in_features
            logger.warning('Output space [%d] is not a multiple of input features. Discarding %d output features' % (
                original_out_features, out_features-original_out_features))
            times_to_stack_v = int(divisor + 1)
        return int(in_features), int(out_features), times_to_stack_v, features_to_pad

    def extract_patches(self, input) -> torch.Tensor:
      #  input = input * torch.ones(self.nmc, 1, 1, 1, 1).to(input.device)
        batch_size = input.size(1)
        batched_input = input.view(input.shape[0] * batch_size, *input.shape[2:])

        patches = torch.nn.functional.unfold(batched_input,
                                             self.kernel_size,
                                             padding=self.padding,
                                             stride=self.stride,
                                             dilation=self.dilation)
        patches = patches.view(input.shape[0], batch_size, -1, self.in_features)

        return patches

    def forward(self, input):
        if self.local_reparameterization:
            output = self.forward_local_reparam(input)
        else:
            output = self.forward_no_local_reparam(input)

        if self.bias:
            return output + self.posterior_bias.sample(input.shape[0] * input.shape[1]).view(input.shape[0], input.shape[1], -1, 1, 1)

        return output

    def forward_no_local_reparam(self, input: torch.Tensor):

        patches = self.extract_patches(input)
        out_shape = compute_output_shape_conv2d(input.shape[-2], input.shape[-1], self.kernel_size, self.padding,
                                                self.dilation, self.stride, self.out_channels)

        G = self.posterior_weights.sample(self.nmc * self.times_to_stack_v).view(self.nmc, self.times_to_stack_v, -1)

        output = fastfood(patches, self.S, self.P, G, self.B, self.in_features_pad, out_shape, self.times_to_stack_v,
                          self.features_to_pad)
        return output

    def forward_local_reparam(self, input):
        # s1 = torch.cuda.Stream()
        # s2 = torch.cuda.Stream()

        patches = self.extract_patches(input)
        out_shape = compute_output_shape_conv2d(input.shape[-2], input.shape[-1], self.kernel_size, self.padding,
                                                self.dilation, self.stride, self.out_channels)

        # mean_G = torch.stack([self.posterior_weights.mean for _ in range(self.times_to_stack_v)]).view(-1)
        # std_G = torch.stack([self.posterior_weights.logvars for _ in range(self.times_to_stack_v)]).view(-1).exp()
        mean_G = self.posterior_weights.mean.view(self.times_to_stack_v, self.in_features_pad)
        std_G = self.posterior_weights.logvars.exp().view(-1)
        G_eps = std_G * torch.randn(input.shape[0], input.shape[1], std_G.shape[0], requires_grad=False, device=std_G.device)
        G_eps = G_eps.view(input.shape[0], input.shape[1], 1, self.times_to_stack_v, self.in_features_pad)
        #G_eps = (std_G * torch.randn_like(std_G)).view(self.times_to_stack_v, self.in_features_pad)

        mean_output = fastfood(patches, self.S, self.P, mean_G, self.B, self.in_features_pad, out_shape,
                               self.times_to_stack_v, self.features_to_pad)


        std_output = fastfood(patches, self.S, self.P, G_eps, self.B, self.in_features_pad, out_shape,
                              self.times_to_stack_v, self.features_to_pad)

       # mean_output = torch.nn.functional.dropout(mean_output, 0.5, training=True)
        return mean_output + std_output

    def kl_divergence(self):
        kl = kl_divergence(self.posterior_weights, self.prior_weights)
        if self.bias:
            kl += kl_divergence(self.posterior_bias, self.prior_bias)
        kl = kl + 0.005 * (torch.norm(self.B, 2) + torch.norm(self.S, 2))
       # kl = kl + 0.005 * (torch.norm(self.B_v, 2) + torch.norm(self.S_v, 2))
        return kl

    def extra_repr(self):
        return 'in_channels=%s, out_channels=%s, kernel_size=%s, padding=%s' % (
            self.in_channels, self.out_channels, self.kernel_size, self.padding)


def fastfood(input, S, P, G, B, in_features_pad, out_shape, times_to_stack_v=1, features_to_pad=0) -> torch.Tensor:
    if len(input.shape) != 4:
        logger.error('Expecting 4D input tensor but got %dD' % len(input.shape))
        raise RuntimeError('Expecting 4D input tensor but got %dD' % len(input.shape))

    if len(out_shape) != 3:
        print(out_shape)
        logger.error('Invalid out_shape')
        raise RuntimeError('Invalid out_shape')

    nmc = input.shape[0]
    batch_size = input.shape[1]
#    G = G.view(-1, times_to_stack_v, in_features_pad)
    input = torch.nn.functional.pad(input, (0, features_to_pad), 'constant', 0).unsqueeze(-2)
    out = B * input
    out = functional.FastWalshHadamardTransform.apply(out)#.view(nmc, batch_size, -1, times_to_stack_v * in_features_pad)
    #PHBx = HBx[..., P.long()]#.view(nmc, batch_size, -1, times_to_stack_v, in_features_pad)
    #out = HBx
    out = (G * out)
  #  out = out.view(nmc, batch_size, -1, times_to_stack_v, in_features_pad)
    out = functional.FastWalshHadamardTransform.apply(out)
    out = S * out
    #out = functional.FastWalshHadamardTransform.apply(out)
    out = out.view(nmc, batch_size, -1, times_to_stack_v * in_features_pad)
    out = out[..., :out_shape[0]].view(nmc * batch_size, -1, out_shape[0]).transpose(-1,-2)
    return out.view(nmc, batch_size, *out_shape)
