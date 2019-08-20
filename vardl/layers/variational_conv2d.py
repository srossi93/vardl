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

from typing import Union

from . import BaseVariationalLayer
from ..distributions import available_distributions, kl_divergence


def compute_output_shape_conv2d(in_height, in_width, kernel_size, padding=0, dilation=1, stride=1, out_features=None):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    out_height = (in_height + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    out_width = (in_width + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return [out_height, out_width] if out_features is None else [out_features, out_height, out_width]


class VariationalConv2d(BaseVariationalLayer):

    def __init__(self, in_channels, out_channels, kernel_size, prior='fully_factorized_matrix_gaussian',
                 posterior='fully_factorized_matrix_gaussian', stride=1,
                 padding=0, dilation=1, bias=False, local_reparameterization=True, **kwargs):
        """

        Args:
            in_channels (int):
            out_channels (int):
            kernel_size (Union[int, tuple]):
            prior (str):
            posterior (str):
            stride (Union[int, tuple]):
            padding (Union[int, tuple]):
            dilation (Union[int, tuple]):
            bias (bool):
            local_reparameterization (bool):
            **kwargs:
        """

        super(VariationalConv2d, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.local_reparameterization = local_reparameterization
        self.bias = bias

        self.filter_size = self.in_channels * (self.kernel_size ** 2)

        self.prior_weights = available_distributions[prior](self.filter_size, self.out_channels, self.dtype)
        self.posterior_weights = available_distributions[posterior](self.filter_size, self.out_channels, self.dtype)

        if self.bias:
            raise NotImplementedError('Bias not implemented yet')

        self.train()
        return

    def kl_divergence(self):
        divergence = kl_divergence(self.posterior_weights, self.prior_weights)
        if self.bias:
            # should be added in the divergence amount
            raise NotImplementedError('Bias not implemented yet')
        return divergence

    def extract_patches(self, input) -> torch.Tensor:
        input = input * torch.ones(self.nmc, 1, 1, 1, 1).to(self.device)
        batch_size = input.size(1)
        in_height, in_width = input.shape[-2], input.shape[-1]
        input = input.view(-1, self.in_channels, in_height, in_width)

        patches = torch.nn.functional.unfold(input,
                                             self.kernel_size,
                                             padding=self.padding,
                                             stride=self.stride,
                                             dilation=self.dilation)
        patches = patches.view(self.nmc, batch_size, self.filter_size, -1)

        return patches

    def forward(self, input):
        in_height, in_width = input.shape[-2], input.shape[-1]
        out_height, out_width = compute_output_shape_conv2d(in_height, in_width, self.kernel_size, self.padding,
                                                                      self.dilation, self.stride)

        if self.local_reparameterization:
            output = self.posterior_weights.sample_local_reparam_conv2d(self.nmc, input, self.out_channels,
                                                                        self.in_channels, in_height,
                                                                        in_width, self.kernel_size, self.stride,
                                                                        self.padding, self.dilation)

        else:
            samples = self.posterior_weights.sample(self.nmc)  # type: torch.nn.Parameter
            samples = samples.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            in_data = input.contiguous().view(-1, self.in_channels, in_height, in_width)
            output = torch.nn.functional.conv2d(in_data, samples,
                                                stride=self.stride,
                                                padding=self.padding,
                                                dilation=self.dilation)  # type: torch.Tensor


            output = output.view(self.nmc, -1, self.out_channels, out_height, out_width)

        if self.bias:
            raise NotImplementedError('Bias not implemented yet')

        return output

    def extra_repr(self):
        return 'in_channels=%s, out_channels=%s, kernel_size=%s, padding=%s' % (
            self.in_channels, self.out_channels, self.kernel_size, self.padding)

