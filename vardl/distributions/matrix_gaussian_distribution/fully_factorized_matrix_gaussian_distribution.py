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

from . import MatrixGaussianDistribution
from ...utils import compute_output_shape_conv2d

import logging
logger = logging.getLogger(__name__)


class FullyFactorizedMatrixGaussian(MatrixGaussianDistribution):
    def __init__(self, n: int, m: int, dtype: torch.dtype = torch.float32):
        super(FullyFactorizedMatrixGaussian, self).__init__(n, m, dtype=dtype)
        self.name = 'Fully factorized matrix Gaussian'
        self.has_local_reparam_linear = True
        self.has_local_reparam_conv2d = True

    def sample(self, n_samples: int):
        epsilon_for_samples = torch.randn(n_samples, self.n, self.m,
                                          dtype=self.dtype,
                                          device=self.mean.device,
                                          requires_grad=False)

        samples = torch.add(torch.mul(epsilon_for_samples,
                                      torch.exp(self.logvars / 2.0)),
                            self.mean)  # type: torch.Tensor
        return samples

    def sample_local_reparam_linear(self, n_sample: int, in_data: torch.Tensor):
        # Retrieve current device (useful generate the data already in the correct device)
        bs = in_data.shape[1]
        device = self.mean.device
        epsilon_for_Y_sample = torch.randn(n_sample, 1, self.mean.size(1),
        # epsilon_for_Y_sample = torch.randn(n_sample, in_data.size(-2), self.mean.size(1),
                                           dtype=self.dtype,
                                           device=device,
                                           requires_grad=False)  # type: torch.Tensor

        mean_Y = torch.matmul(in_data, self.mean)
        var_Y = torch.matmul(in_data ** 2, torch.exp(self.logvars))
        Y = mean_Y + torch.sqrt(var_Y) * epsilon_for_Y_sample  # type: torch.Tensor
        return Y

    def sample_local_reparam_conv2d(self, n_sample, in_data, out_channels, in_channels, in_height, in_width,
                                    kernel_size, stride, padding, dilation):
        """

        Args:
            n_sample:
            in_data (torch.Tensor):
            out_channels:
            in_channels:
            in_height:
            in_width:
            kernel_size:
            stride:
            padding:
            dilation:

        Returns:
            torch.Tensor
        """

        out_height, out_width = compute_output_shape_conv2d(in_height, in_width, kernel_size, padding, dilation, stride)

        mean_weights = self.mean.view(out_channels, in_channels,
                                      kernel_size, kernel_size)  # type: torch.nn.Parameter
        logvars_weights = self.logvars.view(out_channels, in_channels,
                                            kernel_size, kernel_size)  # type: torch.nn.Parameter

        # TODO (@rossi): contiguous might not be necessary
        in_data = in_data.contiguous().view(-1, in_channels, in_height, in_width)
        mean_output = torch.nn.functional.conv2d(in_data, mean_weights,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation)  # type: torch.Tensor

        var_output = torch.nn.functional.conv2d(in_data.pow(2), logvars_weights.exp(),
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation)

        eps = torch.randn_like(mean_output, requires_grad=False)
        output = mean_output + eps * torch.sqrt(var_output + 1e-5)  # type: torch.Tensor

        return output.view(n_sample, -1, out_channels, out_height, out_width)

    def extra_repr(self):
        string = "n=%d, m=%d" % (self.n, self.m)
        return string
