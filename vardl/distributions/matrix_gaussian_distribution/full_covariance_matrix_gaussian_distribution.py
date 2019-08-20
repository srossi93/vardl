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
#
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
import torch.nn as nn

from . import MatrixGaussianDistribution

import logging
logger = logging.getLogger(__name__)


class FullCovarianceMatrixGaussian(MatrixGaussianDistribution):
    def __init__(self, n: int, m: int, dtype: torch.dtype = torch.float32):
        super(FullCovarianceMatrixGaussian, self).__init__(n, m, dtype=dtype)
        self.name = 'Full covariance factorized matrix Gaussian'
        # -- The posterior approximation of the covariance matrix is
        # -- parametrized using Log-Cholesky parametrization
        # -- ref. Unconstrained Parameterizations for Variance-Covariance Matrices (p.2)
        self.cov_lower_triangular = nn.Parameter(
            torch.eye(self.n, self.n, dtype=self.dtype) *
            torch.ones(self.m, 1, 1, dtype=self.dtype) * np.log(1. / self.n),
            requires_grad=False)
        self.has_local_reparam_linear = True
        self.has_local_reparam_conv2d = False

    def sample(self, n_samples: int):
        epsilon_for_samples = torch.randn(n_samples, self.n, self.m,
                                          dtype=self.dtype,
                                          device=self.mean.device,
                                          requires_grad=False)

        samples = torch.zeros_like(epsilon_for_samples, device=self.mean.device)

        for i in range(self.m):
            cov_lower_triangular = torch.tril(self.cov_lower_triangular[i, :, :], -1)
            cov_lower_triangular += torch.diagflat(torch.exp(self.logvars[:, i]))
            samples[:, :, i] = torch.add(torch.matmul(cov_lower_triangular, epsilon_for_samples[:, :, i].t()).t(),
                                         self.mean[:, i])

        samples = torch.add(torch.mul(epsilon_for_samples,
                                      torch.exp(self.logvars / 2.0)),
                            self.mean)  # type: torch.Tensor
        return samples

    def sample_local_reparam_linear(self, n_sample: int, in_data: torch.Tensor):
        # Retrieve current device (useful generate the data already in the correct device)
        device = self.mean.device
        epsilon_for_Y_sample = torch.randn(n_sample, in_data.size(-2), self.mean.size(1),
                                           dtype=self.dtype,
                                           device=device,
                                           requires_grad=False)  # type: torch.Tensor

        mean_Y = torch.matmul(in_data, self.mean)

        var_Y = torch.zeros_like(mean_Y, device=device) * torch.ones(n_sample, 1, 1, device=device)
        for i in range(self.m):
            cov_lower_triangular = torch.tril(self.cov_lower_triangular[i, :, :], -1)
            L_chol = cov_lower_triangular + torch.diagflat(torch.exp(self.logvars[:, i]))

            var_Y[:, :, i] = torch.sum(torch.matmul(in_data, L_chol) ** 2, -1)

        Y = mean_Y + torch.sqrt(var_Y + 1e-5) * epsilon_for_Y_sample  # type: torch.Tensor
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

        if not hasattr(self, '_alert_done'):
            setattr(self, '_alert_done', True)
            logger.warning('Full covariance has no local reparameterization for conv2d operation. Falling back to '
                           'standard reparameterization for %s', self._get_name())

        # Compute output spatial dimensions
        out_height = int((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        out_width = int((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        # Sample using simple reparameterization and apply conv2d operation to them
        samples = self.sample(n_sample)
        samples = samples.view(out_channels, in_channels,
                               kernel_size, kernel_size)  # type: torch.nn.Parameter
        in_data = in_data.contiguous().view(-1, in_channels, in_height, in_width)
        output = torch.nn.functional.conv2d(in_data, samples,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation)  # type: torch.Tensor

        return output.view(n_sample, -1, out_channels, out_height, out_width)

    def extra_repr(self):
        string = "n=%d, m=%d" % (self.n, self.m)
        return string

