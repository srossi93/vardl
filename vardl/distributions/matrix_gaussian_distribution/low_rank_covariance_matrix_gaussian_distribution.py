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

from . import MatrixGaussianDistribution

import logging
logger = logging.getLogger(__name__)


class LowRankCovarianceMatrixGaussian(MatrixGaussianDistribution):
    def __init__(self, n, m, rank, dtype=torch.float32):
        """

        Args:
            n (int):
            m (int):
            rank (int):
            dtype (torch.dtype):
        """
        super(LowRankCovarianceMatrixGaussian, self).__init__(n, m, dtype=dtype)
        self.name = 'Low rank covariance factorized matrix Gaussian'
        self.rank = rank
        if self.rank > n:
            logger.warning('Low rank covariance bigger than full (%d > %d). Setting rank = %d' % (self.rank, n, n))
            self.rank = n

        self.cov_low_rank = nn.Parameter(torch.zeros(self.n, self.rank, dtype=self.dtype) *
                                         torch.ones(self.m, 1, 1, dtype=self.dtype),
                                         requires_grad=False)

        self.has_local_reparam_linear = True
        self.has_local_reparam_conv2d = False

    def sample(self, n_samples):
        device = self.mean.device
        samples = torch.zeros(n_samples, self.n, self.m,
                              dtype=self.dtype,
                              device=device,
                              requires_grad=False)

        epsilon_for_log_diag_W_sample = torch.randn(self.nmc, self.in_features, self.out_features,
                                                    device=device,
                                                    dtype=self.dtype,
                                                    requires_grad=False)
        epsilon_for_low_rank_sample = torch.randn(self.nmc, self.rank, self.out_features,
                                                  device=device,
                                                  dtype=self.dtype,
                                                  requires_grad=False)
        for i in range(self.m):
            samples[:, :, i] = ((torch.matmul(self.cov_low_rank[i, :, :],
                                             epsilon_for_low_rank_sample[:, :, i].t()).t() +
                                torch.matmul(torch.diagflat(torch.exp(self.logvars[i, :] / 2.0)),
                                             epsilon_for_log_diag_W_sample[:, :, i].t()).t()) +
                                self.mean[:, i])

        return samples

    def sample_local_reparam_linear(self, n_sample: int, in_data: torch.Tensor):
        device = self.mean.device
        epsilon_for_Y_sample = torch.randn(n_sample, in_data.size(-2), self.mean.size(1),
                                           dtype=self.dtype,
                                           device=device,
                                           requires_grad=False)  # type: torch.Tensor

        mean_Y = torch.matmul(in_data, self.mean)

        var_Y = torch.zeros_like(mean_Y, device=device) * torch.ones(n_sample, 1, 1, device=device)
        for i in range(self.m):
            var_Y[:, :, i] = torch.add(
                torch.sum(torch.matmul(in_data, self.cov_low_rank[i, :, :]) ** 2, -1),
                torch.matmul(in_data.pow(2), torch.exp(self.logvars[:, i]))
            )

        Y = mean_Y + torch.sqrt(var_Y + 1e-5) * epsilon_for_Y_sample
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
            logger.warning('Low rank covariance has no local reparameterization for conv2d operation. Falling back '
                           'to standard reparameterization for %s', self._get_name())

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

