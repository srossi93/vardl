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
import torch.nn as nn

from . import MultivariateGaussianDistribution

import logging
logger = logging.getLogger(__name__)


class FullCovarianceMultivariateGaussian(MultivariateGaussianDistribution):
    def __init__(self, n: int, dtype: torch.dtype = torch.float32):
        super(FullCovarianceMultivariateGaussian, self).__init__(n, dtype=dtype)
        self.name = 'Full covariance factorized multivariate Gaussian'
        # -- The posterior approximation of the covariance matrix is
        # -- parametrized using Log-Cholesky parametrization
        # -- ref. Unconstrained Parameterizations for Variance-Covariance Matrices (p.2)
        self.cov_lower_triangular = nn.Parameter(
            torch.eye(self.n, self.n, dtype=self.dtype) * np.log(1. / self.n),
            requires_grad=False)
        self.has_local_reparam_linear = True
        self.has_local_reparam_conv2d = False

    def sample(self, n_samples: int):
        epsilon_for_samples = torch.randn(n_samples, self.n,
                                          dtype=self.dtype,
                                          device=self.mean.device,
                                          requires_grad=False)

        cov_lower_triangular = torch.tril(self.cov_lower_triangular, -1)
        cov_lower_triangular += torch.diagflat(torch.exp(self.logvars))
        samples = torch.add(torch.matmul(cov_lower_triangular, epsilon_for_samples.t()).t(),
                                         self.mean)
        return samples

    def sample_local_reparam_linear(self, n_sample: int, in_data: torch.Tensor):
        raise NotImplementedError('Not implemented yet')

    def sample_local_reparam_conv2d(self, n_sample, in_data, out_channels, in_channels, in_height, in_width,
                                    kernel_size, stride, padding, dilation):
        raise NotImplementedError('Not implemented yet')

    def extra_repr(self):
        string = "n=%d" % (self.n )
        return string

