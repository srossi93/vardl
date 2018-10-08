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
import torch.nn as nn
import numpy as np

from . import BaseDistribution


class MatrixGaussianDistribution(BaseDistribution):

    def __init__(self, n: int, m: int, approx: str,
                 dtype: torch.dtype, device: torch.device):
        r"""

        Args:
            n (int): Number of rows (input features)
            m (int): Number of cols (output features)
            approx (str): Covariance approximantion (factorized, full or low-rank)
            dtype (torch.dtype): Datatype
            device (torch.device): Device (cpu/cuda)
        """

        super(MatrixGaussianDistribution, self).__init__()

        if n <= 0:
            raise ValueError("N should be positive")
        if m <= 0:
            raise ValueError("M should be positive")
        if approx not in ['factorized', 'full', 'low-rank']:
            raise ValueError('Current approximations are factorized, full or low-rank')

        self.n = n
        self.m = m
        self.approx = approx
        self.device = device
        self.dtype = dtype

        self._mean = nn.Parameter(
            torch.zeros(self.n, self.m,
                        dtype=self.dtype,
                        device=self.device),
            requires_grad=False)

        self._logvars = nn.Parameter(
            torch.ones(self.n, self.m,   #
                       dtype=dtype,
                       device=self.device) * np.log(1. / self.n),
            requires_grad=False)

        if self.approx == 'full':
            # -- The posterior approximation of the covariance matrix is
            # -- parametrized using Log-Cholesky parametrization
            # -- ref. Unconstrained Parameterizations for Variance-Covariance Matrices (p.2)
            self._cov_lower_triangular = nn.Parameter(
                torch.eye(self.n, self.n,
                          dtype=self.dtype,
                          device=self.device) *
                torch.ones(self.m, 1, 1,
                           dtype=self.dtype,
                           device=self.device) * np.log(1. / self.n),
                requires_grad=False)

        elif self.approx == 'low rank':
            raise NotImplementedError('')
            rank = 10

            self._cov_low_rank = nn.Parameter(torch.zeros(self.n, rank,
                                                          dtype=self.dtype,
                                                          device=self.device) *
                                              torch.ones(self.m, 1, 1,
                                                         dtype=self.dtype,
                                                         device=self.device),
                                              requires_grad=False)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value: torch.Tensor):
        self._mean.data = value

    @property
    def logvars(self):
        return self._logvars

    @logvars.setter
    def logvars(self, value: torch.Tensor):
        self._logvars.data = value

    @property
    def cov_lower_triangular(self):
        return self._cov_lower_triangular

    @cov_lower_triangular.setter
    def cov_lower_triangular(self, value: torch.Tensor):
        self.cov_lower_triangular.data = value

    @property
    def cov_low_rank(self):
        return self.cov_low_rank

    @cov_low_rank.setter
    def cov_low_rank(self, value: torch.Tensor):
        self._cov_low_rank.data = value

    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train

    def sample(self, n_samples: int) -> torch.Tensor:

        epsilon_for_W_sample = torch.randn(n_samples, self.n, self.m,      # TODO: Fix this
                                           dtype=self.dtype,
                                           device=self.device,
                                           requires_grad=False)
        if self.approx == 'factorized':
            w_sample = torch.add(torch.mul(epsilon_for_W_sample,
                                           torch.exp(self.logvars / 2.0)),
                                 self.mean)
            return w_sample

        elif self.approx == 'full':
            w_sample = torch.zeros_like(epsilon_for_W_sample, device=self.device)

            for i in range(self.m):
                cov_lower_triangular = torch.tril(self.cov_lower_triangular[i, :, :], -1)
                L_chol = cov_lower_triangular + \
                    torch.diagflat(torch.exp(self.logvars[i]))
                print(epsilon_for_W_sample.size())
                w_sample[:, :, i] = torch.add(torch.matmul(L_chol, epsilon_for_W_sample[:, :, i].t()).t(),
                                              self.mean[:, i])

            return w_sample

        elif self.approx == 'low-rank':
            pass

        return None

    def sample_local_repr(self, n_sample: int, in_data: torch.Tensor) -> torch.Tensor:
        # For stochastic gradient optimization, high variance in the gradient
        # will fail to make much progress in a reasonable amount of time
        # To avoid this issue, we use local reparameterization
        # ref. Variational Dropout and the Local Reparameterization Trick
        #
        # Instead of sampling W and than computing Y=WX, we directly compute Y,
        # giving the fact that for a Gaussian posterior on the weights W, also the
        # posterior on the outputs Y conditional to the inputs X is factorized
        # Gaussian as well
        epsilon_for_Y_sample = torch.zeros(n_sample, in_data.size(-2), self.mean.size(1),  # TODO: fix this
                                           dtype=self.dtype,
                                           device=self.device,
                                           requires_grad=False)

        mean_Y = torch.matmul(in_data, self.mean)

        if self.approx == 'factorized':
            var_Y = torch.matmul(in_data.pow(2), torch.exp(self.logvars))
            Y = mean_Y + torch.sqrt(var_Y + 1e-5) * epsilon_for_Y_sample

        if self.approx == 'full':
            var_Y = torch.zeros_like(mean_Y, device=self.device) * torch.ones(n_sample, 1, 1, device=self.device)
            # q_W_conv_L_chol = self._q_W_conv_L_chol
            for i in range(self.m):
                cov_lower_triangular = torch.tril(self.cov_lower_triangular[i, :, :], -1)
                L_chol = cov_lower_triangular + \
                         torch.diagflat(torch.exp(self.logvars[i]))

                var_Y[:, :, i] = torch.sum(torch.matmul(in_data, L_chol) ** 2, -1)



        if self.approx == 'low-rank':
            pass
        return Y

    def extra_repr(self):
        string = r"""approx={}""".format(self.approx)
        return string
