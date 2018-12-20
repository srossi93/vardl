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


import numpy as np
import torch
import torch.nn as nn

from . import BaseDistribution
import logging
from ..utils import timing



class MatrixGaussianDistribution(BaseDistribution):

    def __init__(self,
                 n: int,
                 m: int,
                 approx: str = 'factorized',
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu',
                 rank: int = 2) -> None:
        """
        Distribution class for Matrix Gaussian

        Parameters
        ----------
        n : int
            number of input features
        m : int
            number of output features
        approx : str ('factorized')
            approximation of the covariance matrix.
            Possible alternatives are 'factorized' (for diagonal covariance), 'full' and 'low-rank'
        dtype : torch.dtype (torch.float32)
            Data type for samples
        device : str ('cpu')
            Device on which samples will be stored ('cpu' or 'cuda')

        Raises
        --------
        ValueError
            If sanity checks fail
        """
        super(MatrixGaussianDistribution, self).__init__()

        self._logger = logging.getLogger(__name__)
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
                       device=self.device) * np.log(2. / (self.n + self.m)),
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

        if self.approx == 'low-rank':
 #           self._logger.debug('Low rank covariance %d x %d' % (n, rank))
            self.rank = rank
            if rank > n:
                self._logger.warning('Low rank covariance bigger than full (%d > %d). Setting rank = %d' % (rank, n, n))
                self.rank = n

            self._cov_low_rank = nn.Parameter(torch.zeros(self.n, self.rank,
                                                          dtype=self.dtype,
                                                          device=self.device) *
                                              torch.ones(self.m, 1, 1,
                                                         dtype=self.dtype,
                                                         device=self.device),
                                              requires_grad=False)

        return

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
        self._cov_lower_triangular.data = value

    @property
    def cov_low_rank(self):
        return self._cov_low_rank

    @cov_low_rank.setter
    def cov_low_rank(self, value: torch.Tensor):
        self._cov_low_rank.data = value

    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the current distribution

        Parameters
        ----------
        n_samples : int
            Number of independent samples

        Return
        -------
        torch.Tensor
            Tensor of samples with shape [n_samples, n, m]
        """

        epsilon_for_samples = torch.randn(n_samples, self.n, self.m,
                                           dtype=self.dtype,
                                           device=self.device,
                                           requires_grad=False)
        if self.approx == 'factorized':
            samples = torch.add(torch.mul(epsilon_for_samples,
                                           torch.exp(self.logvars / 2.0)),
                                 self.mean)

        elif self.approx == 'full':
            samples = torch.zeros_like(epsilon_for_samples, device=self.device)

            for i in range(self.m):
                cov_lower_triangular = torch.tril(self.cov_lower_triangular[i, :, :], -1)
                cov_lower_triangular += torch.diagflat(torch.exp(self.logvars[:, i]))
                samples[:, :, i] = torch.add(torch.matmul(cov_lower_triangular, epsilon_for_samples[:, :, i].t()).t(),
                                              self.mean[:, i])

        elif self.approx == 'low-rank':
            raise NotImplementedError()

        return samples


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
        epsilon_for_Y_sample = torch.randn(n_sample, in_data.size(-2), self.mean.size(1),  # TODO: fix this
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
                         torch.diagflat(torch.exp(self.logvars[:, i]))

                var_Y[:, :, i] = torch.sum(torch.matmul(in_data, L_chol) ** 2, -1)



        if self.approx == 'low-rank':
            var_Y = torch.zeros_like(mean_Y, device=self.device) * torch.ones(n_sample, 1, 1, device=self.device)
            for i in range(self.m):
                var_Y[:, :, i] = torch.add(
                    torch.sum(torch.matmul(in_data, self.cov_low_rank[i, :, :]) ** 2, -1),
                    torch.matmul(in_data.pow(2), torch.exp(self.logvars[:, i]))
                )

        Y = mean_Y + torch.sqrt(var_Y + 1e-5) * epsilon_for_Y_sample
        return Y

    def sample_local_repr_conv2d(self, n_sample: int,
                                 in_data: torch.Tensor,
                                 out_channels: int,
                                 in_channels: int,
                                 in_height: int,
                                 in_width: int,
                                 kernel_size: int,
                                 stride: int,
                                 padding: int,
                                 dilation: int)-> torch.Tensor:

        out_height = int((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        out_width = int((in_width + 2 * padding - dilation * (kernel_size - 1) -1)/stride + 1)

        mean_weights = self.mean.view(out_channels, in_channels, kernel_size, kernel_size)
        logvars_weights = self.logvars.view(out_channels, in_channels, kernel_size, kernel_size)

        in_data = in_data.contiguous().view(-1, in_channels, in_height, in_width)
        mean_output = torch.nn.functional.conv2d(in_data, mean_weights,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation)
        if self.approx == 'factorized':
            var_output = torch.nn.functional.conv2d(in_data.pow(2), logvars_weights.exp(),
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation)

            eps = torch.randn_like(mean_output, requires_grad=False)
            output = mean_output + eps * torch.sqrt(var_output + 1e-5)

            return output.view(n_sample, -1, out_channels, out_height, out_width)

        if self.approx == 'low-rank':
            self._logger.info(self.cov_low_rank.size())
            print(stride)
            var_output = torch.zeros_like(mean_output, device=self.device) * torch.ones(n_sample, 1, 1, device=self.device)
            for i in range(self.m):
                # TODO: what to do with low rank and local reparameterization for convolution
                var_output[:, :, i] = torch.add(
                    torch.sum(torch.nn.functional.conv2d(in_data, self.cov_low_rank[i].view(1, in_channels, kernel_size,
                                                                                            kernel_size, self.rank),
                                                         stride=stride,
                                                         padding=padding,
                                                         dilation=dilation)** 2, -1),
                    torch.nn.functional.conv2d(in_data.pow(2), torch.exp(logvars_weights[:, i]),
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation)
                )
            eps = torch.randn_like(mean_output, requires_grad=False)
            output = mean_output + eps * torch.sqrt(var_output + 1e-5)
            return output.view(n_sample, -1, out_channels, out_height, out_width)
           # raise NotImplementedError

    def extra_repr(self):
        string = r"""approx={}""".format(self.approx)
        return string
