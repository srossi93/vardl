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

logger = logging.getLogger(__name__)


class MultivariateGaussianDistribution(BaseDistribution):

    def __init__(self,
                 n: int,
                 approx: str = 'factorized',
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu') -> None:
        """
        Distribution class for Matrix Gaussian

        Parameters
        ----------
        n : int
            number of elements
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
        super(MultivariateGaussianDistribution, self).__init__()

        self._logger = logging.getLogger(__name__)
        if n <= 0:
            raise ValueError("N should be positive")
        if approx not in ['factorized']:
            raise ValueError('Current approximations are factorized, full or low-rank')

        self.n = n
        self.approx = approx
        self.device = device
        self.dtype = dtype

        self._mean = nn.Parameter(
            torch.zeros(self.n,
                        dtype=self.dtype,
                        device=self.device),
            requires_grad=False)

        self._logvars = nn.Parameter(
            torch.ones(self.n,    #
                       dtype=dtype,
                       device=self.device) * np.log(1. / (self.n + 1)),
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
            Tensor of samples with shape [n_samples, n]
        """

        epsilon_for_samples = torch.randn(n_samples, self.n,
                                          dtype=self.dtype,
                                          device=self.device,
                                          requires_grad=False)
        if self.approx == 'factorized':
            samples = torch.add(torch.mul(epsilon_for_samples,
                                          torch.exp(self.logvars / 2.0)),
                                self.mean)

        elif self.approx == 'full':
            raise NotImplementedError()
            samples = torch.zeros_like(epsilon_for_samples, device=self.device)

            for i in range(self.m):
                cov_lower_triangular = torch.tril(self.cov_lower_triangular[i, :, :], -1)
                cov_lower_triangular += torch.diagflat(torch.exp(self.logvars[:, i]))
                samples[:, :, i] = torch.add(torch.matmul(cov_lower_triangular, epsilon_for_samples[:, :, i].t()).t(),
                                             self.mean[:, i])

        elif self.approx == 'low-rank':
            raise NotImplementedError()

        return samples


    def extra_repr(self):
        string = r"""n={}, approx={}""".format(self.n, self.approx)
        return string
