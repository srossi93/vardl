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


class Gaussian2DDistribution(BaseDistribution):

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

        super(Gaussian2DDistribution, self).__init__()

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


        self.mean = nn.Parameter(
            torch.zeros(self.n, self.m,
                        dtype=self.dtype,
                        device=self.device),
            requires_grad=False)


        self.logvars = nn.Parameter(
            np.log(1. / self.n) * torch.ones(self.n, self.m,
                                             dtype=dtype,
                                             device=self.device),
            requires_grad=False)

        if self.approx == 'full':
            # -- The posterior approximation of the covariance matrix is
            # -- parametrized using Log-Cholesky parametrization
            # -- ref. Unconstrained Parameterizations for Variance-Covariance Matrices (p.2)
            self.cov_lower_triangular = nn.Parameter(
                np.log(1. / self.n) *
                torch.eye(self.n, self.n,
                          dtype=self.dtype,
                          device=self.device) *
                torch.ones(self.m, 1, 1,
                           dtype=self.dtype,
                           device=self.device),
                requires_grad=False)

        elif self.approx == 'low rank':
            raise NotImplementedError('')
            rank = 10

            self.q_W_low_rank = nn.Parameter(torch.zeros(self.n, rank,
                                                      dtype=self.dtype,
                                                      device=self.device) *
                                             torch.ones(self.m, 1, 1,
                                                     dtype=self.dtype,
                                                     device=self.device),
                                             requires_grad=False)

    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train


    def sample(self, n_samples: int) -> torch.Tensor:

        epsilon_for_W_sample = torch.randn(n_samples, self.n, self.m,
                                           dtype=self.dtype,
                                           device=self.device,
                                           requires_grad=False)
        if self.approx == 'factorized':
            w_sample = torch.add(torch.mul(epsilon_for_W_sample,
                                           torch.exp(self.logvars / 2.0)),
                                 self.mean)
            return w_sample

        elif self.approx == 'full':
            pass
        elif self.approx == 'low-rank':
            pass

        return None





