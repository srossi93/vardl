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

import abc
import torch
import numpy as np
import torch.nn as nn

from .. import BaseDistribution

import logging
logger = logging.getLogger(__name__)


class MatrixGaussianDistribution(BaseDistribution, metaclass=abc.ABCMeta):

    def __init__(self, n: int, m: int, dtype: torch.dtype = torch.float32, **kargs):
        """
        Distribution class for Matrix Gaussian
        """
        super(MatrixGaussianDistribution, self).__init__()

        if n <= 0:
            raise ValueError("N should be positive")
        if m <= 0:
            raise ValueError("M should be positive")

        self.n = n
        self.m = m
        self.dtype = dtype

        self.mean = nn.Parameter(
            torch.zeros(self.n, self.m, dtype=self.dtype),
            requires_grad=False)  # type: torch.nn.Parameter

        self.logvars = nn.Parameter(
            torch.ones(self.n, self.m, dtype=dtype) * np.log(2. / (self.n + self.m)),
            requires_grad=False)

    @abc.abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the current distribution
        """
        raise NotImplementedError('Subclass need to implement this method')

    @abc.abstractmethod
    def sample_local_reparam_linear(self, n_sample: int, in_data: torch.Tensor) -> torch.Tensor:
        # For stochastic gradient optimization, high variance in the gradient
        # will fail to make much progress in a reasonable amount of time
        # To avoid this issue, we use local reparameterization
        # ref. Variational Dropout and the Local Reparameterization Trick
        #
        # Instead of sampling W and than computing Y=WX, we directly compute Y,
        # giving the fact that for a Gaussian posterior on the weights W, also the
        # posterior on the outputs Y conditional to the inputs X is factorized
        # Gaussian as well
        raise NotImplementedError('Subclass need to implement this method')

    @abc.abstractmethod
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
        raise NotImplementedError('Subclass need to implement this method')

    def forward(self, *input):
        return input
