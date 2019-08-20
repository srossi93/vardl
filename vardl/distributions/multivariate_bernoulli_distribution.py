#  Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch
import torch.nn as nn

from . import BaseDistribution
import logging
from ..utils import timing

logger = logging.getLogger(__name__)


class MultivariateBernoulliDistribution(BaseDistribution):

    def __init__(self,
                 n: int,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu') -> None:
        """
        Distribution class for Multivariate Bernoulli

        Parameters
        ----------
        n : int
            number of elements
        dtype : torch.dtype (torch.float32)
            Data type for samples
        device : str ('cpu')
            Device on which samples will be stored ('cpu' or 'cuda')

        Raises
        --------
        ValueError
            If sanity checks fail
        """
        super(MultivariateBernoulliDistribution, self).__init__()

        self._logger = logging.getLogger(__name__)
        if n <= 0:
            raise ValueError("N should be positive")

        self.n = n
        self.device = device
        self.dtype = dtype

        self.logp = nn.Parameter(
            torch.ones(self.n,
                        dtype=self.dtype,
                        device=self.device) * np.log(0.5),
            requires_grad=False)

        # Temperature scaling
        self.t = torch.ones(1) * 1.
        self.support = torch.tensor([-1, 1], dtype=self.dtype)  # TODO: DO NOT TOUCH!! Scaling of samples doesnt
        # work



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
        u = torch.zeros(n_samples, self.n, requires_grad=False).uniform_()
        samples = torch.sigmoid(1 / self.t * (self.logp - (1 - self.logp.exp()).log() + u.log() - (1-u).log()))
        samples = 2 * samples - 1
        return samples
