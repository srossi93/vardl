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


# TODO: fix everything
import torch
import numpy as np
import logging

from . import BaseBayesianLayer
from .. import distributions
from .. import functional

logger = logging.getLogger(__name__)

class BayesianFastfoodLinear(BaseBayesianLayer):
    """
    Implements a Variational Fastfood linear layer using Fast Hadamard transform
    output = V @ input + c
    where V = S H q_G P H B
        P: permutation matrix
        H: Hadamard matrix (implicit thanks to FHT)
        B: Binary scaling (+-1)
        q_G: Gaussian vector
        S: Scaling vector
    q_G and S are learned variationally
    """

    def __init__(self,
                 in_features,
                 out_features,
                 nmc_train: int = 1,
                 nmc_test: int = 1,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        super(BayesianFastfoodLinear, self).__init__(nmc_train, nmc_test, dtype, device)

        self.out_features = out_features

        # S vector
        #self.S = torch.nn.Parameter(torch.randn(out_features))
        self.q_S = distributions.MultivariateGaussianDistribution(out_features)
        self.prior_S = distributions.MultivariateGaussianDistribution(out_features)
        self.q_S.optimize()
        self.q_G = distributions.MultivariateGaussianDistribution(out_features)
        self.prior_G = distributions.MultivariateGaussianDistribution(out_features)
        self.q_G.optimize()
        # self.q_G = torch.nn.Parameter(torch.tensor(np.random.randn(out_features)).float())
        self.B = torch.nn.Parameter(torch.tensor(np.random.choice((-1, 1), size=out_features)).float(),
                                    requires_grad=False)
        self.P = torch.nn.Parameter(torch.tensor(np.random.permutation(out_features)), requires_grad=False)

        self.bias = torch.nn.Parameter(torch.randn(out_features))
        # self.bias = vardl.distributions.MultivariateGaussianDistribution(out_features)
        # self.bias.optimize = True

        self.prior_S.logvars.fill_(np.log(0.01))
        self.prior_G.logvars.fill_(np.log(0.01))

    def forward(self, input):
        # self.P.data = torch.tensor(np.random.permutation(self.out_features))
        # self.B.data = torch.tensor(np.random.choice((-1, 1), size=self.out_features)).float()
        G = self.q_G.sample([self.nmc, input.size(1)])
        S = self.q_S.sample([self.nmc, input.size(1)])

        HBx = functional.HadamardTransform.apply(self.B * input)
        #logger.info(HBx.size())

        PHBx = HBx[..., self.P]
        #logger.info(PHBx.size())

        #logger.info(G.size())
        HGPHBx = functional.HadamardTransform.apply(G * PHBx)
        #logger.info(HGPHBx.size())
        return (S * HGPHBx) + self.bias  # .sample(self.nmc)

    @property
    def dkl(self):
        dkl = distributions.dkl.dkl_matrix_gaussian(self.q_S, self.prior_S)
        dkl += distributions.dkl.dkl_matrix_gaussian(self.q_G, self.prior_G)
        return dkl