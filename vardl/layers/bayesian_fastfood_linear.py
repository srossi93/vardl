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
                 bias: bool = False,
                 nmc_train: int = 1,
                 nmc_test: int = 1,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        super(BayesianFastfoodLinear, self).__init__(nmc_train, nmc_test, dtype, device)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if self.out_features < self.in_features:
            raise NotImplementedError('Corner case to handle')


        self.times_to_stack_v = self.out_features // self.in_features

        # S vector
        #self.S = torch.nn.Parameter(torch.randn(in_features))
        self.q_S = distributions.MultivariateGaussianDistribution(self.in_features)
        self.prior_S = distributions.MultivariateGaussianDistribution(self.in_features)
        self.q_S.optimize()
        self.q_G = distributions.MultivariateGaussianDistribution(self.in_features)
        self.prior_G = distributions.MultivariateGaussianDistribution(self.in_features)
        self.q_G.optimize()
        # self.q_G = torch.nn.Parameter(torch.tensor(np.random.randn(in_features)).float())

        self.B = torch.nn.Parameter(torch.Tensor(np.random.choice((-1, 1), size=[self.times_to_stack_v,
                                                                                 self.in_features])).float(),
                                    requires_grad=False)

        #self.P = torch.nn.Parameter(torch.Tensor(np.random.permutation(self.out_features)), requires_grad=False)
        self.P = torch.nn.Parameter(
            torch.from_numpy(
                np.hstack([(i * self.in_features) + np.random.permutation(self.in_features) for i in range(
                    self.times_to_stack_v)])
            ), requires_grad=False)
#        logger.debug(str(self.P))

        #self.bias = torch.nn.Parameter(torch.randn(self.out_features))
        if bias:
            self.q_bias = distributions.MultivariateGaussianDistribution(self.out_features)
            self.prior_bias = distributions.MultivariateGaussianDistribution(self.out_features)
            self.prior_bias.logvars.fill_(np.log(0.01))
            self.q_bias.optimize()

        self.prior_S.logvars.fill_(np.log(0.01))
        self.prior_G.logvars.fill_(np.log(0.01))

    def forward(self, input: torch.Tensor):
        """
        Perform Vx + b
        Parameters
        ----------
        input torch.Tensor: Input tensor with size (NMC x BS x D_in)

        Returns
        -------

        """
        # self.P.data = torch.tensor(np.random.permutation(self.in_features))
        # self.B.data = torch.tensor(np.random.choice((-1, 1), size=self.in_features)).float()
        batch_size = input.size(1)
        G = self.q_G.sample(self.nmc * batch_size * self.times_to_stack_v).view(self.nmc, batch_size, -1)
        S = self.q_S.sample(self.nmc * batch_size * self.times_to_stack_v).view(self.nmc, batch_size, -1)

        logger.debug('input: %s' % str(input.size()))
        input = input.unsqueeze(-2) # Size: NMC x BS x 1 x D_in
        logger.debug('input unsqueezed: %s' % str(input.size()))
        #input = torch.cat([input for _ in range(self.out_features//self.in_features)], dim=-1)
        logger.debug('B: %s' % str(self.B.size()))
        Bx = (self.B * input)#.squeeze(-1) # Size: NMC x BS x D_out
        logger.debug('Bx: %s' % str(Bx.size()))

        HBx = functional.HadamardTransform.apply(Bx).view(self.nmc, batch_size, -1)#.squeeze(-1)
        logger.debug('HBx: %s' % str(HBx.size()))

        PHBx = HBx[..., self.P.long()]#.unsqueeze(-2)
        logger.debug('PHBx: %s' % str(PHBx.size()))
        #logger.info(PHBx.size())

        logger.debug('G: %s' % str(G.size()))
        GPHBx = G * PHBx
        logger.debug('GPHBx: %s' % str(GPHBx.size()))
        HGPHBx = functional.HadamardTransform.apply(GPHBx)
        logger.debug('HGPHBx: %s' % str(HGPHBx.size()))
        #logger.info(HGPHBx.size())
        logger.debug('S: %s' % str(G.size()))

        if self.bias:
            return (HGPHBx) + self.q_bias.sample(self.nmc * batch_size).view(self.nmc, batch_size, -1)
        else:
            return HGPHBx

    @property
    def dkl(self):
        dkl = distributions.dkl.dkl_matrix_gaussian(self.q_S, self.prior_S)
        dkl += distributions.dkl.dkl_matrix_gaussian(self.q_G, self.prior_G)
        if self.bias:
            dkl += distributions.dkl.dkl_matrix_gaussian(self.q_bias, self.prior_bias)
        return dkl