r"""
   Copyright 2018 Simone Rossi, Maurizio Filippone

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
import torch
import torch.nn as nn
import numpy as np
from ..distributions import MatrixGaussianDistribution
from ..distributions import dkl_matrix_gaussian
from . import BaseBayesianLayer


class BayesianLinear(BaseBayesianLayer):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 approx: str = 'factorized',
                 local_reparameterization: bool = False,
                 nmc_train: int = 10,
                 nmc_test: int = 10,
                 dtype: torch.device = torch.float32,
                 device: torch.device = torch.device('cpu')):

        super(BayesianLinear, self).__init__(nmc_train=nmc_train,
                                             nmc_test=nmc_test,
                                             dtype=dtype,
                                             device=device)
        self.in_features = in_features
        self.out_features = out_features

        self.approx = approx
        self.local_reparameterization = local_reparameterization

        self.prior_W = MatrixGaussianDistribution(n=self.in_features,
                                                  m=self.out_features,
                                                  approx='factorized',
                                                  dtype=self.dtype,
                                                  device=self.device)

        self.q_posterior_W = MatrixGaussianDistribution(n=self.in_features,
                                                        m=self.out_features,
                                                        approx=self.approx,
                                                        dtype=self.dtype,
                                                        device=self.device)

    #    self.prior_W.logvars.data.fill_(np.log(0.05))
        self.prior_W.logvars.data.fill_(np.log(.005))

        self.q_posterior_W.optimize(True)
        self.prior_W.logvars.requires_grad = True

        # -- Scaling factor for weights
        #self.log_scaling_factor = 1 * Parameter(torch.zeros(1, dtype=dtype), requires_grad=True)

        if bias:
            self.bias = True

            # -- Prior on b
            self.prior_b_m = torch.zeros(out_features, dtype=dtype, requires_grad=False)
            self.prior_b_logv = torch.zeros(
                out_features, dtype=dtype, requires_grad=False)

            # -- Posterior approximation on b
            self.q_b_m = nn.Parameter(torch.zeros(out_features))
            self.q_b_logv = nn.Parameter(torch.zeros(out_features))

        else:
            self.bias = False
            self.register_parameter('q_b_m', None)
            self.register_parameter('q_b_logv', None)

        #dkl = self.dkl
        self.train()

    @property
    def dkl(self) -> torch.Tensor:
        #total_dkl = dkl_matrix_gaussian(self.prior_W, self.q_posterior_W)
        total_dkl = dkl_matrix_gaussian(self.q_posterior_W, self.prior_W)

        if self.bias:
            pass
            #raise NotImplementedError()

        return total_dkl

    def forward(self, in_data):
        # Sample nmc times Wr ~ q(Wr | q_W_m, q_W_logv)

        #print(in_data.size())

        if not self.local_reparameterization:
            w_sample = self.q_posterior_W.sample(self.nmc)
            Y = torch.matmul(in_data, w_sample)

        if self.local_reparameterization:
            Y = self.q_posterior_W.sample_local_repr(self.nmc, in_data)

        if self.bias:
            epsilon_for_b_sample = torch.randn(self.nmc, self.out_features, dtype=self.dtype,
                                               requires_grad=False).to(self.device)
            b_sample = torch.add(torch.mul(epsilon_for_b_sample,
                                                     torch.exp(self.q_b_logv / 2.0)),
                                           self.q_b_m)

            Y = Y.permute(1, 0, 2)  # Stupid Broadcasting
            Y = torch.add(Y, b_sample)
            Y = Y.permute(1, 0, 2)  # It works but it sucks

        #print('lin-output', Y.size())

        return Y

    def extra_repr(self):
        string = r"""in_features={}, out_features={}, bias={}, local_repr={}""".format(
            self.in_features, self.out_features, self.bias, self.local_reparameterization)
        return string
