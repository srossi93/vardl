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
from ..distributions import MatrixGaussianDistribution
from ..distributions import dkl_matrix_gaussian
from . import BaseBayesianLayer


class BayesianConv2d(BaseBayesianLayer):

    def __init__(self,
                 in_channels: int,
                 in_height: int,
                 in_width: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 approx: str = 'factorized',
                 local_reparameterization: bool = False,
                 nmc_train: int = 1,
                 nmc_test: int = 1,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):

        super(BayesianConv2d, self).__init__(nmc_train=nmc_train,
                                             nmc_test=nmc_test,
                                             dtype=dtype,
                                             device=device)


        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.approx = approx
        self.local_reparameterization = local_reparameterization
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.dtype = dtype
        self.device = device
        self.bias = bias

        self.filter_size = self.in_channels * (self.kernel_size ** 2)

        self.out_height = int(
            (self.in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.out_width = int(
            (self.in_width + 2*self.padding - self.dilation*(self.kernel_size-1) -1)/self.stride + 1)

        self.prior_W = MatrixGaussianDistribution(n=self.filter_size,
                                                  m=self.out_channels,
                                                  approx='factorized',
                                                  dtype=self.dtype,
                                                  device=self.device)

        self.q_posterior_W = MatrixGaussianDistribution(n=self.filter_size,
                                                        m=self.out_channels,
                                                        approx=self.approx,
                                                        dtype=self.dtype,
                                                        device=self.device)
        self.q_posterior_W.optimize(True)


        self.unfold_engine = nn.Unfold(kernel_size=self.kernel_size,
                                       dilation=self.dilation,
                                       padding=self.padding,
                                       stride=self.stride)


        if self.bias:
            raise NotImplementedError

        self.train()
        return

    @property
    def dkl(self):
        total_dkl = dkl_matrix_gaussian(self.prior_W, self.q_posterior_W)

        if self.bias:
            raise NotImplementedError()

        return total_dkl


    def extract_patches(self, input) -> torch.Tensor:
        input = input * torch.ones(self.nmc, *input.size())

        # print('conv2d-input', input.size())

        # print('input', input.size())

        batched_input = input.contiguous().view(-1, self.in_channels, self.in_height, self.in_width)

        # print('batched_input', batched_input.size())

        patches = self.unfold_engine(batched_input).transpose(-1, -2)
        patches = patches.contiguous().view(self.nmc, -1, self.filter_size)

        return patches


    def forward(self, input):

        input = input * torch.ones(self.nmc, *input.size())

        #print('conv2d-input', input.size())


        #print('input', input.size())

        batched_input = input.contiguous().view(-1, self.in_channels, self.in_height, self.in_width)

        #print('batched_input', batched_input.size())

        patches = self.unfold_engine(batched_input).transpose(-1, -2)
        patches = patches.contiguous().view(self.nmc, -1, self.filter_size)

        #print('patches', patches.size())



        if not self.local_reparameterization:
            w_sample = self.q_posterior_W.sample(self.nmc)

            #print('w_sample', w_sample.size())

            output = torch.matmul(patches, w_sample)
            output = output.transpose(-1, -2).contiguous().view(self.nmc, -1, self.out_channels, self.out_height, self.out_width)

           # print('conv2d-output', output.size())
            return output

        if self.local_reparameterization:
            output = self.q_posterior_W.sample_local_repr(self.nmc, patches)
            output = output.transpose(-1, -2).contiguous().view(self.nmc, -1, self.out_channels, self.out_height,
                                                                self.out_width)
            return output


        raise NotImplementedError


    def extra_repr(self):
        return ''
