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

        self.prior_W = MatrixGaussianDistribution(m=self.filter_size,
                                                  n=self.out_channels,
                                                  approx='factorized',
                                                  dtype=self.dtype,
                                                  device=self.device)
        self.prior_W.logvars.data.fill_(10)

        self.q_posterior_W = MatrixGaussianDistribution(m=self.filter_size,
                                                        n=self.out_channels,
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

        input = input * torch.ones(self.nmc, 1, 1, 1, 1).to(self.device)

        #print('conv2d-input', input.size())


        #print('input', input.size())

        batched_input = input.contiguous().view(-1, self.in_channels, self.in_height, self.in_width)

        #print('batched_input', batched_input.size())

        #patches = self.unfold_engine(batched_input)#.transpose(-1, -2)
        #print('patches_before_reshape:', patches.size())
        #patches = patches.contiguous().view( -1, self.filter_size)
        #print('patches_after_reshape:', patches.size())



        #w_sample = self.q_posterior_W.sample(self.nmc).view(self.nmc, -1)
        #print('w_sample', w_sample.size())

        if not self.local_reparameterization:
            w_sample = self.q_posterior_W.sample(self.nmc).view(-1, self.out_channels, self.in_channels, self.kernel_size,
                                                        self.kernel_size)

            #print('w_sample', w_sample.size())

            #output = torch.matmul(patches, w_sample.view(self.out_channels, self.in_channels * self.kernel_size * self.kernel_size))
            #output = output.contiguous().view(self.nmc, -1, self.out_channels, self.out_height, self.out_width)

           # print('conv2d-output', output.size())
            #input = input.view(-1, self.in_channels, self.in_height, self.in_width)
            #patches = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, padding=self.padding)\
            #    .transpose(-1,-2).contiguous().view(self.nmc, -1, self.in_channels * self.kernel_size ** 2)
            #
            #output = torch.matmul(patches, w_sample).transpose(-1, -2).contiguous().view(self.nmc, -1, self.out_channels, self.out_height, self.out_width)

            if False:
                batch_size = input.size(1)
                patches = torch.nn.functional.unfold(input.view(-1, self.in_channels, self.in_height, self.in_width), self.kernel_size,
                                                 padding=self.padding, stride=self.stride, dilation=self.dilation).view(self.nmc, batch_size, self.filter_size, -1)

                #print('patches:', patches.size())

                w_matrix_full = (w_sample.view(w_sample.size(0), w_sample.size(1), -1) * torch.ones(batch_size, 1, 1, 1, device=self.device)).transpose(0, 1)

                #print('w:', w_matrix_full.size())

                output = torch.matmul(w_matrix_full, patches)#.view(self.nmc, batch_size, self.out_channels, self.out_height, self.out_width)

            else:
                weights = w_sample.view(self.out_channels, self.in_channels, self.kernel_size,
                                                   self.kernel_size)
                input = input.contiguous().view(-1, self.in_channels, self.in_height, self.in_width)
                output = torch.nn.functional.conv2d(input, weights,
                                                stride=self.stride,
                                                padding=self.padding,
                                                dilation=self.dilation)

            return output

        if self.local_reparameterization:
            #output = self.q_posterior_W.sample_local_repr(self.nmc, patches)
            #output = output.transpose(-1, -2).contiguous().view(self.nmc,
            #                                                    -1,
            #                                                    self.out_channels,
            #                                                    self.out_height,
            #                                                    self.out_width)

            mean_weights = self.q_posterior_W.mean.view(self.out_channels, self.in_channels, self.kernel_size,
                                                        self.kernel_size)
            logvars_weights = self.q_posterior_W.logvars.view(self.out_channels, self.in_channels, self.kernel_size,
                                                        self.kernel_size)

            input = input.contiguous().view(-1, self.in_channels, self.in_height, self.in_width)
            mean_output = torch.nn.functional.conv2d(input, mean_weights,
                                                stride=self.stride,
                                                padding=self.padding,
                                                dilation=self.dilation)

            var_output = torch.nn.functional.conv2d(input.pow(2), logvars_weights.exp(),
                                                     stride=self.stride,
                                                     padding=self.padding,
                                                     dilation=self.dilation)

            #print(mean_output.norm())
            #print(var_output.norm())

            eps = torch.randn_like(mean_output, requires_grad=False)
            output = mean_output + eps * torch.sqrt(var_output + 1e-5)



            return output


        raise NotImplementedError


    def extra_repr(self):
        return ''
