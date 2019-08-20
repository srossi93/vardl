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
#
#
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
#

import torch

from . import MultivariateGaussianDistribution
from ...utils import compute_output_shape_conv2d

import logging
logger = logging.getLogger(__name__)


class FullyFactorizedMultivariateGaussian(MultivariateGaussianDistribution):
    def __init__(self, n: int, dtype: torch.dtype = torch.float32):
        super(FullyFactorizedMultivariateGaussian, self).__init__(n,  dtype=dtype)
        self.name = 'Fully factorized matrix Gaussian'
        self.has_local_reparam_linear = False
        self.has_local_reparam_conv2d = False

    def sample(self, n_samples: int):

        epsilon_for_samples = torch.randn(n_samples, self.n,
                                          dtype=self.dtype,
                                          device=self.mean.device,
                                          requires_grad=False)

        samples = torch.add(torch.mul(epsilon_for_samples,
                                      torch.exp(self.logvars / 2.0)),
                            self.mean)  # type: torch.Tensor
        return samples

    def sample_local_reparam_linear(self, n_sample: int, in_data: torch.Tensor):
        raise NotImplementedError('Not implemented yet')

    def sample_local_reparam_conv2d(self, n_sample, in_data, out_channels, in_channels, in_height, in_width,
                                    kernel_size, stride, padding, dilation):
        raise NotImplementedError('Not implemented yet')

    def extra_repr(self):
        string = "n=%d" % (self.n)
        return string
