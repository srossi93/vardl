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

def hadamard_transform_cpu(u, normalize=False):
    """
    Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.

    Parameters
    ----------
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).

    Returns
    -------
        product: Tensor of shape (..., n)
    """

    batch_size, n = u.shape
    m = int(np.log2(n))
    if n != 1 << m:
        raise ValueError('d must be a power of 2')
    x = u[..., np.newaxis]
    for _ in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)


class HadamardTransform(torch.autograd.Function):
    '''
        The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))
    '''
    @staticmethod
    def forward(ctx, u):
        return hadamard_transform_cpu(u)

    @staticmethod
    def backward(ctx, grad):
        return HadamardTransform.apply(grad)