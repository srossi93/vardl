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
from . import MatrixGaussianDistribution


def dkl_matrix_gaussian(q: MatrixGaussianDistribution,
                        p: MatrixGaussianDistribution) -> torch.Tensor:
    if p.approx == 'factorized' and q.approx == 'factorized':
        return _DKL_gaussian_q_diag_p_diag(q.mean, q.logvars, p.mean, p.logvars)
    else:
        raise NotImplementedError()


def _DKL_gaussian_q_diag_p_diag(mq, log_vq, mp, log_vp):
    """
    KL[q || p]
    :param mq: vector of means for q
    :param log_vq: vector of log-variances for q
    :param mp: vector of means for p
    :param log_vp: vector of log-variances for p
    :return: KL divergence between q and p
    """
    # print(mq.is_cuda)
    # print(mp.is_cuda)
    # print(log_vq.is_cuda)
    # print(log_vp.is_cuda)

    return 0.5 * torch.sum(
        log_vp - log_vq + (torch.pow(mq - mp, 2) * torch.exp(-log_vp)) + torch.exp(log_vq - log_vp) - 1.0)
