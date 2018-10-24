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
    """
    Computes the KL divergence between two Matrix Gaussian Distributions

    Parameters
    ----------
    q : MatrixGaussianDistribution
        Approximate distribution

    p : MatrixGaussianDistribution
        Target distribution

    Returns
    -------
    torch.Tensor
        If successful it returns the divergences between q and p

    Raises
    -------
    NotImplementedError
        If some combinations of approximations are not valid

    """
    if p.approx == 'factorized' and q.approx == 'factorized':
        return _dkl_gaussian_q_diag_p_diag(q.mean, q.logvars, p.mean, p.logvars)

    elif p.approx == 'factorized' and q.approx == 'full':
        #print('INFO - KL_d')
        total_dkl = 0
        for i in range(q.m):
            lower_triang_q = torch.tril(q.cov_lower_triangular[i], -1)
            lower_triang_q += torch.diagflat(torch.exp(q.logvars[:,i]))
            total_dkl += _dkl_gaussian_q_full_p_diag(q.mean[:, i], lower_triang_q, p.mean[:, i], p.logvars[:, i])
            del lower_triang_q
        return total_dkl
    else:
        raise NotImplementedError()


def _dkl_gaussian_q_diag_p_diag(mq: torch.Tensor,
                                log_vq: torch.Tensor,
                                mp: torch.Tensor,
                                log_vp: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence when both p and q are fully factorized

    Parameters
    ----------
    mq : torch.Tensor
        means of the approximate Gaussian
    log_vq : torch.Tensor
        log variances of the approximate Gaussian
    mp : torch.Tensor
        means of the target Gaussian
    log_vp : torch.Tensor
        log variances of the target Gaussian

    Returns
    -------
    torch.Tensor
        KL divergence KL(q||p)

    """
    return 0.5 * torch.sum(
        log_vp - log_vq + (torch.pow(mq - mp, 2) * torch.exp(-log_vp)) + torch.exp(log_vq - log_vp) - 1.0)


def _dkl_gaussian_q_full_p_diag(mq: torch.Tensor,
                                lower_triang_q: torch.Tensor,
                                mp: torch.Tensor,
                                log_vp: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence when p is factorized ad q is with full covariance matrix

    Parameters
    ----------
    mq : torch.Tensor
        means of the approximate Gaussian
    lower_triang_q: torch.Tensor
        lower_triangular decomposition of the approximate Gaussian covariance matrix
    mp : torch.Tensor
        means of the target Gaussian
    log_vp : torch.Tensor
        log variances of the target Gaussian

    Returns
    -------
    torch.Tensor
        KL divergence KL(q||p)

    """

    dimension = mq.size(0)

    return 0.5 * (torch.sum(log_vp) - 2.0 * torch.sum(torch.log(torch.diag(lower_triang_q))) +
                  torch.sum(torch.mul(torch.pow(mq - mp, 2),
                                      torch.exp(-log_vp))) +
                  torch.sum(torch.diag(torch.mul(torch.exp(-log_vp),
                                                 torch.matmul(lower_triang_q, lower_triang_q.t())))) -
                  dimension)
