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
from . import MultivariateBernoulliDistribution


def dkl_matrix_gaussian(q: MatrixGaussianDistribution,
                        p: MatrixGaussianDistribution) -> torch.Tensor:
    """
    Computes the KL divergence between two Matrix Gaussian Distributions

    Parameters
    ----------
    q : MultivariateGaussianDistribution
        Approximate distribution

    p : MultivariateGaussianDistribution
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
    elif p.approx == 'factorized' and q.approx == 'low-rank':
        total_dkl = 0
        for i in range(q.m):
            total_dkl += _dkl_gaussian_q_lowrank_p_diag(q.mean[:, i],
                                                        q.cov_low_rank[i],
                                                        q.logvars[:, i],
                                                        p.mean[:, i],
                                                        p.logvars[:, i])
        return total_dkl
    else:
        raise NotImplementedError()


def _dkl_gaussian_q_diag_p_diag(mq: torch.Tensor,
                                log_vq: torch.nn.Parameter,
                                mp: torch.Tensor,
                                log_vp: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence when both p and q are fully factorized

    Parameters
    ----------
    mq : torch.Tensor
        means of the approximate Gaussian
    log_vq : torch.nn.Parameter
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

    kl = 0.5 * torch.sum(
        log_vp - log_vq
        + (torch.pow(mq - mp, 2) * torch.exp(-log_vp))
        + torch.exp(log_vq - log_vp) - 1.0
    )


    return kl


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
                  torch.sum(torch.exp(-log_vp) * torch.sum(torch.pow(lower_triang_q, 2), dim=1)) -
#                  torch.sum(torch.diag(torch.mul(torch.exp(-log_vp),
#                                                 torch.matmul(lower_triang_q, lower_triang_q.t())))) -
                  dimension)


def _dkl_gaussian_q_lowrank_p_diag(mu_q: torch.Tensor,
                                   cov_low_rank_q: torch.Tensor,
                                   logvars_q: torch.Tensor,
                                   mu_p: torch.Tensor,
                                   logvars_p: torch.Tensor) -> torch.Tensor:


    dimension = mu_q.size(0)
    rank = cov_low_rank_q.size(1)
    return 0.5 * (
            torch.sum(logvars_p)
            - torch.logdet(torch.matmul((-logvars_q).exp() * cov_low_rank_q.t(), cov_low_rank_q) +
                           torch.eye(rank, device=mu_q.device))
            #- torch.logdet(torch.matmul(cov_low_rank_q, cov_low_rank_q.t()) + torch.diagflat(logvars_q))
            + torch.sum(torch.mul(torch.pow(mu_q - mu_p, 2), torch.exp(-logvars_p)))
            + torch.sum(torch.exp(-logvars_p) * torch.sum(torch.pow(cov_low_rank_q, 2), dim=1))
            + torch.sum((logvars_p - logvars_q).exp())
            - dimension)


    return 0


def dkl_bernoulli(q: MultivariateBernoulliDistribution,
                  p: MultivariateBernoulliDistribution):

    return torch.sum(q.logp.exp() * torch.log(q.logp.exp() / p.logp.exp()) + (1 - q.logp.exp()) * torch.log((1-q.logp.exp()) / (1-p.logp.exp())))
