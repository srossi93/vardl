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

from functools import total_ordering
import torch

from . import BaseDistribution
from . import FullyFactorizedMatrixGaussian, FullCovarianceMatrixGaussian
from . import FullyFactorizedMultivariateGaussian, FullCovarianceMultivariateGaussian
from . import PlanarFlow

import logging
logger = logging.getLogger(__name__)

_KL_REGISTRY = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_KL_MEMOIZE = {}  # Memoized version mapping many specific (type, type) pairs to functions.


def register_kl(type_q, type_p):
    """
    Decorator to register a pairwise function with kl_divergence.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(q, p):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(q, p): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(q, p): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_q (type): A subclass of `~vardl.distributions.BaseDistribution`.
        type_p (type): A subclass of `~vardl.distributions.BaseDistribution`.

    Note:
        Adapted from PyTorch 1.0
    """
    if not isinstance(type_q, type) and issubclass(type_q, BaseDistribution):
        raise TypeError('Expected type_q to be a Distribution subclass but got {}'.format(type_q))
    if not isinstance(type_p, type) and issubclass(type_p, BaseDistribution):
        raise TypeError('Expected type_p to be a Distribution subclass but got {}'.format(type_p))

    def decorator(fun):
        _KL_REGISTRY[type_q, type_p] = fun
        _KL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


@total_ordering
class _Match(object):
    __slots__ = ['types']

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_kl(type_q, type_p):
    """
    Find the most specific approximate match, assuming single inheritance.
    Note:
        Adapted from PyTorch 1.0
    """
    matches = [(super_q, super_p) for super_q, super_p in _KL_REGISTRY
               if issubclass(type_q, super_q) and issubclass(type_p, super_p)]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    left_q, left_p = min(_Match(*m) for m in matches).types
    right_p, right_q = min(_Match(*reversed(m)) for m in matches).types
    left_fun = _KL_REGISTRY[left_q, left_p]
    right_fun = _KL_REGISTRY[right_q, right_p]
    if left_fun is not right_fun:
        logger.warning('Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.format(
            type_q.__name__, type_p.__name__, left_q.__name__, right_p.__name__))
    return left_fun


def kl_divergence(q, p):
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| p)` between two distributions.

    .. math::

        KL(q \| p) = \int p(x) \log\frac {p(x)} {p(x)} \,dx

    Args:
        q (BaseDistribution):
        p (BaseDistribution):

    Returns:
        torch.Tensor: KL divergence

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    Note:
        Adapted from PyTorch 1.0
    """
    try:
        fun = _KL_MEMOIZE[type(q), type(p)]
    except KeyError:
        fun = _dispatch_kl(type(q), type(p))
        _KL_MEMOIZE[type(q), type(p)] = fun
    if fun is NotImplemented:
        raise NotImplementedError('KL divergence for pair %s - %s not registered' % (type(q).__name__,
                                                                                     type(p).__name__))
    return fun(q, p)


@register_kl(FullyFactorizedMatrixGaussian, FullyFactorizedMatrixGaussian)
def _kl_factorized_matrix_gaussian(q, p):
    """
    Args:
        q (FullyFactorizedMatrixGaussian):
        p (FullyFactorizedMatrixGaussian):
    """

    kl = 0.5 * torch.sum(
        p.logvars - q.logvars
        + (torch.pow(q.mean - p.mean, 2) * torch.exp(-p.logvars))
        + torch.exp(q.logvars - p.logvars) - 1.0
    )
    return kl


@register_kl(FullyFactorizedMultivariateGaussian, FullyFactorizedMultivariateGaussian)
def _kl_factorized_multivariate_gaussian(q, p):
    """
    Args:
        q (FullyFactorizedMultivariateGaussian):
        p (FullyFactorizedMultivariateGaussian):
    """

    kl = 0.5 * torch.sum(
        p.logvars - q.logvars
        + (torch.pow(q.mean - p.mean, 2) * torch.exp(-p.logvars))
        + torch.exp(q.logvars - p.logvars) - 1.0
    )
    return kl


@register_kl(FullCovarianceMultivariateGaussian, FullyFactorizedMultivariateGaussian)
def _kl_fcmvg_ffmvg(q, p):
    """
    Computes the KL divergence between a full covariance multivariate gaussian (FCMVG as q) and a fully factorized
    multivariate Gaussian (FFMVG as p)
    Args:
        q (FullCovarianceMultivariateGaussian):
        p:

    Returns:

    """

    lower_triang_q = torch.tril(q.cov_lower_triangular, -1)
    lower_triang_q += torch.diagflat(torch.exp(q.logvars))

    kl = 0.5 * (torch.sum(p.logvars) - 2.0 * torch.sum(torch.log(torch.diag(lower_triang_q))) +
                torch.sum(torch.mul(torch.pow(q.mean - p.mean, 2), (-p.logvars).exp())) +
                torch.sum(torch.diag((-p.logvars).exp() * torch.matmul(lower_triang_q, lower_triang_q.t()))) -
                q.n)

    return kl


@register_kl(FullCovarianceMatrixGaussian, FullyFactorizedMatrixGaussian)
def _kl_fcmg_ffmg(q, p):
    def _single__dkl_gaussian_q_full_p_diag(mq, lower_triang_q, mp, log_vp):
        dimension = mq.size(0)

        return 0.5 * (torch.sum(log_vp) - 2.0 * torch.sum(torch.log(torch.diag(lower_triang_q))) +
                      torch.sum(torch.mul(torch.pow(mq - mp, 2), torch.exp(-log_vp))) +
                      torch.sum(torch.exp(-log_vp) * torch.sum(torch.pow(lower_triang_q, 2), dim=1)) - dimension)

    kl = 0.
    for i in range(q.m):
        lower_triang_q = torch.tril(q.cov_lower_triangular[i], -1)
        lower_triang_q += torch.diagflat(torch.exp(q.logvars[:, i]))
        kl += _single__dkl_gaussian_q_full_p_diag(q.mean[:, i], lower_triang_q, p.mean[:, i], p.logvars[:, i])
        del lower_triang_q

    return kl


@register_kl(PlanarFlow, FullyFactorizedMultivariateGaussian)
def _kl_nf_ffg(q: PlanarFlow, p: FullyFactorizedMultivariateGaussian):

    kl = kl_divergence(q.initial_distribution, p)
    for f in q.flows:
        kl += f.log_det_jacobian.mean(0).item()
        # f.log_det_jacobian = None
    # sum_log_det_jacobian = sum(f.log_det_jacobian for f in q.flows).mean(0)
    # kl += sum_log_det_jacobian

    return kl
