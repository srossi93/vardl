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

from .base_distribution import BaseDistribution

from .matrix_gaussian_distribution import FullCovarianceMatrixGaussian
from .matrix_gaussian_distribution import FullyFactorizedMatrixGaussian
from .matrix_gaussian_distribution import LowRankCovarianceMatrixGaussian
from .matrix_gaussian_distribution import available_distributions as available_distributions_matrix_gaussian

from .multivariate_gaussian_distribution import FullyFactorizedMultivariateGaussian
from .multivariate_gaussian_distribution import FullCovarianceMultivariateGaussian
from .multivariate_gaussian_distribution import available_distributions as available_distributions_multivariate_gaussian

from .normalizing_flows import PlanarFlow
from .normalizing_flows import available_flows

available_distributions = dict()
available_distributions.update(available_distributions_matrix_gaussian)
available_distributions.update(available_distributions_multivariate_gaussian)
available_distributions.update(available_flows)

from .kl_divergence import kl_divergence
