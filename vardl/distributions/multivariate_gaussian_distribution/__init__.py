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

from .multivariate_gaussian_distribution import MultivariateGaussianDistribution

from .fully_factorized_multivariate_gaussian_distribution import FullyFactorizedMultivariateGaussian
from .full_covariance_multivariate_gaussian_distribution import FullCovarianceMultivariateGaussian

available_distributions = {
    'fully_factorized_multivariate_gaussian': FullyFactorizedMultivariateGaussian,
    'full_covariance_multivariate_gaussian': FullCovarianceMultivariateGaussian,
}
