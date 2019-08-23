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


import torch
import torch.nn
import numpy as np

from . import BaseVariationalLayer

from ..distributions import available_distributions, kl_divergence
from ..distributions import FullyFactorizedMultivariateGaussian
from ..distributions import FullyFactorizedMatrixGaussian


def check_type(arg, type):
    if isinstance(arg, type):
        return arg
    else:
        return type(arg)


def check_str_in_list(arg, list):
    arg = check_type(arg, str)
    if arg in list:
        return arg
    raise ValueError('%s is not a valid option. Choose between %s' % (arg, list))


class VariationalGaussianProcessRFF(BaseVariationalLayer):
    def __init__(self, in_features, out_features, number_rffs, kernel, is_kernel_ard, learn_Omega, learn_theta, add_mean):
        super(VariationalGaussianProcessRFF, self).__init__()
        self.in_features = check_type(in_features, int)
        self.out_features = check_type(out_features, int)
        self.number_rffs = check_type(number_rffs, int)
        self.kernel = check_str_in_list(kernel, ['rbf', 'arccosine'])
        self.is_kernel_ard = check_type(is_kernel_ard, bool)
        self.learn_Omega = check_str_in_list(learn_Omega, ['var_fixed', 'prior_fixed',])
        self.learn_theta = check_str_in_list(learn_theta, ['optim', 'var'])
        self.add_mean = check_type(add_mean, bool)

        # -- Setup theta (kernel parameters)
        if self.learn_theta == 'optim':
            self.log_theta_sigma2 = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)
            if self.is_kernel_ard:
                self.log_theta_lengthscale = torch.nn.Parameter(
                    torch.ones(self.in_features, 1) * 0.5 * np.log(self.in_features * 1.0) - np.log(2.0),
                    requires_grad=True)  # Initialize lengthscale to sqrt(D) / 2
            if not self.is_kernel_ard:
                self.log_theta_lengthscale = torch.nn.Parameter(
                    torch.tensor(0.5 * np.log(self.in_features * 1.0) - np.log(2.)),
                    requires_grad=True)  # Initialize lengthscale to sqrt(D) / 2

        if self.learn_theta == "var":
            self.prior_log_theta_sigma2 = FullyFactorizedMultivariateGaussian(1)
            self.prior_log_theta_sigma2.mean.data = torch.tensor(0.0)
            self.prior_log_theta_sigma2.logvars.data = torch.tensor(-2.)
            self.prior_log_theta_sigma2.optimize(False)

            self.posterior_log_theta_sigma2 = FullyFactorizedMultivariateGaussian(1)
            self.posterior_log_theta_sigma2.mean.data = torch.tensor(0.0)
            self.posterior_log_theta_sigma2.logvars.data = torch.tensor(-2.)
            self.posterior_log_theta_sigma2.optimize(True)

            if not self.is_kernel_ard:
                self.prior_log_theta_lenghtscale = FullyFactorizedMultivariateGaussian(1)
                self.prior_log_theta_lenghtscale.mean.data = torch.tensor(0.5 * np.log(self.in_features) - np.log(2.0))
                self.prior_log_theta_lenghtscale.logvars.data = torch.tensor(-2.)
                self.prior_log_theta_lenghtscale.optimize(False)

                self.posterior_log_theta_lenghtscale = FullyFactorizedMultivariateGaussian(1)
                self.posterior_log_theta_lenghtscale.mean.data = torch.tensor(0.5 * np.log(self.in_features) - np.log(2.0))
                self.posterior_log_theta_lenghtscale.logvars.data = torch.tensor(-2.)
                self.posterior_log_theta_lenghtscale.optimize(True)

            if self.is_kernel_ard:
                self.prior_log_theta_lenghtscale = FullyFactorizedMultivariateGaussian(self.in_features)
                self.prior_log_theta_lenghtscale.mean.data = torch.ones(self.in_features) * 0.5 * np.log(self.in_features) - np.log(2.0)
                self.prior_log_theta_lenghtscale.logvars.data.fill_(-2)
                self.prior_log_theta_lenghtscale.optimize(False)

                self.posterior_log_theta_lenghtscale = FullyFactorizedMultivariateGaussian(self.in_features)
                self.posterior_log_theta_lenghtscale.mean.data = torch.ones(self.in_features) * 0.5 * np.log(self.in_features) - np.log(2.0)
                self.posterior_log_theta_lenghtscale.logvars.data.fill_(-2)
                self.posterior_log_theta_lenghtscale.optimize(True)


        # -- Setup omega (RFF parameters)
        self.prior_Omega = FullyFactorizedMatrixGaussian(self.in_features, self.number_rffs, fixed_randomness=True)
        self.prior_Omega.logvars.fill_(np.log(self.in_features) - np.log(2.0))
        self.prior_Omega.optimize(False)

        if self.learn_Omega == "var_fixed":
            self.posterior_Omega = FullyFactorizedMatrixGaussian(self.in_features, self.number_rffs,
                                                                 fixed_randomness=True)
            self.posterior_Omega.logvars.fill_(np.log(self.in_features) - np.log(2.0))
            self.posterior_Omega.optimize(True)

        # -- Setup weights
        dimension_correction_factor = 1
        dimension_correction_add = 0
        if self.kernel == "rbf":
            dimension_correction_factor = 2
        if self.add_mean:
            dimension_correction_add = self.in_features
        self.hidden_dimension = self.number_rffs * dimension_correction_factor + dimension_correction_add

        self.prior_weights = FullyFactorizedMatrixGaussian(self.hidden_dimension, self.out_features)
        self.prior_weights.optimize(False)
        self.posterior_weights = FullyFactorizedMatrixGaussian(self.hidden_dimension, self.out_features)
        self.posterior_weights.optimize(True)

    def kl_divergence(self):
        kl = kl_divergence(self.posterior_weights, self.prior_weights)
        if self.learn_theta == 'var':
            kl += kl_divergence(self.posterior_log_theta_lenghtscale, self.prior_log_theta_lenghtscale)
            kl += kl_divergence(self.posterior_log_theta_sigma2, self.prior_log_theta_sigma2)
        if self.learn_Omega == 'var_fixed':
            kl += kl_divergence(self.posterior_Omega, self.prior_Omega)
        return kl

    def forward(self, input: torch.Tensor):
        if input.dim() != 3:
            raise RuntimeError('Input shape has to be 3D (Monte Carlo samples x batch size x in_features) but is %dD' %
                               input.dim())
        if input.shape[-1] != self.in_features:
            raise RuntimeError('Missmatch in last dimension. Expected %d (in_features) but got %d' %
                               (self.in_features, input.shape[-1]))
        nmc = input.shape[0]

        if self.learn_theta == 'var':
            log_theta_sigma2_sample = self.posterior_log_theta_sigma2.sample(nmc).unsqueeze(-1)
            log_theta_lenghtscale_sample = self.posterior_log_theta_lenghtscale.sample(nmc).unsqueeze(-1)
        if self.learn_theta == 'optim':
            log_theta_sigma2_sample = self.log_theta_sigma2 * torch.ones(nmc, 1, 1)
            log_theta_lenghtscale_sample = self.log_theta_lengthscale * torch.ones(nmc, 1, 1)

        if self.learn_Omega == 'prior_fixed':
            Omega_sample = self.prior_Omega.sample(nmc)
        if self.learn_Omega == 'var_fixed':
            Omega_sample = self.posterior_Omega.sample(nmc)

        Phi_preactivation = torch.matmul(input/log_theta_lenghtscale_sample.exp(), Omega_sample)
        if self.kernel == 'rbf':
            Phi_postactivation = torch.cat((Phi_preactivation.sin(), Phi_preactivation.cos()), 2) * torch.sqrt(
                log_theta_sigma2_sample.exp() / self.number_rffs)

        if self.kernel == "arccosine":
            Phi_postactivation = (Phi_preactivation).clamp(0) * torch.sqrt(
                    2.0 * log_theta_sigma2_sample.exp() / self.number_rffs)

        if self.add_mean:
            Phi_postactivation = torch.cat((input, Phi_postactivation), 2)

        output = self.posterior_weights.sample_local_reparam_linear(nmc, Phi_postactivation)

        return output
