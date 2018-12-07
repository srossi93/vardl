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


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import torch
import numpy as np
import scipy.linalg
import time

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import humanize

import vardl
from vardl.tools.convert_tensorboard import read_tbevents

logger = vardl.utils.setup_logger('vardl', '/tmp/', 'INFO')


def function(x):
    return np.sin(x) + np.sin(x/2) + np.sin(x/3) - np.sin(x/4) + np.exp(-2) * np.random.rand(*x.shape)


class FastfoodNet(vardl.models.BaseBayesianNet):
    def __init__(self, nfearures: int = 64, activation_function= torch.tanh):
        super(FastfoodNet, self).__init__()
#        self.nmc = 128
        nmc_test = 1024
        self.fastfood_layer = vardl.layers.BayesianFastfoodLinear(1, nfearures, nmc_train=1, nmc_test=nmc_test)
        self.fastfood_layer2 = vardl.layers.BayesianFastfoodLinear(nfearures, nfearures, nmc_train=1, nmc_test=nmc_test)
        self.fc = vardl.layers.BayesianLinear(nfearures, 1, local_reparameterization=True, nmc_train=1, nmc_test=nmc_test,
                                               bias=False)
#        self.fc.prior_W._logvars.data.fill_(np.log(0.01))
        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)

        self.activation_function = activation_function
        self.name = 'BayesianFastfood'

    def forward(self, input):
        x = input * torch.ones(self.fastfood_layer.nmc, *input.size()).to(input.device)
        x = self.fastfood_layer(x)
        x = self.activation_function(x)
        x = self.fastfood_layer2(x)
        x = self.activation_function(x)
        self.basis_functions = x
        x = self.fc(x)
        return x


class BayesianNet(vardl.models.BaseBayesianNet):
    def __init__(self, nfeatures: int = 64, activation_function= torch.tanh):
        super(BayesianNet, self).__init__()
        #        self.nmc = 128
        nmc_test = 1024
        self.linear_layer = vardl.layers.BayesianLinear(1, nfeatures, nmc_train=1, nmc_test=nmc_test)
        self.linear_layer2 = vardl.layers.BayesianLinear(nfeatures, nfeatures, nmc_train=1, nmc_test=nmc_test)
        self.fc = vardl.layers.BayesianLinear(nfeatures, 1, nmc_train=1, nmc_test=nmc_test)
        #        self.fc.prior_W._logvars.data.fill_(np.log(0.01))
        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)

        self.activation_function = activation_function
        self.name = 'BayesianVanilla'

    def forward(self, input):
        x = input * torch.ones(self.linear_layer.nmc, *input.size()).to(input.device)
        x = self.linear_layer(x)
        x = self.activation_function(x)
        x = self.linear_layer2(x)
        x = self.activation_function(x)
        self.basis_functions = x
        x = self.fc(x)
        return x


class MonteCarloNet(torch.nn.Module):
    def __init__(self, nfeatures: int = 64, activation_function= torch.tanh):
        super(MonteCarloNet, self).__init__()
        self.nmc_train = 1
        self.nmc_test = 1024
        self.nfeatures = nfeatures
        self.linear_layer = torch.nn.Linear(1, nfeatures, bias=False)
        self.linear_layer2 = torch.nn.Linear(nfeatures, nfeatures, bias=False)
        self.fc = torch.nn.Linear(nfeatures, 1, bias=False)
        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)
        self.activation_function = activation_function

        self.name = 'MonteCarlo'
        self.dkl = 0

    def train(self, mode=True):
        self.nmc = self.nmc_train if mode else self.nmc_test

    @property
    def trainable_parameters(self):
        return humanize.intword(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, input):
        out = torch.zeros(self.nmc, input.size(0), 1).to(input.device)
        self.basis_functions = torch.zeros(self.nmc, input.size(0), self.nfeatures).to(input.device)
        for i in range(self.nmc):
            x = input
            x = self.linear_layer(x)
            x = torch.nn.functional.dropout(x, 0.5, training=True)
            x = self.activation_function(x)
            x = self.linear_layer2(x)
            x = torch.nn.functional.dropout(x, 0.5, training=True)
            x = self.activation_function(x)
            self.basis_functions[i] = x
            x = self.fc(x)

            out[i] = x

        return out


# ******************* MAIN


def main():
    plot = True
    vardl.utils.set_seed(122018)

    bayesian_fastfood_model = FastfoodNet()
    bayesian_linear_model = BayesianNet()
    montecarlo_model = MonteCarloNet()

    models = [bayesian_fastfood_model,bayesian_linear_model, montecarlo_model]

    for model in models:
        logger.info('Trainable parameters for %s model: %s' % (model.name,
                                                               model.trainable_parameters))

    full_data = np.linspace(-10, 10, 256).reshape(-1, 1)
    gap_data = np.delete(full_data, range(30, 80)).reshape(-1, 1)
    full_data = np.linspace(-12.5, 12.5, 1024).reshape(-1, 1)

    dataloader = DataLoader(TensorDataset(torch.from_numpy(gap_data).float(),
                                          torch.from_numpy(function(gap_data)).float()),
                            batch_size=1024,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

    tb_summary_paths = {}

    for model in models:  # type: torch.nn.Module
        logger.info('Training %s model' % model.name)
        tb_logger = vardl.logger.TensorboardLogger(model=model, directory='./workspace/' + model.name + '/')
        tb_summary_paths[model.name] = tb_logger.directory
        trainer = vardl.trainer.TrainerRegressor(model,
                                                 optimizer='Adam',
                                                 optimizer_config={'lr': 2.5e-3},
                                                 train_dataloader=dataloader,
                                                 test_dataloader=dataloader,
                                                 device='cpu',
                                                 seed=0,
                                                 tb_logger=tb_logger)

        try:
            model.likelihood.log_noise_var.requires_grad = False
            trainer.fit(iterations=25000, test_interval=500, train_verbose=False)
            model.likelihood.log_noise_var.requires_grad = True
            trainer.fit(iterations=15000, test_interval=500, train_verbose=False)
        except KeyboardInterrupt:
            logger.warning('User interruption! Stopping training...')
        trainer.test()

    if plot:
        for model in models:  # type: torch.nn.Module
            fig, ax = plt.subplots()
            ax.plot(gap_data, function(gap_data), 'o', markersize=3)
            predicted_targets = model(torch.from_numpy(full_data).float())

            vardl.utils.tsplot(full_data[..., 0], predicted_targets.detach().numpy()[..., 0], n=4, ax=ax,
                               label=model.name + ' (%s parameters)' % model.trainable_parameters)
            ax.set_ylim(-4.5, 4.5)
            ax.set_title('1D Regression - %s' % model.name)
            ax.legend()
            vardl.utils.ExperimentPlotter.savefig('figures/demo1d/1D-' + model.name, 'pdf')
            vardl.utils.ExperimentPlotter.savefig('figures/demo1d/1D-' + model.name, 'tex')

            fig, ax = plt.subplots()
            basis_functions = model.basis_functions.detach().numpy()
            ax.plot(full_data, basis_functions.mean(0), linewidth=.5)
            ax.set_ylim(-1.25, 1.25)
            ax.set_title('Basis functions - %s' % model.name)
            vardl.utils.ExperimentPlotter.savefig('figures/demo1d/basis_function-' + model.name, 'pdf')
            vardl.utils.ExperimentPlotter.savefig('figures/demo1d/basis_function-' + model.name, 'tex')

    fig, (ax0, ax1) = plt.subplots(2, 1)
    for model, tb_directory_path in tb_summary_paths.items():
        ea = read_tbevents(tb_directory_path)
        nell_test = np.array(ea.Scalars('nell/test'))
        ax0.plot(nell_test[:, 1]+1, nell_test[:, 2], label=model)
        ax0.set_title('Test MNLL')
        ax0.legend()

        error_test = np.array(ea.Scalars('error/test'))
        ax1.plot(error_test[:, 1]+1, error_test[:, 2], label=model)
        ax1.set_title('Test Error')
        ax1.legend()
    ax0.semilogx()
    ax1.semilogx()
    vardl.utils.ExperimentPlotter.savefig('figures/demo1d/test_curves', 'pdf')
    vardl.utils.ExperimentPlotter.savefig('figures/demo1d/test_curves', 'tex')

    fig, (ax0, ax1) = plt.subplots(2, 1)
    for model, tb_directory_path in tb_summary_paths.items():
        ea = read_tbevents(tb_directory_path)
        nell_train = np.array(ea.Scalars('nell/train'))
        ax0.plot(nell_train[:, 1], nell_train[:, 2], label=model)

        error_train = np.array(ea.Scalars('error/train'))
        ax1.plot(error_train[:, 1], error_train[:, 2], label=model)

    ax0.set_title('Train MNLL')
    ax1.set_title('Train Error')
    ax0.legend()
    ax1.legend()
    ax0.semilogx()
    ax1.semilogx()
    # TODO: savefig here
    vardl.utils.ExperimentPlotter.savefig('figures/demo1d/train_curves', 'pdf')
    vardl.utils.ExperimentPlotter.savefig('figures/demo1d/train_curves', 'tex')

    plt.show()
    return 1


if __name__ == '__main__':
    main()
