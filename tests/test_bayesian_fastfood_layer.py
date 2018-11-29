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

import vardl
logger = vardl.utils.setup_logger('vardl', '/tmp/', 'DEBUG')


class BayesianFastfoodLinear(vardl.layers.BaseBayesianLayer):

    def __init__(self, d_out):
        super(BayesianFastfoodLinear, self).__init__(device = 'cpu', nmc_train=1, nmc_test=1, dtype=torch.float32)
        self.d_out = d_out
        #self.S = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.S = vardl.distributions.MultivariateGaussianDistribution(d_out)
        self.S.optimize()
        self.G = vardl.distributions.MultivariateGaussianDistribution(d_out)
        self.G.optimize()
        #self.G = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.B = torch.nn.Parameter(torch.tensor(np.random.choice((-1, 1), size=d_out)).float(), requires_grad=False)
        self.P = torch.nn.Parameter(torch.tensor(np.random.permutation(d_out)), requires_grad=False)

        self.bias = torch.nn.Parameter(torch.randn(d_out))
        #self.bias = vardl.distributions.MultivariateGaussianDistribution(d_out)
        #self.bias.optimize = True

    def forward(self, input):
        #self.P.data = torch.tensor(np.random.permutation(self.d_out))
        #self.B.data = torch.tensor(np.random.choice((-1, 1), size=self.d_out)).float()
        G = self.G.sample(self.nmc)
        S = self.S.sample(self.nmc)

        HBx = vardl.functional.HadamardTransform.apply(self.B * input)
        #tb_logger.debug('HBx = %s' % str(HBx.shape))
        PHBx = HBx[..., self.P]
        #tb_logger.debug('PHBx = %s' % str(PHBx.shape))
        HGPHBx = vardl.functional.HadamardTransform.apply(G * PHBx)
        return (S * HGPHBx) + self.bias#.sample(self.nmc)

    @property
    def dkl(self):
        return 0


class SimpleNN(vardl.models.BaseBayesianNet):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.nmc = 256
        self.fastfood_layer = BayesianFastfoodLinear(64)
        self.fastfood_layer2 = BayesianFastfoodLinear(64)
        #self.fastfood_layer3 = BayesianFastfoodLinear(32)
        #self.fastfood_layer4 = BayesianFastfoodLinear(32)
        self.fc = vardl.layers.BayesianLinear(64, 1, local_reparameterization=True, nmc_test=self.nmc, bias=False)
        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)



    def forward(self, input):
        x = input * torch.ones(self.fc.nmc, *input.size()).to(input.device)
        x = self.fastfood_layer(x)
        x = torch.erf(x)
        x = self.fastfood_layer2(x)
        x = torch.erf(x)
        #x = self.fastfood_layer3(x)
        #x = torch.erf(x)
        #x = self.fastfood_layer4(x)
        #x = torch.erf(x)
        #x = torch.nn.functional.tanh(x)
        #tb_logger.debug(x.size())
        #x = torch.erf(x)
        x = self.fc(x)
        return x

def function(x):
    return np.sin(x) + np.sin(x/2) + np.sin(x/3) - np.sin(x/4) + np.exp(-2) * np.random.rand(*x.shape)


def tsplot(x, y, n=20, percentile_min=1, percentile_max=99, color='r', plot_mean=True, plot_median=False,
           line_color='k', **kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=0)

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 0.9 / n

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()

    if plot_mean:
        l, = ax.plot(x, np.mean(y, axis=0))

    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, edgecolor=None, color=l.get_color())



    return plt.gca()

def main():
    model = SimpleNN()
    #model.train()
    logger.debug(model)

    full_data = np.linspace(-10, 10, 256).reshape(-1, 1)
    gap_data = np.delete(full_data, range(30, 80)).reshape(-1, 1)
    full_data = np.linspace(-10, 10, 2048).reshape(-1, 1)
    #full_targets = function(full_data)




    dataloader = DataLoader(TensorDataset(torch.from_numpy(gap_data).float(),
                                          torch.from_numpy(function(gap_data)).float()),
                            batch_size=1024,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

    tb_logger = vardl.logger.TensorboardLogger(model=model, directory='./workspace/')
    trainer = vardl.trainer.TrainerRegressor(model,
                                             optimizer='Adam',
                                             optimizer_config={'lr': 1e-3},
                                             train_dataloader=dataloader,
                                             test_dataloader=dataloader,
                                             device='cpu',
                                             seed=0,
                                             tb_logger=tb_logger)
    model.likelihood.log_noise_var.requires_grad = False
    trainer.train_per_iterations(iterations=100000, train_verbose=True)
    trainer.test()

    plot = True
    if plot:
        fig, ax = plt.subplots()
        ax.plot(gap_data, function(gap_data), 'o', markersize=3)
        predicted_targets = model(torch.from_numpy(full_data).float())
        predicted_mean, predicted_std = model.likelihood.predict(predicted_targets)
        predicted_mean = predicted_mean.detach().numpy()[:, 0]
        predicted_std = predicted_std.detach().numpy()[:,0]

#        l, = ax.plot(full_data, predicted_mean)
#        for i in [25, 50, 95]:
#
#            percentile = np.percentile(predicted_targets.detach(), i, axis=0)[:,0]
#            ax.fill_between(full_data[:, 0],
#                            predicted_mean - percentile,
#                            predicted_mean + percentile,
#                        alpha=0.1, color=l.get_color())
#        fig.show()

        tsplot(full_data[...,0], predicted_targets.detach().numpy()[...,0], n=3)

    plt.show()
    return 1


if __name__ == '__main__':
    main()