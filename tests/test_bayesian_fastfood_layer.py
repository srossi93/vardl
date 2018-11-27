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
import matplotlib.pyplot as plt
matplotlib.use('agg')

import vardl
logger = vardl.utils.setup_logger(__name__, '/tmp/', 'DEBUG')


class BayesianFastfoodLinear(vardl.layers.BaseBayesianLayer):

    def __init__(self, d_out):
        super(BayesianFastfoodLinear, self).__init__(device = 'cpu', nmc_train=1, nmc_test=1, dtype=torch.float32)

        self.S = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.G = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.B = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.P = torch.nn.Parameter(torch.tensor(np.random.permutation(d_out)), requires_grad=False)

    def forward(self, input):
        HBx = vardl.functional.HadamardTransform.apply(self.B * input.t_())
        PHBx = HBx[:, self.P]
        HGPHBx = vardl.functional.HadamardTransform.apply(self.G * PHBx)
        return self.S * HGPHBx

    def dkl(self):
        return 0


class SimpleNN(vardl.models.BaseBayesianNet):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.nmc = 1
        self.fastfood_layer = BayesianFastfoodLinear(64)
        self.fc = vardl.layers.BayesianLinear(64, 1, local_reparameterization=True)
        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)

    @property
    def dkl(self):
        total_dkl = 0
        for layer in self.modules():
            total_dkl += layer.dkl if issubclass(type(layer), vardl.layers.BaseBayesianLayer) else 0
        return total_dkl

    def forward(self, input):
        input = input * torch.ones(self.nmc, *input.size()).to(input.device)
        x = self.fastfood_layer(input)
        x = x * torch.ones(self.nmc, *x.size())
        x = torch.nn.functional.tanh(x)
        x = self.fc(x)
        return x

def function(x):
    return np.sin(x) + np.sin(x/2) + np.sin(x/3) - np.sin(x/4) + 0.2 * np.random.rand(*x.shape)

def main():
    model = SimpleNN()
    logger.debug(model)

    full_data = np.linspace(-10, 10, 1024)
    full_targets = function(full_data)


    plot = True
    if plot:
        fig, ax = plt.subplots()
        ax.plot(full_data, full_targets, 'o')
        ax.plot(full_data, model(torch.from_numpy(full_data).float()).detach().view(-1).numpy(), )
        fig.show()

    

    plt.show()
    return 1


if __name__ == '__main__':
    main()