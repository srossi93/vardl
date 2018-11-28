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
logger = vardl.utils.setup_logger(__name__, '/tmp/', 'DEBUG')


class BayesianFastfoodLinear(vardl.layers.BaseBayesianLayer):

    def __init__(self, d_out):
        super(BayesianFastfoodLinear, self).__init__(device = 'cpu', nmc_train=1, nmc_test=1, dtype=torch.float32)

        self.S = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.G = torch.nn.Parameter(torch.tensor(np.random.randn(d_out)).float())
        self.B = torch.nn.Parameter(torch.tensor(np.random.choice((-1, 1), size=d_out)).float(), requires_grad=False)
        self.P = torch.nn.Parameter(torch.tensor(np.random.permutation(d_out)), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(d_out))
    def forward(self, input):
        HBx = vardl.functional.HadamardTransform.apply(self.B * input)
        #logger.debug('HBx = %s' % str(HBx.shape))
        PHBx = HBx[..., self.P]
        #logger.debug('PHBx = %s' % str(PHBx.shape))
        HGPHBx = vardl.functional.HadamardTransform.apply(self.G * PHBx)
        return (self.S * HGPHBx) + self.bias

    @property
    def dkl(self):
        return 0


class SimpleNN(vardl.models.BaseBayesianNet):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.nmc = 128
        self.fastfood_layer = BayesianFastfoodLinear(64)
        self.fastfood_layer2 = BayesianFastfoodLinear(64)
        self.fastfood_layer3 = BayesianFastfoodLinear(64)
        self.fastfood_layer4 = BayesianFastfoodLinear(64)
        self.fc = vardl.layers.BayesianLinear(64, 1, local_reparameterization=True, nmc_test=self.nmc, bias=False)
        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)



    def forward(self, input):
        x = input * torch.ones(self.fc.nmc, *input.size()).to(input.device)
        x = self.fastfood_layer(x)
        #logger.debug(x.size())
        x = torch.nn.functional.tanh(x)
        x = self.fastfood_layer2(x)
        x = torch.nn.functional.tanh(x)
        x = self.fastfood_layer3(x)
        x = torch.nn.functional.tanh(x)
        x = self.fastfood_layer4(x)
        x = torch.nn.functional.tanh(x)
        #logger.debug(x.size())
        #x = torch.erf(x)
        x = self.fc(x)
        return x

def function(x):
    return np.sin(x) + np.sin(x/2) + np.sin(x/3) - np.sin(x/4) + 0. * np.random.rand(*x.shape)

def main():
    model = SimpleNN()
    #model.train()
    logger.debug(model)

    full_data = np.linspace(-10, 10, 256).reshape(-1, 1)
    gap_data = np.delete(full_data, range(0, 80)).reshape(-1, 1)
    #full_targets = function(full_data)




    dataloader = DataLoader(TensorDataset(torch.from_numpy(gap_data).float(),
                                          torch.from_numpy(function(gap_data)).float()),
                            batch_size=128,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

    tb_logger = vardl.logger.TensorboardLogger(path='/tmp/', model=model, directory='/tmp/')
    trainer = vardl.trainer.TrainerRegressor(model,
                                             optimizer='Adam',
                                             optimizer_config={'lr': 1e-2},
                                             train_dataloader=dataloader,
                                             test_dataloader=dataloader,
                                             device='cpu',
                                             seed=0,
                                             logger=tb_logger)
    model.likelihood.log_noise_var.requires_grad = False
    trainer.train_per_iterations(iterations=5000, train_verbose=True)
    trainer.test()

    plot = True
    if plot:
        fig, ax = plt.subplots()
        ax.plot(gap_data, function(gap_data), 'o', markersize=3)
        predicted_targets = model(torch.from_numpy(full_data).float())
        predicted_targets, predicted_std = model.likelihood.predict(predicted_targets)
        predicted_targets = predicted_targets.detach().numpy()[:, 0]
        predicted_std = predicted_std.detach().numpy()[:,0]

        l, = ax.plot(full_data, model(torch.from_numpy(full_data).float()).mean(0).detach().numpy())
        ax.fill_between(full_data[:, 0], predicted_targets - 3 * predicted_std, predicted_targets + 3 * predicted_std,
                        alpha=0.1, color=l.get_color())
        fig.show()


    plt.show()
    return 1


if __name__ == '__main__':
    main()