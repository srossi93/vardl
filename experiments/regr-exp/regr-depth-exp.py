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

import sys
sys.path.append('../../')


import torch
import vardl

from sklearn.model_selection import train_test_split

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split

import torch.nn as nn

ex = Experiment('regr-depth-exp')

ex.observers.append(FileStorageObserver.create('work'))


@ex.config
def config():
    batch_size = 64
    iterations = int(100e3)
    lr = 1e-3
    bias = False
    approx = 'factorized'
    local_reparameterization = True
    nmc_train = 1000
    nmc_test = 1000
    random_seed = 0
    train_test_ratio = 0.90
    test_interval = 10
    dataset = 'powerplant'
    init_strategy = 'xavier'
    fold = 0
    device = 'cpu'
    dataset_dir = '~'

ex.add_config('regr-depth-exp-config.yaml')

NOISE_DICT = {'powerplant': -2,
              'concrete': -16,
              'boston': -4,
              'protein': -2,
              'spam': 0}


@ex.automain
def run_experiment(batch_size, iterations, lr, bias, approx, local_reparameterization, nmc_train,
                   nmc_test, random_seed, train_test_ratio, test_interval, dataset, init_strategy,
                   fold, device, dataset_dir):

    logdir = './work/%s/%s' % (dataset, init_strategy)

    X, Y = torch.load('%s/%s/complete_%s.pt' % (dataset_dir, dataset, dataset))

    vardl.utils.set_seed(random_seed + fold)

    complete_dataset = TensorDataset(X, Y)
    size = len(X)
    train_size = int(train_test_ratio * size)
    test_size = size - train_size


    train_dataset, test_dataset = random_split(complete_dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)

    layer_config = {'local_reparameterization': local_reparameterization,
                    'bias': bias,
                    'approx': approx,
                    'nmc_test': nmc_test,
                    'nmc_train': nmc_train,
                    'device': torch.device(device)}

    hidden_units = 100
    arch = nn.Sequential(
        vardl.layers.BayesianLinear(in_features=X.size(1), out_features=hidden_units,
                                    **layer_config),
        nn.ReLU(),
        vardl.layers.BayesianLinear(in_features=hidden_units, out_features=hidden_units,
                                    **layer_config),
        nn.ReLU(),
        vardl.layers.BayesianLinear(in_features=hidden_units, out_features=Y.size(1),
                                    **layer_config),
    )

    model = vardl.models.RegrBayesianNet(architecure=arch)


    tb_logger = vardl.logger.TensorboardLogger(logdir)
    trainer = vardl.trainer.TrainerRegressor(model=model,
                                             train_dataloader=train_dataloader,
                                             test_dataloader=test_dataloader,
                                             optimizer='Adam',
                                             optimizer_config={'lr': float(lr)},
                                             device=device,
                                             tb_logger=tb_logger,
                                             seed=random_seed)

    #initializer = vardl.initializer.OrthogonalInitializer(model)
    #initializer = vardl.initializer.LSUVInitializer(model, train_dataloader=train_dataloader,
    #                                                max_iter=100, tollerance=0.01)
    #initializer = vardl.initializer.BLMInitializer(model=model,
    #                                                   train_dataloader=train_dataloader,
    #                                                   device=torch.device(device),
    #                                                   lognoise=0)

    #initializer.initialize()

    if init_strategy == 'uninformative':
        initializer = vardl.initializer.UninformativeInitializer(model=model)

    elif init_strategy == 'heuristic':
        initializer = vardl.initializer.HeuristicInitializer(model=model)

    elif init_strategy == 'xavier-normal':
        initializer = vardl.initializer.XavierNormalInitializer(model=model)

    elif init_strategy == 'orthogonal':
        initializer = vardl.initializer.OrthogonalInitializer(model=model)

    elif init_strategy == 'lsuv':
        initializer = vardl.initializer.LSUVInitializer(model=model,
                                                        tollerance=0.01,
                                                        max_iter=1000,
                                                        device=device,
                                                        train_dataloader=train_dataloader)

    elif init_strategy == 'blm':
        init_dataloader = DataLoader(train_dataset,
                                      batch_size=256,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=0)
        initializer = vardl.initializer.BLMInitializer(model=model,
                                                       train_dataloader=init_dataloader,
                                                       device=device,
                                                       lognoise=0)

    else:
        raise ValueError()

    initializer.initialize()

    model.likelihood.log_noise_var.requires_grad = False
    trainer.fit(iterations=iterations,
                test_interval=test_interval,
                train_verbose=False,
                train_log_interval=100)
