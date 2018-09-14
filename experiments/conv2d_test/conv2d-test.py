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


from sacred import Experiment
from sacred.observers import FileStorageObserver

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

import torch.nn as nn

ex = Experiment('conv2d-test')

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

ex.add_config('conv2d-test-config.yaml')


class View(nn.Module):
    def __init__(self, *size):
        super(View, self, ).__init__()
        self.size = size

    def forward(self, x):
        x = x.contiguous().view(self.size)
        # print('view-output:', x.size())
        return x

@ex.automain
def run_experiment(batch_size, iterations, lr, bias, approx, local_reparameterization, nmc_train,
                   nmc_test, random_seed, train_test_ratio, test_interval, dataset, init_strategy,
                   fold, device, dataset_dir):

    dataset = 'mnist'

    logdir = './work/%s/%s' % (dataset, init_strategy)

    X, Y = torch.load('%s/%s/complete_%s.pt' % (dataset_dir, dataset, dataset))
    X = X.view(-1, 1, 28, 28)

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

    conv1 = vardl.layers.BayesianConv2d(in_channels=1,
                                        in_height=28,
                                        in_width=28,
                                        out_channels=10,
                                        kernel_size=5,
                                        **layer_config)

    pool = nn.MaxPool2d(2)

    conv2 = vardl.layers.BayesianConv2d(in_channels=10,
                                        in_height=13,
                                        in_width=13,
                                        out_channels=20,
                                        kernel_size=5,
                                        **layer_config)

    layer1 = vardl.layers.BayesianLinear(in_features=500,
                                         out_features=50,
                                         **layer_config)

    layer2 = vardl.layers.BayesianLinear(in_features=50,
                                         out_features=10,
                                         **layer_config)

    arch = nn.Sequential(
        conv1,
        View(-1, 10, 26, 26),
        pool,
        View(layer_config['nmc_train'], -1, 10, 13, 13),
        nn.ReLU(),
        conv2,
        View(-1, 20, 11, 11),
        pool,
        nn.ReLU(),
        View(layer_config['nmc_train'], -1, 500),
        layer1,
        nn.ReLU(),
        layer2
    )

    model = vardl.models.ClassBayesianNet(architecure=arch)


    tb_logger = vardl.logger.TensorboardLogger(logdir)
    trainer = vardl.trainer.TrainerClassifier(model=model,
                                             train_dataloader=train_dataloader,
                                             test_dataloader=test_dataloader,
                                             optimizer='Adam',
                                             optimizer_config={'lr': float(lr)},
                                             device=device,
                                             logger=tb_logger,
                                             seed=random_seed)


    if init_strategy == 'uninformative':
        initializer = vardl.initializer.UninformativeInitializer(model=model)

    elif init_strategy == 'heuristic':
        initializer = vardl.initializer.HeuristicInitializer(model=model)

    elif init_strategy == 'xavier-normal':
        initializer = vardl.initializer.XavierNormalInitializer(model=model)

    elif init_strategy == 'orthogonal':
        initializer = vardl.initializer.OrthogonalInitializer(model=model)

    #elif init_strategy == 'lsuv':
    #    continue
    #    init_dataloader = DataLoader(train_dataset,
    #                                  batch_size=4,
    #                                  shuffle=True,
    #                                  drop_last=True,
    #                                  num_workers=0)
    #    initializer = vardl.initializer.LSUVInitializer(model=model,
    #                                                    tollerance=0.1,
    #                                                    max_iter=100,
    #                                                    device=device,
    #                                                    train_dataloader=init_dataloader)
    #
    #elif init_strategy == 'blm':
    #    continue
    #    init_dataloader = DataLoader(train_dataset,
    #                                  batch_size=256,
    #                                  shuffle=True,
    #                                  drop_last=True,
    #                                  num_workers=0)
    #    initializer = vardl.initializer.BLMInitializer(model=model,
    #                                                   train_dataloader=init_dataloader,
    #                                                   device=device,
    #                                                   lognoise=0)

    else:
        raise ValueError()

    initializer.initialize()

    trainer.fit(iterations=iterations,
                test_interval=test_interval,
                train_verbose=False,
                train_log_interval=100)