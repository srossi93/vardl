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

ex = Experiment('convnet')

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

ex.add_config('convnet-exp-config.yaml')


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


    logdir = './work/%s/%s' % (dataset, init_strategy)


    X_train, Y_train = torch.load('%s/%s/train_%s.pt' % (dataset_dir, dataset, dataset))
    X_test, Y_test = torch.load('%s/%s/test_%s.pt' % (dataset_dir, dataset, dataset))

    if dataset == 'mnist':
        X_train = X_train.view(-1, 1, 28, 28)
        X_test = X_test.view(-1, 1, 28, 28)


    vardl.utils.set_seed(random_seed + fold)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    print('INFO - Dataset: %s' % dataset)
    print('INFO - Train size:', X_train.size(0))
    print('INFO - Test size: ', X_test.size(0))


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    layer_config = {'local_reparameterization': local_reparameterization,
                    'bias': bias,
                    'approx': approx,
                    'nmc_test': nmc_test,
                    'nmc_train': nmc_train,
                    'device': torch.device(device)}


    arch = vardl.architectures.build_lenet_mnist(*X_train.size()[1:], Y_train.size(1), **layer_config)

    model = vardl.models.ClassBayesianNet(architecure=arch)


    tb_logger = vardl.logger.TensorboardLogger(logdir, model)
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

    elif init_strategy == 'lsuv':
        init_dataloader = DataLoader(train_dataset,
                                      batch_size=4,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=0)
        initializer = vardl.initializer.LSUVInitializer(model=model,
                                                        tollerance=0.01,
                                                        max_iter=1000,
                                                        device=device,
                                                        train_dataloader=init_dataloader)

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

    torch.backends.cudnn.benchmark = True

    trainer.fit(iterations=iterations,
                test_interval=test_interval,
                train_verbose=False,
                train_log_interval=100)
