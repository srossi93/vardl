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

ex.add_config('montecarlo-exp-config.yaml')


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


@ex.automain
def run_experiment(batch_size, iterations, lr, bias, approx, local_reparameterization, nmc_train,
                   nmc_test, random_seed, train_test_ratio, test_interval, dataset, init_strategy,
                   fold, device, dataset_dir):

    torch.set_num_threads(6)

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
                                  num_workers=1,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True)

    layer_config = {'local_reparameterization': local_reparameterization,
                    'bias': bias,
                    'approx': approx,
                    'nmc_test': nmc_test,
                    'nmc_train': nmc_train,
                    'device': torch.device(device)}


    if init_strategy == 'mcd':
        if dataset=='mnist':
            model = vardl.mcd.LeNetMNC_MNIST(nmc_test=nmc_test)
        elif dataset == 'cifar10':
            model = vardl.mcd.LeNetMNC_CIFAR10(nmc_test=nmc_test)
        else:
            raise ValueError()


    else:
        if dataset=='mnist':
            arch = vardl.architectures.build_lenet_mnist(*X_train.size()[1:], Y_train.size(1), **layer_config)
        elif dataset == 'cifar10':
            arch = vardl.architectures.build_lenet_cifar10(*X_train.size()[1:], Y_train.size(1), **layer_config)
        else:
            raise ValueError()

        model = vardl.models.ClassBayesianNet(architecure=arch)

    print(model)

    if init_strategy == 'mcd':
        optimizer_config = dict(lr=0.001, weight_decay=0.0005)
        lr_decay_config = dict(gamma=0.0001, p=0.75)
    elif init_strategy == 'blm':
        optimizer_config = dict(lr=float(lr),)
        lr_decay_config = dict(gamma=0.0001, p=0)
    else:
        raise ValueError()


    if dataset == 'mnist':
        prior_optimization = 1000
    elif dataset == 'cifar10':
        prior_optimization = 1000
    else:
        raise ValueError()


    tb_logger = vardl.logger.TensorboardLogger(logdir, model)
    trainer = vardl.trainer.TrainerClassifier(model=model,
                                              train_dataloader=train_dataloader,
                                              test_dataloader=test_dataloader,
                                              optimizer='Adam',
                                              optimizer_config=optimizer_config,
                                              lr_decay_config=lr_decay_config,
                                              device=device,
                                              logger=tb_logger,
                                              seed=random_seed,
                                              prior_update_interval=prior_optimization)

    print(trainer.optimizer)


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

        if dataset == 'mnist':
            lognoise = -0
        elif dataset == 'cifar10':
            lognoise = 3

        initializer = vardl.initializer.BLMInitializer(model=model,
                                                       train_dataloader=init_dataloader,
                                                       device=device,
                                                       lognoise=lognoise)


    if init_strategy != 'mcd':
        initializer.initialize()

    #else:
    #    model.apply(weights_init)

    torch.backends.cudnn.benchmark = True

    #if init_strategy == 'mcd':
    #    iterations *= 100
    trainer.tb_logger.save_model("_after_init")

    trainer.fit(iterations=iterations,
                test_interval=test_interval,
                train_verbose=False,
                train_log_interval=100)
