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
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

import torch.nn as nn

ex = Experiment('prior')

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
    test_interval = 10
    dataset = 'cifar10'
    init_strategy = 'orthogonal'
    fold = 0
    device = 'cpu'
    dataset_dir = '~'
    variance_prior_linear = 0.1
    variance_prior_conv2d = 0.1

ex.add_config('experiment-config.yaml')



@ex.automain
def run_experiment(batch_size, iterations, lr, bias, approx, local_reparameterization, nmc_train,
                   nmc_test, random_seed, test_interval, dataset, init_strategy,
                   fold, device, dataset_dir, prior_variance_linear, prior_variance_conv2d):

    torch.set_num_threads(8)

    logdir = './work/%s/prior_var_lin_%s-prior_var_conv_%s' % (dataset, prior_variance_linear, prior_variance_conv2d)

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

    if dataset=='mnist':
        arch = vardl.architectures.build_lenet_mnist(*X_train.size()[1:], Y_train.size(1), **layer_config)
    elif dataset == 'cifar10':
        arch = vardl.architectures.build_lenet_cifar10(*X_train.size()[1:], Y_train.size(1), **layer_config)
    else:
        raise ValueError()

    model = vardl.models.ClassBayesianNet(architecure=arch)

    print(model)

    optimizer_config = dict(lr=float(lr), weight_decay=0)
    lr_decay_config = dict(gamma=0.0001, p=0)

    print('INFO - Setting prior variance of linear layers at %f' % prior_variance_linear)
    print('INFO - Setting prior variance of conv2d layers at %f' % prior_variance_conv2d)

    for child in model.modules():
        if issubclass(type(child), vardl.layers.BaseBayesianLayer):
            child.prior_W.logvars.requires_grad = False
        if isinstance(child, vardl.layers.BayesianLinear):
            child.prior_W.logvars.fill_(np.log(prior_variance_linear))
        if isinstance(child, vardl.layers.BayesianConv2d):
            child.prior_W.logvars.fill_(np.log(prior_variance_conv2d))

    tb_logger = vardl.logger.TensorboardLogger(logdir, model)
    trainer = vardl.trainer.TrainerClassifier(model=model,
                                              train_dataloader=train_dataloader,
                                              test_dataloader=test_dataloader,
                                              optimizer='Adam',
                                              optimizer_config=optimizer_config,
                                              lr_decay_config=lr_decay_config,
                                              device=device,
                                              logger=tb_logger,
                                              seed=random_seed)

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

    initializer.initialize()

    torch.backends.cudnn.benchmark = True

#    trainer.logger.save_model("_after_init")

    final_test_nell, final_test_error = trainer.fit(iterations=iterations,
                                                    test_interval=test_interval,
                                                    train_verbose=False,
                                                    train_log_interval=100)

    print('INFO - Prior var linear: %.4f' % prior_variance_linear)
    print('INFO - Prior var conv2d: %.4f' % prior_variance_conv2d)
    print('INFO - Final Test NLL: %.4f' % final_test_nell)
    print('INFO - Final Test ERROR RATE: %.4f' % final_test_error)
