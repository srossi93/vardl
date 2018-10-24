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

ex = Experiment('dropout')

ex.observers.append(FileStorageObserver.create('work'))


@ex.config
def config():
    batch_size = 64
    iterations = int(100e3)
    lr = 5e-4
    bias = False
    approx = 'factorized'
    local_reparameterization = True
    nmc_train = 1
    nmc_test = 32
    random_seed = 0
    test_interval = 10
    dataset = 'mnist'
    dropout = 'bernoulli'
    fold = 0
    device = 'cpu'
    dataset_dir = '~'

ex.add_config('experiment-config.yaml')

@ex.automain
def run_experiment(batch_size, iterations, lr, bias, approx, local_reparameterization, nmc_train,
                   nmc_test, random_seed, test_interval, dataset, dropout,
                   fold, device, dataset_dir):

    torch.set_num_threads(6)

    logdir = './work/%s/%s' % (dataset, dropout)


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


    if dropout == 'bernoulli':
        if dataset=='mnist':
            model = vardl.mcd.LeNetMNC_MNIST(nmc_test=nmc_test)
        elif dataset=='cifar10':
            model = vardl.mcd.LeNetMNC_CIFAR10(nmc_test=nmc_test)
        else:
            raise ValueError('Unknown dataset')
    elif dropout == 'beta':
        if dataset=='mnist':
            model = vardl.mcd.LeNetBeta_MNIST(nmc_test)
        elif dataset=='cifar10':
            model = vardl.mcd.LeNetBeta_CIFAR10(nmc_test)
        else:
            raise ValueError('Unknown dataset')
    else:
        raise ValueError('Unknown dropout')

    print(model)

    optimizer_config = dict(lr=float(lr), weight_decay=0.0005)
    lr_decay_config = dict(gamma=0.0001, p=0)


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

    #else:
    #    model.apply(weights_init)

    torch.backends.cudnn.benchmark = True

#    trainer.logger.save_model("_after_init")

    trainer.fit(iterations=iterations,
                test_interval=test_interval,
                train_verbose=False,
                train_log_interval=100)
