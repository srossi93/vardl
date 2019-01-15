#  Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from collections import OrderedDict
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import timeit
import vardl
import humanize
import json


class FastfoodNet(vardl.models.BaseBayesianNet):
    def __init__(self, input_dim, output_dim, nmc_train, nmc_test, nlayers, nfeatures, activation_function):
        """

        Parameters
        ----------
        input_dim: int
        output_dim: int
        nmc_train: int
        nmc_test: int
        nlayers: int
        nfeatures: int
        activation_function
        """
        super(FastfoodNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.nlayers = nlayers
        self.nfearures = nfeatures
        self.activation_function = activation_function

        self.layers = torch.nn.ModuleList()
        if nlayers == 0:
            self.layers.add_module('fc', vardl.layers.BayesianLinear(input_dim, output_dim, bias=True,
                                                                   nmc_train=nmc_train, nmc_test=nmc_test))
        else:
            for i in range(nlayers):
                if i == 0:
                    # First layer
                    name = 'vff0'
                    layer = vardl.layers.BayesianFastfoodLinear(input_dim, nfeatures, bias=True,
                                                                nmc_train=nmc_train,
                                                                nmc_test=nmc_test)

                elif i == nlayers - 1:
                    # Last layer
                    name = 'fc'
                    layer = vardl.layers.BayesianLinear(nfeatures, output_dim, bias=True, nmc_train=nmc_train,
                                                        nmc_test=nmc_test)
                    #layer = torch.nn.Linear(nfeatures, output_dim)
                else:
                    # Everything else in the middle
                    name = 'vff%d' % i
                    layer = vardl.layers.BayesianFastfoodLinear(nfeatures, nfeatures, bias=True,
                                                                nmc_train=nmc_train,
                                                                nmc_test=nmc_test)

                self.layers.add_module(name, layer)

        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)

        self.activation_function = activation_function
        self.basis_functions = None
        self.name = 'BayesianFastfood'
        self.train()

    def forward(self, input):
        x = input * torch.ones(list(self.layers)[0].nmc, *input.size()).to(input.device)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_function(x)
            else:
                self.basis_functions = x
        return x


class BayesianVanillaNet(vardl.models.BaseBayesianNet):
    def __init__(self, input_dim, output_dim, nmc_train, nmc_test, nlayers, nfearures, activation_function):
        """

        Parameters
        ----------
        input_dim: int
        output_dim: int
        nmc_train: int
        nmc_test: int
        nlayers: int
        nfearures: int
        activation_function
        """
        super(BayesianVanillaNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.nlayers = nlayers
        self.nfearures = nfearures
        self.activation_function = activation_function

        self.layers = torch.nn.ModuleList()

        if nlayers == 0:
            self.layers.add_module('fc', vardl.layers.BayesianLinear(input_dim, output_dim, bias=True,
                                                                     nmc_train=nmc_train, nmc_test=nmc_test))
        for i in range(nlayers):
            if i == 0:
               # First layer
                name = 'bll0'
                layer = vardl.layers.BayesianLinear(input_dim, nfearures, bias=True, nmc_train=nmc_train,
                                                           nmc_test=nmc_test)
            elif i == nlayers - 1:
               # Last layer
                name = 'fc'
                layer = vardl.layers.BayesianLinear(nfearures, output_dim, bias=True, nmc_train=nmc_train,
                                                   nmc_test=nmc_test)
            else:
               # Everything else in the middle
                name = 'bll%d' % i
                layer = vardl.layers.BayesianLinear(nfearures, nfearures, bias=True, nmc_train=nmc_train,
                                                           nmc_test=nmc_test)

            self.layers.add_module(name, layer)

        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)

        self.activation_function = activation_function
        self.basis_functions = None
        self.name = 'BayesianVanilla'
        self.train()

    def forward(self, input):
        x = input * torch.ones(list(self.layers)[0].nmc, *input.size()).to(input.device)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_function(x)
            else:
                self.basis_functions = x
        return x


class MCDropoutNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, nmc_train, nmc_test, nlayers, nfearures, activation_function):
        """

        Parameters
        ----------
        input_dim: int
        output_dim: int
        nmc_train: int
        nmc_test: int
        nlayers: int
        nfearures: int
        activation_function
        """
        super(MCDropoutNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.nlayers = nlayers
        self.nfearures = nfearures
        self.activation_function = activation_function

        self.layers = torch.nn.ModuleList()
        print(nlayers)
        if nlayers == 0:
            self.layers.add_module('fc', torch.nn.Linear(input_dim, output_dim, bias=True))

        for i in range(nlayers):
            if i == 0:
               # First layer
                name = 'bll0'
                layer = torch.nn.Linear(input_dim, nfearures, bias=True)
            elif i == nlayers - 1:
               # Last layer
                name = 'fc'
                layer = torch.nn.Linear(nfearures, output_dim, bias=True)
            else:
               # Everything else in the middle
                name = 'bll%d' % i
                layer = torch.nn.Linear(nfearures, nfearures, bias=True)

            self.layers.add_module(name, layer)

        self.likelihood = vardl.likelihoods.Gaussian(dtype=torch.float32)

        self.activation_function = activation_function
        self.basis_functions = None
        self.name = 'BayesianVanilla'
        self.train()

    def train(self, mode=True):
        self.nmc = self.nmc_train if mode else self.nmc_test

    @property
    def dkl(self):
        return torch.tensor(0.)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input):
        out = torch.zeros(self.nmc, input.size(0), 1).to(input.device)

        for j in range(self.nmc):
            x = input
            for i, layer in enumerate(self.layers):
                x = layer(x)

                if i != len(self.layers) - 1:
                    x = torch.nn.functional.dropout(x, 0.5, training=True)
                    x = self.activation_function(x)
                else:
                    self.basis_functions = x
            out[j] = x
        return out

activation_functions = {'tanh': torch.tanh,
                        'relu': torch.nn.functional.relu,
                        'erf': torch.erf}

models = {'fastfood': FastfoodNet,
          'bnn': BayesianVanillaNet,
          'mcd': MCDropoutNet}

def parse_args():

    available_models = models.keys()
    available_activation_functions = ['tanh', 'erf', 'relu']
    available_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'powerplant', 'protein']


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/data/',
                        help='Dataset directory')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Train/test split ratio')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbosity of training steps')
    parser.add_argument('--nmc_train', type=int, default=1,
                        help='Number of Monte Carlo samples during training')
    parser.add_argument('--nmc_test', type=int, default=256,
                        help='Number of Monte Carlo samples during testing')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size during training')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--nfeatures', type=int, default=16,
                        help='Dimensionality of hidden layers',)
    parser.add_argument('--activation_function', choices=available_activation_functions, type=str, default='tanh',
                        help='Activation functions',)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for training', )
    parser.add_argument('--model', choices=available_models, type=str, required=True,
                        help='Type of Bayesian model')
    parser.add_argument('--outdir', type=str,
                        default='workspace/',
                        help='Output directory base path',)
    parser.add_argument('--seed', type=int, default=2018,
                        help='Random seed',)
    parser.add_argument('--iterations_fixed_noise', type=int, default=500000,
                        help='Training iteration without noise optimization')
    parser.add_argument('--iterations_free_noise', type=int, default=500000,
                        help='Training iteration with noise optimization')
    parser.add_argument('--test_interval', type=int, default=500,
                        help='Interval between testing')
    parser.add_argument('--time_budget', type=int, default=720,
                        help='Time budget in minutes')


    args = parser.parse_args()

    args.dataset_dir = os.path.abspath(args.dataset_dir)+'/'
    args.outdir = os.path.abspath(args.outdir)+'/'

    return args



def setup_dataset():
    dataset_path = (args.dataset_dir + args.dataset + '/pytorch/' + args.dataset + '.pth')
    logger.info('Loading dataset from %s' % dataset_path)
    dataset = TensorDataset(*torch.load(dataset_path))

    input_dim = dataset.tensors[0].size(1)
    output_dim = dataset.tensors[1].size(1)
    size = len(dataset)
    train_size = int(args.split_ratio * size)
    test_size = size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=8192,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

    return train_dataloader, test_dataloader, input_dim, output_dim




if __name__ == '__main__':
    args = parse_args()
    
    outdir = vardl.utils.next_path('%s/%s/%s/' % (args.outdir, args.dataset, args.model) + 'run-%04d/')

    if args.verbose:
        logger = vardl.utils.setup_logger('vardl', outdir, 'DEBUG')
    else:
        logger = vardl.utils.setup_logger('vardl', outdir)

    logger.info('Configuration:')
    for key, value in vars(args).items():
        logger.info('  %s = %s' % (key, value))

    # Save experiment configuration as yaml file in logdir
    with open(outdir + 'experiment_config.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)
    vardl.utils.set_seed(args.seed)

    train_dataloader, test_dataloader, input_dim, output_dim = setup_dataset()

    model = models[args.model](input_dim, output_dim, args.nmc_train, args.nmc_test, args.nlayers, args.nfeatures,
                               activation_functions[args.activation_function])

    logger.info("Trainable parameters: %d" % model.trainable_parameters)

    tb_logger = vardl.logger.TensorboardLogger(path=outdir, model=model, directory=None)

    trainer = vardl.trainer.TrainerRegressor(model, 'Adam', {'lr': args.lr}, train_dataloader, test_dataloader, 'cpu',
                                             args.seed, tb_logger)

    model.likelihood.log_noise_var.requires_grad = False
    trainer.fit(args.iterations_fixed_noise, args.test_interval, False, time_budget=args.time_budget//2)

    model.likelihood.log_noise_var.requires_grad = True
    trainer.fit(args.iterations_free_noise, args.test_interval, False, time_budget=args.time_budget//2)


    # Save results
    test_mnll, test_error = trainer.test()

    results = {}

    results['model'] = args.model
    results['dataset'] = args.dataset
    results['nlayers'] = args.nlayers
    results['nfeatures'] = args.nfeatures
    results['trainable_parameters'] = model.trainable_parameters
    results['test_mnll'] = float(test_mnll.item())
    results['test_error'] = float(test_error.item())
    results['total_iters'] = trainer.current_iteration

    with open(outdir + 'results.json', 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)
