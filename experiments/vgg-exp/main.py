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

layer_config = {'local_reparameterization': True,
                'bias': False,
                'approx': 'factorized',
                'nmc_test': 128,
                'nmc_train': 1,
                'device': 'cpu'}

arch = vardl.architectures.build_vgg16_cifar10(**layer_config)

def parse_args():


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/data/',
                        help='Dataset directory')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbosity of training steps')
    parser.add_argument('--nmc_train', type=int, default=1,
                        help='Number of Monte Carlo samples during training')
    parser.add_argument('--nmc_test', type=int, default=128,
                        help='Number of Monte Carlo samples during testing')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for training', )
    parser.add_argument('--model', choices=['iblm', 'mcd'], type=str, required=True,
                        help='Type of Bayesian model')
    parser.add_argument('--outdir', type=str,
                        default='workspace/',
                        help='Output directory base path',)
    parser.add_argument('--seed', type=int, default=2018,
                        help='Random seed',)
    parser.add_argument('--iterations', type=int, default=1000000,
                        help='Interval between testing')
    parser.add_argument('--test_interval', type=int, default=500,
                        help='Interval between testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Interval between testing')
    parser.add_argument('--time_budget', type=int, default=720,
                        help='Time budget in minutes')


    args = parser.parse_args()

    args.dataset_dir = os.path.abspath(args.dataset_dir)+'/'
    args.outdir = os.path.abspath(args.outdir)+'/'

    return args


def setup_dataset():
    dataset_path = (args.dataset_dir + 'cifar10' + '/pytorch/')
    logger.info('Loading dataset from %s' % dataset_path)
    train_dataset = TensorDataset(*torch.load(dataset_path + 'train_cifar10.pth'))
    test_dataset = TensorDataset(*torch.load(dataset_path + 'test_cifar10.pth'))


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=8192,
                                 shuffle=True,
                                 num_workers=2,
                                 pin_memory=True)

    return train_dataloader, test_dataloader




if __name__ == '__main__':
    args = parse_args()

    outdir = vardl.utils.next_path('%s/%s/%s/' % (args.outdir, 'cifar10', args.model) + 'run-%04d/')

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

    train_dataloader, test_dataloader = setup_dataset()


    layer_config = {'local_reparameterization': True,
                    'bias': False,
                    'approx': 'factorized',
                    'nmc_test': args.nmc_test,
                    'nmc_train': args.nmc_train,
                    'device': torch.device(args.device)}



    if args.model == 'mcd':
        model = vardl.mcd.VGG16MCD_CIFAR10(nmc_test=args.nmc_test)
    else:
        arch = vardl.architectures.build_vgg16_cifar10(**layer_config)
        model = vardl.models.ClassBayesianNet(architecure=(arch))

    tb_logger = vardl.logger.TensorboardLogger(path=outdir, model=model, directory=None)
    trainer = vardl.trainer.TrainerClassifier(model=model,
                                              train_dataloader=train_dataloader,
                                              test_dataloader=test_dataloader,
                                              #optimizer='SGD',
                                              #optimizer_config=dict(lr=float(lr), momentum=0.9),
                                              optimizer='Adam',
                                              optimizer_config={'lr': float(args.lr)},
                                              lr_decay_config=dict(gamma=0.0001, p=0),
                                              device=args.device,
                                              logger=tb_logger,
                                              seed=args.seed)


    torch.backends.cudnn.benchmark = True

    if model == 'iblm':
        init_dataloader = DataLoader(train_dataloader.dataset,
                                     batch_size=256,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=0)
        lognoise = 1

        initializer = vardl.initializer.BLMInitializer(model=model,
                                                       train_dataloader=init_dataloader,
                                                       device=args.device,
                                                       lognoise=lognoise)
        initializer.initialize()


    trainer.fit(iterations=args.iterations,
                test_interval=args.test_interval,
                train_verbose=False,
                train_log_interval=1000,
                time_budget=args.time_budget)

    test_mnll, test_error = trainer.test()

    results = {}

    results['model'] = args.model
    results['test_mnll'] = float(test_mnll.item())
    results['test_error'] = float(test_error.item())
    results['total_iters'] = trainer.current_iteration

    with open(outdir + 'results.json', 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)
