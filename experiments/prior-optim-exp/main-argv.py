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

from torch.utils.data import DataLoader, TensorDataset


import numpy as np

import argparse
import sys
import yaml
import os
import time
import humanize

def parse_config() -> argparse.Namespace:

    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", default='config.yaml')

    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}

    if args.conf_file:
        with open(args.conf_file, 'r') as fd:
            config = yaml.load(fd)
        defaults.update(dict(config))

    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
    )
    parser.set_defaults(**defaults)
    parser.add_argument('--name',                       help='Name of the experiment')#, default='experiment',)
    parser.add_argument('--outdir',                     help='Directory path for all output files')#, default='./work',)
    parser.add_argument('--seed',                       help='Random seed')#, default=1234,)
    parser.add_argument('--fold',                       help='Dataset fold')#, default=0,)
    parser.add_argument('--dataset',                    help='Dataset name',)# default='mnist',)
    parser.add_argument('--dataset_dir',                help='Dataset directory',)
    parser.add_argument('--batch_size',                 help='Training and testing batch size')#, default=64,)
    parser.add_argument('--svi',                        help='Initializer for Gaussian-SVI or MCD')#, default='blm-init')
    parser.add_argument('--bias',                       help='Layers with bias or not')#, default=False,)
    parser.add_argument('--approx',                     help='Type of approximation for Gaussian SVI')#,
    # default='factorized',)
    parser.add_argument('--local_reparameterization',   help='Local reparameterization during train/test')#,
    #                         default=True,)
    parser.add_argument('--nmc_train',                  help='Number of MonteCarlo samples in training')#, default=1,)
    parser.add_argument('--nmc_test',                   help='Number of MonteCarlo samples in testing')#, default=16,)
    parser.add_argument('--lr',                         help='Optimizer learning rate')#, default=0.0005,)
    parser.add_argument('--iterations',                 help='Total training iterations', type=int)#, default=1000,)
    parser.add_argument('--prior_update_interval',      help='Interval between prior updates', type=int)#,
    # default=1000,)
    parser.add_argument('--prior_update_conv2d_type',   help='How the prior is shared (layer, outchannels, '
                                                             'outchannels+inchannels', type=str)
    parser.add_argument('--device',                     help='Device backend')#, default='cpu',)
    parser.add_argument('--test_interval',              help='Test every given iterations', type=int)#, default=1000,)
    parser.add_argument('--test_verbose',               help='Verbosity during testing')#, default=True,)
    parser.add_argument('--train_verbose',              help='Verbosity during training')#, default=False,)

    args = parser.parse_args(remaining_argv)
    del args.conf_file
    return args



def run_experiment(name, outdir, seed, fold, dataset, dataset_dir, batch_size, svi, bias, approx,
                   local_reparameterization, nmc_train, nmc_test, lr, iterations, prior_update_interval,
                   prior_update_conv2d_type, device, test_interval, test_verbose, train_verbose, logger, logdir):

    torch.set_num_threads(8)

    try:
        X_train, Y_train = torch.load('%s/%s/train_%s.pt' % (dataset_dir, dataset, dataset))
        X_test, Y_test = torch.load('%s/%s/test_%s.pt' % (dataset_dir, dataset, dataset))
    except FileNotFoundError:
        logger.fatal('Dataset %s not found in %s' % (dataset, dataset_dir))
        sys.exit(-1)

    if dataset == 'mnist':
        X_train = X_train.view(-1, 1, 28, 28)
        X_test = X_test.view(-1, 1, 28, 28)

    vardl.utils.set_seed(seed + fold)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    logger.info('Dataset: %s' % dataset)
    logger.info('  Train size: %d' % X_train.size(0))
    logger.info('  Test size : %d' % X_test.size(0))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    layer_config = {'local_reparameterization': local_reparameterization,
                    'bias': bias,
                    'approx': approx,
                    'nmc_test': nmc_test,
                    'nmc_train': nmc_train,
                    'device': torch.device(device)}

    if svi == 'mcd':
        logger.info('Running experiment with MonteCarlo dropout variational inference')
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

    logger.debug(model)

    if svi == 'mcd':
        optimizer_config = dict(lr=float(lr), weight_decay=0.0005)
    else:
        optimizer_config = dict(lr=float(lr),)

    lr_decay_config = dict(gamma=0.0001, p=0)

    for child in model.modules():
        if issubclass(type(child), vardl.layers.BaseBayesianLayer):
            child.prior_W.logvars.requires_grad = False
        if isinstance(child, vardl.layers.BayesianLinear):
            child.prior_W.logvars.fill_(np.log(0.01))
        if isinstance(child, vardl.layers.BayesianConv2d):
            child.prior_W.logvars.fill_(np.log(0.01))

    prior_optimization = 1

    tb_logger = vardl.logger.TensorboardLogger(path=logdir, model=model, directory=logdir)
    trainer = vardl.trainer.TrainerClassifier(model=model,
                                              train_dataloader=train_dataloader,
                                              test_dataloader=test_dataloader,
                                              optimizer='Adam',
                                              optimizer_config=optimizer_config,
                                              lr_decay_config=lr_decay_config,
                                              device=device,
                                              logger=tb_logger,
                                              seed=seed,
                                              prior_update_interval=prior_update_interval,
                                              prior_update_conv2d_type=prior_update_conv2d_type)

    logger.debug(trainer.optimizer)


    if svi == 'uninformative-init':
        initializer = vardl.initializer.UninformativeInitializer(model=model)

    elif svi == 'heuristic-init':
        initializer = vardl.initializer.HeuristicInitializer(model=model)

    elif svi == 'xavier-normal-init':
        initializer = vardl.initializer.XavierNormalInitializer(model=model)

    elif svi == 'orthogonal-init':
        initializer = vardl.initializer.OrthogonalInitializer(model=model)

    elif svi == 'lsuv-init':
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

    elif svi == 'blm-init':
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

    elif svi != 'mcd':
        logger.warning('Initializer %s not found or not set. Using orthogonal-init' % svi)
        initializer = vardl.initializer.OrthogonalInitializer(model=model)

    if svi != 'mcd':
        initializer.initialize()

    if torch.cuda.is_available() and device != 'cpu':
        logger.debug('Setting CuDNN backend')
        torch.backends.cudnn.benchmark = True

    trainer.fit(iterations=iterations,
                test_interval=test_interval,
                train_verbose=train_verbose,
                train_log_interval=1)


def main():
    # Parse command line arguments
    args = parse_config()


    # Prepare the tb_logger to saving in the right path
    if args.svi != 'mcd':
        logdir = vardl.utils.next_path('%s/%s/%s_prior_update-%s_type-%s/' % (args.outdir, args.dataset, args.svi,
                                                                    args.prior_update_interval, args.prior_update_conv2d_type) +
                                       'run-%04d/')
    else:
        logdir = vardl.utils.next_path('%s/%s/%s/' % (args.outdir, args.dataset, args.svi) +
                                       'run-%04d/')

    logger = vardl.utils.setup_logger('vardl', logdir)

    if args.svi == 'mcd' and (args.prior_update_interval != 0 or args.prior_update_conv2d_type != 'layer'):
        logger.fatal('Clashing options. Experiment canceled')
        sys.exit()

    if args.prior_update_interval == 0 and args.prior_update_conv2d_type != 'layer':
        logger.fatal('Clashing options. Experiment canceled')
        sys.exit()



    # Print current configuration for debug
    logger.info('Running experiment %s' % args.name)
    logger.debug('Current configuration for experiment:')
    for key, value in vars(args).items():
        logger.debug('  %s = %s' % (key, value))
    logger.debug('  logdir = %s' % os.path.abspath(logdir))

    # Save experiment configuration as yaml file in logdir
    with open(logdir + 'experiment-config.yaml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    # Run experiment with current configuration
    t0 = time.time()
    run_experiment(**vars(args), logger=logger, logdir=logdir)
    t_diff = time.time() - t0
    logger.info('Experiment completed in %s', humanize.naturaldelta(t_diff))

if __name__ == '__main__':
    main()


