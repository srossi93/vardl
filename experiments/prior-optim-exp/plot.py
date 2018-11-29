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
sys.path.insert(0, '../..')

import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import matplotlib.pylab as plt
import os
import glob
from itertools import chain
from matplotlib import colors
import numpy as np
import pandas as pd
import vardl

from vardl.tools.convert_tensorboard import read_tbevents, logger, save_tag

import matplotlib2tikz
from matplotlib2tikz import save as tikz_save
import sys

import multiprocessing
import argparse
"""
basedir = './workspace/mnist/'
methods = ['blm-init_prior_update-0', 'blm-init_prior_update-1', 'mcd']
plotter = vardl.utils.ExperimentPlotter(name='prior-updates',
                                        basedir=basedir,
                                        methods=methods,
                                        savepath='./figures/')

plotter.parse()


matplotlib.rc_file('../../../dgp_rfs_svi_pytorch/config/whitepaper.mplrc')
plotter.plot(tags=['error/test', 'loss/test', 'model/dkl'],
             xlim=[100, ],
             ylims=[[0, 0.05], [0.04, .12], [1000, 7500]],
             logx=True,
             save=True)
plt.legend()
plt.close()
"""

def str2bool(v):
    if v.lower() in ('yes', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tbevent_reader(path):
    event_acc = read_tbevents(path)
    for tag in event_acc.Tags()['scalars']:
       save_tag(event_acc, tag, path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_from_tbevents', help='Dumps events only in csv', type=str2bool, default=False)
    parser.add_argument('--datasets', help='Dataset to load', nargs='+', type=str)
    args = parser.parse_args()

    logger.info(args.datasets)
    basedir = './workspace/'
    datasets = ['mnist', 'cifar10']
    methods = ['blm-init_prior_update-0', 'blm-init_prior_update-1', 'blm-init_prior_update-1_type-outchannels',
               'blm-init_prior_update-1_type-outchannels+inchannels',
               'blm-init_prior_update-1_type-outchannels+inchannels+in_rows', 'mcd']
    paths = []
    for dataset in datasets:
        for method in methods:
            paths.append(basedir + dataset + '/' + method + '/run-0001/')

    print(args.read_from_tbevents)
    if args.read_from_tbevents:
        multiprocessing.Pool(32).map(tbevent_reader, paths)

    logger.info('Now can plot')

    tags = ['error/test', 'loss/test', 'model/dkl']
    logx = True
    logy = [False, False, True]

    limits = {'mnist': {
                'error/test': {'x': [1000, ], 'y': [0, 0.05]},
                'loss/test': {'x': [1000, ], 'y': [0.045, 0.10]},
                'model/dkl': {'x': [1000, ], 'y': [1000, 100000]}},
              'cifar10': {
                 'error/test': {'x': [1000, ], 'y': [0.2, 0.4]},
                 'loss/test': {'x': [1000, ], 'y': [1, 2]},
                 'model/dkl': {'x': [1000, ], 'y': [1000, 1000000]}}
    }

    for dataset in datasets:
        # tb_logger.info('Plotting for dataset %s' % dataset)
        fig, axs = plt.subplots(len(tags), 1)

        for i, tag in enumerate(tags):
            for method in methods:
                # tb_logger.info('Plotting ' + method)
                files = list(glob.glob(basedir + dataset + '/' + method + '/**/%s.csv' % tag.replace('/', '_'),
                                       recursive=True))
                try:
                    data = np.loadtxt(files[0]).reshape(-1, 3)
                except:
                    logger.error('File not found. Skip')
                    continue

                if method == 'mcd':
                    linestyle = '--'
                else:
                    linestyle = '-'
                axs[i].plot(data[:, 1] + 1, data[:, 2], label=method.upper(), linestyle=linestyle)
                if logx: axs[i].semilogx()
                if logy[i]: axs[i].semilogy()

            axs[i].set_ylabel(tag.upper())
            axs[i].set_xlim(*limits[dataset][tag]['x'])
            axs[i].set_ylim(*limits[dataset][tag]['y'])
        axs[0].set_title(dataset.upper())
        # axs[0].legend()
    plt.show()

if __name__ == '__main__':
    main()