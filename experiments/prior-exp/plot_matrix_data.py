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


from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

matplotlib.rc_file('~/.config/matplotlib/whitepaper.mplrc')

def main():


    priors_variance_linear = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04,
                            0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    priors_variance_conv2d = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04,
                             0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #priors_variance_linear = [0.001, 0.002, 0.003, 0.004, 0.005]
    #priors_variance_conv2d = [0.001, 0.002, 0.003, 0.004, 0.005]

    table_errors = np.loadtxt('table_errors.csv', delimiter=';')
    table_mnll = np.loadtxt('table_mnll.csv', delimiter=';')

    fig, ax = plt.subplots(figsize=[20, 20])
    im = ax.imshow(table_errors, cmap='RdBu_r')
    ax.grid(False)
    ax.set_xticks(np.arange(len(priors_variance_linear)))
    ax.set_yticks(np.arange(len(priors_variance_conv2d)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(priors_variance_linear)
    ax.set_yticklabels(priors_variance_conv2d)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(table_errors.shape[0]):
        for j in range(table_errors.shape[1]):
            text = ax.text(j, i, '%.3f' % (table_errors[i, j]),
                           ha="center", va="center", )
    ax.set_xlabel('Prior variance for linear layers')
    ax.set_ylabel('Prior variance for conv2d layers')
    ax.set_title('Error rate vs prior')

    vardl.utils.ExperimentPlotter.savefig('figures/table_error_vs_prior_full', 'pdf')
    vardl.utils.ExperimentPlotter.savefig('figures/table_error_vs_prior_full', 'tex')
    plt.close()

    fig, ax = plt.subplots(figsize=[20, 20])
    im = ax.imshow(table_mnll, cmap='RdBu_r')
    ax.grid(False)
    ax.set_xticks(np.arange(len(priors_variance_linear)))
    ax.set_yticks(np.arange(len(priors_variance_conv2d)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(priors_variance_linear)
    ax.set_yticklabels(priors_variance_conv2d)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(table_errors.shape[0]):
        for j in range(table_errors.shape[1]):
            text = ax.text(j, i, '%.3f' % (table_mnll[i, j]),
                           ha="center", va="center", )
    ax.set_xlabel('Prior variance for linear layers')
    ax.set_ylabel('Prior variance for conv2d layers')
    ax.set_title('MNLL vs prior')

    vardl.utils.ExperimentPlotter.savefig('figures/table_mnll_vs_prior_full', 'pdf')
    vardl.utils.ExperimentPlotter.savefig('figures/table_mnll_vs_prior_full', 'tex')
    plt.close()


if __name__ == '__main__':
    main()
