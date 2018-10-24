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
    local_reparameterization = True
    bias = False
    approx = 'factorized'
    nmc_train = 1
    nmc_test = 16
    dataset_dir = '/mnt/workspace/datasets/'
    dataset = 'cifar10'

    X_train, Y_train = torch.load('%s/%s/train_%s.pt' % (dataset_dir, dataset, dataset))
    X_test, Y_test = torch.load('%s/%s/test_%s.pt' % (dataset_dir, dataset, dataset))

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    print('INFO - Dataset: %s' % dataset)
    print('INFO - Test size: ', X_test.size(0))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=1,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)



    layer_config = {'local_reparameterization': local_reparameterization,
                    'bias': bias,
                    'approx': approx,
                    'nmc_test': nmc_test,
                    'nmc_train': nmc_train,
                    'device': torch.device('cuda')}

    arch = vardl.architectures.build_lenet_cifar10(*X_train.size()[1:], Y_train.size(1), **layer_config)
    model_path = 'work/cifar10/prior_var_lin_%s-prior_var_conv_%s/run-0001/model_snapshot_final.pth'

    optimizer_config = dict(lr=float(0.0001), weight_decay=0)
    lr_decay_config = dict(gamma=0.0001, p=0)

    priors_variance_linear = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04,
                            0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    priors_variance_conv2d = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04,
                             0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #priors_variance_linear = [0.001, 0.002, 0.003, 0.004, 0.005]
    #priors_variance_conv2d = [0.001, 0.002, 0.003, 0.004, 0.005]

    table_errors = np.ones([len(priors_variance_conv2d), len(priors_variance_linear)])
    table_mnll = np.ones([len(priors_variance_conv2d), len(priors_variance_linear)])

    for i, prior_variance_conv2d in enumerate(priors_variance_conv2d):
        for j, prior_variance_linear in enumerate(priors_variance_linear):
            print('INFO - Loading model with:')
            print('INFO -     prior variance of linear layers = %.3f' % prior_variance_linear)
            print('INFO -     prior variance of conv2d layers = %.3f' % prior_variance_conv2d)
            model = vardl.models.ClassBayesianNet(architecure=arch)
            try:
                model.load_model(model_path % (prior_variance_linear, prior_variance_conv2d))
                tb_logger = vardl.logger.TensorboardLogger('/tmp/tb/', model)

                trainer = vardl.trainer.TrainerClassifier(model=model,
                                                  train_dataloader=train_dataloader,
                                                  test_dataloader=test_dataloader,
                                                  optimizer='Adam',
                                                  optimizer_config=optimizer_config,
                                                  lr_decay_config=lr_decay_config,
                                                  device='cuda',
                                                  logger=tb_logger,
                                                  seed=0)

                mnll, error = trainer.test(verbose=False)
                table_errors[i, j] = error
                table_mnll[i, j] = mnll
                print('INFO - Evaluation completed')
            except FileNotFoundError:
                print("WARN - File not found - Skip")
                table_errors[i, j] = np.nan
                table_mnll[i, j] = np.nan

    #print(table_errors)a
    np.savetxt('table_errors.csv', table_errors, delimiter=';')
    np.savetxt('table_mnll.csv', table_mnll, delimiter=';')

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


if __name__ == '__main__':
    main()
