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

import torch.nn as nn

from ..layers import BayesianConv2d, BayesianLinear, View


def build_alexnet_cifar10(in_channel, in_height, in_width, out_labels, **config):
    nmc_train = config['nmc_train']
    nmc_test = config['nmc_test']

    conv1 = BayesianConv2d(3, 32, 32, 48, kernel_size=11, stride=4, padding=5, **config)
    conv2 = BayesianConv2d(48, 4, 4, 128, kernel_size=5, padding=2, **config)
    conv3 = BayesianConv2d(128, 2, 2, 192, kernel_size=3, padding=1, **config)
    conv4 = BayesianConv2d(192, 2, 2, 192, kernel_size=3, padding=1, **config)
    conv5 = BayesianConv2d(192, 2, 2, 128, kernel_size=3, padding=1, **config)

    fc1 = BayesianLinear(512, 256, **config)
    fc2 = BayesianLinear(256, 96, **config)
    fc3 = BayesianLinear(96, 10, **config)

    alexnet_cifar10 = nn.Sequential(
        conv1,
        nn.ReLU(),
        View(-1,-1, 48, 8, 8),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 48, 4, 4),


        conv2,
        nn.ReLU(),
        View(-1,-1, 128, 4, 4),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 128, 2, 2),

        conv3,
        nn.ReLU(),

        conv4,
        nn.ReLU(),

        conv5,
        nn.ReLU(),

        View(nmc_train, nmc_test, -1, 512),

        fc1,
        nn.ReLU(),

        fc2,
        nn.ReLU(),

        fc3
    )

    return alexnet_cifar10


def build_alexnet_imagenet(in_channel, in_height, in_width, out_labels, **config):
    nmc_train = config['nmc_train']
    nmc_test = config['nmc_test']

    conv1 = BayesianConv2d(3, 32, 32, 96, kernel_size=11, stride=4, padding=5, **config)
    conv2 = BayesianConv2d(96, 4, 4, 256, kernel_size=5, padding=2, **config)
    conv3 = BayesianConv2d(256, 2, 2, 384, kernel_size=3, padding=1, **config)
    conv4 = BayesianConv2d(384, 2, 2, 384, kernel_size=3, padding=1, **config)
    conv5 = BayesianConv2d(384, 2, 2, 256, kernel_size=3, padding=1, **config)

    fc1 = BayesianLinear(1024, 256, **config)
    fc2 = BayesianLinear(256, 256, **config)
    fc3 = BayesianLinear(256, out_labels, **config)

    alexnet_cifar10 = nn.Sequential(
        conv1,
        nn.ReLU(),
        View(-1,-1, 96, 8, 8),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 96, 4, 4),


        conv2,
        nn.ReLU(),
        View(-1,-1, 256, 4, 4),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 256, 2, 2),

        conv3,
        nn.ReLU(),

        conv4,
        nn.ReLU(),

        conv5,
        nn.ReLU(),

        View(nmc_train, nmc_test, -1, 1024),

        fc1,
        nn.ReLU(),

        fc2,
        nn.ReLU(),

        fc3
    )

    return alexnet_cifar10