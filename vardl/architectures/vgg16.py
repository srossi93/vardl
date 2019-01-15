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

import torch.nn as nn

from ..layers import BayesianConv2d, BayesianLinear, View

def build_vgg16_cifar10(**config):
    nmc_train = config['nmc_train']
    nmc_test = config['nmc_test']

    conv1_1 = BayesianConv2d(3, 32, 32, 32, kernel_size=3, stride=1, padding=1,  **config)
    conv1_2 = BayesianConv2d(32, 32, 32, 32, kernel_size=3, stride=1, padding=1,  **config)
    conv2_1 = BayesianConv2d(32, 16, 16, 64, kernel_size=3, stride=1, padding=1,  **config)
    conv2_2 = BayesianConv2d(64, 16, 16, 64, kernel_size=3, stride=1, padding=1,  **config)
    conv3_1 = BayesianConv2d(64, 8, 8, 128, kernel_size=3, stride=1, padding=1,  **config)
    conv3_2 = BayesianConv2d(128, 8, 8, 128, kernel_size=3, stride=1, padding=1,  **config)
    conv3_3 = BayesianConv2d(128, 8, 8, 128, kernel_size=3, stride=1, padding=1,  **config)
    conv4_1 = BayesianConv2d(128, 4, 4, 256, kernel_size=3, stride=1, padding=1,  **config)
    conv4_2 = BayesianConv2d(256, 4, 4, 256, kernel_size=3, stride=1, padding=1,  **config)
    conv4_3 = BayesianConv2d(256, 4, 4, 256, kernel_size=3, stride=1, padding=1,  **config)
    conv5_1 = BayesianConv2d(256, 2, 2, 256, kernel_size=3, stride=1, padding=1,  **config)
    conv5_2 = BayesianConv2d(256, 2, 2, 256, kernel_size=3, stride=1, padding=1,  **config)
    conv5_3 = BayesianConv2d(256, 2, 2, 256, kernel_size=3, stride=1, padding=1,  **config)

    fc = BayesianLinear(256, 10, **config)



    alexnet_cifar10 = nn.Sequential(
        conv1_1,
        nn.ReLU(),
        conv1_2,
        nn.ReLU(),

        View(-1,-1, 32, 32, 32),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 32, 16, 16),

        conv2_1,
        nn.ReLU(),
        conv2_2,
        nn.ReLU(),

        View(-1,-1, 64, 16, 16),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 64, 8, 8),

        conv3_1,
        nn.ReLU(),
        #conv3_2,   # <--
        #nn.ReLU(),
        conv3_3,
        nn.ReLU(),

        View(-1,-1, 128, 8, 8),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 128, 4, 4),

        conv4_1,
        nn.ReLU(),
        conv4_2,
        nn.ReLU(),
        conv4_3,
        nn.ReLU(),

        View(-1,-1, 256, 4, 4),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 256, 2, 2),

        conv5_1,
        nn.ReLU(),
        conv5_2,
        nn.ReLU(),
        conv5_3,
        nn.ReLU(),

        View(-1,-1, 256, 2, 2),
        nn.MaxPool2d(2),
        View(nmc_train, nmc_test, -1, 256),

        fc
    )

    return alexnet_cifar10
