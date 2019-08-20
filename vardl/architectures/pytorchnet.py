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


def build_pytorchnet(**config) -> nn.Sequential:
    pytorchnet = nn.Sequential(
            BayesianConv2d(in_channels=1,
                                     in_height=28,
                                     in_width=28,
                                     out_channels=10,
                                     kernel_size=5,
                                     **config),


            View(-1, -1, 10, 26, 26),
            nn.MaxPool2d(2),
            View(config['nmc_train'], config['nmc_test'], -1, 10, 13, 13),
            nn.ReLU(),
            BayesianConv2d(in_channels=10,
                                         in_height=13,
                                         in_width=13,
                                         out_channels=20,
                                         kernel_size=5,
                                         **config),
            View(-1, -1, 20, 11, 11),
            nn.MaxPool2d(2),
            View(config['nmc_train'], config['nmc_test'], -1, 500),
            nn.ReLU(),
            BayesianLinear(in_features=500,
                                     out_features=50,
                                     **config),
            nn.ReLU(),
            BayesianLinear(in_features=50,
                                     out_features=10,
                                     **config))
    return pytorchnet