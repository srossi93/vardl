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


import torch
import torch.nn as nn


class BaseBayesianLayer(nn.Module):

    def __init__(self, nmc_train: int, nmc_test: int,
                 dtype: str, device: torch.device):
        super(BaseBayesianLayer, self).__init__()
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.dtype = dtype
        self.device = device

        pass

    @property
    def dkl(self) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, *input):
        raise NotImplementedError()

    def train(self, mode=True):
        self.nmc = self.nmc_train if mode else self.nmc_test