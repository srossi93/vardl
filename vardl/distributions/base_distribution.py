
#  Copyright (c) 2019
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Authors:
#      Simone Rossi <simone.rossi@eurecom.fr>
#      Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#

import abc

import torch.nn as nn


class BaseDistribution(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        super(BaseDistribution, self).__init__()
        NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def sample(self, n_samples):
        NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def sample_local_reparam_linear(self, *args, **kargs):
        NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def sample_local_reparam_conv2d(self, *args, **kargs):
        NotImplementedError("Subclass should implement this.")

    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train
