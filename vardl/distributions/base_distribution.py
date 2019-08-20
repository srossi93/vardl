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

import torch
import torch.nn as nn


class BaseDistribution(nn.Module, metaclass=abc.ABCMeta):
    """
    Implements a metaclass for all distribution
    """
    def __init__(self, *args, **kwargs):
        """
        Implements a metaclass for all distribution
        """
        super(BaseDistribution, self).__init__()
        NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def sample(self, n_samples):
        """
        Abstract method for sampling from the distribution
        """
        NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def sample_local_reparam_linear(self, *args, **kargs):
        """
        Abstract method for sampling from the distribution using
        the local reparameterization trick of linear layers
        """
        NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def sample_local_reparam_conv2d(self, *args, **kargs):
        """
        Abstract method for sampling from the distribution using
        the local reparameterization trick of convolutional layers
        """
        NotImplementedError("Subclass should implement this.")

    def optimize(self, train: bool = True):
        """
        Optimize or not the distribution
        :param train: Flag
        """
        for param in self.parameters():
            param.requires_grad = train
