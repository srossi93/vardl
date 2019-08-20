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

import torch
import abc

import torch.nn as nn

import logging
logger = logging.getLogger(__name__)


class BaseVariationalLayer(nn.Module, abc.ABC):
    def __init__(self, **kwargs):
        super(BaseVariationalLayer, self).__init__()
        self.nmc_train = kwargs.pop('nmc_train') if 'nmc_train' in kwargs else 1
        self.nmc_test = kwargs.pop('nmc_test') if 'nmc_test' in kwargs else 1
        self.dtype = kwargs.pop('dtype') if 'dtype' in kwargs else torch.float32
        self.nmc = None
        self.eval()

    @abc.abstractmethod
    def kl_divergence(self):
        raise NotImplementedError('Subclass should implement this method')

    def train(self, mode=True):
        self.nmc = self.nmc_train if mode else self.nmc_test
