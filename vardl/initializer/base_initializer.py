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

from time import time
import abc
import torch
import torch.nn as nn
from typing import Union

from ..layers import BaseVariationalLayer
from ..models import BaseBayesianNet

import logging
logger = logging.getLogger(__name__)


class BaseInitializer(abc.ABC):
    def __init__(self, model):
        self.model = model  # type: Union[nn.Module, BaseBayesianNet]
        self.layers = []
        self._layers_to_initialize()

    def _layers_to_initialize(self):
        for i, layer in enumerate(self.model.modules()):
            if issubclass(type(layer), BaseVariationalLayer):
                self.layers.append((i, layer))

    @abc.abstractmethod
    def _initialize_layer(self, layer, layer_index=None):
        raise NotImplementedError()

    def initialize(self):
        t_start = time()
        for i, layer in self.layers:
            logger.info('Initialization of layer %d' % i)
            self._initialize_layer(layer, i)
        t_end = time()
        logger.info('Initialization done in %.3f sec.' % (t_end - t_start))

    def __repr__(self):
        return str(self.layers)
