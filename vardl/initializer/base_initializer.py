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

from ..layers import BayesianLinear


class BaseInitializer():
    def __init__(self, model):
        self.model = model
        self.layers = []
        self._layers_to_initialize()

    def _layers_to_initialize(self):
        for i, layer in enumerate(self.model.architecture):
            if isinstance(layer, BayesianLinear):
                self.layers.append((i, layer))

    def _initialize_layer(self, layer: BayesianLinear, layer_index: int=None):
        raise NotImplementedError()

    def initialize(self):
        t_start = time()
        for i, layer in self.layers:
            self._initialize_layer(layer, i)
        t_end = time()
        print('INFO - Initialization done in %.4f sec.' % (t_end - t_start))

    def __repr__(self):
        return str(self.layers)
