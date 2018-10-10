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


class BaseDistribution(nn.Module):
    #    __metaclass__ = abc.ABCMeta

    #    @abc.abstractclassmethod
    def __init__(self):
        super(BaseDistribution, self).__init__()
        NotImplementedError("Subclass should implement this.")

#    @abc.abstractclassmethod
    def sample(self, n_samples):
        NotImplementedError("Subclass should implement this.")

    def optimize(self, train: bool = True):
        NotImplementedError("Subclass should implement this.")
