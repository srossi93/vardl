r"""
   Copyright 2018 Simone Rossi, Maurizio Filippone

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import os
import torch
import humanize
import torch.nn as nn

from ..layers import BayesianConv2d
from ..layers import BayesianLinear

import logging

class BaseBayesianNet(nn.Module):

    def __init__(self):
        super(BaseBayesianNet, self).__init__()
        self.architecture = None
        self.likelihood = None
        self._logger = logging.getLogger(__name__)

    @property
    def dkl(self):
        total_dkl = 0
        for layer in self.architecture:
            total_dkl += layer.dkl if isinstance(layer, BayesianLinear) or isinstance(layer, BayesianConv2d) else 0
        return total_dkl# / 50
    @property
    def trainable_parameters(self):
        return humanize.intword(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def save_model(self, path):
        self._logger.info('Saving model in %s' % path)
        torch.save(self.state_dict(), path)
        self._logger.info('Model saved (%s)' % humanize.naturalsize(os.path.getsize(path), gnu=True))

    def load_model(self, path):
        self._logger.info('Loading model from %s' % path)
        self.load_state_dict(torch.load(path, map_location='cuda'))