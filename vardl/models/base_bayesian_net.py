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

from typing import Union


from ..layers import BaseVariationalLayer

import logging
logger = logging.getLogger(__name__)


class BaseBayesianNet(nn.Module):

    def __init__(self):
        super(BaseBayesianNet, self).__init__()
        self.architecture = None
        self.likelihood = None

    def kl_divergence(self):
        total_dkl = 0.
        for layer in self.modules():  # type: Union[BaseVariationalLayer, nn.Module]
            total_dkl += layer.kl_divergence() if issubclass(type(layer), BaseVariationalLayer) else 0
        return total_dkl

    def forward(self, input):
        input = input * torch.ones(self.architecture.nmc, *input.size()).to(input.device)  # type: torch.Tensor
        # TODO: check how to retrieve nmc
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return nn.parallel.data_parallel(self.architecture, inputs=input, dim=1)
        else:
            return self.architecture(input)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, path):
        logger.info('Saving model in %s' % path)
        torch.save(self.state_dict(), path)
        logger.info('Model saved (%s)' % humanize.naturalsize(os.path.getsize(path), gnu=True))

    def load_model(self, path):
        logger.info('Loading model from %s' % path)
        self.load_state_dict(torch.load(path, map_location='cuda'))
