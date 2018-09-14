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

import torch
import torch.nn as nn

from ..layers import BayesianLinear
from ..likelihoods import Softmax
from . import BaseBayesianNet


class ClassBayesianNet(BaseBayesianNet):

    def __init__(self, architecure: nn.Sequential, dtype: torch.dtype = torch.float32):
        super(ClassBayesianNet, self).__init__()

        self.dtype = dtype
        self.architecture = architecure
        self.likelihood = Softmax()

    def forward(self, input):
        input = input *torch.ones(self.architecture[0].nmc, *input.size()).to(self.architecture[0].device)
        return self.architecture(input)