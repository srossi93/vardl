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
import numpy as np
import torch.distributions as distributions

class BetaDropout(nn.Module):
    def __init__(self, alpha=0.2, beta=0.2):
        super(BetaDropout, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.distribution = distributions.Beta(self.alpha, self.beta)

    def forward(self, input : torch.Tensor) -> torch.Tensor:
#        print('INFO - Inside Betadropout')
        with torch.no_grad():
            size = input.size()
            #size = torch.Size(input.transpose(0, 1).shape[1:])
            #mask = self.distribution.sample(size).to(input.device)
            #mask = mask * torch.ones(input.size(1), *mask.shape).to(input.device)
            #mask = mask.transpose(0,1)
            mask = torch.from_numpy(np.random.beta(self.alpha, self.beta, list(input.size()))).float().to(input.device)
        output = input * mask
#        print('INFO - Masking done')
        return output


