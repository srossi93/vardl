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
# Original code by Karl Krauth
# Changes by Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone

import torch
from . import BaseLikelihood
from .. import utils


class Softmax(BaseLikelihood):
    """
    Implements softmax likelihood for multi-class classification
    """

    def __init__(self):
        pass

    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:
        return torch.sum(output * latent_val, 2) - utils.logsumexp(latent_val, 2)

    def predict(self, latent_val):
        """
        return the probabilty for all the samples, datapoints and calsses
        :param latent_val:
        :return:
        """
        logprob = latent_val - torch.unsqueeze(utils.logsumexp(latent_val, 2), 2)
        return torch.exp(logprob)

    def get_params(self):
        return None