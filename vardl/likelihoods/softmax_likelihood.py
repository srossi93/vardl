
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
#  Original code and updates by:
#      Karl Krauth, Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi

import torch

from . import BaseLikelihood

import logging
logger = logging.getLogger(__name__)


class Softmax(BaseLikelihood):
    """
    Implements softmax likelihood for multi-class classification
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:
        return torch.sum(output * latent_val, 2) - torch.logsumexp(latent_val, 2)

    def predict(self, latent_val):
        """
        return the probabilty for all the samples, datapoints and calsses
        :param latent_val:
        :return:
        """
        logprob = latent_val - torch.unsqueeze(torch.logsumexp(latent_val, 2), 2)
        return logprob.exp()
