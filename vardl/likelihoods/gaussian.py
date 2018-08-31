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
import torch.nn as nn
import numpy as np

from . import BaseLikelihood


class Gaussian(BaseLikelihood):

    def __init__(self, dtype:torch.dtype):#, model: torch.nn.Module):
        super(Gaussian, self).__init__()
        self.dtype = dtype
        self.log_noise_var = nn.Parameter(torch.ones(1, dtype=self.dtype) * -2.0,
                                          requires_grad=True)
        self.log_2_pi_torch = torch.ones(1) * np.log(2.0 * np.pi)

        return
       # self.model = model
       # self.device = model.device
       # self.log_2_pi_torch = torch.ones(1, device=self.device) * np.log(2.0 * np.pi)

    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:

        log_noise_var = self.log_noise_var
        return - 0.5 * (self.log_2_pi_torch + log_noise_var +
                        torch.pow(output - latent_val, 2) * torch.exp(-log_noise_var))

    def get_params(self):
        return self.log_theta_noise_var

    def predict(self, latent_val: torch.Tensor) -> torch.Tensor:
        return torch.mean(latent_val, dim=0), torch.std(latent_val, dim=0)
