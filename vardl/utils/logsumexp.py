# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
#              Maurizio Filippone <maurizio.filippone@eurecom.fr>
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


# Log-sum operation
def logsumexp(vals: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.logsumexp(vals, dim)
#    m = torch.max(vals, dim)[0]
#    if dim is None:
#        return m + torch.log(torch.sum(torch.exp(vals - m), dim))
#    else:
#        return m + torch.log(torch.sum(torch.exp(vals - torch.unsqueeze(m, dim)), dim))
