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
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseVariationalLayer


class TensorConv2d(nn.Conv2d, BaseVariationalLayer):
    def forward(self, input):
        input_shape = input.shape
        c = input.shape[-3]
        h_in = input.shape[-2]
        w_in = input.shape[-1]
        input = input.view(-1, c, h_in, w_in)
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        h_out = output.shape[-2]
        w_out = output.shape[-1]
        output = output.view(*input_shape[:-3], self.out_channels, h_out, w_out)
#        output = F.dropout(output, 0.5, training=True)
        return output

    def kl_divergence(self):
        return 0.0075 * torch.norm(self.weight, 2)
