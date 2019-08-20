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
import torch.nn


class TensorMaxPool2d(torch.nn.modules.pooling._MaxPoolNd):
    def forward(self, input):
        input_shape = input.shape
        c = input.shape[-3]
        h_in = input.shape[-2]
        w_in = input.shape[-1]
        input = input.view(-1, c, h_in, w_in)
        output = torch.nn.functional.max_pool2d(input, self.kernel_size, self.stride,
                                                self.padding, self.dilation, self.ceil_mode,
                                                self.return_indices)
        h_out = output.shape[-2]
        w_out = output.shape[-1]
        return output.view(*input_shape[:-3], c, h_out, w_out)
