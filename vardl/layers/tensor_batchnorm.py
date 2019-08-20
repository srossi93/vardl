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

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class TensorBatchNorm2d(_BatchNorm):
    def forward(self, input):
        input_shape = input.shape
        c = input.shape[-3]
        h_in = input.shape[-2]
        w_in = input.shape[-1]
        input = input.view(-1, c, h_in, w_in)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        output = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        h_out = output.shape[-2]
        w_out = output.shape[-1]
        return output.view(*input_shape[:-3], c, h_out, w_out)


class TensorBatchNorm1d(_BatchNorm):
    def forward(self, input):
        input_shape = input.shape
        f_in = input.shape[-1]

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

#            self.running_mean = exponential_average_factor * self.running_mean + (1 - exponential_average_factor) * input.mean(1)
#            self.running_var = exponential_average_factor * self.running_var + (1 - exponential_average_factor) * input.var(1)
#            mean = self.running_mean.unsqueeze(1)
#            std = self.running_var.unsqueeze(1).sqrt()
#        else:
#            mean = input.mean(1).unsqueeze(1)
#            std = input.std(1).unsqueeze(1)
#
#        output = (input - mean) / (std + self.eps)
#        if self.affine:
#            output = output * self.weight + self.bias
#
#        return output

        input = input.view(-1, f_in)
        output = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        return output.view(*input_shape)
