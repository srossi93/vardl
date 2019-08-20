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

from .logsumexp import logsumexp  # noqa: F401
from .set_seed import set_seed  # noqa: F401
from .path_utils import next_path  # nopa: F401
# from .experiment_plotter import ExperimentPlotter, tsplot
from .glog import *
from .timing import timing
# from .experiment_plotter import savefig
from .cuda_memory_inspector import CUDAMemoryProfiler

from .calibration import calibration_test

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def compute_output_shape_conv2d(in_height, in_width, kernel_size, padding=0, dilation=1, stride=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    out_height = (in_height + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    out_width = (in_width + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return out_height, out_width
