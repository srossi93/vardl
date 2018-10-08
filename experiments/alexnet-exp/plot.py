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


import sys
sys.path.insert(0, '../..')

import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import matplotlib.pylab as plt
import os
import glob
from itertools import chain
from matplotlib import colors
import numpy as np
import pandas as pd
import vardl



import matplotlib2tikz
from matplotlib2tikz import save as tikz_save
import sys

basedir = './work/cifar10/'
methods = ['blm', 'mcd']
plotter = vardl.utils.ExperimentPlotter(name='alexnet',
                                        basedir=basedir,
                                        methods=methods,
                                        savepath='./figures/')

plotter.parse()


matplotlib.rc_file('../../../dgp_rfs_svi_pytorch/config/whitepaper.mplrc')
plotter.plot(tags=['error/test', 'loss/test'],
             xlim=[100, ],
             ylims=[[0.25, 0.6], [1, 1.6]],
             logx=True,
             save=True)
plt.legend()
plt.close()