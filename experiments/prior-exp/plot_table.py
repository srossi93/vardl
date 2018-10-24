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
import numpy as np
import matplotlib.pylab as plt
import vardl

matplotlib.rc_file('~/.config/matplotlib/whitepaper.mplrc')

error = np.array([[40.92, 37.68, 36.99, 39.97, 71.85],
                  [34.41, 31.28, 32.85, 36.38, 51.29],
                  [35.88, 31.90, 33.34, 37.78, 61.87],
                  [43.84, 35.32, 38.22, 42.10, 68.77],
                  [48.07, 40.27, 43.06, 52.30, 55.95]])

of = np.array([["", "", "", "", "*"],
               ["", "", "", "", "*"],
               ["", "", "", "", "*"],
               ["", "", "", "", "*"],
               ["", "", "", "*", "*"],
               ["", "", "", "*", "*"]])

labels = [0.001, 0.01, 0.1, 1, 10]

fig, ax = plt.subplots()
im = ax.imshow(error, cmap='RdBu_r')
ax.grid(False)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for i in range(error.shape[0]):
    for j in range(error.shape[1]):
        text = ax.text(j, i, '%.3f%s' % (error[i, j]/100, of[i, j]),
                       ha="center", va="center", )
ax.set_xlabel('Prior variance for linear layers')
ax.set_ylabel('Prior variance for conv2d layers')
ax.set_title('Error rate vs prior')

vardl.utils.ExperimentPlotter.savefig('figures/table_error_vs_prior', 'pdf')
vardl.utils.ExperimentPlotter.savefig('figures/table_error_vs_prior', 'tex')
plt.close()

mnll = np.array([[1.218, 1.164, 1.151, 1.250, 2.391],
                 [1.098, 1.041, 1.089, 1.315, 1.874],
                 [1.166, 1.089, 1.224, 1.490, 2.099],
                 [1.359, 1.223, 1.401, 1.631, 2.315],
                 [1.464, 1.399, 1.577, 1.910, 2.339]])


of = np.array([["", "", "", "", "*"],
               ["", "", "", "*", "*"],
               ["", "", "", "*", "*"],
               ["", "", "", "*", "*"],
               ["", "", "*", "*", "*"],
               ["", "", "*", "*", "*"]])


labels = [0.001, 0.01, 0.1, 1, 10]

fig, ax = plt.subplots()
im = ax.imshow(mnll, cmap='RdBu_r')
ax.grid(False)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for i in range(error.shape[0]):
    for j in range(error.shape[1]):
        text = ax.text(j, i, '%.3f%s' % (mnll[i, j], of[i, j]),
                       ha="center", va="center", )
ax.set_xlabel('Prior variance for linear layers')
ax.set_ylabel('Prior variance for conv2d layers')
ax.set_title('MNLL vs prior')

vardl.utils.ExperimentPlotter.savefig('figures/table_mnll_vs_prior', 'pdf')
vardl.utils.ExperimentPlotter.savefig('figures/table_mnll_vs_prior', 'tex')
