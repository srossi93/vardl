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
#
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import torch
import numpy as np

def to_numpy_if_tensor(t):
    if hasattr(t, 'data') and torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return t


def calibration_test(p, y, nbins=5):
    '''
    Returns ece:  Expected Calibration Error
           conf: confindence levels (as many as nbins)
           accu: accuracy for a certain confidence level
                 We are interested in the plot confidence vs accuracy
           bin_sizes: how many points lie within a certain confidence level
   '''
    p = to_numpy_if_tensor(p)
    y = to_numpy_if_tensor(y)

    if p.shape[0] != y.shape[0]:
        raise ValueError('Mismatch in first dimension. Expected equal size but got %d and %d' % (p.shape[0],
                                                                                                 y.shape[0]))

    edges = np.linspace(0, 1, nbins+1)
    accu = np.zeros(nbins)
    conf = np.zeros(nbins)
    bin_sizes = np.zeros(nbins)
    APPROXIMATE_CONF = False
    # Multiclass problems are treated by considering the max
    if p.ndim>1 and p.shape[1]!=1:
        pred = np.argmax(p, axis=1)
        p = np.max(p, axis=1)
    elif False:
        # treat binary classification the same was as multiclass (not needed)
        pred = p > 0.5
        pred = pred.astype(int).flatten()
        p[pred==0] = 1-p[pred==0]
    else:
        # the default treatment for binary classification
        pred = np.ones(p.size)
    #
    y = y.flatten()
    p = p.flatten()
    for i in range(nbins):
        idx_in_bin = (p > edges[i]) & (p <= edges[i+1])
        bin_sizes[i] = max(sum(idx_in_bin), 1)
        accu[i] = np.sum(y[idx_in_bin] == pred[idx_in_bin]) / bin_sizes[i]
        if APPROXIMATE_CONF:
            conf[i] = (edges[i+1] + edges[i]) / 2
        else:
            conf[i] = np.sum(p[idx_in_bin]) / bin_sizes[i]
            if conf[i] == 0:
                conf[i] = (edges[i+1] + edges[i]) / 2
    ece = np.sum(np.abs(accu - conf) * bin_sizes) / np.sum(bin_sizes)
    return ece, conf, accu, bin_sizes