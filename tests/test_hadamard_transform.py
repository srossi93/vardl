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

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import torch
import numpy as np
import scipy.linalg
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

import vardl
logger = vardl.utils.setup_logger(__name__, '/tmp/', 'DEBUG')


def main():
    b = 64 # batch size
    d = 128  # feature dimension
    repetitions = 16

    X = torch.randn(b, d, requires_grad=True)
    H = torch.from_numpy(scipy.linalg.hadamard(d)).float()

    matrix_mult = torch.matmul(X, H)
    hadamard_mult = vardl.functional.HadamardTransform.apply(X)

    if (matrix_mult != hadamard_mult).all():
        logger.error('Error in computation of the output')
        logger.debug(matrix_mult)
        logger.debug(hadamard_mult)
        return -1
    logger.info('Transformation ok!')
    logger.info('Test timing forward:')

    grad_matrix_mul = torch.autograd.grad(matrix_mult.sum(), X, retain_graph=True)[0]
    grad_hadamard = torch.autograd.grad(hadamard_mult.sum(), X, retain_graph=True)[0]

    if (grad_matrix_mul - grad_hadamard).abs().sum() > 1e-5:
        logger.error('Error in computation of the gradient')
        logger.debug(grad_matrix_mul)
        logger.debug(grad_hadamard)
        return -1


    times_matmul = []
    times_hadamard = []

    for power in range(2, 15):
        d = 2 ** power
        X = torch.randn(b, d)
        t0 = time.time()
        _ = [torch.matmul(X, torch.from_numpy(scipy.linalg.hadamard(d)).float()) for _ in range(repetitions)]
        t_mean = (time.time() - t0) / repetitions
        times_matmul.append(t_mean)
        logger.info('D = %4d - Matmul   in %1.2e' % (d, t_mean))
        t0 = time.time()
        _ = [vardl.functional.HadamardTransform(X) for _ in range(repetitions)]
        t_mean = (time.time() - t0) / repetitions
        times_hadamard.append(t_mean)
        logger.info('D = %4d - Hadamard in %1.2e' % (d, t_mean))

    fig, ax = plt.subplots()
    ax.plot(times_matmul, label='Matmul')
    ax.plot(times_hadamard, label='FHT')
    #fig.show()
    fig.legend()
    ax.semilogy()
    ax.set_xlabel('log(d)')
    ax.set_ylabel('seconds')
    ax.set_title('Forward')
    fig.show()

    logger.info('Test timing backward:')

    times_matmul = []
    times_hadamard = []

    for power in range(2, 15):
        d = 2 ** power
        X = torch.randn(b, d, requires_grad=True)
        matrix_mult = torch.matmul(X, torch.from_numpy(scipy.linalg.hadamard(d)).float()).sum()
        t0 = time.time()
        _ = [torch.autograd.grad(matrix_mult, X, retain_graph=True) for _ in range(repetitions)]
        t_mean = (time.time() - t0) / repetitions
        times_matmul.append(t_mean)
        logger.info('D = %4d - Matmul   in %1.2e' % (d, t_mean))
        hadamard_mult = vardl.functional.HadamardTransform.apply(X).sum()
        t0 = time.time()
        _ = [torch.autograd.grad(hadamard_mult, X, retain_graph=True) for _ in range(repetitions)]
        t_mean = (time.time() - t0) / repetitions
        times_hadamard.append(t_mean)
        logger.info('D = %4d - Hadamard in %1.2e' % (d, t_mean))

    fig, ax = plt.subplots()
    ax.plot(times_matmul, label='Matmul')
    ax.plot(times_hadamard, label='FHT')
    #fig.show()
    fig.legend()
    ax.semilogy()
    ax.set_xlabel('log(d)')
    ax.set_ylabel('seconds')
    ax.set_title('Backward')
    plt.show()


if __name__ == '__main__':
    main()