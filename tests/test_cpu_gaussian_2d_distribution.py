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
sys.path.insert(0, '.')

import unittest  # noqa: E402
import torch
from vardl.distributions import Gaussian2DDistribution

class Gaussian2DDistributionTest(unittest.TestCase):



    def test_member_n(self):
        n = 10
        m = 5
        approx = 'factorized'
        dtype = torch.float32
        device = torch.device('cpu')
        distr = Gaussian2DDistribution(n, m, approx, dtype, device)
        self.assertTrue(distr.n == n)

    def test_member_m(self):
        n = 10
        m = 5
        approx = 'factorized'
        dtype = torch.float32
        device = torch.device('cpu')
        distr = Gaussian2DDistribution(n, m, approx, dtype, device)
        self.assertTrue(distr.m == m)

    def test_member_approx(self):
        n = 10
        m = 5
        approx = 'factorized'
        dtype = torch.float32
        device = torch.device('cpu')
        distr = Gaussian2DDistribution(n, m, approx, dtype, device)
        self.assertTrue(distr.approx == approx)

    def test_member_dtype(self):
        n = 10
        m = 5
        approx = 'factorized'
        dtype = torch.float32
        device = torch.device('cpu')
        distr = Gaussian2DDistribution(n, m, approx, dtype, device)
        self.assertTrue(distr.dtype == dtype)

    def test_member_device(self):
        n = 10
        m = 5
        approx = 'factorized'
        dtype = torch.float32
        device = torch.device('cpu')
        distr = Gaussian2DDistribution(n, m, approx, dtype, device)
        self.assertTrue(distr.device == device)


    def test_sample_mean(self):
        n = 1
        m = 1
        approx = 'factorized'
        dtype = torch.float32
        device = torch.device('cpu')
        distr = Gaussian2DDistribution(n, m, approx, dtype, device)

        sample_mean = distr.sample(10000000).mean()
        self.assertAlmostEqual(0, sample_mean.numpy(), delta=1e-3)


if __name__ == '__main__':
    unittest.main()
