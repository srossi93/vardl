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
import vardl
from vardl.distributions import Gaussian2DDistribution

class Gaussian2DDistributionTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        vardl.utils.set_seed(0)

    @classmethod
    def setUpClass(cls):
        cls.n = 1
        cls.m = 1
        cls.approx = 'factorized'
        cls.dtype = torch.float32
        cls.device = torch.device('cpu')
        cls.distr = Gaussian2DDistribution(cls.n, cls.m, cls.approx, cls.dtype, cls.device)

    def test_member_n(self):
        self.assertTrue(self.distr.n == self.n)

    def test_member_m(self):
        self.assertTrue(self.distr.m == self.m)

    def test_member_approx(self):
        self.assertTrue(self.distr.approx == self.approx)

    def test_member_dtype(self):
        self.assertTrue(self.distr.dtype == self.dtype)

    def test_member_device(self):
        self.assertTrue(self.distr.device == self.device)


    def test_sample_mean(self):
        sample_mean = self.distr.sample(10000000).mean(dim=0)
        self.assertAlmostEqual(self.distr.mean, sample_mean, delta=1e-3)

    def test_sample_variance(self):
        sample_var = self.distr.sample(10000000).var()
        self.assertAlmostEqual(self.distr.logvars.exp(), sample_var, delta=1e-3)

    def test_optimization_true(self):
        self.distr.optimize(True)
        for param in self.distr.parameters():
            self.assertTrue(param.requires_grad)

    def test_optimization_false(self):
        self.distr.optimize(False)
        for param in self.distr.parameters():
            self.assertFalse(param.requires_grad)

if __name__ == '__main__':
    unittest.main()
