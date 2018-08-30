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
import vardl  # noqa: F401
import vardl.layers  # noqa: F401
import vardl.likelihoods  # noqa: F401
import vardl.logger  # noqa: F401
import vardl.models  # noqa: F401
import vardl.trainer  # noqa: F401
import vardl.utils  # noqa: F401


class VardlImportTest(unittest.TestCase):

    def test_import_vardl(self):
        self.assertTrue('vardl' in sys.modules)

    def test_import_vardl_initializer(self):
        self.assertTrue('vardl.likelihoods' in sys.modules)

    def test_import_vardl_layers(self):
        self.assertTrue('vardl.layers' in sys.modules)

    def test_import_vardl_logger(self):
        self.assertTrue('vardl.logger' in sys.modules)

    def test_import_vardl_trainer(self):
        self.assertTrue('vardl.trainer' in sys.modules)

    def test_import_vardl_utils(self):
        self.assertTrue('vardl.utils' in sys.modules)


if __name__ == '__main__':
    unittest.main()
