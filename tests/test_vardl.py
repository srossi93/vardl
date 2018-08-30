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
