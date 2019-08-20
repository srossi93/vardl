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

import argparse
import bunch
import json


class Experiment:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Experiment')
        self.parser.add_argument('-c', '--config', default=None, type=str,
                            help='config file path (default: None)')
        pass

    def add_config(self):
        """
        It should add arguments to the parser
        """
        pass

    def load_config(self):
        def create_parser():
            parser = argparse.ArgumentParser()

            g = parser.add_argument_group('Configuration')
            g.add_argument('--config',
                type=argparse.FileType(mode='r'))
            return parser

        def parse_args(parser):
            args = parser.parse_args()
            if args.config_file:
                data = json.load(args.config_file)
                delattr(args, 'config')
                arg_dict = args.__dict__
                for key, value in data.items():
                    if isinstance(value, list):
                        for v in value:
                            arg_dict[key].append(v)
                    else:
                        arg_dict[key] = value
            return args

        return parse_args(create_parser())

    def save_config(self):
        pass

    def setup_loggers(self):
        pass

    def load_dataset(self):
        pass

    def load_model(self):
        pass

    def save_reults(self):
        pass

    def __call__(self, *args, **kwargs):
        self.load_config()


if __name__ == '__main__':
    e = Experiment()
    e()
