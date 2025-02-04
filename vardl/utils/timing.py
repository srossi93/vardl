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

from time import time
from functools import wraps
import logging


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        logger = logging.getLogger(__name__)
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.info('func:%r args:[%r] took: %2.4f sec' %
                     (f.__name__, args[0], te - ts))
        return result

    return wrap