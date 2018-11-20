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
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from vardl.utils import setup_logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import time
import humanize
import argparse
import os


logger = setup_logger('vardl', logging_path='/tmp/vardl')


def read_tbevents(filepath) -> EventAccumulator:
    event_acc = EventAccumulator(filepath)
    logger.warning('Starting loading the event file in %s. Might take a while' % (filepath))
    t0 = time.time()
    event_acc.Reload()
    t_diff = time.time() - t0
    logger.info('Loading from %s completed in %s' % (filepath, humanize.naturaldelta(t_diff)))
    return event_acc


def save_tag(event_acc: EventAccumulator, tag, outdir, ext = '.csv'):
    raw_data = np.array(event_acc.Scalars(tag))
    filename = outdir + tag.replace('/', '_') + ext
    if ext == '.npy':
        np.save(filename, raw_data)
    else:
        np.savetxt(filename, raw_data, header='walltime step value')
    logger.info('Tag %s saved in %s (%s)' % (tag, filename,
                                             humanize.naturalsize(os.path.getsize(filename))))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help='File path to Tensorboard event file')
    parser.add_argument('--outdir', help='Directory path to output files')
    args = parser.parse_args()

    # Prepare the event accumulator
    event_acc = read_tbevents(args.filepath)
    logger.info('Tags found for scalars: %s' % ', '.join(event_acc.Tags()['scalars']))

    for tag in event_acc.Tags()['scalars']:
        save_tag(event_acc, tag, args.outdir, '.npy')

    for tag in event_acc.Tags()['scalars']:
        save_tag(event_acc, tag, args.outdir, '.csv')


if __name__ == '__main__':
    main()
