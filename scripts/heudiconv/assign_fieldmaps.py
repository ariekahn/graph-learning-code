#EXAMPLE USAGE
#python assign_fieldmaps_to_IntendedFor_field.py ${sub} ${session}
#neither subject nor session needs the BIDS prefix (i.e. "CBPD" not "sub-CBPD")

import sys
import json
import bisect
import os
import stat
import logging
from glob import glob
from os.path import join, splitext, abspath
from bids import BIDSLayout
from dateutil.parser import parse
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser('Assign fieldmaps to IntendedFor field.')
parser.add_argument('bids_dir', type=str, help='BIDS directory base')
parser.add_argument('subject', type=str, help='Subject (without sub- prefix)')
parser.add_argument('session', type=int, help='Session number (without ses- prefix)')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing IntendedFor field')
parser.add_argument('--dry_run', action='store_true', help="Don't write changes. Implies verbose")
parser.add_argument('--verbose', action='store_true', help='Show intended changes')
parser.add_argument('--database_path', type=str, help='database path for pybids')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

subject = args.subject
session = args.session
overwrite = args.overwrite
bids_dir = abspath(args.bids_dir)
database_path = abspath(args.database_path)
verbose = args.verbose
dry_run = args.dry_run
if dry_run:
    verbose = True

logging.info(f'Subject: {subject}')
logging.info(f'Session: {session}')
logging.info(f'BIDS Dir: {args.bids_dir}')
logging.info(f'Overwrite: {args.overwrite}')

if database_path:
    logging.info(f'Database Path: {database_path}')

layout = BIDSLayout(bids_dir, database_path=database_path)

def bids_path(scan):
    """Return bids-valid path starting at the root of the layout"""
    # Note! This will break if runs are are padded with 0, e.g. 01 instead of 1
    # See: https://github.com/bids-standard/pybids/issues/381
    path = layout.build_path(scan, absolute_paths=False)
    return path.split('/', 1)[-1]

"""
Steps:

    1) Find the date of each functional and fieldmap scan
    2) For each functional scan, find the fieldmap closest in time before it
    3) Assign that functional scan to that fieldmap

"""


# Get a list of tuples of date, path for each scan
scan_niis = layout.get(subject=subject, session=session, datatype='func', extension='.nii.gz')
scan_niis += layout.get(subject=subject, session=session, datatype='dwi', extension='.nii.gz')
scan_pairs = [(x.get_metadata()['AcquisitionTime'], x) for x in scan_niis]
scan_pairs.sort(key=lambda x: x[0])

assert len(scan_niis) > 0, 'No scans!'
logging.info(f'Found {len(scan_niis)} scans')

# Dictionary of intendedfor values, indexed by fieldmap
intendedfor = defaultdict(list)

# Run field map directions independently
for direction in ['AP', 'PA']:
    fmap_jsons = layout.get(subject=subject, session=session, datatype='fmap',
            extension='.json', direction=direction)
    assert len(fmap_jsons) > 0, 'No fieldmaps!'
    logging.info(f'Found {len(fmap_jsons)} {direction} fieldmaps')
    # Date, Scan tuples
    fmap_pairs = [(x.get_dict()['AcquisitionTime'], x) for x in fmap_jsons]
    fmap_pairs.sort(key=lambda x: x[0])
    fmap_dts = [x[0] for x in fmap_pairs]
    fmap_names = [x[1] for x in fmap_pairs]

    # Find the most immediate fieldmap before each scan
    for dt, name in scan_pairs:
        idx = bisect.bisect_left(fmap_dts, dt) - 1
        intendedfor[fmap_names[idx]].append(bids_path(name))

if verbose:
    for fmap, vals in intendedfor.items():
        logging.info(f'{fmap}:')
        logging.info(vals)
if not dry_run:
    # Write out the new jsons
    for fmap in intendedfor.keys():
        with open(fmap.path, 'r') as fi:
            data = json.load(fi)
        # No overwriting, for now
        if overwrite or ('IntendedFor' not in data.keys()):
            logging.info(f'writing {fmap.filename}')
            data['IntendedFor'] = intendedfor[fmap]
            # Make sure it's writable
            # By default, heudiconv seems to make sidecars read-only
            os.chmod(fmap.path, stat.S_IRUSR | stat.S_IWUSR)
            with open(fmap.path, 'w') as fo:
                # Make sure we indent the file so it's still readable
                json.dump(data, fo, indent=2)
            os.chmod(fmap.path, stat.S_IRUSR)
        else:
            logging.info(f'found existing data in {fmap.filename}')
