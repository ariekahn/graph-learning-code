import pandas as pd
import os
import os.path as op
import sys
import argparse
import numpy as np
import subprocess
# import nibabel as nib
from nilearn.image import load_img, index_img

parser = argparse.ArgumentParser('Create BIDS-structured event data.')
parser.add_argument('scan_in', type=str, help='full path to scan file')
parser.add_argument('events_in', type=str, help='full path to events file')
parser.add_argument('n_trs', type=int, help='Number of TRs to discard from the start')
parser.add_argument('--add_suffix', type='store_true', help='Add a suffix to the filenames')
args = parser.parse_args()

scan_filename = op.abspath(args.scan)
events_filename = op.abspath(args.events)
n_trs = args.n_trs
add_suffix = args.add_suffix

print(f'Scan Filename: {scan_filename}')
print(f'Events Filename: {events_filename}')
print(f'# TRs: {n_trs}')
print(f'Add suffix: {add_suffix}')

if add_suffix:
    scan_out = scan_filename[:-7] + '_sliced.nii.gz'
    events_out = events_filename[:-4] + '_sliced.tsv'
else:
    scan_out = scan_filename
    events_out = events_filename

# Load the scan
# Get length of TR in seconds
img = load_img(scan_filename)
tr_length = img.header['pixdim'][4]
offset = n_trs * tr_length

# Slice the file
assert n_trs < img.shape(3)
img_slice = index_img(img, range(n_trs, img.shape[3]))

# Load and modify events
events = pd.read_csv(events_filename, sep='\t')
events['onset'] = events['onset'] - offset
events['onset'] = events['onset'].round(3)
assert np.all(events['onset'] > 0)

img_slice.to_filename(scan_out)
events.to_csv(events_out, sep='\t', index=False)
