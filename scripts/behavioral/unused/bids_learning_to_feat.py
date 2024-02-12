import argparse
import logging
import os.path as op
import pandas as pd

parser = argparse.ArgumentParser('Create FSL-style learning node event files')
parser.add_argument('bids_dir', type=str, help='BIDS directory base')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('n_skip', type=int, help='Number of TRs to skip')
parser.add_argument('tr_length', type=float, help='TR length in seconds')
parser.add_argument('--dry_run', action='store_true', help="Don't write changes.")
parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output.')
args = parser.parse_args()

subject_id = args.subject_id
bids_dir = op.abspath(args.bids_dir)
output_dir = op.abspath(args.output_dir)
dry_run = args.dry_run
n_skip = args.n_skip
tr_length = args.tr_length

if args.verbose:
    logging.basicConfig(level=logging.INFO)

logging.info(f'Subject: {subject_id}')
logging.info(f'BIDS Dir: {bids_dir}')
logging.info(f'Output Dir: {output_dir}')
logging.info(f'TRs to skip: {n_skip}')
logging.info(f'TR Length: {tr_length}')

scan_dir = f'{bids_dir}/sub-{subject_id}/ses-1/func'

for run in range(1,6):
	events = pd.read_csv(f'{scan_dir}/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_events.tsv', sep='\t')
	events['value'] = 1
	for node in range(15):
		shapes = events.loc[(events.trial_type == f'node_{node}'), ['onset', 'duration', 'value']]
		# account for dropped TRs
		shapes['onset'] = (shapes['onset'] - (tr_length * n_skip)).round(3)
		filename = f'{output_dir}/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_node-{node}.txt'
		logging.info(f'Writing {filename}')
		if not args.dry_run:
			shapes.to_csv(filename, sep=' ', header=False, index=False)
