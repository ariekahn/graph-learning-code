import argparse
import logging
import subprocess
import os.path as op
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('Split the rapid learning section of a day-one session')
parser.add_argument('bids_dir', type=str, help='BIDS directory base')
parser.add_argument('fmriprep_dir', type=str, help='fmriprep directory base')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('-d', '--dry_run', action='store_true', help="Don't write changes.")
parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output.')
args = parser.parse_args()

bids_dir = op.abspath(args.bids_dir)
subject_id = args.subject_id
fmriprep_dir = op.abspath(args.fmriprep_dir)

if args.verbose:
    logging.basicConfig(level=logging.INFO)

logging.info(f'Subject: {subject_id}')
logging.info(f'fMRIPrep Dir: {fmriprep_dir}')
logging.info(f'BIDS Dir: {bids_dir}')

tr_length = 0.8  # length of a TR in ms
extra_trs = 5  # How many TRs to buffer with

# We're going to read in the events file, find the last event in the learning phase,
# and add a small buffer onto the end. We then will use fslroi to create a subset of the file

for run in range(1, 6):
    # Figure out how many TRs we want to keep
    events_file = f'{bids_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_events.tsv'
    events = pd.read_csv(events_file, sep='\t')
    last_event = events.iloc[-1]
    last_event_end = last_event.onset + last_event.duration

    # Divide number of seconds by the TR length, then add a few extra TRs
    n_trs = int(np.ceil(last_event_end / tr_length)) + extra_trs

    # Original file:
    # Run fslroi 0 num_trs_to_keep
    original_file = f'{fmriprep_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold_masked.nii.gz'
    split_file = f'{fmriprep_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold_masked_split-learning.nii.gz'

    logging.info(f'fslroi {original_file} {split_file} 0 {n_trs}')
    if not args.dry_run:
        subprocess.run(['fslroi', original_file, split_file, "0", str(n_trs)])

    confound_file = f'{fmriprep_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_desc-confounds_regressors.tsv'
    confounds = pd.read_csv(confound_file, sep='\t')
    confounds_split = confounds.iloc[0:n_trs]
    confounds_split_file = f'{fmriprep_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_desc-confounds_regressors_split-learning.tsv'
    logging.info(f'writing {confounds_split_file}')
    if not args.dry_run:
        confounds_split.to_csv(confounds_split_file, sep='\t', index=False, na_rep='n/a')
