import argparse
import logging
import os.path as op
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('Create FSL-style motion confound event files')
parser.add_argument('fmriprep_dir', type=str, help='fmriprep directory base')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('n_skip', type=int, help='Number of TRs to skip')
parser.add_argument('-d', '--dry_run', action='store_true', help="Don't write changes.")
parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output.')
args = parser.parse_args()

subject_id = args.subject_id
fmriprep_dir = op.abspath(args.fmriprep_dir)
output_dir = op.abspath(args.output_dir)
n_skip = args.n_skip

if args.verbose:
    logging.basicConfig(level=logging.INFO)

logging.info(f'Subject: {subject_id}')
logging.info(f'fMRIPrep Dir: {fmriprep_dir}')
logging.info(f'Output Dir: {output_dir}')
logging.info(f'TRs to skip: {n_skip}')

motion_params =['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
                'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
                'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2',
                'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
                'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
                'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

scan_1_dir = f'{fmriprep_dir}/sub-{subject_id}/ses-1/func'
scan_2_dir = f'{fmriprep_dir}/sub-{subject_id}/ses-2/func'

# For learning scans
for run in range(1,6):
    filename = f'{scan_1_dir}/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_desc-confounds_regressors_split-learning.tsv'
    confounds = pd.read_csv(filename, sep='\t')

    # Orthogonalize motion confounds to prevent some numeric errors
    motion = confounds[motion_params]
    motion = motion.fillna(0)
    motion_qr, _ = np.linalg.qr(motion.values)
    for i, m in enumerate(motion_params):
        confounds[m] = motion_qr[:,i]

    outliers = confounds.columns[confounds.columns.str.startswith('motion_outlier')].values
    nonsteady = confounds.columns[confounds.columns.str.startswith('non_steady_state_outlier')].values
    confound_names = np.concatenate([motion_params, outliers, nonsteady])
    subset = confounds.loc[n_skip:, confound_names]

    # Remove any columns that are zero after the volumes we're skipping
    subset = subset.loc[:,subset.abs().sum() > 0]

    # Remove duplicate outliers/non-steady-state
    for outlier in outliers:
        # Make sure we didn't already eliminate it
        # (can happen with the split data)
        if outlier in subset.columns:
            vol = np.where(subset[outlier])[0][0]
            if f'non_steady_state_outlier{vol:02}' in nonsteady:
                subset = subset.drop(outlier, 1)

    out_name = f'{output_dir}/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_desc-confounds_regressors_split-learning.txt'
    desc_name = f'{output_dir}/sub-{subject_id}_ses-1_task-graphlearning_run-{run}_desc-confounds_regressors_split-learning_desc.txt'
    logging.info(f'Writing {out_name}')
    if not args.dry_run:
        subset.to_csv(out_name, sep=' ', index=False, header=False)
        with open(desc_name, 'w') as f:
            for name in confound_names:
                f.write(f'{name}\n')

# For representation scans
for run in range(1,9):
    filename = f'{scan_2_dir}/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_desc-confounds_regressors.tsv'
    confounds = pd.read_csv(filename, sep='\t')

    # Orthogonalize motion confounds to prevent some numeric errors
    motion = confounds[motion_params]
    motion = motion.fillna(0)
    motion_qr, _ = np.linalg.qr(motion.values)
    for i, m in enumerate(motion_params):
        confounds[m] = motion_qr[:,i]

    outliers = confounds.columns[confounds.columns.str.startswith('motion_outlier')].values
    nonsteady = confounds.columns[confounds.columns.str.startswith('non_steady_state_outlier')].values
    confound_names = np.concatenate([motion_params, outliers, nonsteady])
    subset = confounds.loc[n_skip:, confound_names]

    # Remove any columns that are zero after the volumes we're skipping
    subset = subset.loc[:,subset.abs().sum() > 0]

    # Remove duplicate outliers/non-steady-state
    for outlier in outliers:
        vol = np.where(subset[outlier])[0][0]
        if f'non_steady_state_outlier{vol:02}' in nonsteady:
            subset = subset.drop(outlier, 1)

    out_name = f'{output_dir}/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_desc-confounds_regressors.txt'
    desc_name = f'{output_dir}/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_desc-confounds_regressors_desc.txt'
    logging.info(f'Writing {out_name}')
    if not args.dry_run:
        subset.to_csv(out_name, sep=' ', index=False, header=False)
        with open(desc_name, 'w') as f:
            for name in confound_names:
                f.write(f'{name}\n')

# For localizer scan
filename = f'{scan_2_dir}/sub-{subject_id}_ses-2_task-graphlocalizer_desc-confounds_regressors.tsv'
confounds = pd.read_csv(filename, sep='\t')

# Orthogonalize motion confounds to prevent some numeric errors
motion = confounds[motion_params]
motion = motion.fillna(0)
motion_qr, _ = np.linalg.qr(motion.values)
for i, m in enumerate(motion_params):
    confounds[m] = motion_qr[:,i]

outliers = confounds.columns[confounds.columns.str.startswith('motion_outlier')].values
nonsteady = confounds.columns[confounds.columns.str.startswith('non_steady_state_outlier')].values
confound_names = np.concatenate([motion_params, outliers, nonsteady])
subset = confounds.loc[n_skip:, confound_names]

# Remove any columns that are zero after the volumes we're skipping
subset = subset.loc[:,subset.abs().sum() > 0]

# Remove duplicate outliers/non-steady-state
for outlier in outliers:
    vol = np.where(subset[outlier])[0][0]
    if f'non_steady_state_outlier{vol:02}' in nonsteady:
        subset = subset.drop(outlier, 1)

out_name = f'{output_dir}/sub-{subject_id}_ses-2_task-graphlocalizer_desc-confounds_regressors.txt'
desc_name = f'{output_dir}/sub-{subject_id}_ses-2_task-graphlocalizer_desc-confounds_regressors_desc.txt'
logging.info(f'Writing {out_name}')
if not args.dry_run:
    subset.to_csv(out_name, sep=' ', index=False, header=False)
    with open(desc_name, 'w') as f:
        for name in confound_names:
            f.write(f'{name}\n')
