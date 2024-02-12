import argparse
import logging
import os.path
import pandas as pd
from nilearn.image import load_img
from nilearn.masking import apply_mask
from nipype.algorithms.confounds import is_outlier
import matplotlib.pyplot as plt

"""
Creates diagnostic information about non-steady and outlying TRs. Run after fMRIPrep.

Creates a 1d timeseries by either taking the mean or the max of each volume, and
feeds those into `is_outlier` from nipype to identify outlying volumes.
"""

parser = argparse.ArgumentParser('Summarize outlying TRs')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
args = parser.parse_args()

output_dir = args.output_dir
subject_id = args.subject_id
project_dir = args.project_dir
fmriprep_dir = f'{project_dir}/derived/fmriprep/sub-{subject_id}'

logging.basicConfig(filename=f'{output_dir}/sub-{subject_id}_outlying_trs.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(f'Subject: {subject_id}')

space = 'T1w'

runs = {
    'rest-1': f'ses-1/func/sub-{subject_id}_ses-1_task-rest_acq-pre_run-1',
    'rest-2': f'ses-1/func/sub-{subject_id}_ses-1_task-rest_acq-pre_run-2',
    'rest-3': f'ses-1/func/sub-{subject_id}_ses-1_task-rest_acq-post_run-1',
    'rest-4': f'ses-1/func/sub-{subject_id}_ses-1_task-rest_acq-post_run-2',
    'learning-1': f'ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-1',
    'learning-2': f'ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-2',
    'learning-3': f'ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-3',
    'learning-4': f'ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-4',
    'learning-5': f'ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-5',
    'rep-1': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-1',
    'rep-2': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-2',
    'rep-3': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-3',
    'rep-4': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-4',
    'rep-5': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-5',
    'rep-6': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-6',
    'rep-7': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-7',
    'rep-8': f'ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-8',
}

i = 0
n_runs = len(runs)

outliers = []

f, ax = plt.subplots(n_runs * 2, 2, figsize=(20, 8 * n_runs))
for runname, runpath in runs.items():
    logging.info(runname)
    logging.info(runpath)
    mask_path = f'{fmriprep_dir}/{runpath}_space-{space}_desc-brain_mask.nii.gz'
    bold_path = f'{fmriprep_dir}/{runpath}_space-{space}_desc-preproc_bold.nii.gz'
    if os.path.exists(bold_path):
        mask = load_img(mask_path)
        scan = load_img(bold_path)
        data = apply_mask(scan, mask)

        tr_avg = data.mean(1)
        flagged_avg = is_outlier(tr_avg)
        logging.info(f'avg: {flagged_avg}')

        tr_max = data.max(1)
        flagged_max = is_outlier(tr_max)
        logging.info(f'max: {flagged_max}')

        outliers.append(dict(subject=subject_id,
                             run=runname,
                             dummy_avg=flagged_avg,
                             dummy_max=flagged_max))

        ax[i, 0].plot(tr_avg[:])
        ax[i, 0].axhline(tr_avg[flagged_avg - 1])
        ax[i, 0].plot(tr_avg[:flagged_avg], 'r', label=f'Average Value sub-{subject_id}/{runname}, Flagged TRs ({flagged_avg})')
        ax[i, 0].set_ylabel('Amplitude')
        ax[i, 0].legend()

        ax[i, 1].plot(tr_avg[:50])
        ax[i, 1].axhline(tr_avg[flagged_avg - 1])
        ax[i, 1].plot(tr_avg[:flagged_avg], 'r', label=f'Average Value sub-{subject_id}/{runname}, Flagged TRs ({flagged_avg})')
        ax[i, 1].set_ylabel('Amplitude')
        ax[i, 1].legend()

        i += 1

        ax[i, 0].plot(tr_max[:])
        ax[i, 0].axhline(tr_max[flagged_max - 1])
        ax[i, 0].plot(tr_max[:flagged_max], 'r', label=f'Max Value sub-{subject_id}/{runname}, Flagged TRs ({flagged_max})')
        ax[i, 0].set_ylabel('Amplitude')
        ax[i, 0].legend()

        ax[i, 1].plot(tr_max[:50])
        ax[i, 1].axhline(tr_max[flagged_max - 1])
        ax[i, 1].plot(tr_max[:flagged_max], 'r', label=f'Max Value sub-{subject_id}/{runname}, Flagged TRs ({flagged_max})')
        ax[i, 1].set_ylabel('Amplitude')
        ax[i, 1].legend()

        i += 1
    else:  # file was missing
        outliers.append(dict(subject=subject_id,
                             run=runname,
                             dummy_avg=None,
                             dummy_max=None))

        i += 2

f.tight_layout()
f.savefig(f'{output_dir}/sub-{subject_id}_nonsteady.pdf')

outlier_df = pd.DataFrame(outliers)
outlier_df.to_csv(f'{output_dir}/sub-{subject_id}_outliers.tsv', sep='\t')
