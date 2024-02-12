import argparse
import logging
import numpy as np
from nilearn.image import load_img, new_img_like

parser = argparse.ArgumentParser('Run MVPA analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('--zscored', action='store_true', help='Use zscored RDM computations')
args = parser.parse_args()

subject_id = args.subject_id
project_dir = args.project_dir
output_dir = args.output_dir
zscored = args.zscored

logging.basicConfig(filename=f'{output_dir}/rsa_consistency_{subject_id}.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(f'Subject: {subject_id}')

data_dir = f'{project_dir}/data'

# for subject in subjects:
if zscored:
    run_files = [f'{project_dir}/derived/cosmo_make_rdm_searchlight_per_run/sub-{subject_id}/results/sub-{subject_id}_run-{run}_rdm_nvox-100_searchlight-lss-zfiles.nii.gz' for run in range(1, 9)]
else:
    run_files = [f'{project_dir}/derived/cosmo_make_rdm_searchlight_per_run/sub-{subject_id}/results/sub-{subject_id}_run-{run}_rdm_nvox-100_searchlight-lss-zfiles-zscored.nii.gz' for run in range(1, 9)]

runs = [load_img(f) for f in run_files]

n_voxels = 81 * 96 * 81
n_runs = 8

data_flat = np.stack([runs[i].get_fdata().reshape(n_voxels, 105) for i in range(8)])

results = np.zeros(n_voxels)
for run_ind in range(n_runs):
    x1 = np.concatenate([1 - data_flat[0:run_ind, :, :], 1 - data_flat[run_ind + 1:, :, :]]).mean(0)
    x2 = 1 - data_flat[run_ind, :, :]
    for voxel in range(n_voxels):
        results[voxel] += np.corrcoef(x1[voxel, :], x2[voxel, :])[0][1]
results /= n_runs

results_img = new_img_like(runs[0], np.nan_to_num(results.reshape(81, 96, 81)))
if zscored:
    results_img.to_filename(f'{output_dir}/sub-{subject_id}_across-run_lower-bound_zscored.nii.gz')
else:
    results_img.to_filename(f'{output_dir}/sub-{subject_id}_across-run_lower-bound.nii.gz')
