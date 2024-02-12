import argparse
import os
import logging
from itertools import combinations, product
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.utils import parallel_backend
from nilearn.image import load_img, math_img, new_img_like

parser = argparse.ArgumentParser('Run Dimensionality analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('roi', type=str, choices=['loc-localized', 'postcentral-lh', 'postcentral-rh'])
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('n_threads', type=int, help='Number of threads')
args = parser.parse_args()

subject_id = args.subject_id
project_dir = args.project_dir
output_dir = args.output_dir
n_threads = args.n_threads
roi = args.roi

os.makedirs(output_dir, exist_ok=True)
# os.makedirs(f'{output_dir}/images', exist_ok=True)

space = 'MNI152NLin2009cAsym'

logging.basicConfig(filename=f'{output_dir}/dimensionality_{subject_id}_{roi}.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(f'Subject: {subject_id}')

fmriprep_dir = f'{project_dir}/derived/fmriprep'
lss_dir = f'{project_dir}/derived/feat_representation_lss'
localizer_dir = f'{project_dir}/derived/feat_localizer'
data_dir = f'{project_dir}/data'


# Create some helper functions for
# 1. Loading the T1 template
# 2. Loading the LOC template


def load_masked_template(subject_id):
    """
    Load the T1w template, and mask it

    derived/fmriprep/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz

    Usage:
    template = load_masked_template('GLS011')
    """
    # Load T1w template and mask
    template = load_img(f'{fmriprep_dir}/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
    template_mask = load_img(f'{fmriprep_dir}/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    template = math_img("img1 * img2", img1=template, img2=template_mask)
    return template


def load_aparc(subject_id):
    """
    Load Freesurfer aparc for localizer

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    aparc = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz')
    return aparc


def load_aparc_roi(left_ind, right_ind, subject_id, hemi):
    roi_str = ''
    if hemi == 'left':
        roi_str = f'img == {left_ind}'
    elif hemi == 'right':
        roi_str = f'img == {right_ind}'
    elif hemi == 'both':
        roi_str = f'(img == {left_ind}) | (img == {right_ind})'
    elif hemi == 'separate':
        roi_str = f'(img == {left_ind}) + 2*(img == {right_ind})'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    roi = math_img(roi_str, img=aparc)
    return roi


def load_benson_parc(subject_id):
    """
    Load Benson varea
    """
    varea = load_img(f'{project_dir}/derived/template_benson/benson14_varea/sub-{subject_id}/sub-{subject_id}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_benson14_varea.nii.gz')
    return varea


def load_benson_roi(subject_id, roi):
    """
    Load Benson varea roi

    roi : integer
        Value of 1-14
    """
    if roi < 1 or roi > 12:
        raise ValueError('ROI must be between 1 and 12')
    varea = load_benson_parc(subject_id)
    mask = math_img(f'img == {roi}', img=varea)
    return mask


def load_loc(subject_id, hemi='both'):
    """
    Load a mask for LOC

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-lateraloccipital = 1
    ctx-rh-lateraloccipital = 2

    We want 1011 ctx-lh-lateraloccipital and 2011 ctx-rh-lateraloccipital
    """
    return load_aparc_roi('1011', '2011', subject_id, hemi)


def load_entorhinal(subject_id, hemi='both'):
    """
    Load a mask for Entorhinal Cortex

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-entorhinal = 1
    ctx-rh-entorhinal = 2

    We want 1006 ctx-lh-entorhinal and 2006 ctx-rh-entorhinal
    """
    return load_aparc_roi('1006', '2006', subject_id, hemi)


def load_hippocampus(subject_id, hemi='both'):
    """
    Load a mask for Hippocampus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    Left-Hippocampus = 1
    Right-Hippocampus = 2

    We want 17 Left-Hippocampus and 53 Right-Hippocampus
    """
    return load_aparc_roi('17', '53', subject_id, hemi)


def load_postcentral(subject_id, hemi='both'):
    """
    Load a mask for Postcentral Gyrus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-postcentral = 1
    ctx-rh-postcentral = 2

    We want 1022 ctx-lh-postcentral and 2022 ctx-rh-postcentral
    """
    return load_aparc_roi('1022', '2022', subject_id, hemi)


def load_pericalcarine(subject_id, hemi='both'):
    """
    Load a mask for Pericalcarine Cortex

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-pericalcarine = 1
    ctx-rh-pericalcarine = 2

    We want 1021 ctx-lh-pericalcarine and 2021 ctx-rh-pericalcarine
    """
    return load_aparc_roi('1021', '2021', subject_id, hemi)


def load_cuneus(subject_id, hemi='both'):
    """
    Load a mask for Cuneus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-cuneus = 1
    ctx-rh-cuneus = 2

    We want 1005 ctx-lh-cuneus and 2005 ctx-rh-cuneus
    """
    return load_aparc_roi('1005', '2005', subject_id, hemi)


def load_lingual(subject_id, hemi='both'):
    """
    Load a mask for Lingual Cortex

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-lingual = 1
    ctx-rh-lingual = 2

    We want 1013 ctx-lh-lingual and 2013 ctx-rh-lingual
    """
    return load_aparc_roi('1013', '2013', subject_id, hemi)


def load_zstat_index(subject_id):
    """
    Loads zstat index where thresholded cluster voxels are labeled
    derived/localizer/clusters/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
    """
    img = load_img(f'{localizer_dir}/sub-{subject_id}/clusters/fwhm-5.0_zstat1_index.nii.gz')
    return img


def load_zstat(subject_id):
    """
    Loads zstat statistical map
    derived/localizer/zfiles/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
    """
    img = load_img(f'{localizer_dir}/sub-{subject_id}/zfiles/fwhm-5.0_zstat1.nii.gz')
    return img


# Skip pe 0, not included
n_excluded_pes = 1
n_pes = 60 - n_excluded_pes


def load_pes_lss(mask, subtract_mean=True, rescale=True, sort_by=None):
    """LS-S regression with one regressor for each event

       Run a separate GLM for each event

    """
    pes = []
    for run in range(1, 9):
        pes_run = []
        for event in range(n_excluded_pes, 60):
            pe = load_img(f'{lss_dir}/sub-{subject_id}/run-{run}/parameter_estimates/fwhm-5.0_event-{event}_pe1.nii.gz')
            # Load the zstat data for the mask
            data = pe.get_fdata()[np.where(mask.get_fdata())]
            if np.isnan(data).any():
                raise ValueError
            pes_run.append(data)
        pes_run = np.vstack(pes_run)
        pes.append(pes_run)
    pes = np.stack(pes)

    if sort_by is not None:
        pes = pes[:, :, sort_by]

    if subtract_mean:
        # Find and subtract the mean of each voxel, for each run
        # Take the mean over events, then re-add a single dimension, and subtract
        pes = pes - np.expand_dims(pes.mean(1), 1)

    # Find and mask unvarying voxels
    nonzero_voxels = (pes.std(1) > 0).all(0)
    pes = pes[:, :, nonzero_voxels]

    if rescale:
        # Rescale each voxel to 1
        pes = pes / np.expand_dims(pes.std(1), 1)

    if np.isnan(pes).any():
        raise ValueError

    # Reshape to single array
    pes = pes.reshape(pes.shape[0] * pes.shape[1], -1)

    return pes


# Nodes and blocks
logging.info('Loading nodes and blocks')

events = []
for run in range(1, 9):
    filename = f'{data_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_events.tsv'
    data = pd.read_csv(filename, sep='\t').drop(columns=['stim_file', 'trial_type'])
    data['steadystate'] = True
    data.loc[:n_excluded_pes - 1, 'steadystate'] = False
    data['block'] = run - 1
    events.append(data)
events = pd.concat(events).reset_index(drop=True)
events['subject'] = subject_id
nodes_lss = events[events['steadystate']]['node'].values
blocks_lss = events[events['steadystate']]['block'].values

subtract_mean = True
rescale = True
n_loc_features = 400

if roi == 'loc-localized':
    # Bold Mask
    bold_mask = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-{space}_desc-brain_mask.nii.gz')
    bold_mask = new_img_like(bold_mask, np.rint(bold_mask.get_fdata()) > 0)

    ctx_lateraloccipital = load_loc(subject_id)
    zstat_localizer = load_zstat(subject_id)

    # Load cluster assignments
    cluster_img = load_zstat_index(subject_id)
    # We're taking the top 3 clusters, since a few subjects have misidentified clusters
    n_clusters = cluster_img.get_fdata().max()
    mask_localizer = math_img(f'(img == {n_clusters}) | (img == {n_clusters - 1})', img=cluster_img)
    if subject_id in ('GLS024'):
        mask_localizer = math_img(f'(img == {n_clusters}) | (img == {n_clusters - 2})', img=cluster_img)
    elif subject_id in ('GLS017'):
        mask_localizer = math_img(f'(img == {n_clusters - 1}) | (img == {n_clusters - 2})', img=cluster_img)

    mask_loc = math_img('img1 & img2', img1=mask_localizer, img2=ctx_lateraloccipital)

    # Argsort zstats to order the voxels in importance
    zstat_masked = np.abs(zstat_localizer.get_fdata()[np.where(mask_loc.get_fdata())])
    argsort_zstat_masked = np.argsort(zstat_masked)[::-1]

    pes_lss = load_pes_lss(mask_loc, sort_by=argsort_zstat_masked, subtract_mean=subtract_mean, rescale=rescale)
    pes_lss = pes_lss[:, :n_loc_features]
elif roi == 'postcentral-lh':
    mask = load_postcentral(subject_id, hemi='left')
    pes_lss = load_pes_lss(mask, subtract_mean=subtract_mean, rescale=rescale)
elif roi == 'postcentral-rh':
    mask = load_postcentral(subject_id, hemi='right')
    pes_lss = load_pes_lss(mask, subtract_mean=subtract_mean, rescale=rescale)

max_iter = 400000
clf = LinearSVC(max_iter=max_iter)
gkf = LeaveOneGroupOut()

freq_factor_perms = np.array((1, 1, 1, 1, 4, 13, 13, 13, 17, 11, 7, 2, 1, 1, 1))
freq_factor_assignments = np.array((1, 1, 1, 1, 1, 1, 3, 7, 7, 13, 19, 41, 37, 11, 1))

for m in range(2, 16):
    logging.info(f'set size {m}')
    accuracies = []
    counts = np.zeros((15, 2))
    shape_combinations = combinations(range(15), m)
    freq_perm = freq_factor_perms[m - 1]
    freq_assignment = freq_factor_assignments[m - 1]
    for shape_ind, shape_combination in enumerate(shape_combinations):
        # logging.info(f'shape combination: {shape_combination}')
        if shape_ind % freq_perm == 0:
            accuracies_combination = []
            binary_assignments = product([0, 1], repeat=m - 1)
            for i, assignment in enumerate(binary_assignments):
                # logging.info(f'assignment: {assignment}')
                # Skip assignment of all zeros
                if (i > 0) and (i % freq_assignment == 0):
                    inds = np.isin(nodes_lss, shape_combination)
                    # For each node in m, replace it with its assignment
                    nodes_lss_binary = np.zeros_like(nodes_lss)
                    for j in range(m - 1):
                        nodes_lss_binary[nodes_lss == shape_combination[j]] = assignment[j]
                        counts[shape_combination[j], assignment[j]] += 1
                    counts[shape_combination[-1], 0] += 1
                    X = pes_lss[inds, :]
                    Y = nodes_lss_binary[inds]
                    blocks = blocks_lss[inds]
                    with parallel_backend('threading', n_jobs=n_threads):
                        predictions = cross_val_predict(clf, X, Y, cv=gkf, groups=blocks, verbose=False)
                    accuracies_combination.append((np.array(predictions) == Y).mean())
            accuracies.append(accuracies_combination)
    accuracies_arr = np.stack(accuracies)
    np.save(f'{output_dir}/sub-{subject_id}_{roi}_accuracy-binary_m-{m}.npy', accuracies_arr, allow_pickle=False)
    np.save(f'{output_dir}/sub-{subject_id}_{roi}_accuracy-binary_m-{m}_counts.npy', counts, allow_pickle=False)
